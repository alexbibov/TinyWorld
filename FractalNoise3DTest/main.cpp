#define DEFAULT_WINDOW_WIDTH 1024
#define DEFAULT_WINDOW_HEIGHT 768

#include <cstdlib>
#include <AntTweakBar.h>

#include "../TinyWorld/Screen.h"
#include "../TinyWorld/SSFilter.h"
#include "../TinyWorld/CompleteShaderProgram.h"
#include "../TinyWorld/TextureUnitBlock.h"
#include "../TinyWorld/ImmutableTexture2D.h"
#include "../TinyWorld/ImmutableTexture3D.h"
#include "../TinyWorld/VectorTypes.h"
#include "../TinyWorld/QuaternionTypes.h"
#include "../TinyWorld/MatrixTypes.h"
#include "../TinyWorld/AbstractProjectingDevice.h"
#include "../TinyWorld/FractalNoise.h"
#include "../TinyWorld/KTXTexture.h"
#include "../TinyWorld/TransparentBox.h"
#include "../TinyWorld/Framebuffer.h"
#include "../TinyWorld/SSFilter_HDRBloom.h"
#include "../TinyWorld/Light.h"


using namespace tiny_world;



//Class implementing the principal rendering context
class RenderContext final
{
private:
	static RenderContext* p_myself;
	Screen& screen;
	vec2 v2ScreenRatio;
	PerspectiveProjectingDevice main_camera;
	FractalNoise3D fractal_noise;
	ImmutableTexture3D noise_map;
	Framebuffer canvas;
	ImmutableTexture2D canvas_color_texture;
	ImmutableTexture2D canvas_depth_texture;
	ImmutableTexture2D canvas_bloom_texture;
	SSFilter_HDRBloom hdr_bloom_ss_filter;
	DirectionalLight light;
	TransparentBox transparent_box;

	vec3 v3NoiseLocation;
	float noise_rotation[4];
	float light_source_direction[3];

	TwBar* p_control_bar;


	
	RenderContext(Screen& output_screen) : screen{ output_screen },
		v2ScreenRatio{ static_cast<float>(DEFAULT_WINDOW_WIDTH) / std::max(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT),
		static_cast<float>(DEFAULT_WINDOW_HEIGHT) / std::max(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT) },
		main_camera{ "MainCamera", v2ScreenRatio.x, v2ScreenRatio.y, 1.0f, 1000.0f },
		fractal_noise{ 128, 128, 64, 16, 16, 16, 3 }, noise_map{ "NoiseMap3D" }, canvas{ "MainCanvas" },
		canvas_color_texture{ "CanvasColorTexture" }, canvas_depth_texture{ "CanvasDepthTexture" }, canvas_bloom_texture{ "CanvasBloomTexture" },
		light{ "MainSceneLight" }, transparent_box{ "NoisyBox" }, v3NoiseLocation{ 0 }
	{
		noise_rotation[0] = 0; noise_rotation[1] = 0; noise_rotation[2] = 0; noise_rotation[3] = 1;
		light_source_direction[0] = 1; light_source_direction[1] = 1; light_source_direction[2] = -1;

		//Setup screen
		output_screen.setClearColor(vec4{ 0 });
		output_screen.applyOpenGLContextSettings();
		
		//Setup main canvas framebuffer
		canvas.setClearColor(vec4{ 0, 0, 0, 1 });
		canvas.defineViewport(0, 0, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);
		canvas.setCullTestEnableState(false);
		canvas.setDepthTestEnableState(false);
		canvas.setDepthBufferClampFlag(true);
		canvas_color_texture.allocateStorage(1, 1, TextureSize{ DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT, 1 }, InternalPixelFormat::SIZED_FLOAT_RGBA32);
		canvas_depth_texture.allocateStorage(1, 1, TextureSize{ DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT, 1 }, InternalPixelFormat::SIZED_DEPTH32);
		canvas_bloom_texture.allocateStorage(1, 1, TextureSize{ DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT, 1 }, InternalPixelFormat::SIZED_FLOAT_RGBA32);
		canvas.attachTexture(FramebufferAttachmentPoint::COLOR0, FramebufferAttachmentInfo{ 0, 0, &canvas_color_texture });
		canvas.attachTexture(FramebufferAttachmentPoint::COLOR1, FramebufferAttachmentInfo{ 0, 0, &canvas_bloom_texture });
		canvas.attachTexture(FramebufferAttachmentPoint::DEPTH, FramebufferAttachmentInfo{ 0, 0, &canvas_depth_texture });
		canvas.attachRenderer(renderToCanvas);
		canvas.applyOpenGLContextSettings();

		//Setup HDR-Bloom filter
		hdr_bloom_ss_filter.defineColorTexture(canvas_color_texture);
		hdr_bloom_ss_filter.defineBloomTexture(canvas_bloom_texture);
		hdr_bloom_ss_filter.setBloomImpact(0.3f);
		hdr_bloom_ss_filter.initialize();

		//Setup the main camera
		main_camera.setLocation(vec3{ 0, 0, 25 });

		//Setup the scene light
		light.setDirection(vec3{ 1, 1, 1 });
		light.setColor(vec3{ 1, 1, 1 });

		//Generate noise map
		fractal_noise.setContinuity(true);
		fractal_noise.setEvolutionRate(0.3f);
		fractal_noise.generateNoiseMap();
		noise_map = fractal_noise.retrieveNoiseMap();

		//Setup the transparent box that contains the noise
		transparent_box.setDimensions(vec3{ 10 });
		transparent_box.installPointCloud(noise_map);
		transparent_box.addLightSourceDirection(light);
		transparent_box.setScreenSize(uvec2{ DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT });
		transparent_box.setBloomMinimalThreshold(0.4f);
		transparent_box.setBloomMaximalThreshold(1.0f);
		transparent_box.setBloomIntensity(1.7f);
		transparent_box.selectRenderingMode(TW_RENDERING_MODE_RAY_CAST_GAS);
		transparent_box.useColormap(false);
		transparent_box.setMediumUniformColor(vec3{ 1.0f, 1.0f, 1.0f });
		transparent_box.setSolidAngle(3);

		//Attach GLFW events
		glfwSetMouseButtonCallback(screen, onMouseClick);
		glfwSetCursorPosCallback(screen, onCursorPositionChange);
		glfwSetScrollCallback(screen, onScrollCallback);
		glfwSetKeyCallback(screen, onKeyPress);
		glfwSetCharCallback(screen, onCharInput);

		//Initialize control bar
		TwInit(TwGraphAPI::TW_OPENGL_CORE, nullptr);
		std::pair<uint32_t, uint32_t> screenSize = screen.getScreenSize();
		TwWindowSize(screenSize.first, screenSize.second);
		p_control_bar = TwNewBar("Control panel");
		TwAddVarRW(p_control_bar, "NoiseRotation", TW_TYPE_QUAT4F, noise_rotation, " label='Rotation' ");
		TwAddVarRW(p_control_bar, "LightDirection", TW_TYPE_DIR3F, light_source_direction, " label='Light source direction' ");
	}

	~RenderContext()
	{
		TextureUnitBlock::reset();
		p_myself = nullptr;
		TwTerminate();
	}


	//Executes when rendering to the canvas should be performed
	static void renderToCanvas(Framebuffer& canvas)
	{
		p_myself->fractal_noise.generateNoiseMap();
		canvas.clearBuffers(BufferClearTarget::COLOR_DEPTH);

		p_myself->transparent_box.resetObjectRotation();
		p_myself->transparent_box.applyRotation(quaternion{ p_myself->noise_rotation[3], p_myself->noise_rotation[0], p_myself->noise_rotation[1], p_myself->noise_rotation[2] }, RotationFrame::LOCAL);
		p_myself->light.setDirection(vec3{ p_myself->light_source_direction[0], p_myself->light_source_direction[1], p_myself->light_source_direction[2] });
		p_myself->transparent_box.applyViewProjectionTransform(p_myself->main_camera);
		for (uint32_t i = 0; i < p_myself->transparent_box.getNumberOfRenderingPasses(TW_RENDERING_MODE_RAY_CAST_GAS); ++i)
		{
			p_myself->transparent_box.prepareRendering(canvas, i);
			p_myself->transparent_box.render();
			p_myself->transparent_box.finalizeRendering();
		}
	}

	//Executes rendering commands
	static void render(Screen& screen)
	{
		if (!p_myself) return;

		p_myself->hdr_bloom_ss_filter.pass(p_myself->main_camera, screen);

		TwDraw();
	}


	//GLFW mouse button callback
	static void onMouseClick(GLFWwindow* p_glfw_window, int button, int action, int mods)
	{
		TwEventMouseButtonGLFW(button, action);
	}

	//GLFW mouse position callback
	static void onCursorPositionChange(GLFWwindow* p_glfw_window, double xpos, double ypos)
	{
		TwEventMousePosGLFW(static_cast<int>(std::round(xpos)), static_cast<int>(std::round(ypos)));
	}

	//GLFW scrolling callback
	static void onScrollCallback(GLFWwindow* p_glfw_window, double xoffset, double yoffset)
	{
		TwEventMouseWheelGLFW(static_cast<int>(std::round(xoffset)));
	}

	//GLFW key press callback
	static void onKeyPress(GLFWwindow* p_glfw_window, int key, int scancode, int action, int modes)
	{
		TwEventKeyGLFW(key, action);

		switch (key)
		{
		case GLFW_KEY_W:
			p_myself->main_camera.translate(vec3{ 0, 0, -0.1f });
			break;

		case GLFW_KEY_S:
			p_myself->main_camera.translate(vec3{ 0, 0, 0.1f });
			break;
		}
	}

	//GLFW character input callback
	static void onCharInput(GLFWwindow* p_glfw_window, unsigned int codepoint)
	{
		TwEventCharGLFW(codepoint, 0);
	}

public:
	//Creates new rendering context or returns a pointer to the existing context
	static RenderContext* createContext(Screen& output_screen)
	{
		if (!p_myself)
		{
			//Initialize TinyWorld engine
			output_screen.attachRenderer(render);
			ShaderProgram::setShaderBaseCatalog("../tw_shaders/");
			AbstractRenderableObjectTextured::defineTextureLookupPath("/");
			AbstractRenderableObjectTextured::defineTextureUnitBlockGlobalPointer(TextureUnitBlock::initialize());
			p_myself = new RenderContext{ output_screen };
		}
		return p_myself;
	}

	//Destroys existing rendering context. Does nothing if there is no rendering context
	static void reset()
	{
		if (p_myself) delete p_myself;
	}

	//Runs drawing commands assumed by the rendering context and returns 'true' if the window attached to the context 
	//should remain alive upon completion of the drawing tasks. Returns 'false' if the window should be closed as soon
	//as the last drawing command has been executed
	bool process()
	{
		//Setup the scene


		//Render the scene
		canvas.makeActive();
		canvas.refresh();

		screen.makeActive();
		screen.refresh();

		return !screen.shouldClose();
	}
};

RenderContext* RenderContext::p_myself = nullptr;

int main(int argc, char* argv[])
{
	Screen main_screen{ 45, 45, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT };
	main_screen.setScreenVideoMode(main_screen.getVideoModes()[0]);
	main_screen.defineViewport(0, 0, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);
	main_screen.setStringName("FractalNoise3DTest");

	RenderContext* p_rendering_context = RenderContext::createContext(main_screen);

	while (p_rendering_context->process());
	p_rendering_context->reset();
	
	return EXIT_SUCCESS;
}