#define DEFAULT_WINDOW_WIDTH 1920
#define DEFAULT_WINDOW_HEIGHT 1080

#include <stdlib.h>
#include <crtdbg.h>


#include "../TinyWorld/Screen.h"
#include "../TinyWorld/FullscreenRectangle.h"
#include "../TinyWorld/CompleteShaderProgram.h"
#include "../TinyWorld/TextureUnitBlock.h"
#include "../TinyWorld/ImmutableTexture2D.h"
#include "../TinyWorld/VectorTypes.h"
#include "../TinyWorld/QuaternionTypes.h"
#include "../TinyWorld/MatrixTypes.h"
#include "../TinyWorld/AbstractProjectingDevice.h"
#include "../TinyWorld/FractalNoise.h"
#include "../TinyWorld/KTXTexture.h"

using namespace tiny_world;


//Class implementing the principal rendering context
class RenderContext final
{
private:
	static RenderContext* p_myself;
	Screen& screen;
	vec2 v2ScreenRatio;

	FractalNoise2D _2d_fractal_noise;
	ImmutableTexture2D noise_map;
	FullscreenRectangle texture_viewer;
	PerspectiveProjectingDevice main_camera;
	SeparateShaderProgram filter_program;

	RenderContext(Screen& output_screen) : screen{ output_screen }, 
		v2ScreenRatio{ static_cast<float>(DEFAULT_WINDOW_WIDTH) / std::max(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT) ,
		static_cast<float>(DEFAULT_WINDOW_HEIGHT) / std::max(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT) },
		_2d_fractal_noise(2048, 2048, 512, 512, 8U), 
		main_camera{ "MainCamera", v2ScreenRatio.x, v2ScreenRatio.y, 1.0f, 1e3f },
		filter_program{ "FilterProgram" }
	{
		//Setup screen
		output_screen.setClearColor(vec4{ 0 });
		output_screen.applyOpenGLContextSettings();
		
		//Generate fractal noise
		_2d_fractal_noise.setContinuity(true);
		_2d_fractal_noise.setPeriodicity(true);
		_2d_fractal_noise.setEvolutionRate(0.1f);
		_2d_fractal_noise.generateNoiseMap();
		noise_map = _2d_fractal_noise.retrieveNoiseMap();

		//Setup the full screen rectangle that is employed to draw the noise map
		texture_viewer.setDimensions(v2ScreenRatio.x, v2ScreenRatio.y);
		texture_viewer.setLocation(vec3{ 0, 0, -1 });
		texture_viewer.selectRenderingMode(TW_RENDERING_MODE_DEFAULT);
		texture_viewer.installTexture(noise_map);
		TextureSampler periodic_sampler{};
		periodic_sampler.setMinFilter(SamplerMinificationFilter::LINEAR_MIPMAP_NEAREST);
		periodic_sampler.setMagFilter(SamplerMagnificationFilter::LINEAR);
		periodic_sampler.setWrapping(SamplerWrapping{ SamplerWrappingMode::REPEAT, SamplerWrappingMode::REPEAT, SamplerWrappingMode::CLAMP_TO_EDGE });
		texture_viewer.installSampler(periodic_sampler);

		const char* custom_filter_shader_source =
			"#version 430 core\n"
			"uniform sampler2D source0;\n"
			"in vec2 tex_coord;\n"
			"out vec4 v4Color;\n"
			"void main()\n"
			"{\n"
			"    v4Color = vec4(texture(source0, tex_coord).r);\n"
			"}\n";
		Shader custom_filter_shader{ GLSLSourceCode{ custom_filter_shader_source, strlen(custom_filter_shader_source) }, ShaderType::FRAGMENT_SHADER, "FilterEffectShader" };
		filter_program.addShader(custom_filter_shader);
		filter_program.link();
		texture_viewer.setFilterEffect(filter_program,
			[](const AbstractProjectingDevice& projecting_device, const AbstractRenderingDevice& rendering_device, int first_vacant_texture_unit)->bool
		{
			return true;
		});

	}

	~RenderContext()
	{
		TextureUnitBlock::reset();
		p_myself = nullptr;
	}


	//Executes rendering commands
	static void render(Screen& screen)
	{
		if (!p_myself) return;

		screen.clearBuffers(BufferClearTarget::COLOR);
		p_myself->_2d_fractal_noise.generateNoiseMap();


		p_myself->texture_viewer.applyViewProjectionTransform(p_myself->main_camera);
		for (uint32_t i = 0; i < p_myself->texture_viewer.getNumberOfRenderingPasses(TW_RENDERING_MODE_DEFAULT); ++i)
		{
			p_myself->texture_viewer.prepareRendering(screen, i);
			p_myself->texture_viewer.render();
			p_myself->texture_viewer.finalizeRendering();
		}
			
	}

public:
	//Creates new rendering context or returns a pointer to the existing context
	static RenderContext* createContext(Screen& output_screen)
	{
		if(!p_myself)
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
		screen.makeActive();
		screen.refresh();

		return !screen.shouldClose();
	}
};

RenderContext* RenderContext::p_myself = nullptr;


int main(int argc, char* argv[])
{
	//Initialize TinyWorld's screen
	Screen main_screen{ 0, 0, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT };
	main_screen.setStringName("2D Fractal Noise Test");
	main_screen.setScreenVideoMode(main_screen.getVideoModes()[0]);
	main_screen.defineViewport(0, 0, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);

	RenderContext* p_render_context = RenderContext::createContext(main_screen);
	while (p_render_context->process());

	RenderContext::reset();

	return EXIT_SUCCESS;
}