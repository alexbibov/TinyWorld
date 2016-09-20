#define DEFAULT_WINDOW_WIDTH 1024
#define DEFAULT_WINDOW_HEIGHT 768

#include <AntTweakBar.h>


#include "../TinyWorld/Screen.h"
#include "../TinyWorld/Framebuffer.h"
#include "../TinyWorld/CompleteShaderProgram.h"
#include "../TinyWorld/TextureUnitBlock.h"
#include "../TinyWorld/KTXTexture.h"
#include "../TinyWorld/ImmutableTexture1D.h"
#include "../TinyWorld/ImmutableTexture2D.h"
#include "../TinyWorld/VectorTypes.h"
#include "../TinyWorld/MatrixTypes.h"
#include "../TinyWorld/Light.h"
#include "../TinyWorld/AbstractProjectingDevice.h"
#include "../TinyWorld/SSFilter_HDRBloom.h"
#include "../TinyWorld/SSFilter_Blur.h"
#include "../TinyWorld/SSFilter_SSAO.h"
#include "../TinyWorld/SSFilter_ImmediateShader.h"
#include "../TinyWorld/CylindricalSurface.h"
#include "../TinyWorld/QuaternionTypes.h"

using namespace tiny_world;


//Main rendering context
class RenderContext
{
private:
	static RenderContext* p_myself;
	Screen& main_screen;

	PerspectiveProjectingDevice main_camera;
	CylindricalSurface surface;
	LightingConditions lighting_conditions;
	AmbientLight ambient_light;
	PointLight main_light;
	Framebuffer render_data;
	ImmutableTexture2D render_color, render_bloom, normal_map, render_depth, linear_depth, ad_map;
	ImmutableTexture2D side_surface_texture, face_surface_texture;
	ImmutableTexture2D white_texture;
	
	//SS-filters
	tiny_world::SSFilter_HDRBloom hdr_bloom;	//HDR-Bloom screen space filter
	tiny_world::SSFilter_SSAO ssao;	//screen-space ambient occlusion filter
	tiny_world::SSFilter_Blur bloom_blur_x, bloom_blur_y, ssao_blur_x, ssao_blur_y;	//blur filters
	tiny_world::SSFilter_ImmediateShader immediate_shader;	//immediate renderer
	tiny_world::CascadeFilterLevel CL0, CL1, CL2, CL3, CL4, CL5;	//cascade filter levels
	tiny_world::CascadeFilter postprocess;	//cascade filter that performs post-processing

	TwBar* p_twbar;

	float surface_rotation[4];
	float light_location[3];
	bool diffuse_texture_enabled;
	bool bump_texture_enabled;
	bool specular_texture_enabled;

	//Rendering context can only be constructed when the screen has already been made available
	RenderContext(Screen& main_screen) : main_screen{ main_screen }, surface{ 180, 50 }
	{
		//Initialize TinyWorld
		AbstractRenderableObjectTextured::defineTextureUnitBlockGlobalPointer(TextureUnitBlock::initialize());

		//Initialize AntTweakBar (3rd party library employed for visualization of GUI)
		TwInit(TwGraphAPI::TW_OPENGL_CORE, NULL);
		Rectangle window_rectangle = main_screen.getViewportRectangle(0);
		TwWindowSize(static_cast<uint32_t>(window_rectangle.w), static_cast<uint32_t>(window_rectangle.h));
		p_twbar = TwNewBar("main_bar");
		TwDefine(" main_bar label='Visual appearance'\n help='Change visual appearance parameters to affect visualization of the surface'\n color='117 134 41'\n"
			" alpha='100'\n text='light'\n iconifiable='false'\n movable='false'\n resizable='false'\n buttonalign='center'\n size='350 350'\n valueswidth='180' ");
		
		surface_rotation[0] = 0; surface_rotation[1] = 0; surface_rotation[2] = 0; surface_rotation[3] = 1;
		TwAddVarRW(p_twbar, "cylindrical_surface_rotation", TW_TYPE_QUAT4F, surface_rotation,
			" label='Object rotation'\n help='Rotates cylindrical surface around its mass center'");

		light_location[0] = 0; light_location[1] = 10; light_location[2] = 0;
		TwAddVarRW(p_twbar, "log_position_x", TW_TYPE_DIR3F, light_location,
			" label='Light source location'\n help='Sets position of the light' ");

		diffuse_texture_enabled = true; bump_texture_enabled = true; specular_texture_enabled = true;
		TwAddVarRW(p_twbar, "diffuse_texture_enabled", TwType::TW_TYPE_BOOLCPP, &diffuse_texture_enabled, " label='Diffuse texture'\n help='Switches diffuse texture on and off' ");
		TwAddVarRW(p_twbar, "bump_texture_enabled", TwType::TW_TYPE_BOOLCPP, &bump_texture_enabled, " label='Normal map'\n help='Switches normal map on and off' ");
		TwAddVarRW(p_twbar, "specular_texture_enabled", TwType::TW_TYPE_BOOLCPP, &specular_texture_enabled, " label='Specular map'\n help='Switches specular map on and off' ");


		//Initialize framebuffer used for HDR rendering
		render_data.setStringName("SurfaceViewer::main_framebuffer");
		render_data.setClearColor(vec4{ 1.0f });
		render_data.setCullTestEnableState(true);
		render_data.setDepthTestEnableState(true);
		render_data.defineViewport(window_rectangle);

		render_color.setStringName("SurfaceViewer::render_color_buffer");
		render_bloom.setStringName("SurfaceViewer::render_bloom_buffer");
		render_depth.setStringName("SurfaceViewer::render_depth_buffer");
		render_color.allocateStorage(1, 1, TextureSize{ static_cast<uint32_t>(window_rectangle.w), static_cast<uint32_t>(window_rectangle.h) }, InternalPixelFormat::SIZED_FLOAT_RGBA32);
		render_bloom.allocateStorage(1, 1, TextureSize{ static_cast<uint32_t>(window_rectangle.w), static_cast<uint32_t>(window_rectangle.h) }, InternalPixelFormat::SIZED_FLOAT_RGBA32);
		normal_map.allocateStorage(1, 1, TextureSize{ static_cast<uint32_t>(window_rectangle.w), static_cast<uint32_t>(window_rectangle.h) }, InternalPixelFormat::SIZED_FLOAT_RGB32);
		linear_depth.allocateStorage(1, 1, TextureSize{ static_cast<uint32_t>(window_rectangle.w), static_cast<uint32_t>(window_rectangle.h) }, InternalPixelFormat::SIZED_FLOAT_R32);
		ad_map.allocateStorage(1, 1, TextureSize{ static_cast<uint32_t>(window_rectangle.w), static_cast<uint32_t>(window_rectangle.h) }, InternalPixelFormat::SIZED_FLOAT_RGB32);
		render_depth.allocateStorage(1, 1, TextureSize{ static_cast<uint32_t>(window_rectangle.w), static_cast<uint32_t>(window_rectangle.h) }, InternalPixelFormat::SIZED_DEPTH16);
		render_data.attachTexture(FramebufferAttachmentPoint::COLOR0, FramebufferAttachmentInfo{ 0, 0, &render_color });
		render_data.attachTexture(FramebufferAttachmentPoint::COLOR1, FramebufferAttachmentInfo{ 0, 0, &render_bloom });
		render_data.attachTexture(FramebufferAttachmentPoint::COLOR2, FramebufferAttachmentInfo{ 0, 0, &normal_map });
		render_data.attachTexture(FramebufferAttachmentPoint::COLOR3, FramebufferAttachmentInfo{ 0, 0, &linear_depth });
		render_data.attachTexture(FramebufferAttachmentPoint::COLOR4, FramebufferAttachmentInfo{ 0, 0, &ad_map });
		render_data.attachTexture(FramebufferAttachmentPoint::DEPTH, FramebufferAttachmentInfo{ 0, 0, &render_depth });
		render_data.attachRenderer(std::bind(&RenderContext::framebufferRenderer, this, std::placeholders::_1));


		//Configure textures
		white_texture.allocateStorage(1, 0, TextureSize{ 1, 1, 0 }, InternalPixelFormat::SIZED_RGB8);
		white_texture.setMipmapLevelData(0, PixelLayout::RGB, PixelDataType::BYTE, std::array < char, 3 > {1, 1, 1}.data());

		KTXTexture side_surface_COLOR, side_surface_NRM, side_surface_SPEC;
		KTXTexture face_surface_COLOR, face_surface_NRM, face_surface_SPEC;
		side_surface_COLOR.loadTexture("bark.ktx");
		side_surface_NRM.loadTexture("bark_NRM.ktx");
		side_surface_SPEC.loadTexture("bark_SPEC.ktx");
		face_surface_COLOR.loadTexture("sapwood.ktx");
		face_surface_NRM.loadTexture("sapwood_NRM.ktx");
		face_surface_SPEC.loadTexture("sapwood_SPEC.ktx");

		side_surface_texture = TEXTURE_2D(side_surface_COLOR.getContainedTexture());
		face_surface_texture = TEXTURE_2D(face_surface_COLOR.getContainedTexture());

		surface.applyNormalMapSourceTexture(TEXTURE_2D(side_surface_NRM.getContainedTexture()), TEXTURE_2D(face_surface_NRM.getContainedTexture()));
		surface.applySpecularMapSourceTexture(TEXTURE_2D(side_surface_SPEC.getContainedTexture()), TEXTURE_2D(face_surface_SPEC.getContainedTexture()));


		//Configure main scene camera
		const float focal_plane_width = window_rectangle.w / std::max(window_rectangle.w, window_rectangle.h);
		const float focal_plane_height = window_rectangle.h / std::max(window_rectangle.w, window_rectangle.h);
		main_camera.setProjectionVolume(-focal_plane_width / 2, focal_plane_width / 2, -focal_plane_height / 2, focal_plane_height / 2, 1.0f, 1000.0f);
		main_camera.setStringName("SurfaceViewer::main_camera");
		main_camera.setLocation(vec3{ 0, 0, 15.0 });



		//Configure  cascade filter
		CL0.setOutputColorBitWidth(tiny_world::CascadeFilterLevel::ColorWidth::_32bit);
		CL0.setOutputColorComponents(true, true, true, false);
		CL0.addFilter(&bloom_blur_x);

		CL1.setOutputColorBitWidth(tiny_world::CascadeFilterLevel::ColorWidth::_32bit);
		CL1.setOutputColorComponents(true, false, false, false);
		CL1.addFilter(&ssao);

		CL2.setOutputColorBitWidth(tiny_world::CascadeFilterLevel::ColorWidth::_32bit);
		CL2.setOutputColorComponents(true, true, true, false);
		CL2.addFilter(&bloom_blur_y);
		CL2.addFilter(&ssao_blur_x);

		CL3.setOutputColorBitWidth(tiny_world::CascadeFilterLevel::ColorWidth::_32bit);
		CL3.setOutputColorComponents(true, true, true, false);
		CL3.addFilter(&ssao_blur_y);

		CL4.setOutputColorBitWidth(tiny_world::CascadeFilterLevel::ColorWidth::_32bit);
		CL4.setOutputColorComponents(true, true, true, false);
		CL4.addFilter(&immediate_shader);

		CL5.setOutputColorBitWidth(tiny_world::CascadeFilterLevel::ColorWidth::_32bit);
		CL5.setOutputColorComponents(true, true, true, false);
		CL5.addFilter(&hdr_bloom);

		postprocess.addBaseLevel(CL0);
		postprocess.addLevel(CL1, std::list < tiny_world::CascadeFilter::DataFlow > {});
		postprocess.addLevel(CL2,
			std::list < tiny_world::CascadeFilter::DataFlow > { tiny_world::CascadeFilter::DataFlow{ 0, 0, 0 }, tiny_world::CascadeFilter::DataFlow{ 1, 1, 0 } });
		postprocess.addLevel(CL3,
			std::list < tiny_world::CascadeFilter::DataFlow > { tiny_world::CascadeFilter::DataFlow{ 0, 2, 1 } });
		postprocess.addLevel(CL4,
			std::list < tiny_world::CascadeFilter::DataFlow > { tiny_world::CascadeFilter::DataFlow{ 2, 3, 0 } });
		postprocess.addLevel(CL5,
			std::list < tiny_world::CascadeFilter::DataFlow > { tiny_world::CascadeFilter::DataFlow{ 0, 4, 0 }, tiny_world::CascadeFilter::DataFlow{ 1, 2, 0 } });


		postprocess.useCommonOutputResolution(true);
		postprocess.setCommonOutputResolutionValue(tiny_world::uvec2{ static_cast<uint32_t>(window_rectangle.w), static_cast<uint32_t>(window_rectangle.h) });


		//Configure bloom blur filters
		bloom_blur_x.defineInputTexture(render_bloom);
		bloom_blur_x.setKernelMipmap(0);
		bloom_blur_x.setKernelScale(5.0f);
		bloom_blur_x.setKernelSize(16);

		bloom_blur_y.setKernelMipmap(0);
		bloom_blur_y.setKernelScale(5.0f);
		bloom_blur_y.setKernelSize(16);
		bloom_blur_y.setDirection(tiny_world::SSFilter_Blur::BlurDirection::VERTICAL);


		//Configure SSAO
		ssao.setNumberOfSamples(64);
		ssao.defineScreenSpaceNormalMap(normal_map);
		ssao.defineLinearDepthBuffer(linear_depth);

		//Configure SSAO-blur filters
		ssao_blur_x.setKernelMipmap(0);
		ssao_blur_x.setKernelScale(0.5f);
		ssao_blur_x.setKernelSize(4);
		ssao_blur_y.setKernelMipmap(0);
		ssao_blur_y.setKernelScale(0.5f);
		ssao_blur_y.setKernelSize(4);


		//Configure immediate shader
		immediate_shader.defineColorMap(render_color);
		immediate_shader.defineADMap(ad_map);


		//Configure HDR-Bloom filter
		hdr_bloom.setBloomImpact(1.0f);


		postprocess.initialize();




		//Configure scene light
		ambient_light.setColor(vec3{ 0.2f, 0.2f, 0.2f });
		main_light.setLocation(vec3{ light_location[0], light_location[1], light_location[2] });
		main_light.setColor(vec3{ 1, 1, 0.7f });
		main_light.setAttenuation(1.0f, 0.05f, 0.01f);
		lighting_conditions.addLight(ambient_light);
		lighting_conditions.addLight(main_light);


		//Configure surface map object
		surface.applyLightingConditions(lighting_conditions);
		surface.setScreenSize(static_cast<uvec2>(vec2{ window_rectangle.w, window_rectangle.h }));
		surface.setBloomMinimalThreshold(0.4f);
		surface.setBloomMaximalThreshold(1.0f);
		surface.setBloomIntensity(0.5f);
		surface.setLength(1000.0f);
		surface.scale(vec3{ 0.01f });
		surface.setSpecularExponent(0.6f);
		surface.setSpecularColor(vec3{ 1.0f, 1.0f, 1.0f });


		//Set mouse button callback
		glfwSetMouseButtonCallback(main_screen, RenderContext::mouseAction);

		//Set mouse scroll callback
		glfwSetScrollCallback(main_screen, RenderContext::mouseScroll);

		//Set mouse move callback
		glfwSetCursorPosCallback(main_screen, RenderContext::mouseMove);
	}


	RenderContext(const RenderContext& other) = delete;
	RenderContext(RenderContext&& other) = delete;
	RenderContext& operator=(const RenderContext& other) = delete;
	RenderContext& operator=(RenderContext&& other) = delete;


	~RenderContext()
	{
		TextureUnitBlock::reset();
		TwTerminate();
		RenderContext::p_myself = nullptr;
	}


	static void mouseMove(GLFWwindow* p_window, double x_pos, double y_pos)
	{
		TwEventMousePosGLFW(static_cast<int>(x_pos), static_cast<int>(y_pos));
	}


	static void mouseAction(GLFWwindow* p_window, int button, int action, int mods)
	{
		TwEventMouseButtonGLFW(button, action);
	}


	static void mouseScroll(GLFWwindow* p_window, double x_offset, double y_offset)
	{
		static int wheel_pos = 0;
		wheel_pos = static_cast<int>(wheel_pos + y_offset);

		TwEventMouseWheelGLFW(wheel_pos);
	}


	void framebufferRenderer(Framebuffer& framebuffer)
	{
		framebuffer.clearBuffers(BufferClearTarget::COLOR_DEPTH);
		surface.applyViewProjectionTransform(main_camera);

		for (unsigned int i = 0; i < surface.getNumberOfRenderingPasses(surface.getActiveRenderingMode()); ++i)
		{
			surface.prepareRendering(framebuffer, i);
			surface.render();
			surface.finalizeRendering();
		}
	}

	
	static void screenRenderer(Screen& screen)
	{
		if (!p_myself) return;
		p_myself->postprocess.pass(p_myself->main_camera, screen);

		TwDraw();
	}


public:
	//Creates new rendering context and returns a pointer to its instance
	static RenderContext* createRenderingContext(Screen& main_screen)
	{
		if (!p_myself)
		{
			//Configure TinyWorld directories
			ShaderProgram::setShaderBaseCatalog("../tw_shaders/");
			AbstractRenderableObjectTextured::defineTextureLookupPath("tw_textures/");

			//Attach renderer to the main screen
			main_screen.attachRenderer(screenRenderer);

			//Create rendering context
			p_myself = new RenderContext{ main_screen };
		}
		
		return p_myself;
	}

	//Destroys previously created rendering context
	static void reset()
	{
		if (p_myself)
		{
			delete p_myself;
			p_myself = nullptr;
		}
	}




	//Draws the scene and returns 'true' if the scene should yet be rendered next time.
	//If function returns 'false', this means that the process should exit
	bool process()
	{
		surface.resetObjectRotation();
		surface.applyRotation(quaternion{ surface_rotation[3], surface_rotation[0], surface_rotation[1], surface_rotation[2] }, RotationFrame::LOCAL);

		main_light.setLocation(vec3{ light_location[0], light_location[1], light_location[2] });

		surface.setNormalMapEnableState(bump_texture_enabled);
		surface.setSpecularMapEnableState(specular_texture_enabled);

		if (diffuse_texture_enabled)
			surface.installTexture(side_surface_texture, face_surface_texture);
		else
			surface.installTexture(white_texture, white_texture);


		render_data.makeActive();
		render_data.refresh();

		main_screen.makeActive();
		main_screen.refresh();

		return !main_screen.shouldClose();
	}


	//Returns pointer to encapsulated cylindrical surface object
	CylindricalSurface* getCylindricalSurfacePointer() { return &surface; }
};


RenderContext* RenderContext::p_myself = nullptr;



int main(int argc, char* argv[])
{
	Screen main_screen{ 45, 45, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT };
	main_screen.setScreenVideoMode(main_screen.getVideoModes()[0]);
	main_screen.defineViewport(0, 0, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);
	main_screen.setStringName("SurfaceViewer::main_screen");

	RenderContext* p_rendering_context = RenderContext::createRenderingContext(main_screen);
	p_rendering_context->getCylindricalSurfacePointer()->defineSurface("surface_map_corrected.txt");


	while (p_rendering_context->process());
	p_rendering_context->reset();
	return EXIT_SUCCESS;
}