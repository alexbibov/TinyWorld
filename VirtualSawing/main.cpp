#define DEFAULT_WINDOW_WIDTH 1024
#define DEFAULT_WINDOW_HEIGHT 768


#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <string>
#include <list>

#include <AntTweakBar.h>


#include "../TinyWorld/Screen.h"
#include "../TinyWorld/Framebuffer.h"
#include "../TinyWorld/CompleteShaderProgram.h"
#include "../TinyWorld/TextureUnitBlock.h"
#include "../TinyWorld/ImmutableTexture1D.h"
#include "../TinyWorld/ImmutableTexture2D.h"
#include "../TinyWorld/ImmutableTexture3D.h"
#include "../TinyWorld/VectorTypes.h"
#include "../TinyWorld/MatrixTypes.h"
#include "../TinyWorld/Light.h"
#include "../TinyWorld/AbstractProjectingDevice.h"
#include "../TinyWorld/SSFilter_HDRBloom.h"
#include "../TinyWorld/TransparentBox.h"
#include "../TinyWorld/Misc.h"
#include "../TinyWorld/QuaternionTypes.h"


#include "vlog/vlog.h"


using namespace tiny_world;


class RenderContext
{
private:
	static RenderContext* p_myself;		//pointer of the context to itself

	PerspectiveProjectingDevice main_camera;
	TransparentBox log_point_cloud;
	DirectionalLight light;
	Framebuffer framebuffer;
	ImmutableTexture2D color_texture, bloom_texture, depth_texture;
	ImmutableTexture1D colormap_texture;
	const uint32_t colormap_resolution = 256;	//default resolution of the colormap
	SSFilter_HDRBloom hdr_bloom_filter;
	TwBar* p_tw_bar;

	static Screen* p_main_screen;	//main screen of the program


	//Variables representing parameters of the objects being rendered
	typedef enum{
		GAS = TW_RENDERING_MODE_RAY_CAST_GAS,
		ABSORBENT = TW_RENDERING_MODE_RAY_CAST_ABSORBENT,
		ABSORBENT2 = TW_RENDERING_MODE_PROXY_GEOMETRY_ABSORBENT
	} optical_mode;

	typedef struct{
		uint32_t num_primary_samples;
		uint32_t num_secondary_samples;		//number of secondary samples does not affect rendering when optical mode ABSORBENT2 is active
		vec3 v3MediumColor;
		float solid_angle;	//this parameter has no affect for optical modes ABSORBENT and ABSORBENT2
	}optical_properties;


	float log_rotation[4];	//raw quaternion representation accepted by AntTweakBar. Quaternion is packed as (x, y, z, w), where w is the scalar part
	float log_location[3];	//location of the center of the log
	optical_mode log_optical_mode;	//optical mode used to visualize the log
	optical_properties log_optical_properties[3];	//optical properties stored in array for each of the optical modes: GAS, ABSORBENT, ABSORBENT2 in this order
	std::list<std::array<float, 3>> color_map;	//stores vector containing key-values of 1D colormap function. These key-values are used for polynomial reconstruction
	//*******************************************************************


	//Default constructor
	RenderContext()
	{
		//Set initial rendering parameters
		log_rotation[0] = 0;
		log_rotation[1] = 0;
		log_rotation[2] = 0;
		log_rotation[3] = 1;

		log_location[0] = 0;
		log_location[1] = 0;
		log_location[2] = -10;

		log_optical_mode = ABSORBENT2;

		//Apply initial setting for optical mode GAS
		log_optical_properties[0].num_primary_samples = 3;
		log_optical_properties[0].num_secondary_samples = 5;
		log_optical_properties[0].solid_angle = 1.0f;
		log_optical_properties[0].v3MediumColor = 10.0f;

		//Apply initial setting for optical mode ABSORBENT
		log_optical_properties[1].num_primary_samples = 15;
		log_optical_properties[1].num_secondary_samples = 5;
		log_optical_properties[1].solid_angle = 0.0f;
		log_optical_properties[1].v3MediumColor = 1.0f;

		//Apply initial setting for optical mode ABSORBENT2
		log_optical_properties[2].num_primary_samples = 50;
		log_optical_properties[2].num_secondary_samples = 0;
		log_optical_properties[2].solid_angle = 0.0f;
		log_optical_properties[2].v3MediumColor = 1.0f;


		//Initialize AntTweakBar (3rd-party library employed for visualization of GUI)
		TwInit(TW_OPENGL_CORE, NULL);
		TwWindowSize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);
		p_tw_bar = TwNewBar("main_bar");
		TwDefine(" main_bar label='Appearance'\n help='Change optical parameters of the rendering'\n color='117 134 41'\n alpha='100'\n text='light'\n "
			"iconifiable='false'\n movable='false'\n resizable='false'\n buttonalign='center'");

		TwAddVarRW(p_tw_bar, "log_point_cloud_rotation", TW_TYPE_QUAT4F, log_rotation,
			" label='Rotation' help='Rotates chosen piece of the log' ");

		TwAddVarRW(p_tw_bar, "log_position_x", TW_TYPE_DIR3F, log_location,
			" label='Location'\n help='Sets position of the center of the log' ");

		TwEnumVal OpticalModes[] = { { GAS, "Gas" }, { ABSORBENT, "Absorbent" }, { ABSORBENT2, "Opaque absorbent" } };
		TwType tw_optical_mode = TwDefineEnum("optical_mode", OpticalModes, 3);
		TwAddVarCB(p_tw_bar, "log_optical_mode", tw_optical_mode, setOpticalModeCallback, getOpticalModeCallback, nullptr, " label='Optical modes' ");

		//Below the reinterpet_cast<void*>() casts are used to pass identification data to the callback functions so that the callback
		//can deduce, which parameter exactly has been updated
		TwAddVarCB(p_tw_bar, "num_primary_samples", TW_TYPE_UINT32, setOpticalModeParamsCallback, getOpticalModeParamsCallback, reinterpret_cast<void*>(0),
			" group='Optical parameters'\n min='2'\n label='Primary samples' ");
		TwAddVarCB(p_tw_bar, "num_secondary_samples", TW_TYPE_UINT32, setOpticalModeParamsCallback, getOpticalModeParamsCallback, reinterpret_cast<void*>(1),
			" group='Optical parameters'\n min='2'\n label='Secondary samples' ");
		TwAddVarCB(p_tw_bar, "v3MediumColor", TW_TYPE_COLOR3F, setOpticalModeParamsCallback, getOpticalModeParamsCallback, reinterpret_cast<void*>(2),
			" group='Optical parameters'\n label='Medium color' ");
		TwAddVarCB(p_tw_bar, "solid_angle", TW_TYPE_FLOAT, setOpticalModeParamsCallback, getOpticalModeParamsCallback, reinterpret_cast<void*>(3),
			" group='Optical parameters'\n min='0.1'\n max='100'\n step='0.01'\n label='Solid angle size' ");

		//reinterpret_cast<void*>(1) is needed to inform callback that colormap recalculation is required on insertion of a new key color value
		TwAddButton(p_tw_bar, "button_add_colormap_keyval", addColormapKeyvalCallback, reinterpret_cast<void*>(1), "group='Colormap settings'\n label='Add key color' ");

		TwAddButton(p_tw_bar, "button_remove_colormap_keyval", removeColormapKeyvalCallback, nullptr, "group='Colormap settings'\n label='Remove key color' ");


		////Initialize framebuffer used for HDR filtering
		framebuffer.setClearColor(0.0f, 0.0f, 0.0, 0.0f);
		framebuffer.setCullTestEnableState(true);
		framebuffer.defineViewport(Rectangle{ 0, 0, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT });
		framebuffer.setDepthTestEnableState(true);
		color_texture.setStringName("MainScene_Color");
		bloom_texture.setStringName("MainScene_Bloom");
		depth_texture.setStringName("MainScene_Depth");
		color_texture.allocateStorage(1, 0, TextureSize{ DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT }, InternalPixelFormat::SIZED_FLOAT_RGBA32);
		bloom_texture.allocateStorage(1, 0, TextureSize{ DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT }, InternalPixelFormat::SIZED_FLOAT_RGBA32);
		depth_texture.allocateStorage(1, 0, TextureSize{ DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT }, InternalPixelFormat::SIZED_DEPTH16);
		framebuffer.attachTexture(FramebufferAttachmentPoint::COLOR0, FramebufferAttachmentInfo{ 0, 0, &color_texture });
		framebuffer.attachTexture(FramebufferAttachmentPoint::COLOR1, FramebufferAttachmentInfo{ 0, 0, &bloom_texture });
		framebuffer.attachTexture(FramebufferAttachmentPoint::DEPTH, FramebufferAttachmentInfo{ 0, 0, &depth_texture });
		framebuffer.attachRenderer(framebufferOnRender);

		//Setup main scene camera
		float width = static_cast<float>(DEFAULT_WINDOW_WIDTH) / std::max(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);
		float height = static_cast<float>(DEFAULT_WINDOW_HEIGHT) / std::max(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);
		main_camera.setProjectionVolume(-width / 2, width / 2, -height / 2, height / 2, 1.0f, 100.0f);
		main_camera.setStringName("MainCamera");
		main_camera.setLocation(vec3{ 0, 0, 6.5f });



		//Initialize filter
		hdr_bloom_filter.setBloomImpact(0.3f);
		hdr_bloom_filter.defineColorTexture(color_texture);
		hdr_bloom_filter.defineBloomTexture(bloom_texture);
		hdr_bloom_filter.initialize();


		//Setup main light
		light.setDirection(vec3{ 0, 0, 1 });
		light.setColor(vec3{ 1.0, 1.0, 1.0 });


		//Load virtual log 3D point cloud
		XRAY_DOMAIN xray_domain;
		std::vector<XRAY_SOURCE> xray_sources;
		std::vector<XRAY_DETECTOR_PIXEL> xray_detector_pixels;
		load_vlog_data("vlog/vlogdata.vlog", &xray_domain, &xray_sources, &xray_detector_pixels);

		//Retrieve dimensions of the virtual log 3D density point cloud
		uint32_t point_cloud_width = static_cast<uint32_t>(std::floor(xray_domain.defined_width / xray_domain.defined_vx + 0.5));
		uint32_t point_cloud_height = static_cast<uint32_t>(std::floor(xray_domain.defined_height / xray_domain.defined_vy + 0.5));
		uint32_t point_cloud_depth = static_cast<uint32_t>(std::floor(xray_domain.defined_depth / xray_domain.defined_vz + 0.5));

		//Retrieve maximal supported 3D texture depth
		GLint max_3dtexture_width_height_depth;
		glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_3dtexture_width_height_depth);

		uint32_t target_width = std::min<uint32_t>(point_cloud_width, max_3dtexture_width_height_depth);
		uint32_t target_height = std::min<uint32_t>(point_cloud_height, max_3dtexture_width_height_depth);
		uint32_t target_depth = std::min<uint32_t>(point_cloud_depth, max_3dtexture_width_height_depth);


		//Perform up-sampling of the point cloud if necessary
		double* vlog_density_point_cloud = static_cast<double*>(malloc(sizeof(double)*target_width*target_height*target_depth));
		upsample_vlog_data(xray_domain.get_defined_data_ptr(), point_cloud_width, point_cloud_height, point_cloud_depth, target_width, target_height, target_depth, vlog_density_point_cloud);

		float* vlog_density_point_cloud_float = reinterpret_cast<float*>(vlog_density_point_cloud);
		for (int i = 0; i < static_cast<int>(target_width*target_height*target_depth); ++i)
			vlog_density_point_cloud_float[i] = static_cast<float>(vlog_density_point_cloud[i]);


		//Create 3D point cloud representing virtual log data
		ImmutableTexture3D _3d_point_cloud_texture{ "virtual_log_point_cloud" };
		uint32_t num_mipmaps = static_cast<uint32_t>(std::max(std::max(std::log2(target_width), std::log2(target_height)), std::log2(target_depth)));
		_3d_point_cloud_texture.allocateStorage(1, 1, TextureSize{ target_width, target_height, target_depth }, InternalPixelFormat::SIZED_FLOAT_R32);
		_3d_point_cloud_texture.setUnpackPadding(static_cast<TextureStorageAlignment>(1));
		_3d_point_cloud_texture.setMipmapLevelData(0, PixelLayout::RED, PixelDataType::FLOAT, vlog_density_point_cloud_float);
		_3d_point_cloud_texture.generateMipmapLevels();
		free(vlog_density_point_cloud);

		log_point_cloud.installPointCloud(_3d_point_cloud_texture);
		log_point_cloud.addLightSourceDirection(light);
		log_point_cloud.setScreenSize(uvec2{ DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT });

		log_point_cloud.setDimensions(3.0f, 3.0f, 15.0f);

		log_point_cloud.setBloomMinimalThreshold(0.4f);
		log_point_cloud.setBloomMaximalThreshold(1.0f);
		log_point_cloud.setBloomIntensity(1.7f);


		//Init keyboard callback
		glfwSetKeyCallback(*p_main_screen, GLFWKeyboardCallback);

		//Init mouse button callback
		glfwSetMouseButtonCallback(*p_main_screen, GLFWMouseButtonCallback);

		//Init scrolling callback
		glfwSetScrollCallback(*p_main_screen, GLFWScrollCallback);

		//Init cursor position callback
		glfwSetCursorPosCallback(*p_main_screen, GLFWMouseCallback);
	}

	//Copy constructor
	RenderContext(const RenderContext& other) = delete;

	//Move constructor
	RenderContext(RenderContext&& other) = delete;

	//Copy assignment operator
	RenderContext& operator=(const RenderContext& other) = delete;

	//Move assignment operator
	RenderContext& operator=(RenderContext&& other) = delete;


	//Destructor
	~RenderContext()
	{
		//Shutdown TinyWorld resources
		TextureUnitBlock::reset();

		//Shutdown AntTweakBar
		TwTerminate();

		RenderContext::p_myself = nullptr;
	}


	//Updates current colormap data
	inline void update_color_map_data()
	{
		//Update colormap settings
		GLfloat *colormap_data = new GLfloat[3 * colormap_resolution];

		for (int step = 0; step < static_cast<int>(colormap_resolution); ++step)
		{
			float t = step / (colormap_resolution - 1.0f);
			std::array<float, 3> S = { 0.0f, 0.0f, 0.0f };
			std::list<std::array<float, 3>>::const_iterator F = color_map.begin();

			for (int i = 0; i < static_cast<int>(color_map.size()); ++i)
			{
				float x_i = color_map.size() > 1 ? i / (color_map.size() - 1.0f) : 0.5f;
				float D = 1.0f;
				for (int j = 0; j < static_cast<int>(color_map.size()); ++j)
				{
					float x_j = color_map.size() > 1 ? j / (color_map.size() - 1.0f) : 0.5f;
					if (j != i) D *= (t - x_j) / (x_i - x_j);
				}

				S[0] += (*F)[0] * D;
				S[1] += (*F)[1] * D;
				S[2] += (*F)[2] * D;

				++F;
			}

			colormap_data[step * 3 + 0] = S[0];
			colormap_data[step * 3 + 1] = S[1];
			colormap_data[step * 3 + 2] = S[2];
		}


		colormap_texture.setMipmapLevelData(0, PixelLayout::RGB, PixelDataType::FLOAT, colormap_data);
		delete[] colormap_data;
	}



	//*******************************************************************************************************GUI callback functions***********************************************************************************************************
	static void TW_CALL setOpticalModeCallback(const void* value, void* aux_data)
	{
		p_myself->log_optical_mode = *static_cast<const optical_mode*>(value);

		switch (p_myself->log_optical_mode)
		{
		case GAS:
			TwDefine(" main_bar/num_secondary_samples visible='true' ");
			TwDefine(" main_bar/solid_angle visible='true' ");
			break;

		case ABSORBENT:
			TwDefine(" main_bar/num_secondary_samples visible='true' ");
			TwDefine(" main_bar/solid_angle visible='false' ");
			break;

		case ABSORBENT2:
			TwDefine(" main_bar/num_secondary_samples visible='false' ");
			TwDefine(" main_bar/solid_angle visible='false' ");
			break;
		}
	}

	static void TW_CALL getOpticalModeCallback(void* value, void* aux_data)
	{
		*static_cast<optical_mode*>(value) = p_myself->log_optical_mode;
	}



	static void TW_CALL setOpticalModeParamsCallback(const void* value, void* aux_data)
	{
		uint32_t update_idx = static_cast<uint32_t>(reinterpret_cast<size_t>(aux_data));
		uint32_t optical_mode_idx;
		float medium_color_multiplier = 1.0f;	//needed to adjust visibility for certain optical modes
		switch (p_myself->log_optical_mode)
		{
		case GAS:
			optical_mode_idx = 0;
			medium_color_multiplier = 10.0f;
			break;

		case ABSORBENT:
			optical_mode_idx = 1;
			break;

		case ABSORBENT2:
			optical_mode_idx = 2;
			break;
		}

		switch (update_idx)
		{
		case 0:
			p_myself->log_optical_properties[optical_mode_idx].num_primary_samples = *static_cast<const uint32_t*>(value);
			break;

		case 1:
			p_myself->log_optical_properties[optical_mode_idx].num_secondary_samples = *static_cast<const uint32_t*>(value);
			break;

		case 2:
		{
			const float *fColor = static_cast<const float*>(value);
			p_myself->log_optical_properties[optical_mode_idx].v3MediumColor.x = fColor[0] * medium_color_multiplier;
			p_myself->log_optical_properties[optical_mode_idx].v3MediumColor.y = fColor[1] * medium_color_multiplier;
			p_myself->log_optical_properties[optical_mode_idx].v3MediumColor.z = fColor[2] * medium_color_multiplier;
			break;
		}

		case 3:
			p_myself->log_optical_properties[optical_mode_idx].solid_angle = *static_cast<const float*>(value);
			break;
		}
	}

	static void TW_CALL getOpticalModeParamsCallback(void* value, void* aux_data)
	{
		uint32_t update_idx = static_cast<uint32_t>(reinterpret_cast<size_t>(aux_data));
		uint32_t optical_mode_idx;
		float medium_color_multiplier = 1.0f;	//needed to adjust visibility for certain optical modes
		switch (p_myself->log_optical_mode)
		{
		case GAS:
			optical_mode_idx = 0;
			medium_color_multiplier = 10.0f;
			break;

		case ABSORBENT:
			optical_mode_idx = 1;
			break;

		case ABSORBENT2:
			optical_mode_idx = 2;
			break;
		}

		switch (update_idx)
		{
		case 0:
			*static_cast<uint32_t*>(value) = p_myself->log_optical_properties[optical_mode_idx].num_primary_samples;
			break;

		case 1:
			*static_cast<uint32_t*>(value) = p_myself->log_optical_properties[optical_mode_idx].num_secondary_samples;
			break;

		case 2:
		{
			*static_cast<float*>(value) = p_myself->log_optical_properties[optical_mode_idx].v3MediumColor.x / medium_color_multiplier;
			*(static_cast<float*>(value) + 1) = p_myself->log_optical_properties[optical_mode_idx].v3MediumColor.y / medium_color_multiplier;
			*(static_cast<float*>(value) + 2) = p_myself->log_optical_properties[optical_mode_idx].v3MediumColor.z / medium_color_multiplier;
			break;
		}

		case 3:
			*static_cast<float*>(value) = p_myself->log_optical_properties[optical_mode_idx].solid_angle;
			break;
		}
	}



	static void TW_CALL addColormapKeyvalCallback(void* aux_data)
	{
		if (!p_myself->color_map.size())
		{
			std::array<float, 3> new_keyval = { 1.0f, 1.0f, 1.0f };
			p_myself->color_map.push_back(new_keyval);
			TwAddVarCB(p_myself->p_tw_bar, "color0", TW_TYPE_COLOR3F, setColormapKeyValue, getColormapKeyValue,
				p_myself->color_map.back().data(), " group='Colormap settings'\n label='Key color 1' ");
			p_myself->log_point_cloud.useColormap(true);
		}
		else
		{
			std::array<float, 3> new_keyval = p_myself->color_map.back();
			p_myself->color_map.push_back(new_keyval);
			TwAddVarCB(p_myself->p_tw_bar, ("color" + std::to_string(p_myself->color_map.size() - 1)).c_str(), TW_TYPE_COLOR3F,
				setColormapKeyValue, getColormapKeyValue, p_myself->color_map.back().data(),
				(" group='Colormap settings'\n label='Key color " + std::to_string(p_myself->color_map.size()) + "' ").c_str());
		}

		if (aux_data)
			p_myself->update_color_map_data();
	}

	static void TW_CALL removeColormapKeyvalCallback(void* aux_data)
	{
		if (p_myself->color_map.size())
		{
			TwRemoveVar(p_myself->p_tw_bar, ("color" + std::to_string(p_myself->color_map.size() - 1)).c_str());
			p_myself->color_map.pop_back();

			if (!p_myself->color_map.size())
				p_myself->log_point_cloud.useColormap(false);
			else
				p_myself->update_color_map_data();
		}
	}



	static void TW_CALL setColormapKeyValue(const void* value, void* aux_data)
	{
		float* pData = static_cast<float*>(aux_data);
		pData[0] = *static_cast<const float*>(value);
		pData[1] = *(static_cast<const float*>(value)+1);
		pData[2] = *(static_cast<const float*>(value)+2);

		p_myself->update_color_map_data();
	}

	static void TW_CALL getColormapKeyValue(void* value, void* aux_data)
	{
		float* pValue = static_cast<float*>(value);
		float* pData = static_cast<float*>(aux_data);

		pValue[0] = pData[0];
		pValue[1] = pData[1];
		pValue[2] = pData[2];
	}
	//*****************************************************************************************************************************************************************************************************************************************



	//***********************************************************************************************Screen handling callback functions******************************************************************************************************

	static void framebufferOnRender(Framebuffer& framebufer)
	{
		if (!p_myself) return;

		framebufer.clearBuffers(BufferClearTarget::COLOR_DEPTH);

		p_myself->log_point_cloud.applyViewProjectionTransform(p_myself->main_camera);
		for (int i = 0;
			i < static_cast<int>(p_myself->log_point_cloud.getNumberOfRenderingPasses(p_myself->log_point_cloud.getActiveRenderingMode()));
			++i)
		{
			p_myself->log_point_cloud.prepareRendering(framebufer, i);
			p_myself->log_point_cloud.render();
			p_myself->log_point_cloud.finalizeRendering();
		}

	}


	static void screenOnRender(Screen& screen)
	{
		if (!p_myself) return;

		screen.clearBuffers(BufferClearTarget::COLOR_DEPTH);
		p_myself->hdr_bloom_filter.pass(p_myself->main_camera, screen);

		TwDraw();
	}


	static void screenOnChangeSize(Screen& screen, int new_width, int new_height)
	{

	}

	//*****************************************************************************************************************************************************************************************************************************************



	//*****************************************************************************************************GLFW callback functions**********************************************************************************************************

	static void GLFWKeyboardCallback(GLFWwindow* p_window, int key, int scancode, int actions, int mods)
	{
		const float translation_speed = 0.1f;
		tiny_world::vec3 camera_location = p_myself->main_camera.getLocation();

		switch (key)
		{
		case GLFW_KEY_A:
		case GLFW_KEY_LEFT:
			camera_location.x -= translation_speed;
			break;

		case GLFW_KEY_D:
		case GLFW_KEY_RIGHT:
			camera_location.x += translation_speed;
			break;

		case GLFW_KEY_S:
		case GLFW_KEY_DOWN:
			camera_location.z += translation_speed;
			break;

		case GLFW_KEY_W:
		case GLFW_KEY_UP:
			camera_location.z -= translation_speed;
			break;
		}

		p_myself->main_camera.setLocation(camera_location);
	}

	static void GLFWMouseCallback(GLFWwindow* p_window, double xpos, double ypos)
	{
		TwEventMousePosGLFW(static_cast<int>(xpos), static_cast<int>(ypos));
	}

	static void GLFWMouseButtonCallback(GLFWwindow* p_window, int button, int action, int mods)
	{
		TwEventMouseButtonGLFW(button, action);
	}

	static void GLFWScrollCallback(GLFWwindow* p_window, double xoffset, double yoffset)
	{
		static int wheel_pos = 0;
		wheel_pos = static_cast<int>(wheel_pos + yoffset);

		TwEventMouseWheelGLFW(wheel_pos);
	}

	//*****************************************************************************************************************************************************************************************************************************************



public:

	//Creates new rendering context and returns a pointer for its instance
	static RenderContext* createRenderingContext()
	{
		if (!p_myself)
		{
			RenderContext::p_main_screen = new Screen{ 25, 25, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT };
			//Configure main screen of the program
			p_main_screen->setScreenVideoMode(p_main_screen->getVideoModes()[0]);
			p_main_screen->attachRenderer(screenOnRender);
			p_main_screen->registerOnChangeSizeCallback(screenOnChangeSize);
			p_main_screen->setStringName("BINTEC Virtual Sawing Demo (for internal use only)");
			p_main_screen->defineViewport(Rectangle{ 0, 0, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT });
			p_main_screen->setDepthTestEnableState(true);

			//Initialize TinyWorld engine and create rendering context structure
			ShaderProgram::setShaderBaseCatalog("../tw_shaders/");
			AbstractRenderableObjectTextured::defineTextureLookupPath("../tw_textures/");
			TextureUnitBlock* p_texture_unit_block = TextureUnitBlock::initialize();
			AbstractRenderableObjectTextured::defineTextureUnitBlockGlobalPointer(p_texture_unit_block);

			p_myself = new RenderContext{};

			//Configure basic OpenGL context parameters
			p_main_screen->setClearColor(1.0f, 1.0f, 1.0f, 1.0f);
			p_main_screen->setCullTestEnableState(true);


			//Apply initial settings for the color map
			p_myself->colormap_texture.allocateStorage(1, 1, TextureSize{ p_myself->colormap_resolution, 0, 0 }, InternalPixelFormat::SIZED_FLOAT_RGB32);
			p_myself->colormap_texture.setUnpackPadding(static_cast<TextureStorageAlignment>(1));
			p_myself->colormap_texture.setStringName("virtual_log_colormap");

			addColormapKeyvalCallback(nullptr);
			p_myself->color_map.back().data()[0] = 0;
			p_myself->color_map.back().data()[1] = 0;
			p_myself->color_map.back().data()[2] = 0.01f;

			addColormapKeyvalCallback(nullptr);
			p_myself->color_map.back().data()[0] = 0.3f;
			p_myself->color_map.back().data()[1] = 0.3f;
			p_myself->color_map.back().data()[2] = 0;

			addColormapKeyvalCallback(nullptr);
			p_myself->color_map.back().data()[0] = 0.8f;
			p_myself->color_map.back().data()[1] = 0;
			p_myself->color_map.back().data()[2] = 0;

			p_myself->update_color_map_data();
			p_myself->log_point_cloud.installColormap(p_myself->colormap_texture);
			p_myself->log_point_cloud.useColormap(true);
		}

		return p_myself;
	}


	//Destroys previously created rendering context
	static void reset()
	{
		if (p_myself)
		{
			delete p_myself;
			delete p_main_screen;
		}
	}


	//Draws the scene and returns 'true' if the scene should yet be rendered next time.
	//If function returns 'false', this means that the process should exit
	bool process()
	{
		//Apply rendering parameters
		log_point_cloud.resetObjectRotation();
		log_point_cloud.applyRotation(tiny_world::quaternion{ log_rotation[3], log_rotation[0], log_rotation[1], log_rotation[2] }, RotationFrame::LOCAL);

		log_point_cloud.setLocation(tiny_world::vec3{ log_location[0], log_location[1], log_location[2] });

		log_point_cloud.selectRenderingMode(log_optical_mode);


		uint32_t optical_mode_idx;
		switch (log_optical_mode)
		{
		case GAS:
			optical_mode_idx = 0;
			break;

		case ABSORBENT:
			optical_mode_idx = 1;
			break;

		case ABSORBENT2:
			optical_mode_idx = 2;
			break;
		}

		log_point_cloud.setNumberOfPrimarySamples(log_optical_properties[optical_mode_idx].num_primary_samples);
		log_point_cloud.setNumberOfSecondarySamples(log_optical_properties[optical_mode_idx].num_secondary_samples);
		log_point_cloud.setMediumUniformColor(log_optical_properties[optical_mode_idx].v3MediumColor);
		log_point_cloud.setSolidAngle(log_optical_properties[optical_mode_idx].solid_angle);


		//Draw scene
		framebuffer.makeActive();
		framebuffer.refresh();

		p_main_screen->makeActive();
		p_main_screen->refresh();

		return !p_main_screen->shouldClose();
	}
};

RenderContext* RenderContext::p_myself = nullptr;
Screen* RenderContext::p_main_screen = nullptr;


int main(int argc, char* argv[])
{
	RenderContext* p_render_context = RenderContext::createRenderingContext();

	//Start the main rendering loop
	while (p_render_context->process());

	p_render_context->reset();

	return EXIT_SUCCESS;
}

