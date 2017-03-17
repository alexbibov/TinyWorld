//Implements functionality of the shallow water demo

#ifndef CLOUDS_SCENE
#define CLOUDS_SCENE


#include <cstdlib>


#include "AntTweakBar.h"


#include "../TinyWorld/Screen.h"
#include "../TinyWorld/Framebuffer.h"
#include "../TinyWorld/KTXTexture.h"
#include "../TinyWorld/Cube.h"
#include "../TinyWorld/TessellatedTerrain.h"
#include "../TinyWorld/FullscreenRectangle.h"
#include "../TinyWorld/Skydome.h"
#include "../TinyWorld/SSFilter_HDRBloom.h"
#include "../TinyWorld/SSFilter_SSAO.h"
#include "../TinyWorld/SSFilter_Blur.h"
#include "../TinyWorld/SSFilter_ImmediateShader.h"
#include "../TinyWorld/SSFilter_AtmosphericFog.h"
#include "../TinyWorld/SSFilter_LightHaze.h"
#include "../TinyWorld/StaticClouds.h"


using namespace tiny_world;


class CloudsScene final
{
private:
    struct RayleighScatteringCoefficients
    {
        float wavelength_red;
        float wavelength_green;
        float wavelength_blue;
    };


    static CloudsScene* p_myself;		//pointer to the instance of the class (the class is a singleton)
    static Screen* p_screen;	//pointer to the screen object which receives the rendered data
    static const float pi;	//the value of pi

    static const vec3 v3SunLight;	//light intencity of the sun
    static const vec3 v3MoonLight;	//light intencity of the moon


    bool error_state;	//equals 'true' if the object is in an erroneous state, equals 'false' otherwise

    PerspectiveProjectingDevice main_camera;	//main camera of the scene
    PerspectiveProjectingDevice reflection_camera;	//camera used to generate the global cubemap
    Framebuffer rendering_composition_framebuffer;	//framebuffer used for rendering composition
    Framebuffer reflection_framebuffer;	//reflection framebuffer
    ImmutableTextureCubeMap reflection_map;	//global reflection map used by the scene
    ImmutableTexture2D reflection_map_depth_buffer;		//depth buffer employed by reflection map
    ImmutableTexture2D refraction_map;	//global refraction map used by the scene
    ImmutableTexture2D normal_map;	//normal map of the scene
    ImmutableTexture2D ad_map;	//ambient modulated diffuse color map
    ImmutableTexture2D linear_depth_buffer;	//linear depth buffer of the scene
    ImmutableTexture2D color_buffer;	//screen color texture
    ImmutableTexture2D bloom_texture;	//scene bloom texture
    ImmutableTexture2D depth_buffer;	//screen depth buffer
    ImmutableTexture2D selection_buffer;	//selection buffer storage
    TextureUnitBlock* p_tex_unit_block;	//pointer to the texture unit block
    TessellatedTerrain tess_terrain;	//tessellated terrain
    AmbientLight ambient_light;	//ambient light of the scene
    DirectionalLight skybody_light;		//light of the sky body (sun or moon)
    StaticClouds clouds;    //Clouds
    Skydome skydome;	//Skydome
    LightingConditions lighting;	//lighting conditions descriptor

                                    //SS-filters
    SSFilter_HDRBloom hdr_bloom;	//HDR-Bloom screen space filter
    SSFilter_SSAO ssao;	//screen-space ambient occlusion filter
    SSFilter_Blur bloom_blur_x, bloom_blur_y, ssao_blur_x, ssao_blur_y;	//blur filters
    SSFilter_ImmediateShader immediate_shader;	//immediate renderer
    SSFilter_AtmosphericFog atmospheric_fog_filter;	//screen-space filter implementing the atmospheric fog
    SSFilter_LightHaze light_haze_filter;	//screen-space filter implementing light haze effect
    CascadeFilterLevel CL0, CL1, CL2, CL3, CL4, CL5, CL6, CL7;	//cascade filter levels
    CascadeFilter postprocess;	//cascade filter that performs post-processing


    float* p_raw_topography_data;	//raw topography height map as it appears before interpolation
    uint32_t topography_x_res, topography_y_res;	//resolution of the topography height map
    float topography_offset, topography_max_height;	//the offset height and the maximal height of the topography
    float modification_strength;	//strength of modifications that could be applied via user interaction


                                    //Control bar parameters
    TwBar* p_main_bar;
    bool suppress_mouse_user_input;	//when equals 'true' camera movements are suppressed
    float daytime;	//current time of the day
    float fog_density;	//density of the atmospheric fog
    float fog_height_fall_off;	//height fall off rate applied to the atmospheric fog

    typedef enum { RAISE_TERRAIN, LOWER_TERRAIN, SLIDE_TERRAIN } UserInteractionMode; //allowed user interaction modes

    UserInteractionMode interaction_mode;


    CloudsScene(const std::string& topography_file_name, uint32_t reflection_map_resolution);	//initialization of the object requires the caller to provide a screen object for which to perform the rendering
    CloudsScene(const CloudsScene& other) = delete;	//no copying allowed
    CloudsScene(CloudsScene&& other) = delete;	//no moves allowed
    CloudsScene& operator=(const CloudsScene& other) = delete;	//no copy assignments are possible
    CloudsScene& operator=(CloudsScene&& other) = delete;	//no move assignments are possible
    ~CloudsScene();	//destructor


                            //Event handlers
    static void onScreenRedraw(Screen& target_screen);
    static void onScreenSizeChange(Screen& screen, int width, int height);
    static void onSceneRedraw(Framebuffer& target_framebuffer);
    static void onReflectionMapUpdate(Framebuffer& target_framebuffer);
    static void onKeyPress(GLFWwindow* p_window, int key, int scancode, int actions, int mods);
    static void onMouseMove(GLFWwindow* p_window, double xpos, double ypos);
    static void onMouseClick(GLFWwindow* p_glfw_window, int button, int action, int mods);
    static void onScroll(GLFWwindow* p_glfw_window, double xoffset, double yoffset);
    static void onCharInput(GLFWwindow* p_glfw_window, unsigned int codepoint);

    static void TW_CALL onParamSet(const void* value, void* client_data);
    static void TW_CALL onParamGet(void* value, void* client_data);


    //Miscellaneous
    inline void init_ss_filters();	//initializes the screen-space filters
    inline void init_rendering_composer();	//initializes rendering compositing framebuffer
    inline void init_reflection_buffer(uint32_t reflection_buffer_resolution);	//initializes reflection framebuffer
    inline void init_toolbar();	//initializes the toolbar



public:
    static CloudsScene* initializeScene(const std::string& title, uint32_t init_screen_width, uint32_t init_screen_height, const std::string& topography_file_name,
        const std::string& shader_base_catalog, const std::string& texture_base_catalog, uint32_t reflection_map_resolution = 1024U);	//creates an instance of the scene

    static void destroyScene();	//destroys the scene

    static bool updateScene();	//updates the scene and returns 'true' if the scene should be updated also next time or 'false' if the application should exit
};



#endif