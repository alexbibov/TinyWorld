//Implements concept of the skydome with dynamically changing daytime, sky light scattering and dynamical sky bodies (sun, moon, clouds and stars)
//Skydome does not provide support for levitating camera, i.e. the camera is assumed to remain close to earth for the scattering effects to work correctly
//For planet-size light scattering allowing in- and out- atmosphere rendering use object Atmosphere instead of Skydome

#ifndef TW__SKYDOME_H__

#include <random>
#include <cstdint>

#include "VectorTypes.h"
#include "AbstractRenderableObject.h"
#include "AbstractRenderableObjectHDRBloomEx.h"
#include "CompleteShaderProgram.h"
#include "ImageUnit.h"

namespace tiny_world
{
    //Describes scattering properties of the atmosphere
    struct LightScatteringSettings
    {
        float red_wavelength;		//wavelength of the red channel
        float green_wavelength;		//wavelength of the green channel
        float blue_wavelength;		//wavelength of the blue channel

        float rayleigh_scale;	//Scale factor applied to Rayleigh scattering coefficient
        float mie_coefficient;		//Mie scattering coefficient
        float mie_phase_function_param;		//Parameter of the Mie phase function

        //Ratio between a single unit of length in the scattering space and the distance to the far clipping plane
        //in the world space. In other words, this parameter defines how many distances to the far clipping plane
        //can be tiled along the vertical line shot towards the top of the sky sphere
        float scattering_to_far_world_ratio;

        LightScatteringSettings();		//initializes the structure using default scattering parameters


        bool operator==(const LightScatteringSettings& other) const;	//field-to-field comparison
    };




    class Skydome : public AbstractRenderableObjectTextured,  public AbstractRenderableObjectExtensionAggregator<AbstractRenderableObjectHDRBloomEx>
    {
        friend class LightingConditions_Core;

    private:
        static const std::string skydome_rendering_program_name;
        static const std::string star_field_rendering_program_name;
        static const std::string moon_rendering_program_name;

        const uint32_t num_vertices;	//total number of vertices needed to represent the sky dome, stars, moon and sun
        const uint32_t vertex_buffer_capacity;	//capacity of the vertex buffer needed to store general vertex attributes
        const std::uniform_real_distribution<float> scintillation_parameter;	//parameter used to model scintillations of stars

        std::default_random_engine default_random_generator;	//random generator used to produce positions and colors of stars and to model star scintillations

        uint32_t *ref_counter;	//reference counter

        //Radius of the sphere representing sky "surface".
        float radius;

        //A value between 0 and 1 determining time of the day.
        //The values between 0 and 0.5 correspond to the day time, and the values between 0.5 and 1.0 correspond to the night time.
        float daytime;

        uint32_t num_stars;		//number of stars to render in the night sky

        uint32_t latitude_steps;	//number of latitude steps for sphere discretization. The minimal value for this parameter is 2
        uint32_t longitude_steps;	//number of longitude steps for discretization of each of the hemispheres. The minimal value for this parameter is 3

        vec3 sun_direction;	//normalized direction vector to the sun light
        vec3 sun_intensity;	//intensity of the sun light

        vec3 moon_direction;	//normalized direction vector to the moon light
        vec3 moon_intensity;	//intensity of the moon light
        float moon_nominal_size;	//nominal radius of the moon disk as it appears to the viewer


        LightScatteringSettings scattering_settings;	//scattering settings applied to the atmosphere

        ShaderProgramReferenceCode skydome_rendering_program_ref_code;	//reference code of the shader program implementing shading of the sky dome
        ShaderProgramReferenceCode star_field_rendering_program_ref_code;	//reference code of the shader program implementing shading of a star field
        ShaderProgramReferenceCode moon_rendering_program_ref_code;		//reference code of the shader program implementing moon shading

        TextureReferenceCode star_ref_code;			//reference code of the star texture
        TextureReferenceCode sun_ref_code;			//reference code of the texture of sun
        TextureReferenceCode moon_ref_code;			//reference code of of the texture of moon

        TextureSamplerReferenceCode clamping_sampler_ref_code;	//clamping sampler applied to the textures (sun, moon, stars, and clouds)


        ImmutableTexture2D in_scattering_sun;	//contains in-scattering of each vertex of the sky dome due to the sun light
        ImmutableTexture2D in_scattering_moon;	//contains in-scattering of each vertex of the sky dome due to the moon light



        //**********************Raw OpenGL resources**********************
        GLuint ogl_vertex_attribute_objects[4];
        GLuint ogl_array_buffer_objects[2];
        //****************************************************************

        uint32_t current_rendering_pass;	//currently active rendering pass

        //Structure containing values of the OpenGL state components that get affected by rendering of the sky dome
        struct gl_state{
            GLboolean primitive_restart_enable_state;	//equals 'true' if GL_PRIMITIVE_RESTART capability was enabled before invocation of prepareRendering(...)
            GLboolean blend_enable_state;	//equals 'true' if GL_BLEND capability was enabled before invocation of prepareRendering(...)
            GLboolean program_point_size_enable_state;	//equals 'true' if GL_PROGRAM_POINT_SIZE capability was enabled before rendering of the sky dome
            GLboolean depth_mask_state;		//equals 'true' if the depth buffer was enabled for writing before rendering of the sky dome
            GLint primitive_restart_index;	//value of primitive restart index valid prior to rendering of the sky dome
            GLint src_rgb;					//value of the source RGB-channel blend function valid prior to rendering of the sky dome
            GLint src_alpha;				//value of the source Alpha-channel blend function valid prior to rendering of the sky dome
            GLint dst_rgb;					//value of the destination RGB-channel blend function valid prior to rendering of the sky dome
            GLint dst_alpha;				//value of the destination Alpha-channel blend function valid prior to rendering of the sky dome
            GLint blend_eq_rgb;				//blend equation that was in use for RGB-channel blending before rendering of the sky dome
            GLint blend_eq_alpha;			//blend equation that was in use for Alpha-channel blending before rendering of the sky dome
        }gl_context_state;


        void applyScreenSize(const uvec2& screen_size) override;
        bool configureRendering(AbstractRenderingDevice& render_target, uint32_t rendering_pass) override;
        void configureViewProjectionTransform(const AbstractProjectingDevice& projecting_device) override;
        bool configureRenderingFinalization() override;

        void init_random_generator();	//initializes random generator to ensure that it produces different results on each run
        void generate_vertex_data();	//generates vertex data for the sky dome
        void init_skydome();	//runs initialization routines for the sky dome object
        void free_ogl_resources();	//destroys OpenGL resources associated with the object

        vec3 celestial_to_direction(const vec2& celestial_location) const;	//converts celestial coordinates into direction vector
        vec2 direction_to_celestial(const vec3& direction_vector) const;	//converts direction vector into coordinates of celestial location

        vec3 celestial_to_cartesian(const vec2& celestial_location) const;	//converts celestial coordinates to default Cartesian coordinates

    public:
        //Initializes new sky dome with given discretization parameters and with given daytime
        Skydome(float daytime = 0.0f, uint32_t num_night_sky_stars = 1000, uint32_t latitude_steps = 128, uint32_t longitude_steps = 128);

        //Initializes new sky dome with given radius, discretization parameters and day time.
        //The sky dome will use weak identification by the user-defined string name
        Skydome(std::string skydome_string_name, float radius, float daytime = 0.0f, uint32_t num_night_sky_stars = 1000, uint32_t latitude_steps = 128, uint32_t longitude_steps = 128);

        //Copy constructor
        Skydome(const Skydome& other);

        //Assignment operator
        Skydome& operator = (const Skydome& other);

        //Destructor
        ~Skydome();

        void setRadius(float new_radius);	//assigns new radius for the sky dome
        float getRadius() const;	//returns radius of the sky dome

        //This function applies new daytime value to the sky dome. Note, that calling this function will also update location and light intensity of
        //sun and moon. If you want to manually control sun and moon behavior with respect to the day time change, you
        //would prefer to avoid calling this function as doing so will override the custom settings.
        //NOTE: The usual 24-hour day cycle is normalized to the range of [0, 1), i.e. any value for the day time outside this range will be periodically mapped onto it
        void setDaytime(float new_daytime);

        float getDaytime() const;	//returns actual day time of the sky dome
        bool isDay() const;		//returns 'true' if the current normalized time of day belongs to [0, 0.5)
        bool isNight() const;	//returns 'true' if the current normalized time of day belongs to [0.5, 1)


        void applySunTexture(const ImmutableTexture2D& new_sun_texture);	//assigns new texture for sun
        void applyMoonTexture(const ImmutableTexture2D& new_moon_texture);	//assigns new texture for moon

        void setAtmosphereScatteringSettings(const LightScatteringSettings& scattering_settings);	//assigns new settings for atmospheric scattering
        LightScatteringSettings getAtmosphericScatteringSettings() const;		//returns atmospheric scattering parameters currently in use

        void setSunLightIntensity(const vec3& intensity);		//sets intensity of the sun light
        vec3 getSunLightIntensity() const;		//returns intensity of the sun light

        //Assigns new sun location represented in horizontal celestial coordinates.
        //Location latitude is periodically mapped onto the range [-pi / 6, pi + pi / 6] and longitude is periodically mapped onto the range [0, 2*pi].
        void setSunLocation(const vec2& location);

        vec2 getSunLocation() const;	//returns valid sun location represented in horizontal celestial coordinates

        vec3 getSunDirection() const;    //returns current direction to sun

        void setMoonLightIntensity(const vec3& intensity);		//sets intensity for the moon light
        vec3 getMoonLightIntensity() const;		//returns intensity of the moon light

        //Assigns new moon location represented in horizontal celestial coordinates.
        //Location latitude is periodically mapped onto the range [-pi / 6, pi + pi / 6] and longitude is periodically mapped onto the range [0, 2*pi].
        void setMoonLocation(const vec2& location);

        vec2 getMoonLocation() const;	//returns valid moon location represented in horizontal celestial coordinates

        vec3 getMoonDirection() const;     //returns current direction to moon

        void setNominalMoonSize(float size);	//sets nominal size of the moon as appears from viewer's position
        float getNominalMoonSize() const;	//returns current nominal size of the moon disk


        //Rendering infrastructure of the object
        bool supportsRenderingMode(uint32_t rendering_mode) const override;
        uint32_t getNumberOfRenderingPasses(uint32_t rendering_mode) const override;
        bool render() override;
    };
}


#define TW__SKYDOME_H__
#endif