#ifndef TW__STATIC_CLOUDS__
#define TW__STATIC_CLOUDS__

#include "AbstractRenderableObject.h"
#include "AbstractRenderableObjectHDRBloomEx.h"
#include "AbstractRenderableObjectSelectionEx.h"
#include "LightingConditions.h"
#include "std140UniformBuffer.h"
#include "LightingConditions.h"

#include <random>

namespace tiny_world
{

class StaticClouds final : virtual public AbstractRenderableObjectTextured,
    virtual public AbstractRenderableObjectExtensionAggregator<
    AbstractRenderableObjectHDRBloomEx,
    AbstractRenderableObjectSelectionEx>
{
private:
    static const std::string preprocess_program_name;
    static const std::string rendering_program_name;
    static const uint32_t cloud_particle_texture_resolution = 32U;
    static const uint32_t preprocess_group_size_x = 8U;
    static const uint32_t preprocess_group_size_y = 8U;
    static const uint32_t preprocess_group_size_z = 8U;

    vec3 v3DomainSize;    // size of the domain
    uvec3 uv3DomainResolution;    //vertex resolution of the cloud domain
    TextureSamplerReferenceCode texture_sampler_ref_code;    //texture sampler employed by cloud renderer
    TextureReferenceCode density_pattern_texture_ref_code;    //reference code of the density pattern texture
    TextureReferenceCode density_pattern_random_shifts_ref_code;    //reference code of the buffer texture containing random shifts for particle densities
    ImmutableTexture3D out_scattering_values;    //buffer texture containing preprocessed out-scattering values
    TextureReferenceCode out_scattering_values_ref_code;    //reference code of the out-scattering values texture. Used to automatically handle texture binding units.
    TextureReferenceCode sun_in_scattering_ref_code;
    TextureReferenceCode moon_in_scattering_ref_code;

    std::default_random_engine random_engine;
    std::uniform_real_distribution<float> density_pattern_shift_distribution;

    const uvec2 v2CloudDensityPatternResolution;    //resolution of the density pattern texture
    float albedo;    //albedo of the clouds
    float domega;    //solid angle differential
    float density_scale;    //scaling factor applied to density kernels
    float particle_size;    //size of single cloud particle

    GLuint ogl_vertex_attribute_object;    //native OpenGL vertex attribute object
    GLuint ogl_vertex_buffer;    //native OpenGL vertex buffer

    ShaderProgramReferenceCode preprocess_program_ref_code;    //reference code of compute program used for preprocessing
    ShaderProgramReferenceCode rendering_program_ref_code;    //reference code of shading program employed for final rendering

    int rendering_pass;    //active rendering pass

    const LightingConditions* p_lighting_conditions;    //lighting conditions

    AbstractRenderingDevice* p_last_render_targer;

    void applyScreenSize(const uvec2& screen_size) override;
    bool configureRendering(AbstractRenderingDevice& render_target, uint32_t rendering_pass) override;
    void configureViewProjectionTransform(const AbstractProjectingDevice& projecting_device) override;
    bool configureRenderingFinalization() override;


    inline void setup_object();	//runs initial procedures required on object's initialization

public:
    StaticClouds();

    //Initializes clouds simulation given the spatial size and the resolution of the simulation domain
    StaticClouds(const vec3& cloud_domain_size, const uvec3& cloud_domain_resolution);

    //Copy initialization
    StaticClouds(const StaticClouds& other);

    //Move initialization
    StaticClouds(StaticClouds&& other);

    //Copy assignment
    StaticClouds& operator=(const StaticClouds& other);

    //Move assignment
    StaticClouds& operator=(StaticClouds&& other);

    //Destructor
    ~StaticClouds();


    vec3 getDomainDimensions() const;	//retrieves spatial dimensions of the cloud domain

    void setAlbedo(float albedo);    //sets albedo of the clouds
    float getAlbedo() const;    //retrieves albedo of the clouds

    void setSolidAngleDifferentialScale(float scale);    //applies scale of solid angle differential
    float getSolidAngleDifferentialScale() const;    //retrieves scale of solid angle differential

    void setDensityScale(float scale);    //applies scaling factor applied to cloud density kernels
    float getDensityScale() const;    //returns scaling factor applied to cloud density kernels

    void setParticleSize(float size);    //sets size of cloud particles
    float getParticleSize() const;    //retrieves size of cloud particles

    void setLightingConditions(const LightingConditions& lighting_conditions);
    const LightingConditions* getLightingConditions() const;

   //Standard infrastructure of a drawable object

    bool supportsRenderingMode(uint32_t rendering_mode) const override;
    uint32_t getNumberOfRenderingPasses(uint32_t rendering_mode) const override;
    bool render() override;
};

}

#endif