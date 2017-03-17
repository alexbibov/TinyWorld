#include "StaticClouds.h"
#include "ImageUnit.h"

using namespace tiny_world;

const std::string StaticClouds::preprocess_program_name = "static_clouds_preprocess_program";
const std::string StaticClouds::rendering_program_name = "static_clouds_rendering_program";

namespace {

uvec3 calculateNumberOfGroupsForDispatchCompute(uint32_t problem_size_x, uint32_t problem_size_y, uint32_t problem_size_z,
    uint32_t group_size_x, uint32_t group_size_y, uint32_t group_size_z)
{
    return uvec3{
        problem_size_x / group_size_x + static_cast<uint32_t>(problem_size_x % group_size_x != 0),
        problem_size_y / group_size_y + static_cast<uint32_t>(problem_size_y % group_size_y != 0),
        problem_size_z / group_size_z + static_cast<uint32_t>(problem_size_z % group_size_z != 0)
    };
}

uint16_t to_half_precision_float(float fp_float)
{
    uint32_t fp_float_bits = *reinterpret_cast<uint32_t*>(&fp_float);
    uint8_t fp_exp_bits = ((fp_float_bits & 0x7F800000) << 1) >> 24;
    uint32_t fp_fraction_bits = fp_float_bits & 0x7FFFFF;
    uint16_t hp_float_bits = 0x0;

    hp_float_bits |= fp_float_bits >> 31 << 15;    //copy sign bit

    int hp_exp_bits = fp_exp_bits - 127 + 15;
    if (hp_exp_bits < 0) return hp_float_bits;
    if (hp_exp_bits > 31)
    {
        hp_float_bits |= 0x7C00;
        return hp_float_bits;
    }

    hp_float_bits |= hp_exp_bits << 10;
    hp_float_bits |= hp_float_bits >> 13;

    return hp_float_bits;
}

}

void StaticClouds::applyScreenSize(const uvec2& screen_size)
{
    
}

bool StaticClouds::configureRendering(AbstractRenderingDevice& render_target, uint32_t rendering_pass)
{
    this->rendering_pass = rendering_pass;

    switch (rendering_pass)
    {
    case 0:
        COMPLETE_SHADER_PROGRAM_CAST(retrieveShaderProgram(preprocess_program_ref_code)).activate();
        return true;

    case 1:
        if (!render_target.isActive())
            render_target.makeActive();

        COMPLETE_SHADER_PROGRAM_CAST(retrieveShaderProgram(rendering_program_ref_code)).activate();

        auto viewport = render_target.getViewportRectangle(0);
        retrieveShaderProgram(rendering_program_ref_code)->assignUniformVector("v4Viewport", vec4{ viewport.x, viewport.y, viewport.w, viewport.h });

        //Bind object's data buffer
        glBindVertexArray(ogl_vertex_attribute_object);

        glPointSize(particle_size);

        render_target.pushOpenGLContextSettings();
        render_target.setDepthTestEnableState(false);
        render_target.setColorBlendEnableState(true);
        render_target.setRGBSourceBlendFactor(ColorBlendFactor::DST_COLOR);
        render_target.setRGBDestinationBlendFactor(ColorBlendFactor::ZERO);
        render_target.setBlendEquation(ColorBlendEquation::ADD);

        p_last_render_targer = &render_target;
        

        return true;
    }


    return false;
}

void StaticClouds::configureViewProjectionTransform(const AbstractProjectingDevice& projecting_device)
{
    mat4 ModelViewTransform = projecting_device.getViewTransform()*getObjectTransform()*getObjectScaleTransform();
    retrieveShaderProgram(rendering_program_ref_code)->assignUniformVector("v3Scale", getObjectScale());
    retrieveShaderProgram(rendering_program_ref_code)->assignUniformMatrix("m4ModelViewTransform", ModelViewTransform);
    retrieveShaderProgram(rendering_program_ref_code)->assignUniformMatrix("m4ProjectionTransform", projecting_device.getProjectionTransform());

    mat4 m4WorldToScaledObjectTransform = getObjectTransform().inverse();
    vec4 v4ObserverLocationInScaledObjectSpace = m4WorldToScaledObjectTransform*vec4{ projecting_device.getLocation(), 1.f };

    retrieveShaderProgram(rendering_program_ref_code)->assignUniformMatrix("m4ProjectionToScaledObjectTransform",
        m4WorldToScaledObjectTransform*projecting_device.getViewTransform().inverse()*projecting_device.getProjectionTransform().inverse());
    retrieveShaderProgram(rendering_program_ref_code)->assignUniformVector("v3ObserverLocation", 
        vec3{v4ObserverLocationInScaledObjectSpace.x, v4ObserverLocationInScaledObjectSpace.y, v4ObserverLocationInScaledObjectSpace.z}
    / v4ObserverLocationInScaledObjectSpace.w);



    vec3 light_direction = p_lighting_conditions->getSkydome()->isDay() ?
        p_lighting_conditions->getSkydome()->getSunDirection() :
        p_lighting_conditions->getSkydome()->getMoonDirection();
    vec4 light_direction_homogeneous{ light_direction, 0.f };

    light_direction_homogeneous = m4WorldToScaledObjectTransform*light_direction_homogeneous;
    light_direction.x = light_direction_homogeneous.x;
    light_direction.y = light_direction_homogeneous.y;
    light_direction.z = light_direction_homogeneous.z;
    retrieveShaderProgram(preprocess_program_ref_code)->assignUniformVector("v3LightDirection", light_direction.get_normalized());
    retrieveShaderProgram(preprocess_program_ref_code)->assignUniformVector("v3Scale", getObjectScale());

    retrieveShaderProgram(rendering_program_ref_code)->assignUniformVector("v3LightDirection", light_direction.get_normalized());
}

bool StaticClouds::configureRenderingFinalization()
{
    if(p_last_render_targer)
        p_last_render_targer->popOpenGLContextSettings();
    return true;
}

inline void StaticClouds::setup_object()
{
    texture_sampler_ref_code = createTextureSampler("Clouds::texture_sampler");

    //generate texture containing cloud densities
    float* density_pattern_data = new float[cloud_particle_texture_resolution*cloud_particle_texture_resolution];

    for(uint32_t i = 0 ; i < cloud_particle_texture_resolution; ++i)
        for (uint32_t j = 0; j < cloud_particle_texture_resolution; ++j)
        {
            float x = -1.f + 2.f*i / (cloud_particle_texture_resolution - 1.f);
            float y = -1.f + 2.f*j / (cloud_particle_texture_resolution - 1.f);
            density_pattern_data[i*cloud_particle_texture_resolution + j] = std::exp(-(x*x + y*y));
        }

    ImmutableTexture2D density_pattern_texture{ "Clouds::density_pattern_texture" };
    density_pattern_texture.allocateStorage(1, 1, TextureSize{ cloud_particle_texture_resolution, cloud_particle_texture_resolution, 1 }, InternalPixelFormat::SIZED_FLOAT_R16);
    density_pattern_texture.setMipmapLevelData(0, PixelLayout::RED, PixelDataType::FLOAT, density_pattern_data);
    delete[] density_pattern_data;

    if (!density_pattern_texture_ref_code)
        density_pattern_texture_ref_code = registerTexture(density_pattern_texture, texture_sampler_ref_code);


    //calculate random shifts
    BufferTexture random_shifts_texture{ "Clouds::density_pattern_random_shifts_texture" };
    random_shifts_texture.allocateStorage(uv3DomainResolution.x*uv3DomainResolution.y*uv3DomainResolution.z * 4, BufferTextureInternalPixelFormat::SIZED_FLOAT_RG16);
    uint32_t* random_shifts = static_cast<uint32_t*>(random_shifts_texture.map(BufferTextureAccessPolicy::WRITE));
    for (uint32_t i = 0; i < uv3DomainResolution.x*uv3DomainResolution.y*uv3DomainResolution.z; ++i)
    {
        uint16_t u_shift = to_half_precision_float(density_pattern_shift_distribution(random_engine));
        uint16_t v_shift = to_half_precision_float(density_pattern_shift_distribution(random_engine));
        random_shifts[i] = static_cast<uint32_t>(v_shift) << 16 | u_shift;
    }
    random_shifts_texture.unmap();
    if (!density_pattern_random_shifts_ref_code)
        density_pattern_random_shifts_ref_code = registerTexture(random_shifts_texture);


    out_scattering_values.allocateStorage(1, 1, TextureSize{ uv3DomainResolution.x, uv3DomainResolution.y, uv3DomainResolution.z }, InternalPixelFormat::SIZED_FLOAT_R16);
    if (!out_scattering_values_ref_code)
        out_scattering_values_ref_code = registerTexture(out_scattering_values, texture_sampler_ref_code);


    //setup OpenGL objects
    glGenVertexArrays(1, &ogl_vertex_attribute_object);
    glBindVertexArray(ogl_vertex_attribute_object);
    glEnableVertexAttribArray(vertex_attribute_position::getId());
    vertex_attribute_position::setVertexAttributeBufferLayout(0, 0);

    //create and populate vertex buffers
    glGenBuffers(1, &ogl_vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, ogl_vertex_buffer);


    const size_t vb_typed_element_size = vertex_attribute_position::getSize();
    const size_t vb_untyped_element_size = vertex_attribute_position::getCapacity();
    const size_t vb_size = vb_untyped_element_size*uv3DomainResolution.x*uv3DomainResolution.y*uv3DomainResolution.z;


    void* vertex_buf = malloc(vb_size);

    for (uint32_t k = 0; k < uv3DomainResolution.z; ++k)
    {
        size_t offset_layer = k*uv3DomainResolution.x*uv3DomainResolution.y;

        for (uint32_t i = 0; i < uv3DomainResolution.x; ++i)
        {
            size_t offset_row = offset_layer + i*uv3DomainResolution.y;

            for (uint32_t j = 0; j < uv3DomainResolution.y; ++j)
            {
                size_t offset = (offset_row + j)*vb_typed_element_size;

                float x = -1.f + 2.f*i / (uv3DomainResolution.x - 1.f);
                float y = -1.f + 2.f*j / (uv3DomainResolution.y - 1.f);
                float z = -1.f + 2.f*k / (uv3DomainResolution.z - 1.f);

                static_cast<float*>(vertex_buf)[offset + 0] = x;
                static_cast<float*>(vertex_buf)[offset + 1] = y;
                static_cast<float*>(vertex_buf)[offset + 2] = z;
                static_cast<float*>(vertex_buf)[offset + 3] = 1.f;
            }
        }
    }

    glBufferData(GL_ARRAY_BUFFER, vb_size, vertex_buf, GL_STATIC_DRAW);
    delete[] vertex_buf;

    glBindVertexBuffer(0U, ogl_vertex_buffer, 0, static_cast<GLsizei>(vb_untyped_element_size));



    //setup shader programs
    if (!preprocess_program_ref_code)
    {
        preprocess_program_ref_code =
            createCompleteShaderProgram(preprocess_program_name, { PipelineStage::COMPUTE_SHADER });

        Shader compute_shader{ ShaderProgram::getShaderBaseCatalog() + "CloudsPreprocess.cs.glsl", ShaderType::COMPUTE_SHADER,
            "CalculateCloudsOutScattering" };

        retrieveShaderProgram(preprocess_program_ref_code)->addShader(compute_shader);
        retrieveShaderProgram(preprocess_program_ref_code)->link();
    }

    if (!rendering_program_ref_code)
    {
        rendering_program_ref_code =
            createCompleteShaderProgram(rendering_program_name, { PipelineStage::VERTEX_SHADER, PipelineStage::FRAGMENT_SHADER });

        Shader vertex_shader{ ShaderProgram::getShaderBaseCatalog() + "CloudsInScattering.vp.glsl", ShaderType::VERTEX_SHADER,
            "CalculateCloudsInScattering.vp" };
        Shader fragment_shader{ ShaderProgram::getShaderBaseCatalog() + "CloudsInScattering.fp.glsl", ShaderType::FRAGMENT_SHADER,
            "CalculateCloudsInScattering.fp" };

        retrieveShaderProgram(rendering_program_ref_code)->addShader(vertex_shader);
        retrieveShaderProgram(rendering_program_ref_code)->addShader(fragment_shader);

        retrieveShaderProgram(rendering_program_ref_code)->bindVertexAttributeId("v4VertexPosition", vertex_attribute_position::getId());

        retrieveShaderProgram(rendering_program_ref_code)->link();
    }
}

StaticClouds::StaticClouds():
    v3DomainSize{ 1.f },
    uv3DomainResolution{ 20U },
    out_scattering_values{"Clouds::out_scattering_values"},
    v2CloudDensityPatternResolution{ cloud_particle_texture_resolution },
    albedo{ 1.f },
    domega{ .01f },
    density_scale{ 1.f },
    particle_size{ 10.f },
    rendering_pass{ -1 }
{
    setup_object();
}

StaticClouds::StaticClouds(const vec3& cloud_domain_size, const uvec3& cloud_domain_resolution):
    v3DomainSize{ cloud_domain_size },
    uv3DomainResolution{ cloud_domain_resolution },
    out_scattering_values{ "Clouds::out_scattering_values" },
    v2CloudDensityPatternResolution{ cloud_particle_texture_resolution },
    albedo{ 1.f },
    domega{ .01f },
    density_scale{ 1.f },
    particle_size{ 10.f },
    rendering_pass{ -1 }
{
    setup_object();
}

StaticClouds::StaticClouds(const StaticClouds& other)
{
}

StaticClouds::StaticClouds(StaticClouds&& other)
{
}

StaticClouds& tiny_world::StaticClouds::operator=(const StaticClouds& other)
{
    if (this == &other)
        return *this;

    return *this;
}

StaticClouds& StaticClouds::operator=(StaticClouds&& other)
{
    if (this == &other)
        return *this;

    return *this;
}

StaticClouds::~StaticClouds()
{
    glDeleteVertexArrays(1, &ogl_vertex_attribute_object);
    glDeleteBuffers(1, &ogl_vertex_buffer);
}

void StaticClouds::setDomainDimensions(float cloud_domain_size_x, float cloud_domain_size_y, float cloud_domain_size_z)
{
    setDomainDimensions(vec3{ cloud_domain_size_x, cloud_domain_size_y, cloud_domain_size_z });
}

void StaticClouds::setDomainDimensions(const vec3& cloud_domain_size)
{
    v3DomainSize = cloud_domain_size;
}

vec3 StaticClouds::getDomainDimensions() const
{
    return v3DomainSize;
}

void StaticClouds::setAlbedo(float albedo)
{
    this->albedo = albedo;
}

float StaticClouds::getAlbedo() const
{
    return albedo;
}

void StaticClouds::setSolidAngleDifferentialScale(float scale)
{
    domega = scale;
}

float StaticClouds::getSolidAngleDifferentialScale() const
{
    return domega;
}

void StaticClouds::setDensityScale(float scale)
{
    density_scale = scale;
}

float StaticClouds::getDensityScale() const
{
    return density_scale;
}

void StaticClouds::setParticleSize(float size)
{
    particle_size = size;
}

float StaticClouds::getParticleSize() const
{
    return particle_size;
}

void StaticClouds::setLightingConditions(const LightingConditions& lighting_conditions)
{
    p_lighting_conditions = &lighting_conditions;

    std::pair<ImmutableTexture2D, ImmutableTexture2D> sun_moon_in_scattering_contribution = p_lighting_conditions->retrieveInScatteringTextures();
    if (!sun_in_scattering_ref_code)
        sun_in_scattering_ref_code = registerTexture(sun_moon_in_scattering_contribution.first, texture_sampler_ref_code);
    if (!moon_in_scattering_ref_code)
        moon_in_scattering_ref_code = registerTexture(sun_moon_in_scattering_contribution.second, texture_sampler_ref_code);
}

const LightingConditions* StaticClouds::getLightingConditions() const
{
    return p_lighting_conditions;
}

bool StaticClouds::supportsRenderingMode(uint32_t rendering_mode) const
{
    switch (rendering_mode)
    {
    case TW_RENDERING_MODE_DEFAULT:
        return true;

    default:
        return false;
    }
}

uint32_t StaticClouds::getNumberOfRenderingPasses(uint32_t rendering_mode) const
{
    return 2U;
}

bool StaticClouds::render()
{
    if (getActiveRenderingMode() == TW_RENDERING_MODE_DEFAULT)
    {
        switch (rendering_pass)
        {
        case 0:
        {
            retrieveShaderProgram(preprocess_program_ref_code)->assignUniformVector("uv3DomainResolution", uv3DomainResolution);
            retrieveShaderProgram(preprocess_program_ref_code)->assignUniformVector("v3DomainSize", v3DomainSize);

            ImageUnit preprocessed_data_storage_texture;
            preprocessed_data_storage_texture.setStringName("Clouds::preprocessed_data_storage_texture");
            preprocessed_data_storage_texture.attachTexture(out_scattering_values, 0, BufferAccess::WRITE, InternalPixelFormat::SIZED_FLOAT_R16);
            retrieveShaderProgram(preprocess_program_ref_code)->assignUniformScalar("i3dOutScatteringValues", preprocessed_data_storage_texture.getBinding());

            retrieveShaderProgram(preprocess_program_ref_code)->assignUniformScalar("s2dDensityPatternTexture", getBindingUnit(density_pattern_texture_ref_code));
            retrieveShaderProgram(preprocess_program_ref_code)->assignUniformScalar("sbDensityPatternRandomShiftsTexture", getBindingUnit(density_pattern_random_shifts_ref_code));

            retrieveShaderProgram(preprocess_program_ref_code)->assignUniformScalar("fAlbedo", albedo);
            retrieveShaderProgram(preprocess_program_ref_code)->assignUniformScalar("fGamma", domega);
            retrieveShaderProgram(preprocess_program_ref_code)->assignUniformScalar("fDensityScale", density_scale);

            uvec3 num_groups = calculateNumberOfGroupsForDispatchCompute(uv3DomainResolution.x, uv3DomainResolution.y, uv3DomainResolution.z,
                preprocess_group_size_x, preprocess_group_size_y, preprocess_group_size_z);
            glDispatchCompute(num_groups.x, num_groups.y, num_groups.z);
            break;
        }

        case 1:
        {
            retrieveShaderProgram(rendering_program_ref_code)->assignUniformScalar("s2dDensityPatternTexture", getBindingUnit(density_pattern_texture_ref_code));
            retrieveShaderProgram(rendering_program_ref_code)->assignUniformScalar("s3dOutScatteringTexture", getBindingUnit(out_scattering_values_ref_code));
            retrieveShaderProgram(rendering_program_ref_code)->assignUniformScalar("s2dCelestialBodyInScattering",
                p_lighting_conditions->getSkydome()->isDay() ? getBindingUnit(sun_in_scattering_ref_code) : getBindingUnit(moon_in_scattering_ref_code));
            retrieveShaderProgram(rendering_program_ref_code)->assignUniformVector("v3DomainSize", v3DomainSize);
            retrieveShaderProgram(rendering_program_ref_code)->assignUniformScalar("sbDensityPatternRandomShiftsTexture", getBindingUnit(density_pattern_random_shifts_ref_code));

            retrieveShaderProgram(rendering_program_ref_code)->assignUniformScalar("fAlbedo", albedo);
            retrieveShaderProgram(rendering_program_ref_code)->assignUniformScalar("fGamma", domega);
            retrieveShaderProgram(rendering_program_ref_code)->assignUniformScalar("fDensityScale", density_scale);


            if (density_pattern_texture_ref_code)
                bindTexture(density_pattern_texture_ref_code);

            if (out_scattering_values_ref_code)
                bindTexture(out_scattering_values_ref_code);

            if (p_lighting_conditions->getSkydome()->isDay())
            {
                if (sun_in_scattering_ref_code)
                    bindTexture(sun_in_scattering_ref_code);
            }
            else if (moon_in_scattering_ref_code)
                bindTexture(moon_in_scattering_ref_code);

            glDrawArrays(GL_POINTS, 0, uv3DomainResolution.x*uv3DomainResolution.y*uv3DomainResolution.z);
            break;
        }
        }
    }


    return true;
}
