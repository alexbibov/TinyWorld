#include <chrono>

#include "KPWater.h"
#include "KTXTexture.h"

#pragma comment(lib, "libswecore")

using namespace tiny_world;

#define pi 3.1415926535897932384626433832795f


typedef AbstractRenderableObjectExtensionAggregator < AbstractRenderableObjectLightEx, AbstractRenderableObjectHDRBloomEx, AbstractRenderableObjectSelectionEx > ExtensionAggregator;

const std::string KPWater::water_rendering_program0_name = "KPWater::rendering_program0";
const std::string KPWater::fft_compute_program_name = "KPWater::fft_compute_program";
const uint32_t KPWater::fractal_noise_update_period = 10;	//update fractal noise map once per 10 rendering inquiries 


//Default boundary conditions compute procedure implementing reflecting boundaries
void reflectingBoundariesComputeProcedure(const SaintVenantSystem::SVSCore* p_svs, const SaintVenantSystem::Numeric* p_interpolated_topography,
    KPWater::BoundaryCondition& west, KPWater::BoundaryCondition& east, KPWater::BoundaryCondition& south, KPWater::BoundaryCondition& north)
{
    if (p_svs->isInitialized())
    {
        const SaintVenantSystem::Numeric4* p_state = p_svs->getSystemState();
        SaintVenantSystem::SVSCore::DomainSettings domain_settings = p_svs->getDomainSettings();

        //compute east and west boundary conditions
        west = KPWater::BoundaryCondition{ domain_settings.height };
        east = KPWater::BoundaryCondition{ domain_settings.height };
        for (unsigned int i = 0; i < domain_settings.height; ++i)
        {
            SaintVenantSystem::Numeric4 state_west_value = p_state[(domain_settings.width + 2)*(i + 1) + 1];
            west.setElement(i, /*p_interpolated_topography[(2 * domain_settings.width + 1)*(2 * i + 1)] + */state_west_value.x,
                /*p_interpolated_topography[(2 * domain_settings.width + 1)*(2 * i + 1)] + */state_west_value.x, 0, 0, 0, 0);

            SaintVenantSystem::Numeric4 state_east_value = p_state[(domain_settings.width + 2)*(i + 1) + domain_settings.width];
            east.setElement(i, /*p_interpolated_topography[(2 * domain_settings.width + 1)*(2 * i + 1) + 2 * domain_settings.width] +*/ state_east_value.x,
               /* p_interpolated_topography[(2 * domain_settings.width + 1)*(2 * i + 1) + 2 * domain_settings.width] + */state_east_value.x, 0, 0, 0, 0);
        }

        //compute south and north boundary conditions
        south = KPWater::BoundaryCondition{ domain_settings.width };
        north = KPWater::BoundaryCondition{ domain_settings.width };
        for (unsigned int j = 0; j < domain_settings.width; ++j)
        {
            SaintVenantSystem::Numeric4 state_south_value = p_state[(domain_settings.width + 2) + j + 1];
            south.setElement(j, /*p_interpolated_topography[2 * j + 1] +*/ state_south_value.x, /*p_interpolated_topography[2 * j + 1] + */state_south_value.x, 0, 0, 0, 0);

            SaintVenantSystem::Numeric4 state_north_value = p_state[(domain_settings.width + 2) * (domain_settings.height) + j + 1];
            north.setElement(j,/* p_interpolated_topography[(2 * domain_settings.width + 1) * 2 * domain_settings.height + 2 * j + 1] + */state_north_value.x,
                /*p_interpolated_topography[(2 * domain_settings.width + 1) * 2 * domain_settings.height + 2 * j + 1] +*/ state_north_value.x, 0, 0, 0, 0);
        }
    }
}




KPWater::BoundaryCondition::BoundaryCondition() : length{ 0 }, capacity{ 0 }
{
    w_center = new SaintVenantSystem::Numeric[0];
    w_edge = new SaintVenantSystem::Numeric[0];

    hu_center = new SaintVenantSystem::Numeric[0];
    hu_edge = new SaintVenantSystem::Numeric[0];

    hv_center = new SaintVenantSystem::Numeric[0];
    hv_edge = new SaintVenantSystem::Numeric[0];
}

KPWater::BoundaryCondition::BoundaryCondition(uint32_t length) : length{ length }, capacity{ length*sizeof(SaintVenantSystem::Numeric) }
{
    w_center = new SaintVenantSystem::Numeric[length];
    w_edge = new SaintVenantSystem::Numeric[length];

    hu_center = new SaintVenantSystem::Numeric[length];
    hu_edge = new SaintVenantSystem::Numeric[length];

    hv_center = new SaintVenantSystem::Numeric[length];
    hv_edge = new SaintVenantSystem::Numeric[length];
}

KPWater::BoundaryCondition::BoundaryCondition(const BoundaryCondition& other) : length{ other.length }, capacity{ other.capacity }
{
    w_center = new SaintVenantSystem::Numeric[length];
    memcpy(w_center, other.w_center, capacity);
    w_edge = new SaintVenantSystem::Numeric[length];
    memcpy(w_edge, other.w_edge, capacity);

    hu_center = new SaintVenantSystem::Numeric[length];
    memcpy(hu_center, other.hu_center, capacity);
    hu_edge = new SaintVenantSystem::Numeric[length];
    memcpy(hu_edge, other.hu_edge, capacity);

    hv_center = new SaintVenantSystem::Numeric[length];
    memcpy(hv_center, other.hv_center, capacity);
    hv_edge = new SaintVenantSystem::Numeric[length];
    memcpy(hv_edge, other.hv_edge, capacity);
}

KPWater::BoundaryCondition::BoundaryCondition(BoundaryCondition&& other) : length{ other.length }, capacity{ other.capacity },
w_center{ other.w_center }, w_edge{ other.w_edge }, 
hu_center{ other.hu_center }, hu_edge{ other.hu_edge },
hv_center{ other.hv_center }, hv_edge{ other.hv_edge }
{
    other.w_center = nullptr;
    other.w_edge = nullptr;

    other.hu_center = nullptr;
    other.hu_edge = nullptr;

    other.hv_center = nullptr;
    other.hv_edge = nullptr;
}

KPWater::BoundaryCondition::~BoundaryCondition()
{
    if (w_center)
    {
        delete[] w_center;
        delete[] w_edge;

        delete[] hu_center;
        delete[] hu_edge;

        delete[] hv_center;
        delete[] hv_edge;
    }
}

KPWater::BoundaryCondition& KPWater::BoundaryCondition::operator=(const KPWater::BoundaryCondition& other)
{
    if (this == &other)
        return *this;

    if (capacity < other.capacity)
    {
        delete[] w_center;
        delete[] w_edge;
        w_center = new SaintVenantSystem::Numeric[other.length];
        w_edge = new SaintVenantSystem::Numeric[other.length];

        delete[] hu_center;
        delete[] hu_edge;
        hu_center = new SaintVenantSystem::Numeric[other.length];
        hu_edge = new SaintVenantSystem::Numeric[other.length];

        delete[] hv_center;
        delete[] hv_edge;
        hv_center = new SaintVenantSystem::Numeric[other.length];
        hv_edge = new SaintVenantSystem::Numeric[other.length];

        capacity = other.capacity;
    }

    length = other.length;

    memcpy(w_center, other.w_center, capacity);
    memcpy(w_edge, other.w_edge, capacity);

    memcpy(hu_center, other.hu_center, capacity);
    memcpy(hu_edge, other.hu_edge, capacity);

    memcpy(hv_center, other.hv_center, capacity);
    memcpy(hv_edge, other.hv_edge, capacity);

    return *this;
}

KPWater::BoundaryCondition& KPWater::BoundaryCondition::operator=(KPWater::BoundaryCondition&& other)
{
    if (this == &other)
        return *this;

    delete[] w_center;
    delete[] w_edge;
    w_center = other.w_center;
    w_edge = other.w_edge;
    other.w_center = nullptr;
    other.w_edge = nullptr;

    delete[] hu_center;
    delete[] hu_edge;
    hu_center = other.hu_center;
    hu_edge = other.hu_edge;
    other.hu_center = nullptr;
    other.hu_edge = nullptr;

    delete[] hv_center;
    delete[] hv_edge;
    hv_center = other.hv_center;
    hv_edge = other.hv_edge;
    other.hv_center = nullptr;
    other.hv_edge = nullptr;

    length = other.length;
    capacity = other.capacity;

    return *this;
}

void KPWater::BoundaryCondition::setElement(uint32_t index, SaintVenantSystem::Numeric w_center_value, SaintVenantSystem::Numeric w_edge_value,
    SaintVenantSystem::Numeric hu_center_value, SaintVenantSystem::Numeric hu_edge_value,
    SaintVenantSystem::Numeric hv_center_value, SaintVenantSystem::Numeric hv_edge_value)
{
    w_center[index] = w_center_value;
    w_edge[index] = w_edge_value;

    hu_center[index] = hu_center_value;
    hu_edge[index] = hu_edge_value;

    hv_center[index] = hv_center_value;
    hv_edge[index] = hv_edge_value;
}

void KPWater::BoundaryCondition::getElement(uint32_t index, SaintVenantSystem::Numeric& w_center_value, SaintVenantSystem::Numeric& w_edge_value,
    SaintVenantSystem::Numeric& hu_center_value, SaintVenantSystem::Numeric& hu_edge_value,
    SaintVenantSystem::Numeric& hv_center_value, SaintVenantSystem::Numeric& hv_edge_value) const
{
    w_center_value = w_center[index];
    w_edge_value = w_edge[index];

    hu_center_value = hu_center[index];
    hu_edge_value = hu_edge[index];

    hv_center_value = hv_center[index];
    hv_edge_value = hv_edge[index];
}

uint32_t KPWater::BoundaryCondition::getLength() const { return length; }

void KPWater::BoundaryCondition::getRawData(SaintVenantSystem::Numeric** p_w_center_values, SaintVenantSystem::Numeric** p_w_edge_values,
    SaintVenantSystem::Numeric** p_hu_center_values, SaintVenantSystem::Numeric** p_hu_edge_values,
    SaintVenantSystem::Numeric** p_hv_center_values, SaintVenantSystem::Numeric** p_hv_edge_values)
{
    *p_w_center_values = w_center;
    *p_w_edge_values = w_edge;

    *p_hu_center_values = hu_center;
    *p_hu_edge_values = hu_edge;

    *p_hv_center_values = hv_center;
    *p_hv_edge_values = hv_edge;
}

void KPWater::BoundaryCondition::getRawData(const SaintVenantSystem::Numeric** p_w_center_values, const SaintVenantSystem::Numeric** p_w_edge_values,
    const SaintVenantSystem::Numeric** p_hu_center_values, const SaintVenantSystem::Numeric** p_hu_edge_values,
    const SaintVenantSystem::Numeric** p_hv_center_values, const SaintVenantSystem::Numeric** p_hv_edge_values) const
{
    const_cast<BoundaryCondition*>(this)->getRawData(const_cast<SaintVenantSystem::Numeric**>(p_w_center_values), 
        const_cast<SaintVenantSystem::Numeric**>(p_w_edge_values), 
        const_cast<SaintVenantSystem::Numeric**>(p_hu_center_values), 
        const_cast<SaintVenantSystem::Numeric**>(p_hu_edge_values), 
        const_cast<SaintVenantSystem::Numeric**>(p_hv_center_values), 
        const_cast<SaintVenantSystem::Numeric**>(p_hv_edge_values));
}




void KPWater::applyScreenSize(const uvec2& screen_size)
{
    retrieveShaderProgram(water_rendering_program_ref_code)->assignUniformVector("uv2ScreenSize", screen_size);

    if (refraction_texture_with_caustics_tex_res.first.isInitialized()) refraction_texture_with_caustics_tex_res.first = ImmutableTexture2D{ refraction_texture_with_caustics_tex_res.first.getStringName() };
    refraction_texture_with_caustics_tex_res.first.allocateStorage(1, 1, TextureSize{ screen_size.x, screen_size.y, 1 }, InternalPixelFormat::SIZED_FLOAT_RGB32);
    if (!refraction_texture_with_caustics_tex_res.second)
        refraction_texture_with_caustics_tex_res.second = registerTexture(refraction_texture_with_caustics_tex_res.first, refraction_texture_sampler_ref_code);
    else
        updateTexture(refraction_texture_with_caustics_tex_res.second, refraction_texture_with_caustics_tex_res.first, refraction_texture_sampler_ref_code);

    retrieveShaderProgram(water_rendering_program_ref_code)->assignUniformScalar("s2dRefractionTexture", getBindingUnit(refraction_texture_with_caustics_tex_res.second));
    

    caustics_framebuffer.defineViewport(Rectangle{ 0, 0, static_cast<float>(screen_size.x), static_cast<float>(screen_size.y) });
    caustics_framebuffer.attachTexture(FramebufferAttachmentPoint::COLOR0, FramebufferAttachmentInfo{ 0, 0, &refraction_texture_with_caustics_tex_res.first });

    caustics_rendering_program.assignUniformVector("v4Viewport", vec4{ 0, 0, static_cast<float>(screen_size.x), static_cast<float>(screen_size.y) });
    retrieveShaderProgram(water_rendering_program_ref_code)->assignUniformVector("v4Viewport", vec4{ 0, 0, static_cast<float>(screen_size.x), static_cast<float>(screen_size.y) });
}


bool KPWater::configureRendering(AbstractRenderingDevice& render_target, uint32_t rendering_pass)
{
    if (rendering_pass > getNumberOfRenderingPasses(getActiveRenderingMode())) return false;
    current_rendering_pass = rendering_pass;
    p_render_target = &render_target;

    switch (rendering_pass)
    {
    case 0:
    {
        //Bind texture containing the Phillips spectrum
        if (phillips_spectrum_tex_res.second)
            bindTexture(phillips_spectrum_tex_res.second);


        ImageUnit imageOutput1{}; imageOutput1.attachTexture(fft_ripples_tex_res.first, 0, BufferAccess::READ_WRITE, InternalPixelFormat::SIZED_FLOAT_RG32);
        ImageUnit imageOutput2{}; imageOutput2.attachTexture(fft_displacement_map_tex_res.first, 0, BufferAccess::READ_WRITE, InternalPixelFormat::SIZED_FLOAT_RGBA32);
        ImageUnit imageOutput3{}; imageOutput3.attachTexture(fft_ripples_normal_map_global_scale_tex_res.first, 0, BufferAccess::READ_WRITE, InternalPixelFormat::SIZED_FLOAT_RGBA32);
        ImageUnit imageOutput4{}; imageOutput4.attachTexture(fft_ripples_normal_map_capillary_scale_tex_res.first, 0, BufferAccess::READ_WRITE, InternalPixelFormat::SIZED_FLOAT_RGBA32);

        //Bind image units
        retrieveShaderProgram(fft_compute_program_ref_code)->assignUniformScalar("i2dOutput1", imageOutput1.getBinding());
        retrieveShaderProgram(fft_compute_program_ref_code)->assignUniformScalar("i2dOutput2", imageOutput2.getBinding());
        retrieveShaderProgram(fft_compute_program_ref_code)->assignUniformScalar("i2dOutput3", imageOutput3.getBinding());
        retrieveShaderProgram(fft_compute_program_ref_code)->assignUniformScalar("i2dOutput4", imageOutput4.getBinding());

        COMPLETE_SHADER_PROGRAM_CAST(retrieveShaderProgram(fft_compute_program_ref_code)).activate();

        retrieveShaderProgram(fft_compute_program_ref_code)->assignUniformScalar("uiStep", 0U);
        glDispatchCompute(1, fft_size, 1);

        imageOutput1.flush();
        imageOutput2.flush();
        imageOutput3.flush();
        imageOutput4.flush();

        retrieveShaderProgram(fft_compute_program_ref_code)->assignUniformScalar("uiStep", 1U);
        glDispatchCompute(1, fft_size, 1);

        imageOutput1.flush();
        imageOutput2.flush();
        imageOutput3.flush();
        imageOutput4.flush();

        
        fft_ripples_normal_map_global_scale_tex_res.first.generateMipmapLevels();
        fft_ripples_normal_map_capillary_scale_tex_res.first.generateMipmapLevels();

        return true;
    }

    case 1:
    {
        caustics_framebuffer.makeActive();
        return true;
    }

    case 2:
    {
        //Update fractal noise map if necessary
        if (!fractal_noise_update_counter) fractal_noise.generateNoiseMap();


        render_target.makeActive();

        //Update water height map texture
        const SaintVenantSystem::Numeric4* p_system_state = kpwater_cuda.getSystemState();
        for (unsigned int i = 0; i < domain_settings.height; ++i)
            for (unsigned int j = 0; j < domain_settings.width; ++j)
                p_water_heightmap_data[i*domain_settings.width + j] = static_cast<float>(p_system_state[(i + 1)*(domain_settings.width + 2) + j + 1].x);
        water_heightmap_tex_res.first.setMipmapLevelData(0, PixelLayout::RED, PixelDataType::FLOAT, p_water_heightmap_data);
        if (force_water_heightmap_update)
        {
            updateTexture(water_heightmap_tex_res.second, water_heightmap_tex_res.first);
            force_water_heightmap_update = false;
        }


        //Configure vertex source objects
        glBindVertexArray(ogl_vertex_array_object);



        //Initialize uniform values
        ShaderProgram* p_shader_program = retrieveShaderProgram(water_rendering_program_ref_code);
        p_shader_program->assignUniformScalar("fLOD", lod_factor);
        p_shader_program->assignUniformScalar("fMaxLightPenetrationDepth", max_light_penetration_depth);
        //p_shader_program->assignUniformScalar("fFresnelPower", fresnel_power);
        p_shader_program->assignUniformVector("v3ColorExtinctionFactors", v3ColorExtinctionFactors);


        //Bind texture units
        if (refraction_texture_with_caustics_tex_res.second)
            bindTexture(refraction_texture_with_caustics_tex_res.second);

        if (water_heightmap_tex_res.second)
            bindTexture(water_heightmap_tex_res.second);

        if (topography_heightmap_tex_res.second)
            bindTexture(topography_heightmap_tex_res.second);

        if (fft_ripples_tex_res.second)
            bindTexture(fft_ripples_tex_res.second);

        if (fft_ripples_normal_map_global_scale_tex_res.second)
            bindTexture(fft_ripples_normal_map_global_scale_tex_res.second);

        if (fft_ripples_normal_map_capillary_scale_tex_res.second)
            bindTexture(fft_ripples_normal_map_capillary_scale_tex_res.second);

        if (fft_displacement_map_tex_res.second)
            bindTexture(fft_displacement_map_tex_res.second);

        if (fractal_noise_map_tex_res.second)
            bindTexture(fractal_noise_map_tex_res.second);



        //Initialize rendering pipeline 
        COMPLETE_SHADER_PROGRAM_CAST(p_shader_program).activate();



        //Set miscellaneous parameters
        glPatchParameteri(GL_PATCH_VERTICES, 4);

        return true;
    }

    default:
        return false;
    }
}


void KPWater::configureViewProjectionTransform(const AbstractProjectingDevice& projecting_device)
{
    mat4 m4ProjectionTransform = projecting_device.getProjectionTransform();
    mat4 m4ViewTransform = projecting_device.getViewTransform() * getObjectTransform();

    retrieveShaderProgram(water_rendering_program_ref_code)->assignUniformVector("v3Scale", vec3{ domain_settings.width*domain_settings.dx, getObjectScale().y, domain_settings.height*domain_settings.dy });
    retrieveShaderProgram(water_rendering_program_ref_code)->assignUniformMatrix("m4ProjectionTransform", m4ProjectionTransform);
    retrieveShaderProgram(water_rendering_program_ref_code)->assignUniformMatrix("m4ModelViewTransform", m4ViewTransform);
    //retrieveShaderProgram(water_rendering_program)->assignUniformScalar("fFarClipPlane", projecting_device.getFarClipPlane());



    //Set location and dimensions of caustics canvas so that it covers the whole focal plane
    float left, right, bottom, top, near, far;
    projecting_device.getProjectionVolume(&left, &right, &bottom, &top, &near, &far);

    caustics_canvas.setLocation(vec3{ (left + right) / 2.0f, (bottom + top) / 2.0f, -near });
    caustics_canvas.setDimensions(right - left, top - bottom);
    caustics_canvas.applyViewProjectionTransform(projecting_device);

    caustics_rendering_program.assignUniformVector("v3Scale", vec3{ domain_settings.width*domain_settings.dx, getObjectScale().y, domain_settings.height*domain_settings.dy });
    caustics_rendering_program.assignUniformScalar("fFocalDistance", near);
    caustics_rendering_program.assignUniformVector("v4FocalPlane", vec4{ left, right, bottom, top });
    caustics_rendering_program.assignUniformMatrix("m4VS2SOS", m4ViewTransform.inverse());
}


bool KPWater::configureRenderingFinalization()
{
    return true;
}


//Reverses bits in the input value x given number of significant bits.
//Maximal length of the bit sequence to be reverted cannot exceed 32 bits
uint32_t reverse_bits(uint32_t x, unsigned char significant_bits)
{
    x = (x & 0xAAAAAAAA) >> 1 | (x & 0x55555555) << 1;
    x = (x & 0xCCCCCCCC) >> 2 | (x & 0x33333333) << 2;
    x = (x & 0xF0F0F0F0) >> 4 | (x & 0x0F0F0F0F) << 4;
    x = (x & 0xFF00FF00) >> 8 | (x & 0x00FF00FF) << 8;
    x = (x & 0xFFFF0000) >> 16 | (x & 0x0000FFFF) << 16;

    return x >> (32 - significant_bits);
}


//Generates Phillips spectrum for the given parameters (see definition for details)
//NOTE1: The output is written using bit-reversed order in columns and rows meaning that for example 6-th column,
//which is read as 110 in radix-2 will be stored in column 011, or 3.
//NOTE2: Number of rows N and number of columns M of the output table must both be even numbers. Otherwise
//the results might not be valid
//NOTE3: Objects random_number_generator and gaussian_distribution passed by non-const references are used to
//produce draws from Gaussian distribution during calculation of the Phillips spectrum. Therefore their states 
//get updated as a side effect of the function call. Note that object gaussian_distribution passed to the 
//function should implement standard normal distribution with zero mean and unit variance for the output to be correct
void generatePhillipsSpectrum(float Lx, float Lz, uint32_t N, uint32_t M, 
    vec2 v2WindVelocity, float gravity_constant, float minimal_wave_length_ratio,
    std::default_random_engine& random_number_generator, std::normal_distribution<float>& gaussian_distribution, 
    void* output_spectrum)
{
    uint32_t significant_bits_M = static_cast<uint32_t>(std::log2(static_cast<float>(M)));
    uint32_t significant_bits_N = static_cast<uint32_t>(std::log2(static_cast<float>(N)));

    float absolute_wind_speed = v2WindVelocity.norm();
    ogl_type_mapper<float>::ogl_type* p_gl_output = static_cast<ogl_type_mapper<float>::ogl_type*>(output_spectrum);

    for (int i = -static_cast<int>(M) / 2; i < static_cast<int>(M) / 2; ++i)
    {
        int ri = i < 0 ? i + M : i;
        if (ri) ri = M - ri;
        ri = reverse_bits(ri, significant_bits_M);

        for (int j = -static_cast<int>(N) / 2; j < static_cast<int>(N) / 2; ++j)
        {
            int rj = j < 0 ? j + N : j;
            if (rj) rj = N - rj;
            rj = reverse_bits(rj, significant_bits_N);

            vec2 v2K{ 2 * pi*j / Lx, 2 * pi*i / Lz };
            float k = v2K.norm();
            if (k == 0.0f)
            {
                //h(K) (see notation in the Tessendorf's paper)
                p_gl_output[4 * (ri*N + rj) + 0] = 0.0f;
                p_gl_output[4 * (ri*N + rj) + 1] = 0.0f;

                //conj(h(-K))
                p_gl_output[4 * (ri*N + rj) + 2] = 0.0f;
                p_gl_output[4 * (ri*N + rj) + 3] = 0.0f;

                continue;
            }
            v2K /= k;
            float L = absolute_wind_speed*absolute_wind_speed / gravity_constant;	//largest possible wave arising from a continuous wind of the given speed
            float DP = v2WindVelocity.dot_product(v2K) / absolute_wind_speed;
            DP *= DP;
            float minimal_wave_length = L*minimal_wave_length_ratio;
            float P = std::exp(-1.0f / (L*L*k*k)) / (k*k*k*k)*DP * std::exp(-k*k*minimal_wave_length*minimal_wave_length);
            P = std::sqrt(P / 2.0f);


            //h(K) (see notation in the Tessendorf's paper)
            p_gl_output[4 * (ri*N + rj) + 0] = gaussian_distribution(random_number_generator)*P;
            p_gl_output[4 * (ri*N + rj) + 1] = gaussian_distribution(random_number_generator)*P;
        }
    }


    //It is still left to generate conj(h(-K)), where conj(*) denotes complex-conjugate
    for (int i = -static_cast<int>(M) / 2; i < static_cast<int>(M) / 2; ++i)
    {
        int ri = i < 0 ? i + M : i;
        int ric = ri;
        if (ri) ri = M - ri;
        ri = reverse_bits(ri, significant_bits_M);
        ric = reverse_bits(ric, significant_bits_M);

        for (int j = -static_cast<int>(N) / 2; j < static_cast<int>(N) / 2; ++j)
        {
            int rj = j < 0 ? j + N : j;
            int rjc = rj;
            if (rj) rj = N - rj;
            rj = reverse_bits(rj, significant_bits_N);
            rjc = reverse_bits(rjc, significant_bits_N);

            p_gl_output[4 * (ri*N + rj) + 2] = p_gl_output[4 * (ric*N + rjc) + 0];
            p_gl_output[4 * (ri*N + rj) + 3] = -p_gl_output[4 * (ric*N + rjc) + 1];
        }
    }
}


void KPWater::setup_deep_water_waves()
{
    if (!phillips_spectrum_tex_res.first.isInitialized())
        phillips_spectrum_tex_res.first.allocateStorage(1, 1, TextureSize{ fft_size, fft_size, 1 }, InternalPixelFormat::SIZED_FLOAT_RGBA32);

    if (!fft_ripples_tex_res.first.isInitialized())
        fft_ripples_tex_res.first.allocateStorage(1, 1, TextureSize{ fft_size, fft_size, 1 }, InternalPixelFormat::SIZED_FLOAT_RG32);

    if (!fft_displacement_map_tex_res.first.isInitialized())
        fft_displacement_map_tex_res.first.allocateStorage(1, 1, TextureSize{ fft_size, fft_size, 1 }, InternalPixelFormat::SIZED_FLOAT_RGBA32);

    if (!fft_ripples_normal_map_global_scale_tex_res.first.isInitialized())
        fft_ripples_normal_map_global_scale_tex_res.first.allocateStorage(TW__KPWATER_MIPMAPS__, 1, TextureSize{ fft_size, fft_size, 1 }, InternalPixelFormat::SIZED_FLOAT_RGBA32);

    if (!fft_ripples_normal_map_capillary_scale_tex_res.first.isInitialized())
        fft_ripples_normal_map_capillary_scale_tex_res.first.allocateStorage(TW__KPWATER_MIPMAPS__, 1, TextureSize{ fft_size, fft_size, 1 }, InternalPixelFormat::SIZED_FLOAT_RGBA32);

    
    ogl_type_mapper<float>::ogl_type* p_phillips_spectrum_data = new ogl_type_mapper<float>::ogl_type[4 * fft_size*fft_size];
    
    generatePhillipsSpectrum(domain_settings.dx*domain_settings.width, domain_settings.dy*domain_settings.height, fft_size, fft_size,
        v2WindVelocity, g, 1e-3f, random_number_generator, standard_gaussian_distribution, p_phillips_spectrum_data);
    phillips_spectrum_tex_res.first.setMipmapLevelData(0, PixelLayout::RGBA, PixelDataType::FLOAT, p_phillips_spectrum_data);
    
    delete[] p_phillips_spectrum_data;

    if (!phillips_spectrum_tex_res.second)
    {
        phillips_spectrum_tex_res.second = registerTexture(phillips_spectrum_tex_res.first);
        retrieveShaderProgram(fft_compute_program_ref_code)->assignUniformScalar("s2dPhillipsSpectrum", getBindingUnit(phillips_spectrum_tex_res.second));
    }


    if (!fft_ripples_tex_res.second)
    {
        fft_ripples_tex_res.second = registerTexture(fft_ripples_tex_res.first, ripple_texture_sampler_ref_code);
        retrieveShaderProgram(water_rendering_program_ref_code)->assignUniformScalar("s2dFFTRipples", getBindingUnit(fft_ripples_tex_res.second));
    }

    if (!fft_displacement_map_tex_res.second)
    {
        fft_displacement_map_tex_res.second = registerTexture(fft_displacement_map_tex_res.first, ripple_texture_sampler_ref_code);
        retrieveShaderProgram(water_rendering_program_ref_code)->assignUniformScalar("s2dFFTDisplacementMap", getBindingUnit(fft_displacement_map_tex_res.second));
    }

    if (!fft_ripples_normal_map_global_scale_tex_res.second)
    {
        fft_ripples_normal_map_global_scale_tex_res.second = registerTexture(fft_ripples_normal_map_global_scale_tex_res.first, normal_texture_sampler_ref_code);
        retrieveShaderProgram(water_rendering_program_ref_code)->assignUniformScalar("s2dFFTRipplesNormalMapGlobalScale", getBindingUnit(fft_ripples_normal_map_global_scale_tex_res.second));
    }

    if (!fft_ripples_normal_map_capillary_scale_tex_res.second)
    {
        fft_ripples_normal_map_capillary_scale_tex_res.second = registerTexture(fft_ripples_normal_map_capillary_scale_tex_res.first, normal_texture_sampler_ref_code);
        retrieveShaderProgram(water_rendering_program_ref_code)->assignUniformScalar("s2dFFTRipplesNormalMapCapillaryScale", getBindingUnit(fft_ripples_normal_map_capillary_scale_tex_res.second));
    }

    retrieveShaderProgram(fft_compute_program_ref_code)->assignUniformScalar("fChoppiness", choppiness);
    retrieveShaderProgram(water_rendering_program_ref_code)->assignUniformVector("uv2DeepWaterRippleMapTilingFactor", uv2DeepWaterRippleMapTilingFactor);
    retrieveShaderProgram(water_rendering_program_ref_code)->assignUniformScalar("fMaxDeepWaterWaveAmplitude", max_deep_water_wave_amplitude);
    retrieveShaderProgram(water_rendering_program_ref_code)->assignUniformScalar("fMaxWaveHeightAsElevationFraction", max_wave_height_as_elevation_fraction);
}


void KPWater::setup_buffers()
{
    //Create OpenGL vertex and index buffers
    glGenBuffers(2, ogl_buffers);


    //Create and setup vertex array object
    glGenVertexArrays(1, &ogl_vertex_array_object);

    glBindVertexArray(ogl_vertex_array_object);

    glEnableVertexAttribArray(vertex_attribute_position::getId());
    vertex_attribute_position::setVertexAttributeBufferLayout(0, 0);

    //glEnableVertexAttribArray(vertex_attribute_texcoord::getId());
    //vertex_attribute_texcoord::setVertexAttributeBufferLayout(vertex_attribute_position::getCapacity(), 0);


    //Create tessellation billet for the water height map
    glBindBuffer(GL_ARRAY_BUFFER, ogl_buffers[0]);
    char* p_tessellation_billet_data = new char[tess_billet_horizontal_resolution*tess_billet_vertical_resolution*vertex_attribute_position::getCapacity()];
    for (unsigned int i = 0; i < tess_billet_vertical_resolution; ++i)
        for (unsigned int j = 0; j < tess_billet_horizontal_resolution; ++j)
        {
            vertex_attribute_position::value_type* p_current_vertex =
                reinterpret_cast<vertex_attribute_position::value_type*>(p_tessellation_billet_data + vertex_attribute_position::getCapacity()*(i*tess_billet_horizontal_resolution + j));
            //vertex_attribute_texcoord::value_type* p_current_texcoord = reinterpret_cast<vertex_attribute_texcoord::value_type*>(reinterpret_cast<char*>(p_current_vertex)+vertex_attribute_position::getCapacity());

            float dx = j / (tess_billet_horizontal_resolution - 1.0f);
            float dy = i / (tess_billet_vertical_resolution - 1.0f);

            p_current_vertex[0] = -0.5f + dx;	//x-coordinate
            p_current_vertex[1] = 0.0f;			//y-coordinate
            p_current_vertex[2] = 0.5f - dy;	//z-coordinate
            p_current_vertex[3] = 1.0f;			//w-coordinate

            //p_current_texcoord[0] = dx / tess_billet_horizontal_resolution;
            //p_current_texcoord[1] = dy / tess_billet_vertical_resolution;
        }
    glBufferData(GL_ARRAY_BUFFER, tess_billet_horizontal_resolution*tess_billet_vertical_resolution*vertex_attribute_position::getCapacity(), p_tessellation_billet_data, GL_STATIC_DRAW);
    glBindVertexBuffer(0, ogl_buffers[0], 0, vertex_attribute_position::getCapacity());
    delete[] p_tessellation_billet_data;

    //Create index data for the tessellation billet
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ogl_buffers[1]);
    uint32_t* p_index_data = new uint32_t[4 * (tess_billet_horizontal_resolution - 1)*(tess_billet_vertical_resolution - 1)];
    for (unsigned int i = 0; i < tess_billet_vertical_resolution - 1; ++i)
        for (unsigned int j = 0; j < tess_billet_horizontal_resolution - 1; ++j)
        {
            p_index_data[4 * (i*(tess_billet_horizontal_resolution - 1) + j)] = i*tess_billet_horizontal_resolution + j;
            p_index_data[4 * (i*(tess_billet_horizontal_resolution - 1) + j) + 1] = i*tess_billet_horizontal_resolution + j + 1;
            p_index_data[4 * (i*(tess_billet_horizontal_resolution - 1) + j) + 2] = (i + 1)*tess_billet_horizontal_resolution + j + 1;
            p_index_data[4 * (i*(tess_billet_horizontal_resolution - 1) + j) + 3] = (i + 1)*tess_billet_horizontal_resolution + j;
        }
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * sizeof(uint32_t) * (tess_billet_horizontal_resolution - 1)*(tess_billet_vertical_resolution - 1), p_index_data, GL_STATIC_DRAW);
    delete[] p_index_data;
}


void KPWater::setup_object()
{
    //Configure OpenGL buffers used to render the object
    setup_buffers();


    //Initialize shader program, which performs shading of the water surface
    water_rendering_program_ref_code = createCompleteShaderProgram(water_rendering_program0_name,
    { PipelineStage::VERTEX_SHADER, PipelineStage::TESS_CONTROL_SHADER, PipelineStage::TESS_EVAL_SHADER, PipelineStage::GEOMETRY_SHADER, PipelineStage::FRAGMENT_SHADER });
    Shader vertex_shader{ ShaderProgram::getShaderBaseCatalog() + "KPWater.vp.glsl", ShaderType::VERTEX_SHADER, "KPWater::rendering_program0::vertex_shader" };
    Shader tess_control_shader{ ShaderProgram::getShaderBaseCatalog() + "KPWater.tcp.glsl", ShaderType::TESS_CONTROL_SHADER, "KPWater::rendering_program0::tessellation_control_shader" };
    Shader tess_eval_shader{ ShaderProgram::getShaderBaseCatalog() + "KPWater.tep.glsl", ShaderType::TESS_EVAL_SHADER, "KPWater::rendering_program0::tessellation_evaluation_shader" };
    Shader geometry_shader{ ShaderProgram::getShaderBaseCatalog() + "KPWater.gp.glsl", ShaderType::GEOMETRY_SHADER, "KPWater::rendering_program0::geometry_shader" };
    Shader fragment_shader{ ShaderProgram::getShaderBaseCatalog() + "KPWater.fp.glsl", ShaderType::FRAGMENT_SHADER, "KPWater::rendering_program0::fragment_shader" };
    retrieveShaderProgram(water_rendering_program_ref_code)->addShader(vertex_shader);
    retrieveShaderProgram(water_rendering_program_ref_code)->addShader(tess_control_shader);
    retrieveShaderProgram(water_rendering_program_ref_code)->addShader(tess_eval_shader);
    retrieveShaderProgram(water_rendering_program_ref_code)->addShader(geometry_shader);
    retrieveShaderProgram(water_rendering_program_ref_code)->addShader(fragment_shader);
    retrieveShaderProgram(water_rendering_program_ref_code)->bindVertexAttributeId("v4TessellationBilletVertex", vertex_attribute_position::getId());
    retrieveShaderProgram(water_rendering_program_ref_code)->bindVertexAttributeId("v2TessellationBilletTexCoord", vertex_attribute_texcoord::getId());
    retrieveShaderProgram(water_rendering_program_ref_code)->link();


    //Initialize compute program responsible for calculation of the deep water waves
    fft_compute_program_ref_code = createCompleteShaderProgram(fft_compute_program_name, { PipelineStage::COMPUTE_SHADER });
    Shader compute_shader{ ShaderProgram::getShaderBaseCatalog() + "KPWaterFFT.cp.glsl", ShaderType::COMPUTE_SHADER, "KPWater::fft_compute_program::compute_shader" };
    retrieveShaderProgram(fft_compute_program_ref_code)->addShader(compute_shader);
    retrieveShaderProgram(fft_compute_program_ref_code)->link();
    

    //Setup parameters responsible for implementation of caustics
    Shader caustics_shader{ ShaderProgram::getShaderBaseCatalog() + "KPWaterCaustics.fp.glsl", ShaderType::FRAGMENT_SHADER, "KPWater::caustics_rendering_program::fragment_shader" };
    caustics_rendering_program.addShader(caustics_shader);
    caustics_rendering_program.link();
    caustics_canvas.setFilterEffect(caustics_rendering_program, std::bind(&KPWater::setup_caustics_parameters, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    caustics_canvas.installSampler(*retrieveTextureSampler(refraction_texture_sampler_ref_code));
    caustics_rendering_program.assignUniformScalar("fCausticsPower", caustics_power);
    caustics_rendering_program.assignUniformScalar("fCausticsAmplification", caustics_amplification);
    caustics_rendering_program.assignUniformScalar("fCausticsSampleArea", caustics_sample_area);


    //Install default water normal map
    defineProceduralNormalMap("FFTRipples");

    //Install default water specular map
    defineProceduralSpecularMap("FFTRipplesSpecularModulation");
    
    //Install cube map sample retriever
    defineCustomCubeEnvironmentMapSampleRetriever("KPWaterCubicEnvironmentMapSampleRetriever");

    //By default water has no diffuse color
    setDiffuseColor(vec4{ 1.0f });


    //Initialize random generator used by simulation of the capillary waves
    long long seed_value = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    random_number_generator = std::default_random_engine{ static_cast<std::default_random_engine::result_type>(seed_value) };
}


bool KPWater::setup_caustics_parameters(const AbstractProjectingDevice& projecting_device, const AbstractRenderingDevice& render_target, int first_texture_unit_available)
{
    caustics_rendering_program.assignUniformScalar("s2dFFTRipplesNormalMapGlobalScale", first_texture_unit_available);
    caustics_rendering_program.assignUniformScalar("s2dFFTRipplesNormalMapCapillaryScale", first_texture_unit_available + 1);
    caustics_rendering_program.assignUniformScalar("s2dRefractionDepthMap", first_texture_unit_available + 2);
    caustics_rendering_program.assignUniformScalar("s2dWaterHeightMap", first_texture_unit_available + 3);

    TextureUnitBlock* p_texture_unit_block = AbstractRenderableObjectTextured::getTextureUnitBlockPointer();

    p_texture_unit_block->switchActiveTextureUnit(first_texture_unit_available);
    p_texture_unit_block->bindTexture(fft_ripples_normal_map_global_scale_tex_res.first);
    p_texture_unit_block->bindSampler(*retrieveTextureSampler(normal_texture_sampler_ref_code));

    p_texture_unit_block->switchActiveTextureUnit(first_texture_unit_available + 1);
    p_texture_unit_block->bindTexture(fft_ripples_normal_map_capillary_scale_tex_res.first);
    p_texture_unit_block->bindSampler(*retrieveTextureSampler(normal_texture_sampler_ref_code));

    p_texture_unit_block->switchActiveTextureUnit(first_texture_unit_available + 2);
    p_texture_unit_block->bindTexture(refraction_texture_depth_map);
    p_texture_unit_block->bindSampler(*retrieveTextureSampler(refraction_texture_sampler_ref_code));

    p_texture_unit_block->switchActiveTextureUnit(first_texture_unit_available + 3);
    p_texture_unit_block->bindTexture(water_heightmap_tex_res.first);
    p_texture_unit_block->bindSampler(*retrieveTextureSampler(refraction_texture_sampler_ref_code));


    //Compute average direction and intensity of the light source that generates caustics
    const Skydome* p_skydome = retrieveLightingConditionsPointer()->getSkydome();

    vec3 v3LightIntensity{ 0 };
    if (p_skydome)
        v3LightIntensity = p_skydome->isDay() ? p_skydome->getSunLightIntensity().get_normalized() : p_skydome->getMoonLightIntensity().get_normalized();
    else
    {
        auto direction_lights_iterator = retrieveLightingConditionsPointer()->getDirectionalLightSources().begin();
        for (uint32_t i = 0; i < retrieveLightingConditionsPointer()->getDirectionalLightSources().size(); ++i)
        {
            v3LightIntensity = (v3LightIntensity*i + (*direction_lights_iterator)->getColor()) / (i + 1);
            ++direction_lights_iterator;
        }
    }
    caustics_rendering_program.assignUniformVector("v3LightIntensity", v3LightIntensity);

    return true;
}


//Rounds x to the nearest power of 2
uint32_t _2power_round(uint32_t x)
{
    uint32_t r = x - 1;
    r |= r >> 1;
    r |= r >> 2;
    r |= r >> 4;
    r |= r >> 8;
    r |= r >> 16;
    ++r;

    uint32_t l = r / 2;

    return r - x > x - l ? l : r;
}


KPWater::KPWater(uint32_t tess_billet_horizontal_resolution /* = 128 */, uint32_t tess_billet_vertical_resolution /* = 128 */) : 

AbstractRenderableObject("KPWater"),

ExtensionAggregator(std::initializer_list<std::pair<PipelineStage, std::string>>({ std::make_pair(PipelineStage::FRAGMENT_SHADER, ShaderProgram::getShaderBaseCatalog() + "KPWater.fp.ext.glsl") })),

is_initialized{ false }, ode_solver{ KPWater::ODESolver::RungeKutta33 }, bc_callback{ reflectingBoundariesComputeProcedure }, 
g{ 9.8f }, theta{ 1.1f }, eps{ 0.0f }, p_water_heightmap_data{ nullptr }, p_topography_heightmap_data{ nullptr }, 
p_interpolated_topography_heightmap_data{ nullptr }, lod_factor{ 20.0f }, max_light_penetration_depth{ 1.0f }, 
water_heightmap_tex_res{ ImmutableTexture2D{ "KPWater::water_height_map" }, TextureReferenceCode{} }, 
topography_heightmap_tex_res{ ImmutableTexture2D{ "KPWater::topography_height_map" }, TextureReferenceCode{} },
refraction_texture_with_caustics_tex_res{ ImmutableTexture2D{ "KPWater::refraction_texture_with_caustics" }, TextureReferenceCode{} },
force_water_heightmap_update{ false }, max_deep_water_wave_amplitude{ 10.0f },
max_capillary_wave_amplitude{ 10.0f }, v2RippleSimulationTime{ 0.0f }, v2RippleSimulationDt{ 1e-5f, 1e-4f }, /*fresnel_power{ 0.9f },*/
v3ColorExtinctionFactors{ 1535, 92, 23 }, v2WindVelocity{ 0.0f, 1.0f }, standard_gaussian_distribution{ 0, 1 }, 
phillips_spectrum_tex_res{ ImmutableTexture2D{ "KPWater::phillips_spectrum" }, TextureReferenceCode{} },
fft_ripples_tex_res{ ImmutableTexture2D{ "KPWater::fft_ripples" }, TextureReferenceCode{} }, 
fft_displacement_map_tex_res{ ImmutableTexture2D{ "KPWater::fft_displacement_map" }, TextureReferenceCode{} }, 
fft_ripples_normal_map_global_scale_tex_res{ ImmutableTexture2D{ "KPWater::fft_ripples_normal_map_global_scale" }, TextureReferenceCode{} }, 
fft_ripples_normal_map_capillary_scale_tex_res{ ImmutableTexture2D{ "KPWater::fft_ripples_normal_map_capillary_scale" }, TextureReferenceCode{} },
choppiness{ 0.1f }, uv2DeepWaterRippleMapTilingFactor{ 5U, 20U }, max_wave_height_as_elevation_fraction{ 0.1f },
fractal_noise{ 128, 128, 64U, 64U, 4U }, fractal_noise_update_counter{ 0 },
caustics_power{ 1e3f }, caustics_amplification{ 1.0f }, caustics_sample_area{ 1e-3f },
current_rendering_pass{ 0 }, p_render_target{ nullptr }

{
    this->tess_billet_horizontal_resolution = _2power_round(tess_billet_horizontal_resolution);
    this->tess_billet_vertical_resolution = _2power_round(tess_billet_vertical_resolution);

    refraction_texture_sampler_ref_code = createTextureSampler("KPWater::refraction_texture_sampler", SamplerMagnificationFilter::LINEAR, SamplerMinificationFilter::LINEAR,
        SamplerWrapping{ SamplerWrappingMode::MIRRORED_REPEAT, SamplerWrappingMode::MIRRORED_REPEAT, SamplerWrappingMode::MIRRORED_REPEAT });
    ripple_texture_sampler_ref_code = createTextureSampler("KPWater::ripple_texture_sampler", SamplerMagnificationFilter::LINEAR, SamplerMinificationFilter::LINEAR,
        SamplerWrapping{ SamplerWrappingMode::REPEAT, SamplerWrappingMode::REPEAT, SamplerWrappingMode::CLAMP_TO_EDGE });
    normal_texture_sampler_ref_code = createTextureSampler("KPWater::normal_texture_sampler", SamplerMagnificationFilter::LINEAR, SamplerMinificationFilter::LINEAR_MIPMAP_LINEAR,
        SamplerWrapping{ SamplerWrappingMode::REPEAT, SamplerWrappingMode::REPEAT, SamplerWrappingMode::CLAMP_TO_EDGE });

    caustics_framebuffer.setStringName("KPWater::caustics_framebuffer");
    caustics_rendering_program.setStringName("KPWater::caustics_rendering_program");
    caustics_canvas.setStringName("KPWater::caustics_canvas");

    fractal_noise.setContinuity(true);
    fractal_noise.setPeriodicity(true);
    fractal_noise.setEvolutionRate(0.05f);
    fractal_noise_map_tex_res.first = fractal_noise.retrieveNoiseMap();
    fractal_noise_map_tex_res.second = registerTexture(fractal_noise_map_tex_res.first, normal_texture_sampler_ref_code);

    setup_object();
    retrieveShaderProgram(water_rendering_program_ref_code)->assignUniformScalar("s2dFractalNoiseMap", getBindingUnit(fractal_noise_map_tex_res.second));
}


KPWater::KPWater(const SaintVenantSystem::Numeric* init_water_levels, const SaintVenantSystem::Numeric* init_horizontal_speed_flux, const SaintVenantSystem::Numeric* init_vertical_speed_flux,
    uint32_t domain_width, uint32_t domain_height, SaintVenantSystem::Numeric dx, SaintVenantSystem::Numeric dy, SaintVenantSystem::Numeric eps, const SaintVenantSystem::Numeric* p_interpolated_topography,
    SaintVenantSystem::Numeric g /* = 9.8 */, SaintVenantSystem::Numeric theta /* = 1.1 */, ODESolver solver /* = ODESolver::RungeKutta33 */, 
    uint32_t tess_billet_horizontal_resolution /* = 128 */, uint32_t tess_billet_vertical_resolution /* = 128 */) :

    AbstractRenderableObject("KPWater"),

    ExtensionAggregator(std::initializer_list<std::pair<PipelineStage, std::string>>({ std::make_pair(PipelineStage::FRAGMENT_SHADER, ShaderProgram::getShaderBaseCatalog() + "KPWater.fp.ext.glsl") })),

    bc_callback{ reflectingBoundariesComputeProcedure }, g{ g }, theta{ static_cast<float>(theta) }, eps{ static_cast<float>(eps) }, 
    p_water_heightmap_data{ nullptr }, p_topography_heightmap_data{ nullptr },
    p_interpolated_topography_heightmap_data{ nullptr }, lod_factor{ 20.0f }, max_light_penetration_depth{ 1.0f }, 
    water_heightmap_tex_res{ ImmutableTexture2D{ "KPWater::water_height_map" }, TextureReferenceCode{} },
    topography_heightmap_tex_res{ ImmutableTexture2D{ "KPWater::topography_height_map" }, TextureReferenceCode{} },
    refraction_texture_with_caustics_tex_res{ ImmutableTexture2D{ "KPWater::refraction_texture_with_caustics" }, TextureReferenceCode{} },
    force_water_heightmap_update{ false }, max_deep_water_wave_amplitude{ 10.0f }, max_capillary_wave_amplitude{ 10.0f }, 
    v2RippleSimulationTime{ 0.0f }, v2RippleSimulationDt{ 1e-5f, 1e-4f }, /*fresnel_power{ 0.9f },*/ v3ColorExtinctionFactors{ 1535, 92, 23 }, 
    v2WindVelocity{ 0.0f, 1.0f }, standard_gaussian_distribution{ 0, 1 }, phillips_spectrum_tex_res{ ImmutableTexture2D{ "KPWater::phillips_spectrum" }, 
    TextureReferenceCode{} }, fft_ripples_tex_res{ ImmutableTexture2D{ "KPWater::fft_ripples" }, TextureReferenceCode{} }, 
    fft_displacement_map_tex_res{ ImmutableTexture2D{ "KPWater::fft_displacement_map" }, TextureReferenceCode{} }, 
    fft_ripples_normal_map_global_scale_tex_res{ ImmutableTexture2D{ "KPWater::fft_ripples_normal_map_global_scale" }, TextureReferenceCode{} },
    fft_ripples_normal_map_capillary_scale_tex_res{ ImmutableTexture2D{ "KPWater::fft_ripples_normal_map_capillary_scale" }, TextureReferenceCode{} },
    choppiness{ 0.1f }, uv2DeepWaterRippleMapTilingFactor{ 5U, 20U }, max_wave_height_as_elevation_fraction{ 0.1f },
    fractal_noise{ 128, 128, 64U, 64U, 4U }, fractal_noise_update_counter{ 0 },
    caustics_power{ 1e3f }, caustics_amplification{ 1.0f }, caustics_sample_area{ 1e-3f },
    current_rendering_pass{ 0 }, p_render_target{ nullptr }

{
    this->tess_billet_horizontal_resolution = _2power_round(tess_billet_horizontal_resolution);
    this->tess_billet_vertical_resolution = _2power_round(tess_billet_vertical_resolution);

    refraction_texture_sampler_ref_code = createTextureSampler("KPWater::refraction_texture_sampler", SamplerMagnificationFilter::LINEAR, SamplerMinificationFilter::LINEAR,
        SamplerWrapping{ SamplerWrappingMode::MIRRORED_REPEAT, SamplerWrappingMode::MIRRORED_REPEAT, SamplerWrappingMode::MIRRORED_REPEAT });
    ripple_texture_sampler_ref_code = createTextureSampler("KPWater::ripple_texture_sampler", SamplerMagnificationFilter::LINEAR, SamplerMinificationFilter::LINEAR,
        SamplerWrapping{ SamplerWrappingMode::REPEAT, SamplerWrappingMode::REPEAT, SamplerWrappingMode::CLAMP_TO_EDGE });
    normal_texture_sampler_ref_code = createTextureSampler("KPWater::normal_texture_sampler", SamplerMagnificationFilter::LINEAR, SamplerMinificationFilter::LINEAR_MIPMAP_LINEAR,
        SamplerWrapping{ SamplerWrappingMode::REPEAT, SamplerWrappingMode::REPEAT, SamplerWrappingMode::CLAMP_TO_EDGE });

    caustics_framebuffer.setStringName("KPWater::caustics_framebuffer");
    caustics_rendering_program.setStringName("KPWater::caustics_rendering_program");
    caustics_canvas.setStringName("KPWater::caustics_canvas");

    fractal_noise.setContinuity(true);
    fractal_noise.setPeriodicity(true);
    fractal_noise.setEvolutionRate(0.05f);
    fractal_noise_map_tex_res.first = fractal_noise.retrieveNoiseMap();
    fractal_noise_map_tex_res.second = registerTexture(fractal_noise_map_tex_res.first, normal_texture_sampler_ref_code);

    setup_object();
    retrieveShaderProgram(water_rendering_program_ref_code)->assignUniformScalar("s2dFractalNoiseMap", getBindingUnit(fractal_noise_map_tex_res.second));
    initialize(init_water_levels, init_horizontal_speed_flux, init_vertical_speed_flux, domain_width, domain_height, dx, dy, eps, p_interpolated_topography, g, theta, solver);
}


KPWater::KPWater(const KPWater& other) : 

AbstractRenderableObject(other), AbstractRenderableObjectTextured(other), 
AbstractRenderableObjectExtensionAggregator<AbstractRenderableObjectLightEx, AbstractRenderableObjectHDRBloomEx, AbstractRenderableObjectSelectionEx>(other),

kpwater_cuda{ other.kpwater_cuda }, domain_settings(other.domain_settings), is_initialized{ other.is_initialized }, ode_solver{ other.ode_solver },
bc_callback{ other.bc_callback }, g{ other.g }, theta{ other.theta }, eps{ other.eps }, 
p_water_heightmap_data{ nullptr }, p_topography_heightmap_data{ nullptr }, p_interpolated_topography_heightmap_data{ nullptr },
lod_factor{ other.lod_factor }, max_light_penetration_depth{ other.max_light_penetration_depth }, 
water_heightmap_tex_res{ ImmutableTexture2D{ "KPWater::water_height_map" }, other.water_heightmap_tex_res.second },
topography_heightmap_tex_res{ ImmutableTexture2D{ "KPWater::topography_height_map" }, other.topography_heightmap_tex_res.second },
refraction_texture_with_caustics_tex_res{ other.refraction_texture_with_caustics_tex_res }, refraction_texture_depth_map{ other.refraction_texture_depth_map },
refraction_texture_sampler_ref_code{ other.refraction_texture_sampler_ref_code },
ripple_texture_sampler_ref_code{ other.ripple_texture_sampler_ref_code }, 
normal_texture_sampler_ref_code{ other.normal_texture_sampler_ref_code },
force_water_heightmap_update{ true }, water_rendering_program_ref_code{ other.water_rendering_program_ref_code }, fft_compute_program_ref_code{ other.fft_compute_program_ref_code }, 
tess_billet_horizontal_resolution{ other.tess_billet_horizontal_resolution }, tess_billet_vertical_resolution{ other.tess_billet_vertical_resolution }, 
max_deep_water_wave_amplitude{ other.max_deep_water_wave_amplitude }, max_capillary_wave_amplitude{ other.max_capillary_wave_amplitude }, 
v2RippleSimulationTime{ other.v2RippleSimulationTime }, v2RippleSimulationDt{ other.v2RippleSimulationDt }, /*fresnel_power{ other.fresnel_power },*/ 
v3ColorExtinctionFactors{ other.v3ColorExtinctionFactors }, v2WindVelocity{ other.v2WindVelocity }, standard_gaussian_distribution{ other.standard_gaussian_distribution }, 
phillips_spectrum_tex_res{ other.phillips_spectrum_tex_res }, fft_ripples_tex_res{ other.fft_ripples_tex_res }, fft_displacement_map_tex_res{ other.fft_displacement_map_tex_res }, 
fft_ripples_normal_map_global_scale_tex_res{ other.fft_ripples_normal_map_global_scale_tex_res }, fft_ripples_normal_map_capillary_scale_tex_res{ other.fft_ripples_normal_map_capillary_scale_tex_res },
choppiness{ other.choppiness }, uv2DeepWaterRippleMapTilingFactor{ other.uv2DeepWaterRippleMapTilingFactor }, max_wave_height_as_elevation_fraction{ other.max_wave_height_as_elevation_fraction }, 
fractal_noise{ other.fractal_noise }, fractal_noise_map_tex_res{ fractal_noise.retrieveNoiseMap(), other.fractal_noise_map_tex_res.second },
fractal_noise_update_counter{ other.fractal_noise_update_counter }, caustics_framebuffer{ other.caustics_framebuffer }, caustics_canvas{ other.caustics_canvas },
caustics_rendering_program{ other.caustics_rendering_program }, caustics_power{ other.caustics_power }, caustics_amplification{ other.caustics_amplification }, caustics_sample_area{ other.caustics_sample_area },
current_rendering_pass{ other.current_rendering_pass }, p_render_target{ other.p_render_target }

{
    //Configure OpenGL buffer employed for rendering
    setup_buffers();

    //Configure the random number generator
    long long seed_value = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    random_number_generator = std::default_random_engine{ static_cast<std::default_random_engine::result_type>(seed_value) };

    //Copy water and topography height map data if necessary
    if (is_initialized)
    {
        p_water_heightmap_data = new float[domain_settings.width*domain_settings.height];
        water_heightmap_tex_res.first.allocateStorage(1, 1, TextureSize{ domain_settings.width, domain_settings.height, 0 }, InternalPixelFormat::SIZED_FLOAT_R32);

        p_topography_heightmap_data = new float[domain_settings.width*domain_settings.height];
        memcpy(p_topography_heightmap_data, other.p_topography_heightmap_data, domain_settings.width*domain_settings.height*sizeof(float));
        topography_heightmap_tex_res.first.allocateStorage(1, 1, TextureSize{ domain_settings.width, domain_settings.height, 0 }, InternalPixelFormat::SIZED_FLOAT_R32);
        topography_heightmap_tex_res.first.setMipmapLevelData(0, PixelLayout::RED, PixelDataType::FLOAT, p_topography_heightmap_data);
        updateTexture(topography_heightmap_tex_res.second, topography_heightmap_tex_res.first);

        p_interpolated_topography_heightmap_data = new SaintVenantSystem::Numeric[(2 * domain_settings.width + 1)*(2 * domain_settings.height + 1)];
        memcpy(p_interpolated_topography_heightmap_data, other.p_interpolated_topography_heightmap_data, 
            (2 * domain_settings.width + 1)*(2 * domain_settings.height + 1)*sizeof(SaintVenantSystem::Numeric));
    }

    //Update filter effect used by caustics generator
    caustics_canvas.setFilterEffect(caustics_rendering_program, std::bind(&KPWater::setup_caustics_parameters, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

    //Update fractal noise texture so that the local texture instance is used by this object
    updateTexture(fractal_noise_map_tex_res.second, fractal_noise_map_tex_res.first, normal_texture_sampler_ref_code);
}


KPWater::KPWater(KPWater&& other) : 

AbstractRenderableObject(std::move(other)), AbstractRenderableObjectTextured(std::move(other)), ExtensionAggregator(std::move(other)),

kpwater_cuda{ std::move(other.kpwater_cuda) }, domain_settings(std::move(other.domain_settings)), is_initialized{ other.is_initialized }, 
ode_solver{ other.ode_solver }, bc_callback(std::move(other.bc_callback)), g{ other.g }, theta{ other.theta }, eps{ other.eps }, 
p_water_heightmap_data{ other.p_water_heightmap_data }, p_topography_heightmap_data{ other.p_topography_heightmap_data },
p_interpolated_topography_heightmap_data{ other.p_interpolated_topography_heightmap_data }, lod_factor{ other.lod_factor }, 
max_light_penetration_depth{ other.max_light_penetration_depth }, water_heightmap_tex_res{ std::move(other.water_heightmap_tex_res) }, 
topography_heightmap_tex_res{ std::move(other.topography_heightmap_tex_res) }, refraction_texture_depth_map{ std::move(other.refraction_texture_depth_map) },
refraction_texture_sampler_ref_code{ std::move(other.refraction_texture_sampler_ref_code) },
ripple_texture_sampler_ref_code{ std::move(other.ripple_texture_sampler_ref_code) }, 
normal_texture_sampler_ref_code{ std::move(other.normal_texture_sampler_ref_code) },
refraction_texture_with_caustics_tex_res{ std::move(other.refraction_texture_with_caustics_tex_res) },
force_water_heightmap_update{ false }, water_rendering_program_ref_code{ std::move(other.water_rendering_program_ref_code) }, 
fft_compute_program_ref_code{ std::move(other.fft_compute_program_ref_code) }, tess_billet_horizontal_resolution{ other.tess_billet_horizontal_resolution }, 
tess_billet_vertical_resolution{ other.tess_billet_vertical_resolution }, max_deep_water_wave_amplitude{ other.max_deep_water_wave_amplitude }, 
max_capillary_wave_amplitude{ other.max_capillary_wave_amplitude }, v2RippleSimulationTime{ std::move(other.v2RippleSimulationTime) }, 
v2RippleSimulationDt{ std::move(other.v2RippleSimulationDt) }, /*fresnel_power{ other.fresnel_power },*/ v3ColorExtinctionFactors{ std::move(other.v3ColorExtinctionFactors) }, 
v2WindVelocity{ std::move(other.v2WindVelocity) }, standard_gaussian_distribution{ std::move(other.standard_gaussian_distribution) }, 
phillips_spectrum_tex_res{ std::move(other.phillips_spectrum_tex_res) }, fft_ripples_tex_res{ std::move(other.fft_ripples_tex_res) }, 
fft_displacement_map_tex_res{ std::move(other.fft_displacement_map_tex_res) }, 
fft_ripples_normal_map_global_scale_tex_res{ std::move(other.fft_ripples_normal_map_global_scale_tex_res) }, fft_ripples_normal_map_capillary_scale_tex_res{ std::move(other.fft_ripples_normal_map_capillary_scale_tex_res) },
choppiness{ other.choppiness }, uv2DeepWaterRippleMapTilingFactor{ std::move(other.uv2DeepWaterRippleMapTilingFactor) }, max_wave_height_as_elevation_fraction{ other.max_wave_height_as_elevation_fraction },
fractal_noise{ std::move(other.fractal_noise) }, fractal_noise_map_tex_res{ std::move(other.fractal_noise_map_tex_res) }, fractal_noise_update_counter{ other.fractal_noise_update_counter },
caustics_framebuffer{ std::move(other.caustics_framebuffer) }, caustics_canvas{ std::move(other.caustics_canvas) }, caustics_rendering_program{ std::move(other.caustics_rendering_program) },
caustics_power{ other.caustics_power }, caustics_amplification{ other.caustics_amplification }, caustics_sample_area{ other.caustics_sample_area },
current_rendering_pass{ other.current_rendering_pass }, p_render_target{ other.p_render_target }, ogl_vertex_array_object{ other.ogl_vertex_array_object }

{
    long long seed_value = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    random_number_generator = std::default_random_engine{ static_cast<std::default_random_engine::result_type>(seed_value) };

    other.ogl_vertex_array_object = 0;
    ogl_buffers[0] = other.ogl_buffers[0]; other.ogl_buffers[0] = 0;
    ogl_buffers[1] = other.ogl_buffers[1]; other.ogl_buffers[1] = 0;
    other.p_water_heightmap_data = nullptr;
    other.p_topography_heightmap_data = nullptr;
    other.p_interpolated_topography_heightmap_data = nullptr;

    caustics_canvas.setFilterEffect(caustics_rendering_program, std::bind(&KPWater::setup_caustics_parameters, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
}



KPWater::~KPWater()
{
    if (p_water_heightmap_data)
        delete[] p_water_heightmap_data;

    if (p_topography_heightmap_data)
        delete[] p_topography_heightmap_data;

    if (p_interpolated_topography_heightmap_data)
        delete[] p_interpolated_topography_heightmap_data;

    if (ogl_buffers[0]) glDeleteBuffers(2, ogl_buffers);
    if(ogl_vertex_array_object) glDeleteVertexArrays(1, &ogl_vertex_array_object);
}


KPWater& KPWater::operator=(const KPWater& other)
{
    if (this == &other)
        return *this;

    AbstractRenderableObject::operator=(other);
    AbstractRenderableObjectTextured::operator=(other);
    ExtensionAggregator::operator=(other);


    if (is_initialized)
        reset();

    kpwater_cuda = other.kpwater_cuda;
    domain_settings = other.domain_settings;
    is_initialized = other.is_initialized;
    ode_solver = other.ode_solver;
    bc_callback = other.bc_callback;
    g = other.g;
    theta = other.theta;
    eps = other.eps;

    
    water_heightmap_tex_res.first = ImmutableTexture2D{ "KPWater::water_height_map" };
    topography_heightmap_tex_res.first = ImmutableTexture2D{ "KPWater::topography_height_map" };
    refraction_texture_with_caustics_tex_res = other.refraction_texture_with_caustics_tex_res;
    refraction_texture_depth_map = other.refraction_texture_depth_map;
    refraction_texture_sampler_ref_code = other.refraction_texture_sampler_ref_code;
    ripple_texture_sampler_ref_code = other.ripple_texture_sampler_ref_code;
    normal_texture_sampler_ref_code = other.normal_texture_sampler_ref_code;
    if (other.is_initialized)
    {
        p_water_heightmap_data = new float[domain_settings.width*domain_settings.height];
        water_heightmap_tex_res.first.allocateStorage(1, 1, TextureSize{ domain_settings.width, domain_settings.height, 0 }, InternalPixelFormat::SIZED_FLOAT_R32);

        p_topography_heightmap_data = new float[domain_settings.width*domain_settings.height];
        memcpy(p_topography_heightmap_data, other.p_topography_heightmap_data, domain_settings.width*domain_settings.height*sizeof(float));
        topography_heightmap_tex_res.first.allocateStorage(1, 1, TextureSize{ domain_settings.width, domain_settings.height, 0 }, InternalPixelFormat::SIZED_FLOAT_R32);
        topography_heightmap_tex_res.first.setMipmapLevelData(0, PixelLayout::RED, PixelDataType::FLOAT, p_topography_heightmap_data);
        updateTexture(topography_heightmap_tex_res.second, topography_heightmap_tex_res.first);

        p_interpolated_topography_heightmap_data = new SaintVenantSystem::Numeric[(2 * domain_settings.width + 1)*(2 * domain_settings.height + 1)];
        memcpy(p_interpolated_topography_heightmap_data, other.p_interpolated_topography_heightmap_data, 
            (2 * domain_settings.width + 1)*(2 * domain_settings.height + 1)*sizeof(SaintVenantSystem::Numeric));
    }


    lod_factor = other.lod_factor;
    max_light_penetration_depth = other.max_light_penetration_depth;
    force_water_heightmap_update = other.is_initialized;	//this is intentional! Height map gets updated on the next KP-scheme step only if the scheme was initialized upon the assignment!
    water_rendering_program_ref_code = other.water_rendering_program_ref_code;
    fft_compute_program_ref_code = other.fft_compute_program_ref_code;
    tess_billet_horizontal_resolution = other.tess_billet_horizontal_resolution;
    tess_billet_vertical_resolution = other.tess_billet_vertical_resolution;
    max_deep_water_wave_amplitude = other.max_deep_water_wave_amplitude;
    max_capillary_wave_amplitude = other.max_capillary_wave_amplitude;
    v2RippleSimulationTime = other.v2RippleSimulationTime;
    v2RippleSimulationDt = other.v2RippleSimulationDt;
    //fresnel_power = other.fresnel_power;
    v3ColorExtinctionFactors = other.v3ColorExtinctionFactors;
    v2WindVelocity = other.v2WindVelocity;
    standard_gaussian_distribution = other.standard_gaussian_distribution;
    phillips_spectrum_tex_res = other.phillips_spectrum_tex_res;
    fft_ripples_tex_res = other.fft_ripples_tex_res;
    fft_displacement_map_tex_res = other.fft_displacement_map_tex_res;
    fft_ripples_normal_map_global_scale_tex_res = other.fft_ripples_normal_map_global_scale_tex_res;
    fft_ripples_normal_map_capillary_scale_tex_res = other.fft_ripples_normal_map_capillary_scale_tex_res;
    choppiness = other.choppiness;
    uv2DeepWaterRippleMapTilingFactor = other.uv2DeepWaterRippleMapTilingFactor;
    max_wave_height_as_elevation_fraction = other.max_wave_height_as_elevation_fraction;

    fractal_noise = other.fractal_noise;
    fractal_noise_map_tex_res.first = fractal_noise.retrieveNoiseMap();
    fractal_noise_map_tex_res.second = other.fractal_noise_map_tex_res.second;
    updateTexture(fractal_noise_map_tex_res.second, fractal_noise_map_tex_res.first, normal_texture_sampler_ref_code);
    fractal_noise_update_counter = other.fractal_noise_update_counter;

    caustics_framebuffer = other.caustics_framebuffer;

    caustics_rendering_program = other.caustics_rendering_program;
    //Note: after this operation caustics_canvas of "this" object will refer to caustics_rendering_program of the "other" object. This should be corrected!
    caustics_canvas = other.caustics_canvas;
    caustics_canvas.setFilterEffect(caustics_rendering_program, std::bind(&KPWater::setup_caustics_parameters, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    
    caustics_power = other.caustics_power;
    caustics_amplification = other.caustics_amplification;
    caustics_sample_area = other.caustics_sample_area;

    current_rendering_pass = other.current_rendering_pass;
    p_render_target = other.p_render_target;

    return *this;
}


KPWater& KPWater::operator=(KPWater&& other)
{
    if (this == &other)
        return *this;

    AbstractRenderableObject::operator=(std::move(other));
    AbstractRenderableObjectTextured::operator=(std::move(other));
    ExtensionAggregator::operator=(std::move(other));


    kpwater_cuda = std::move(other.kpwater_cuda);
    domain_settings = std::move(other.domain_settings);
    is_initialized = other.is_initialized;
    ode_solver = other.ode_solver;
    bc_callback = std::move(other.bc_callback);
    g = other.g;
    theta = other.theta;
    eps = other.eps;
    p_water_heightmap_data = other.p_water_heightmap_data; other.p_water_heightmap_data = nullptr;
    p_topography_heightmap_data = other.p_topography_heightmap_data; other.p_topography_heightmap_data = nullptr;
    p_interpolated_topography_heightmap_data = other.p_interpolated_topography_heightmap_data; other.p_interpolated_topography_heightmap_data = nullptr;
    lod_factor = other.lod_factor;
    max_light_penetration_depth = other.max_light_penetration_depth;
    water_heightmap_tex_res = std::move(other.water_heightmap_tex_res);
    topography_heightmap_tex_res = std::move(other.topography_heightmap_tex_res);
    refraction_texture_with_caustics_tex_res = std::move(other.refraction_texture_with_caustics_tex_res);
    refraction_texture_depth_map = std::move(other.refraction_texture_depth_map);
    refraction_texture_sampler_ref_code = std::move(other.refraction_texture_sampler_ref_code);
    ripple_texture_sampler_ref_code = std::move(other.ripple_texture_sampler_ref_code);
    normal_texture_sampler_ref_code = std::move(other.normal_texture_sampler_ref_code);
    force_water_heightmap_update = false;
    water_rendering_program_ref_code = std::move(other.water_rendering_program_ref_code);
    fft_compute_program_ref_code = std::move(other.fft_compute_program_ref_code);
    tess_billet_horizontal_resolution = other.tess_billet_horizontal_resolution;
    tess_billet_vertical_resolution = other.tess_billet_vertical_resolution;
    max_deep_water_wave_amplitude = other.max_deep_water_wave_amplitude;
    max_capillary_wave_amplitude = other.max_capillary_wave_amplitude;
    v2RippleSimulationTime = std::move(other.v2RippleSimulationTime);
    v2RippleSimulationDt = std::move(other.v2RippleSimulationDt);
    //fresnel_power = other.fresnel_power;
    v3ColorExtinctionFactors = std::move(other.v3ColorExtinctionFactors);
    v2WindVelocity = std::move(other.v2WindVelocity);
    standard_gaussian_distribution = std::move(other.standard_gaussian_distribution);
    phillips_spectrum_tex_res = std::move(other.phillips_spectrum_tex_res);
    fft_ripples_tex_res = std::move(other.fft_ripples_tex_res);
    fft_displacement_map_tex_res = std::move(other.fft_displacement_map_tex_res);
    fft_ripples_normal_map_global_scale_tex_res = std::move(other.fft_ripples_normal_map_global_scale_tex_res);
    fft_ripples_normal_map_capillary_scale_tex_res = std::move(other.fft_ripples_normal_map_capillary_scale_tex_res);
    choppiness = other.choppiness;
    uv2DeepWaterRippleMapTilingFactor = std::move(other.uv2DeepWaterRippleMapTilingFactor);
    max_wave_height_as_elevation_fraction = other.max_wave_height_as_elevation_fraction;

    fractal_noise = std::move(other.fractal_noise);
    fractal_noise_map_tex_res = std::move(other.fractal_noise_map_tex_res);
    fractal_noise_update_counter = other.fractal_noise_update_counter;

    caustics_framebuffer = std::move(other.caustics_framebuffer);
    
    caustics_rendering_program = std::move(other.caustics_rendering_program);
    //Note: after this operation caustics_canvas of "this" object will refer to caustics_rendering_program of the "other" object. This should be corrected!
    caustics_canvas = std::move(other.caustics_canvas);
    caustics_canvas.setFilterEffect(caustics_rendering_program, std::bind(&KPWater::setup_caustics_parameters, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

    caustics_power = other.caustics_power;
    caustics_amplification = other.caustics_amplification;
    caustics_sample_area = other.caustics_sample_area;

    current_rendering_pass = other.current_rendering_pass;
    p_render_target = other.p_render_target;
    ogl_vertex_array_object = other.ogl_vertex_array_object; other.ogl_vertex_array_object = 0;
    ogl_buffers[0] = other.ogl_buffers[0]; other.ogl_buffers[0] = 0;
    ogl_buffers[1] = other.ogl_buffers[1]; other.ogl_buffers[1] = 0;

    return *this;
}


void KPWater::initialize(const SaintVenantSystem::Numeric* init_water_levels,
    const SaintVenantSystem::Numeric* init_horizontal_speed_flux, const SaintVenantSystem::Numeric* init_vertical_speed_flux,
    uint32_t domain_width, uint32_t domain_height, SaintVenantSystem::Numeric dx, SaintVenantSystem::Numeric dy, SaintVenantSystem::Numeric eps, 
    const SaintVenantSystem::Numeric* p_interpolated_topography, SaintVenantSystem::Numeric g /* = 9.8 */, SaintVenantSystem::Numeric theta /* = 1.1 */, ODESolver solver /* = ODESolver::RungeKutta33 */)
{
    if (is_initialized) return;

    SaintVenantSystem::SVSVarU init_state;
    SaintVenantSystem::Numeric* init_state_scaled = new SaintVenantSystem::Numeric[3*domain_width*domain_height];

    init_state.w = init_state_scaled;
    init_state.hu = init_state_scaled + domain_width*domain_height;
    init_state.hv = init_state_scaled + 2 * domain_width*domain_height;

    vec3 v3Scale = getObjectScale();
    p_topography_heightmap_data = new float[domain_width*domain_height];
    p_interpolated_topography_heightmap_data = new SaintVenantSystem::Numeric[(2 * domain_width + 1)*(2 * domain_height + 1)];
    for (uint32_t i = 0; i < domain_height; ++i)
    {
        for (uint32_t j = 0; j < domain_width; ++j)
        {
            p_topography_heightmap_data[i*domain_width + j] = static_cast<float>(p_interpolated_topography[(2 * i + 1)*(2 * domain_width + 1) + 2 * j + 1]) * v3Scale.y;
            init_state_scaled[i*domain_width + j] = init_water_levels[i*domain_width + j] * v3Scale.y;
            init_state_scaled[domain_width*domain_height + i*domain_width + j] = init_horizontal_speed_flux[i*domain_width + j] * v3Scale.x;
            init_state_scaled[2 * domain_width*domain_height + i*domain_width + j] = init_vertical_speed_flux[i*domain_width + j] * v3Scale.z;


            p_interpolated_topography_heightmap_data[2 * i*(2 * domain_width + 1) + 2 * j] = p_interpolated_topography[2 * i*(2 * domain_width + 1) + 2 * j] * v3Scale.y;
            p_interpolated_topography_heightmap_data[2 * i*(2 * domain_width + 1) + (2 * j + 1)] = p_interpolated_topography[2 * i*(2 * domain_width + 1) + (2 * j + 1)] * v3Scale.y;
            p_interpolated_topography_heightmap_data[(2 * i + 1)*(2 * domain_width + 1) + 2 * j] = p_interpolated_topography[(2 * i + 1)*(2 * domain_width + 1) + 2 * j] * v3Scale.y;
            p_interpolated_topography_heightmap_data[(2 * i + 1)*(2 * domain_width + 1) + (2 * j + 1)] = p_interpolated_topography[(2 * i + 1)*(2 * domain_width + 1) + (2 * j + 1)] * v3Scale.y;
        }
        p_interpolated_topography_heightmap_data[2 * i*(2 * domain_width + 1) + 2 * domain_width] = p_interpolated_topography[2 * i*(2 * domain_width + 1) + 2 * domain_width] * v3Scale.y;
        p_interpolated_topography_heightmap_data[(2 * i + 1)*(2 * domain_width + 1) + 2 * domain_width] = p_interpolated_topography[(2 * i + 1)*(2 * domain_width + 1) + 2 * domain_width] * v3Scale.y;
    }
    for (uint32_t j = 0; j < 2 * domain_width + 1; ++j)
    {
        p_interpolated_topography_heightmap_data[2 * domain_height*(2 * domain_width + 1) + j] = p_interpolated_topography[2 * domain_height*(2 * domain_width + 1) + j] * v3Scale.y;
    }
            

    topography_heightmap_tex_res.first.allocateStorage(1, 1, TextureSize{ domain_width, domain_height, 1 }, InternalPixelFormat::SIZED_FLOAT_R32);
    topography_heightmap_tex_res.first.setMipmapLevelData(0, PixelLayout::RED, PixelDataType::FLOAT, p_topography_heightmap_data);
    if (!topography_heightmap_tex_res.second)
        topography_heightmap_tex_res.second = registerTexture(topography_heightmap_tex_res.first);
    else
        updateTexture(topography_heightmap_tex_res.second, topography_heightmap_tex_res.first);
    retrieveShaderProgram(water_rendering_program_ref_code)->assignUniformScalar("s2dTopographyHeightmap", getBindingUnit(topography_heightmap_tex_res.second));

    dx *= v3Scale.x;
    dy *= v3Scale.z;
    kpwater_cuda.initialize(init_state, domain_height, domain_width, dx, dy, g, theta, eps, p_interpolated_topography_heightmap_data);
    domain_settings.width = domain_width; domain_settings.height = domain_height;
    domain_settings.dx = dx; domain_settings.dy = dy;

    ode_solver = solver;
    this->g = static_cast<float>(g);
    this->theta = static_cast<float>(theta);
    this->eps = static_cast<float>(eps);

    retrieveShaderProgram(fft_compute_program_ref_code)->assignUniformScalar("fGravityConstant", static_cast<float>(g));
    retrieveShaderProgram(fft_compute_program_ref_code)->assignUniformVector("v2DomainSize", vec2{ domain_settings.dx*domain_settings.width, domain_settings.dy*domain_settings.height });

    water_heightmap_tex_res.first.allocateStorage(1, 1, TextureSize{ domain_settings.width, domain_settings.height, 1 }, InternalPixelFormat::SIZED_FLOAT_R32);
    p_water_heightmap_data = new float[domain_settings.width*domain_settings.height];
    for (unsigned int i = 0; i < domain_settings.width*domain_settings.height; ++i)
        p_water_heightmap_data[i] = static_cast<float>(init_state_scaled[i]);
    water_heightmap_tex_res.first.setMipmapLevelData(0, PixelLayout::RED, PixelDataType::FLOAT, p_water_heightmap_data);
    if (!water_heightmap_tex_res.second)
        water_heightmap_tex_res.second = registerTexture(water_heightmap_tex_res.first);
    else
        updateTexture(water_heightmap_tex_res.second, water_heightmap_tex_res.first);
    delete[] init_state_scaled;

    retrieveShaderProgram(water_rendering_program_ref_code)->assignUniformScalar("s2dWaterHeightMap", getBindingUnit(water_heightmap_tex_res.second));

    //Configure deep water waves
    setup_deep_water_waves();

    retrieveShaderProgram(water_rendering_program_ref_code)->assignUniformScalar("fMaxCapillaryWaveAmplitude", max_capillary_wave_amplitude);


    //Generate initial noise map
    fractal_noise.generateNoiseMap();


    is_initialized = true;
}


void KPWater::reset()
{
    if (!is_initialized) return;
    kpwater_cuda.reset();
    is_initialized = false;
    domain_settings.width = 0; domain_settings.height = 0;
    domain_settings.dx = 0; domain_settings.dy = 0;

    water_heightmap_tex_res.first = ImmutableTexture2D{ "KPWater::water_height_map" };
    topography_heightmap_tex_res.first = ImmutableTexture2D{ "KPWater::topography_height_map" };

    if (p_water_heightmap_data) { delete[] p_water_heightmap_data; p_water_heightmap_data = nullptr; }
    if (p_topography_heightmap_data) { delete[] p_topography_heightmap_data; p_topography_heightmap_data = nullptr; }
    if (p_interpolated_topography_heightmap_data) { delete[] p_interpolated_topography_heightmap_data; p_interpolated_topography_heightmap_data = nullptr; }
    glDeleteBuffers(2, ogl_buffers);
    glDeleteVertexArrays(1, &ogl_vertex_array_object);
}


void KPWater::setODESolver(ODESolver ode_solver)
{
    this->ode_solver = ode_solver;
}


KPWater::ODESolver KPWater::getODESolver() const { return ode_solver; }


void KPWater::registerBoundaryConditionsComputeProcedure(const BoundaryConditionsComputeProcedure& callback)
{
    bc_callback = callback;
}


void KPWater::step(SaintVenantSystem::Numeric dt)
{
    if (!is_initialized) return;

    SaintVenantSystem::SVSBoundary wb, eb, sb, nb;
    BoundaryCondition west_boundary, east_boundary, south_boundary, north_boundary;
    bc_callback(&kpwater_cuda, p_interpolated_topography_heightmap_data, west_boundary, east_boundary, south_boundary, north_boundary);
    west_boundary.getRawData(&wb.w, &wb.w_edge, &wb.hu, &wb.hu_edge, &wb.hv, &wb.hv_edge);
    east_boundary.getRawData(&eb.w, &eb.w_edge, &eb.hu, &eb.hu_edge, &eb.hv, &eb.hv_edge);
    south_boundary.getRawData(&sb.w, &sb.w_edge, &sb.hu, &sb.hu_edge, &sb.hv, &sb.hv_edge);
    north_boundary.getRawData(&nb.w, &nb.w_edge, &nb.hu, &nb.hu_edge, &nb.hv, &nb.hv_edge);

    switch (ode_solver)
    {
    case ODESolver::RungeKutta22:
        kpwater_cuda.solveSSPRK22(wb, eb, sb, nb, dt);
        break;
    case ODESolver::RungeKutta33:
        kpwater_cuda.solveSSPRK33(wb, eb, sb, nb, dt);
        break;
    case ODESolver::Euler:
        kpwater_cuda.solveEuler(wb, eb, sb, nb, dt);
        break;
    }

    v2RippleSimulationTime += v2RippleSimulationDt;
    retrieveShaderProgram(fft_compute_program_ref_code)->assignUniformScalar("fTimeGlobalScale", v2RippleSimulationTime.x);
    retrieveShaderProgram(fft_compute_program_ref_code)->assignUniformScalar("fTimeCapillaryScale", v2RippleSimulationTime.y);
    fractal_noise_update_counter = ++fractal_noise_update_counter % fractal_noise_update_period;
}


void KPWater::updateSystemState(const std::vector<float>& water_levels, const std::vector<float>& horizontal_velocities, const std::vector<float>& vertical_velocities)
{
    if (!is_initialized) return;

    SaintVenantSystem::Numeric* p_new_system_state = new SaintVenantSystem::Numeric[3 * domain_settings.width*domain_settings.height];
    SaintVenantSystem::Numeric* p_water_levels = p_new_system_state;
    SaintVenantSystem::Numeric* p_hu = p_new_system_state + domain_settings.width*domain_settings.height;
    SaintVenantSystem::Numeric* p_hv = p_new_system_state + 2 * domain_settings.width*domain_settings.height;

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(domain_settings.width*domain_settings.height); ++i)
    {
        p_water_levels[i] = static_cast<SaintVenantSystem::Numeric>(water_levels[i]);
        p_hu[i] = static_cast<SaintVenantSystem::Numeric>(horizontal_velocities[i]);
        p_hv[i] = static_cast<SaintVenantSystem::Numeric>(vertical_velocities[i]);

        p_water_heightmap_data[i] = p_water_levels[i];
    }

    kpwater_cuda.updateSystemState(SaintVenantSystem::SVSVarU{ p_water_levels, p_hu, p_hv });
    water_heightmap_tex_res.first.setMipmapLevelData(0, PixelLayout::RED, PixelDataType::FLOAT, p_water_heightmap_data);

    delete[] p_new_system_state;
}


void KPWater::updateSystemState(const std::vector<float>& water_levels)
{
    if (!is_initialized) return;

    const SaintVenantSystem::Numeric4* p_current_system_state = kpwater_cuda.getSystemState();

    SaintVenantSystem::Numeric* p_new_system_state = new SaintVenantSystem::Numeric[3 * domain_settings.width*domain_settings.height];
    SaintVenantSystem::Numeric* p_water_levels = p_new_system_state;
    SaintVenantSystem::Numeric* p_hu = p_new_system_state + domain_settings.width*domain_settings.height;
    SaintVenantSystem::Numeric* p_hv = p_new_system_state + 2 * domain_settings.width*domain_settings.height;

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(domain_settings.height); ++i)
    {
        for (int j = 0; j < static_cast<int>(domain_settings.width); ++j)
        {
            p_hu[i*domain_settings.width + j] = p_current_system_state[(i + 1)*(domain_settings.width + 2) + j + 1].y;
            p_hv[i*domain_settings.width + j] = p_current_system_state[(i + 1)*(domain_settings.width + 2) + j + 1].z;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(domain_settings.width*domain_settings.height); ++i)
    {
        p_water_levels[i] = static_cast<SaintVenantSystem::Numeric>(water_levels[i]);

        p_water_heightmap_data[i] = p_water_levels[i];
    }


    kpwater_cuda.updateSystemState(SaintVenantSystem::SVSVarU{ p_water_levels, p_hu, p_hv });
    water_heightmap_tex_res.first.setMipmapLevelData(0, PixelLayout::RED, PixelDataType::FLOAT, p_water_heightmap_data);

    delete[] p_new_system_state;
}


void KPWater::retrieveCurrentSystemState(std::vector<float>& water_levels, std::vector<float>& horizontal_velocities, std::vector<float>& vertical_velocities) const
{
    if (!is_initialized) return;

    const SaintVenantSystem::Numeric4* p_current_system_state = kpwater_cuda.getSystemState();
    water_levels.reserve(domain_settings.width*domain_settings.height);
    horizontal_velocities.reserve(domain_settings.width*domain_settings.height);
    vertical_velocities.reserve(domain_settings.width*domain_settings.height);

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(domain_settings.height); ++i)
        for (int j = 0; j < static_cast<int>(domain_settings.width); ++j)
        {
            water_levels[i*domain_settings.width + j] = p_current_system_state[(i + 1)*(domain_settings.width + 2) + j + 1].x;
            horizontal_velocities[i*domain_settings.width + j] = p_current_system_state[(i + 1)*(domain_settings.width + 2) + j + 1].y;
            vertical_velocities[i*domain_settings.width + j] = p_current_system_state[(i + 1)*(domain_settings.width + 2) + j + 1].z;
        }
}


void KPWater::updateTopography(const std::vector<float>& interpolated_topography_values)
{
    if (!is_initialized) return;

    float object_scale_y = getObjectScale().y;

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(domain_settings.height); ++i)
    {
        for (int j = 0; j < static_cast<int>(domain_settings.width); ++j)
        {
            p_interpolated_topography_heightmap_data[(2 * i + 1)*(2 * domain_settings.width + 1) + 2 * j + 1] =
                interpolated_topography_values[(2 * i + 1)*(2 * domain_settings.width + 1) + 2 * j + 1] * object_scale_y;
            p_interpolated_topography_heightmap_data[2 * i*(2 * domain_settings.width + 1) + 2 * j + 1] =
                interpolated_topography_values[2 * i*(2 * domain_settings.width + 1) + 2 * j + 1] * object_scale_y;
            p_interpolated_topography_heightmap_data[(2 * i + 1)*(2 * domain_settings.width + 1) + 2 * j] =
                interpolated_topography_values[(2 * i + 1)*(2 * domain_settings.width + 1) + 2 * j] * object_scale_y;
            p_interpolated_topography_heightmap_data[2 * i*(2 * domain_settings.width + 1) + 2 * j] =
                interpolated_topography_values[2 * i*(2 * domain_settings.width + 1) + 2 * j] * object_scale_y;


            p_topography_heightmap_data[i*domain_settings.width + j] =
                static_cast<float>(p_interpolated_topography_heightmap_data[(2 * i + 1)*(2 * domain_settings.width + 1) + 2 * j + 1]);
        }

        p_interpolated_topography_heightmap_data[(2 * i + 1)*(2 * domain_settings.width + 1) + 2 * domain_settings.width] =
            interpolated_topography_values[(2 * i + 1)*(2 * domain_settings.width + 1) + 2 * domain_settings.width] * object_scale_y;
        p_interpolated_topography_heightmap_data[2 * i*(2 * domain_settings.width + 1) + 2 * domain_settings.width] =
            interpolated_topography_values[2 * i*(2 * domain_settings.width + 1) + 2 * domain_settings.width] * object_scale_y;
    }

    #pragma omp parallel for
    for (int j = 0; j < static_cast<int>(2 * domain_settings.width + 1); ++j)
    {
        p_interpolated_topography_heightmap_data[2 * domain_settings.height*(2 * domain_settings.width + 1) + j] =
            interpolated_topography_values[2 * domain_settings.height*(2 * domain_settings.width + 1) + j] * object_scale_y;
    }


    kpwater_cuda.updateTopographyData(p_interpolated_topography_heightmap_data);
    topography_heightmap_tex_res.first.setMipmapLevelData(0, PixelLayout::RED, PixelDataType::FLOAT, p_topography_heightmap_data);
}


void KPWater::setLODFactor(float factor) { lod_factor = factor; }

float KPWater::getLODFactor() const { return lod_factor; }


void KPWater::setColorExtinctionBoundary(float water_level) { max_light_penetration_depth = water_level; }

float KPWater::getColorExtinctionBoundary() const { return max_light_penetration_depth; }


void KPWater::setMaximalDeepWaveAmplitude(float amplitude)
{
    max_deep_water_wave_amplitude = amplitude;
    retrieveShaderProgram(water_rendering_program_ref_code)->assignUniformScalar("fMaxDeepWaterWaveAmplitude", max_deep_water_wave_amplitude);
}

float KPWater::getMaximalDeepWaveAmplitude() const { return max_deep_water_wave_amplitude; }


void KPWater::setMaximalRippleWaveAmplitude(float amplitude)
{
    max_capillary_wave_amplitude = amplitude;
    retrieveShaderProgram(water_rendering_program_ref_code)->assignUniformScalar("fMaxCapillaryWaveAmplitude", max_capillary_wave_amplitude);
}

float KPWater::getMaximalRippleWaveAmplitude() const { return max_capillary_wave_amplitude; }


void KPWater::setRippleSimulationTimeStep(const vec2& dt) { v2RippleSimulationDt = dt; }

vec2 KPWater::getRippleSimulationTimeStep() const { return v2RippleSimulationDt; }


//void KPWater::setFresnelPower(float value) { fresnel_power = value; }

//float KPWater::getFresnelPower() const { return fresnel_power; }


void KPWater::setColorExtinctionFactors(float red_extinction_factor, float green_extinction_factor, float blue_extinction_factor)
{
    v3ColorExtinctionFactors.x = red_extinction_factor;
    v3ColorExtinctionFactors.y = green_extinction_factor;
    v3ColorExtinctionFactors.z = blue_extinction_factor;
}

void KPWater::setColorExtinctionFactors(const vec3& color_extinction_factors) { v3ColorExtinctionFactors = color_extinction_factors; }

vec3 KPWater::getColorExtinctionFactors() const { return v3ColorExtinctionFactors; }


void KPWater::setWindVelocity(const vec2& wind_speed)
{
    v2WindVelocity = wind_speed;
    if(is_initialized) setup_deep_water_waves();
}

vec2 KPWater::getWindVelocity() const { return v2WindVelocity; }


void KPWater::setDeepWaterWavesChoppiness(float choppiness) 
{ 
    this->choppiness = choppiness;
    retrieveShaderProgram(fft_compute_program_ref_code)->assignUniformScalar("fChoppiness", choppiness);
}

float KPWater::getDeepWaterWavesChoppiness() const { return choppiness; }


void KPWater::setDeepWaterRippleMapTilingFactor(uint32_t geometry_waves_tiling_factor, uint32_t bump_map_tiling_factor)
{
    uv2DeepWaterRippleMapTilingFactor.x = geometry_waves_tiling_factor;
    uv2DeepWaterRippleMapTilingFactor.y = bump_map_tiling_factor;
    retrieveShaderProgram(water_rendering_program_ref_code)->assignUniformVector("uv2DeepWaterRippleMapTilingFactor", uv2DeepWaterRippleMapTilingFactor);
    caustics_rendering_program.assignUniformVector("uv2DeepWaterRippleMapTilingFactor", uv2DeepWaterRippleMapTilingFactor);
}

uvec2 KPWater::getDeepWaterRippleMapTilingFactor() const { return uv2DeepWaterRippleMapTilingFactor; }


void KPWater::setMaximalWaveHeightAsElevationFraction(float ratio)
{
    max_wave_height_as_elevation_fraction = ratio;
    retrieveShaderProgram(water_rendering_program_ref_code)->assignUniformScalar("fMaxWaveHeightAsElevationFraction", max_wave_height_as_elevation_fraction);
}

float KPWater::getMaximalWaveHeightAsElevationFraction() const { return max_wave_height_as_elevation_fraction; }


void KPWater::setCausticsPower(float power)
{
    caustics_power = power;
    caustics_rendering_program.assignUniformScalar("fCausticsPower", caustics_power);
}

float KPWater::getCausticsPower() const { return caustics_power; }


void KPWater::setCausticsAmplificationFactor(float factor)
{
    caustics_amplification = factor;
    caustics_rendering_program.assignUniformScalar("fCausticsAmplification", caustics_amplification);
}

float KPWater::getCausticsAmplificationFactor() const { return caustics_amplification; }


void KPWater::setCausticsSampleArea(float area)
{
    caustics_sample_area = area;
    caustics_rendering_program.assignUniformScalar("fCausticsSampleArea", area);
}

float KPWater::getCausticsSampleArea() const { return caustics_sample_area; }


KPWater::DomainSettings KPWater::getDomainDetails() const
{
    return DomainSettings{ domain_settings.dx*domain_settings.width, domain_settings.dy*domain_settings.height,
        domain_settings.width, domain_settings.height };
}



void KPWater::updateRefractionTexture(const ImmutableTexture2D& refraction_texture, const ImmutableTexture2D& refraction_texture_depth_map)
{
    /*if (!refraction_texture_tex_res.second)
        refraction_texture_tex_res.second = registerTexture(refraction_texture, &refraction_texture_sampler);
    else
        updateTexture(refraction_texture_tex_res.second, refraction_texture);
    retrieveShaderProgram(water_rendering_program_ref_code)->assignUniformScalar("s2dRefractionTexture", getBindingUnit(refraction_texture_tex_res.second));
    refraction_texture_tex_res.first = refraction_texture;*/

    caustics_canvas.installTexture(refraction_texture);
    this->refraction_texture_depth_map = refraction_texture_depth_map;
}



void KPWater::scale(float x_scale_factor, float y_scale_factor, float z_scale_factor)
{
    if (is_initialized)
    {
        const SaintVenantSystem::Numeric4* p_kpwater_system_state = kpwater_cuda.getSystemState();
        SaintVenantSystem::Numeric* p_init_water_heights = new SaintVenantSystem::Numeric[domain_settings.width*domain_settings.height * 3];
        SaintVenantSystem::Numeric* p_init_horizontal_flux = p_init_water_heights + domain_settings.width*domain_settings.height;
        SaintVenantSystem::Numeric* p_init_vertical_flux = p_init_water_heights + domain_settings.width*domain_settings.height * 2;

        for (uint32_t i = 0; i < domain_settings.height; ++i)
        {
            for (uint32_t j = 0; j < domain_settings.width; ++j)
            {
                p_topography_heightmap_data[i * domain_settings.width + j] *= y_scale_factor;
                
                p_init_water_heights[i*domain_settings.width + j] = p_kpwater_system_state[(i + 1)*(domain_settings.width + 2) + j + 1].x * y_scale_factor;
                p_init_horizontal_flux[i*domain_settings.width + j] = p_kpwater_system_state[(i + 1)*(domain_settings.width + 2) + j + 1].y * x_scale_factor;
                p_init_vertical_flux[i*domain_settings.width + j] = p_kpwater_system_state[(i + 1)*(domain_settings.width + 2) + j + 1].z * z_scale_factor;

                p_water_heightmap_data[i*domain_settings.width + j] = static_cast<float>(p_init_water_heights[i*domain_settings.width + j]);

                p_interpolated_topography_heightmap_data[2 * i*(2 * domain_settings.width + 1) + 2 * j] *= y_scale_factor;
                p_interpolated_topography_heightmap_data[2 * i*(2 * domain_settings.width + 1) + (2 * j + 1)] *= y_scale_factor;
                p_interpolated_topography_heightmap_data[(2 * i + 1)*(2 * domain_settings.width + 1) + 2 * j] *= y_scale_factor;
                p_interpolated_topography_heightmap_data[(2 * i + 1)*(2 * domain_settings.width + 1) + (2 * j + 1)] *= y_scale_factor;
            }
            p_interpolated_topography_heightmap_data[2 * i*(2 * domain_settings.width + 1) + 2 * domain_settings.width] *= y_scale_factor;
            p_interpolated_topography_heightmap_data[(2 * i + 1)*(2 * domain_settings.width + 1) + 2 * domain_settings.width] *= y_scale_factor;
        }
        for (uint32_t j = 0; j < 2 * domain_settings.width + 1; ++j)
        {
            p_interpolated_topography_heightmap_data[2 * domain_settings.height*(2 * domain_settings.width + 1) + j] *= y_scale_factor;
        }

        topography_heightmap_tex_res.first.setMipmapLevelData(0, PixelLayout::RED, PixelDataType::FLOAT, p_topography_heightmap_data);
        water_heightmap_tex_res.first.setMipmapLevelData(0, PixelLayout::RED, PixelDataType::FLOAT, p_water_heightmap_data);
        domain_settings.dx *= x_scale_factor;
        domain_settings.dy *= z_scale_factor;


        kpwater_cuda.reset();
        SaintVenantSystem::SVSVarU init_state;
        init_state.w = p_init_water_heights;
        init_state.hu = p_init_horizontal_flux;
        init_state.hv = p_init_vertical_flux;
        kpwater_cuda.initialize(init_state, domain_settings.height, domain_settings.width, domain_settings.dx, domain_settings.dy, g, theta, eps, p_interpolated_topography_heightmap_data);
    }

    AbstractRenderableObject::scale(x_scale_factor, y_scale_factor, z_scale_factor);
}


void KPWater::scale(const vec3& new_scale_factors)
{
    scale(new_scale_factors.x, new_scale_factors.y, new_scale_factors.z);
}



bool KPWater::supportsRenderingMode(uint32_t rendering_mode) const
{
    if (rendering_mode == TW_RENDERING_MODE_DEFAULT) return true;
    else return false;
}


uint32_t KPWater::getNumberOfRenderingPasses(uint32_t rendering_mode) const
{
    if (rendering_mode == TW_RENDERING_MODE_DEFAULT) return 3;
    else return 0;
}


bool KPWater::render()
{
    switch (current_rendering_pass)
    {
    case 0:
        return true;

    case 1:
    {
        caustics_canvas.selectRenderingMode(TW_RENDERING_MODE_DEFAULT);
        for (uint32_t i = 0; i < caustics_canvas.getNumberOfRenderingPasses(caustics_canvas.getActiveRenderingMode()); ++i)
        {
            caustics_canvas.prepareRendering(caustics_framebuffer, i);
            caustics_canvas.render();
            caustics_canvas.finalizeRendering();
        }
        return true;
    }

    case 2:
        glDrawElements(GL_PATCHES, 4 * (tess_billet_horizontal_resolution - 1)*(tess_billet_vertical_resolution - 1), GL_UNSIGNED_INT, 0);
        return true;

    default:
        return false;
    }
    
    return true;
}




void KPWater::computeTopographyBilinearInterpolation(const float* topography, uint32_t topography_width, uint32_t topography_height, float* interpolated_topography)
{
    for (unsigned int i = 0; i < topography_height - 1; ++i)
    {
        for (unsigned int j = 0; j < topography_width - 1; ++j)
        {
            SaintVenantSystem::Numeric aux1 = (topography[i*topography_width + j] + topography[i*topography_width + j + 1]) / 2;
            SaintVenantSystem::Numeric aux2 = (topography[(i + 1)*topography_width + j] + topography[(i + 1)*topography_width + j + 1]) / 2;
            interpolated_topography[(2 * i + 1) * (2 * topography_width + 1) + 2 * j + 1] = (aux1 + aux2) / 2;
            interpolated_topography[(2 * i) * (2 * topography_width + 1) + 2 * j + 1] = aux1;
            interpolated_topography[(2 * i + 1) * (2 * topography_width + 1) + 2 * j] = (topography[i*topography_width + j] + topography[(i + 1)*topography_width + j]) / 2;
            interpolated_topography[(2 * i) * (2 * topography_width + 1) + 2 * j] = topography[i*topography_width + j];
        }
        SaintVenantSystem::Numeric aux1 = topography[i*topography_width + topography_width - 1];
        SaintVenantSystem::Numeric aux2 = topography[(i + 1)*topography_width + topography_width - 1];
        interpolated_topography[(2 * i + 1) * (2 * topography_width + 1) + 2 * topography_width - 1] = (aux1 + aux2) / 2;

        interpolated_topography[(2 * i) * (2 * topography_width + 1) + 2 * topography_width - 1] = aux1;

        interpolated_topography[(2 * i + 1) * (2 * topography_width + 1) + 2 * topography_width - 2] = (aux1 + aux2) / 2;
        interpolated_topography[(2 * i + 1) * (2 * topography_width + 1) + 2 * topography_width] = (aux1 + aux2) / 2;

        interpolated_topography[(2 * i) * (2 * topography_width + 1) + 2 * topography_width - 2] = topography[i * topography_width + topography_width - 1];
        interpolated_topography[(2 * i) * (2 * topography_width + 1) + 2 * topography_width] = topography[i * topography_width + topography_width - 1];
    }

    for (unsigned int j = 0; j < topography_width - 1; ++j)
    {
        SaintVenantSystem::Numeric aux = (topography[(topography_height - 1)*topography_width + j] + topography[(topography_height - 1)*topography_width + j + 1]) / 2;
        interpolated_topography[(2 * topography_height - 1) * (2 * topography_width + 1) + 2 * j + 1] = aux;

        interpolated_topography[(2 * topography_height - 2) * (2 * topography_width + 1) + 2 * j + 1] = aux;
        interpolated_topography[(2 * topography_height) * (2 * topography_width + 1) + 2 * j + 1] = aux;

        interpolated_topography[(2 * topography_height - 1) * (2 * topography_width + 1) + 2 * j] = topography[(topography_height - 1)*topography_width + j];

        interpolated_topography[(2 * topography_height - 2) * (2 * topography_width + 1) + 2 * j] = topography[(topography_height - 1)*topography_width + j];
        interpolated_topography[(2 * topography_height) * (2 * topography_width + 1) + 2 * j] = topography[(topography_height - 1)*topography_width + j];
    }

    SaintVenantSystem::Numeric aux = topography[(topography_height - 1)*topography_width + topography_width - 1];
    interpolated_topography[(2 * topography_height - 1) * (2 * topography_width + 1) + 2 * topography_width - 1] = aux;

    interpolated_topography[(2 * topography_height - 2) * (2 * topography_width + 1) + 2 * topography_width - 1] = aux;
    interpolated_topography[(2 * topography_height) * (2 * topography_width + 1) + 2 * topography_width - 1] = aux;

    interpolated_topography[(2 * topography_height - 1) * (2 * topography_width + 1) + 2 * topography_width - 2] = topography[(topography_height - 1)*topography_width + topography_width - 1];
    interpolated_topography[(2 * topography_height - 1) * (2 * topography_width + 1) + 2 * topography_width] = topography[(topography_height - 1)*topography_width + topography_width - 1];

    interpolated_topography[(2 * topography_height - 2) * (2 * topography_width + 1) + 2 * topography_width - 2] = topography[(topography_height - 1)*topography_width + topography_width - 1];
    interpolated_topography[(2 * topography_height - 2) * (2 * topography_width + 1) + 2 * topography_width] = topography[(topography_height - 1)*topography_width + topography_width - 1];
    interpolated_topography[(2 * topography_height) * (2 * topography_width + 1) + 2 * topography_width - 2] = topography[(topography_height - 1)*topography_width + topography_width - 1];
    interpolated_topography[(2 * topography_height) * (2 * topography_width + 1) + 2 * topography_width] = topography[(topography_height - 1)*topography_width + topography_width - 1];
}


void KPWater::extractTopographyPrincipalComponent(const float* interpolated_topography, uint32_t topography_width, uint32_t topography_height, float* topography_principal_component)
{
    for (uint32_t i = 0; i < topography_height; ++i)
        for (uint32_t j = 0; j < topography_width; ++j)
            topography_principal_component[i*topography_width + j] = interpolated_topography[(2 * i + 1)*(2 * topography_width + 1) + 2 * j + 1];
}