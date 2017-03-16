#include <chrono>

#include "Skydome.h"
#include "KTXTexture.h"
#include "Misc.h"

#define pi 3.1415926535897932384626433832795f

#define VA_SKYDOME 0		//offset to vertex attribute object used for rendering of the sky dome
#define VA_STARS 1			//offset to vertex attribute object used for rendering of the star field
#define VA_MOON 2			//offset to vertex attribute object used for moon rendering
#define VA_SUN 3			//offset to vertex attribute object used for sun rendering

#define AB_VERTEX 0	//offset to array buffer used for vertex attribute storage
#define AB_INDEX 1	//offset to array buffer used for index storage

using namespace tiny_world;




LightScatteringSettings::LightScatteringSettings() : red_wavelength{ 0.700f }, green_wavelength{ 0.570f }, blue_wavelength{ 0.475f },
rayleigh_scale{ 0.0055f }, mie_coefficient{ 0.002f }, mie_phase_function_param{ -0.99f }, scattering_to_far_world_ratio{ 100.0f }
{

}

bool LightScatteringSettings::operator==(const LightScatteringSettings& other) const
{
    return red_wavelength == other.red_wavelength && green_wavelength == other.green_wavelength && blue_wavelength == other.blue_wavelength &&
        rayleigh_scale == other.rayleigh_scale && mie_coefficient == other.mie_coefficient && mie_phase_function_param == other.mie_phase_function_param &&
        scattering_to_far_world_ratio == other.scattering_to_far_world_ratio;
}




typedef AbstractRenderableObjectExtensionAggregator<AbstractRenderableObjectHDRBloomEx> ExtensionAggregator;

//Define static string names of the shader programs implementing the sky dome
const std::string Skydome::skydome_rendering_program_name = "Skydome::rendering_program";
const std::string Skydome::star_field_rendering_program_name = "Skydome::star_field_rendering_program";
const std::string Skydome::moon_rendering_program_name = "Skydome::moon_rendering_program";

//Type describing generic vertex attribute representing size of a sky body (of sun, moon or of a star)
typedef VertexAttributeSpecification<TW_RESERVED_VA_IDs, ogl_type_mapper<float>::ogl_type, 4> vertex_attribute_sky_body_luminosity;


void Skydome::applyScreenSize(const uvec2& screen_size)
{

}


void Skydome::init_random_generator()
{
    std::chrono::system_clock::time_point current_system_time = std::chrono::system_clock::now();	//retrieve current system time
    std::default_random_engine::result_type seed =		//generate seed value for random-number generator from the system time
        static_cast<std::default_random_engine::result_type>(std::chrono::duration_cast<std::chrono::seconds>(current_system_time.time_since_epoch()).count());
    default_random_generator = std::default_random_engine{ seed };
}

void Skydome::generate_vertex_data()
{
    void* vdata = malloc(vertex_buffer_capacity);

    float lon_step_size = (2.0f * pi) / longitude_steps;
    float lat_step_size = (pi / 2.0f - atmospheric_scattering_constants::horizon_angle) / latitude_steps;


    for (int i = 0; i < static_cast<int>(latitude_steps); ++i)
    {
        for (int j = 0; j < static_cast<int>(longitude_steps); ++j)
        {
            //Currently processed vertex position in the upper hemisphere
            vertex_attribute_position::value_type* p_uh_vertex =
                reinterpret_cast<vertex_attribute_position::value_type*>(static_cast<char*>(vdata)+
                (i*longitude_steps + j) * (vertex_attribute_position::getCapacity() + vertex_attribute_texcoord::getCapacity()));

            //x-coordinate of the currently processed vertex in the upper hemisphere
            p_uh_vertex[0] = atmospheric_scattering_constants::sky_sphere_radius * std::cos(i * lat_step_size + atmospheric_scattering_constants::horizon_angle) * std::cos(j * lon_step_size);

            //y-coordinate of the currently processed vertex in the upper hemisphere
            p_uh_vertex[1] = atmospheric_scattering_constants::sky_sphere_radius * std::sin(i * lat_step_size + atmospheric_scattering_constants::horizon_angle) - atmospheric_scattering_constants::planet_radius;

            //z-coordinate of the currently processed vertex in the upper hemisphere
            p_uh_vertex[2] = -atmospheric_scattering_constants::sky_sphere_radius * std::cos(i * lat_step_size + atmospheric_scattering_constants::horizon_angle) * std::sin(j * lon_step_size);

            //w-coordinate of the currently processed vertex in the upper hemisphere
            p_uh_vertex[3] = 1.0f;

            //u-texture coordinate of the currently processed vertex in the upper hemisphere
            p_uh_vertex[4] = static_cast<float>(j);

            //v-texture coordinate of the currently processed vertex in the upper hemisphere
            p_uh_vertex[5] = static_cast<float>(i);


            if (i > 0)
            {
                vertex_attribute_position::value_type* p_lh_vertex =
                    reinterpret_cast<vertex_attribute_position::value_type*>(reinterpret_cast<char*>(p_uh_vertex)+
                    (longitude_steps * (latitude_steps - 1) + 1) * (vertex_attribute_position::getCapacity() + vertex_attribute_texcoord::getCapacity()));


                //define x-, y-, and z-coordinates of the currently processed vertex in the bottom hemisphere
                p_lh_vertex[0] = p_uh_vertex[0];
                p_lh_vertex[1] = -p_uh_vertex[1];
                p_lh_vertex[2] = p_uh_vertex[2];
                p_lh_vertex[3] = 1.0f;

                //the values of u- and v- texture coordinates here simply indicate a vertex from bottom hemisphere
                p_lh_vertex[4] = -1.0f;
                p_lh_vertex[5] = -1.0f;
            }
        }
    }

    //Write top and bottom vertices positions and texture coordinates
    vertex_attribute_position::value_type* p_top_vertex = reinterpret_cast<vertex_attribute_position::value_type*>(static_cast<char*>(vdata)+
        longitude_steps * latitude_steps * (vertex_attribute_position::getCapacity() + vertex_attribute_texcoord::getCapacity()));
    p_top_vertex[0] = 0.0f;	//x-coordinate
    p_top_vertex[1] = atmospheric_scattering_constants::sky_sphere_radius - atmospheric_scattering_constants::planet_radius;	//y-coordinate
    p_top_vertex[2] = 0.0f;	//z-coordinate
    p_top_vertex[3] = 1.0f;	//w-coordinate

    //negative texture u- or v- coordinate indicate that the in-scattering value from top of the sky-dome should not be stored into the in-scattering textures
    //Here u-coordinate is positive to indicate the top of the sky sphere
    p_top_vertex[4] = 1.0f;
    p_top_vertex[5] = -1.0f;


    vertex_attribute_position::value_type* p_bottom_vertex = reinterpret_cast<vertex_attribute_position::value_type*>(static_cast<char*>(vdata)+
        ((2 * latitude_steps - 1)*longitude_steps + 1) * (vertex_attribute_position::getCapacity() + vertex_attribute_texcoord::getCapacity()));
    p_bottom_vertex[0] = 0.0f;	//x-coordinate
    p_bottom_vertex[1] = atmospheric_scattering_constants::planet_radius - atmospheric_scattering_constants::sky_sphere_radius;	//y-coordinate
    p_bottom_vertex[2] = 0.0f;	//z-coordinate
    p_bottom_vertex[3] = 1.0f;	//w-coordinate

    //negative texture u- or v- coordinate indicate that the in-scattering value from bottom of the sky-dome should not be stored into the in-scattering textures
    //Here u-coordinate is negative to indicate the bottom of the sky sphere
    p_bottom_vertex[4] = -1.0f;
    p_bottom_vertex[5] = -1.0f;



    //Generate positions for stars
    init_random_generator();
    std::uniform_real_distribution<float> latitude_distribution{ -pi/6.0f, pi+pi/6.0f };
    std::uniform_real_distribution<float> longitude_distribution{ 0.0f, 2 * pi };
    std::uniform_real_distribution<float> star_color_index{ 0.0f, 1.0f };
    std::uniform_real_distribution<float> star_size{ 0.5f, 1.5f };

    void* sky_bodies_vdata_offset = static_cast<char*>(vdata)+((2 * latitude_steps - 1) * longitude_steps + 2) * (vertex_attribute_position::getCapacity() + vertex_attribute_texcoord::getCapacity());
    vec3 blue_star{ 0.1f, 0.1f, 0.3f }, red_star{ 0.4f, 0.2f, 0.3f };

    for (int i = 0; i < static_cast<int>(num_stars); ++i)
    {
        //Write star position into vertex buffer
        vertex_attribute_position::value_type* current_sky_body_position =
            reinterpret_cast<vertex_attribute_position::value_type*>(static_cast<char*>(sky_bodies_vdata_offset)+
            i * (vertex_attribute_position::getCapacity() + vertex_attribute_sky_body_luminosity::getCapacity()));


        float latitude = latitude_distribution(default_random_generator);	//latitude of the currently generated star position
        float longitude = longitude_distribution(default_random_generator);	//longitude of the currently generated star position

        //Convert latitude and longitude values represented in horizontal celestial coordinates to the default Cartesian coordinates (with origin aligned with the viewer)
        vec3 star_position = celestial_to_cartesian(vec2{ latitude, longitude });

        current_sky_body_position[0] = star_position.x;
        current_sky_body_position[1] = star_position.y;
        current_sky_body_position[2] = star_position.z;
        current_sky_body_position[3] = 1.0f;


        //Write star color and brightness into generic vertex attribute representing luminosity
        vertex_attribute_sky_body_luminosity::value_type* current_sky_body_luminosity =
            reinterpret_cast<vertex_attribute_sky_body_luminosity::value_type*>(reinterpret_cast<char*>(current_sky_body_position)+vertex_attribute_position::getCapacity());

        vec3 star_color = red_star + star_color_index(default_random_generator) * (blue_star - red_star);
        current_sky_body_luminosity[0] = star_color.x;
        current_sky_body_luminosity[1] = star_color.y;
        current_sky_body_luminosity[2] = star_color.z;
        current_sky_body_luminosity[3] = star_size(default_random_generator);
    }

    glBindBuffer(GL_ARRAY_BUFFER, ogl_array_buffer_objects[AB_VERTEX]);
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertex_buffer_capacity, vdata);

    free(vdata);
}

void Skydome::init_skydome()
{
    //Initialize reference counter
    ref_counter = new uint32_t{ 1 };

    //Create and configure OpenGL vertex attribute objects
    glGenVertexArrays(4, ogl_vertex_attribute_objects);

    glBindVertexArray(ogl_vertex_attribute_objects[VA_SKYDOME]);
    glEnableVertexAttribArray(vertex_attribute_position::getId());
    vertex_attribute_position::setVertexAttributeBufferLayout(0, 0);
    glEnableVertexAttribArray(vertex_attribute_texcoord::getId());
    vertex_attribute_texcoord::setVertexAttributeBufferLayout(vertex_attribute_position::getCapacity(), 0);

    glBindVertexArray(ogl_vertex_attribute_objects[VA_STARS]);
    glEnableVertexAttribArray(vertex_attribute_position::getId());
    vertex_attribute_position::setVertexAttributeBufferLayout(0, 0);
    glEnableVertexAttribArray(vertex_attribute_sky_body_luminosity::getId());
    vertex_attribute_sky_body_luminosity::setVertexAttributeBufferLayout(vertex_attribute_position::getCapacity(), 0);

    glBindVertexArray(ogl_vertex_attribute_objects[VA_MOON]);
    glEnableVertexAttribArray(vertex_attribute_position::getId());
    vertex_attribute_position::setVertexAttributeBufferLayout(0, 0);
    glEnableVertexAttribArray(vertex_attribute_sky_body_luminosity::getId());
    vertex_attribute_sky_body_luminosity::setVertexAttributeBufferLayout(vertex_attribute_position::getCapacity(), 0);

    glBindVertexArray(ogl_vertex_attribute_objects[VA_SUN]);
    glEnableVertexAttribArray(vertex_attribute_position::getId());
    vertex_attribute_position::setVertexAttributeBufferLayout(0, 0);
    glEnableVertexAttribArray(vertex_attribute_sky_body_luminosity::getId());
    vertex_attribute_sky_body_luminosity::setVertexAttributeBufferLayout(vertex_attribute_position::getCapacity(), 0);


    //Create OpenGL array buffers and populate them with data
    glGenBuffers(2, ogl_array_buffer_objects);

    //Allocate storage for indices
    const uint32_t num_indices = (2 * (longitude_steps + 1)*(latitude_steps - 1) + (longitude_steps + 2)) * 2 + 2 * (latitude_steps - 1);
    GLushort* idata = static_cast<GLushort*>(malloc(sizeof(GLushort) * num_indices));
    const uint16_t bottom_hemisphere_vertex_offset = longitude_steps * latitude_steps + 1;	//index of the first vertex located in the bottom hemisphere

    for (int i = 0; i < static_cast<int>(latitude_steps) - 1; ++i)
    {
        //Set primitive restart indexes to separate the stripes
        idata[2 * i * (2 * longitude_steps + 3) + (2 * longitude_steps + 2)] = 0xFFFF;
        if (i < static_cast<int>(latitude_steps)-2)
            idata[(2 * i + 1) * (2 * longitude_steps + 3) + (2 * longitude_steps + 2)] = 0xFFFF;

        for (int j = 0; j < static_cast<int>(longitude_steps) + 1; ++j)
        {
            //Top hemisphere
            idata[2 * i * (2 * longitude_steps + 3) + 0 + 2 * j] = i*longitude_steps + j % longitude_steps;	//lower-left corner

            idata[2 * i * (2 * longitude_steps + 3) + 1 + 2 * j] = (i + 1)*longitude_steps + j % longitude_steps;	//upper-left corner

            //Bottom hemisphere
            idata[(2 * i + 1) * (2 * longitude_steps + 3) + 0 + 2 * j] =
                bottom_hemisphere_vertex_offset + i*longitude_steps + j % longitude_steps;	//lower-left corner

            idata[(2 * i + 1) * (2 * longitude_steps + 3) + 1 + 2 * j] =
                (i > 0 ? bottom_hemisphere_vertex_offset + (i - 1) * longitude_steps : 0) + j % longitude_steps;	//upper-left corner
        }
    }

    //Set vertex and index data for triangle fan center points on top and bottom of the sphere
    idata[2 * (2 * longitude_steps + 2) * (latitude_steps - 1) + 2 * (latitude_steps - 1) - 1] = longitude_steps * latitude_steps;	//vertex at the top of hemisphere
    idata[2 * (2 * longitude_steps + 2) * (latitude_steps - 1) + 2 * (latitude_steps - 1) + longitude_steps + 1] = 0xFFFF;	//primitive restart
    idata[2 * (2 * longitude_steps + 2) * (latitude_steps - 1) + 2 * (latitude_steps - 1) + longitude_steps + 2] = longitude_steps * (2 * latitude_steps - 1) + 1;	//vertex at the bottom of the hemisphere

    //Generate index data describing triangle fans on top and bottom of the sphere
    for (int j = 0; j < static_cast<int>(longitude_steps) + 1; ++j)
    {
        //Top hemisphere
        idata[2 * (2 * longitude_steps + 2) * (latitude_steps - 1) + 2 * (latitude_steps - 1) + j] =
            longitude_steps * (latitude_steps - 1) + (longitude_steps - j) % longitude_steps;

        //Bottom hemisphere
        idata[2 * (2 * longitude_steps + 2) * (latitude_steps - 1) + 2 * (latitude_steps - 1) + longitude_steps + 3 + j] =
            2* longitude_steps * (latitude_steps - 1) + 1 + j % longitude_steps;
    }


    //Bind index buffer to the context and copy the index data to GPU-side
    glBindVertexArray(ogl_vertex_attribute_objects[VA_SKYDOME]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ogl_array_buffer_objects[AB_INDEX]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, num_indices*sizeof(GLushort), idata, GL_STATIC_DRAW);
    free(idata);


    //Create vertex buffer object and allocate space for it
    glBindBuffer(GL_ARRAY_BUFFER, ogl_array_buffer_objects[AB_VERTEX]);
    glBufferData(GL_ARRAY_BUFFER, vertex_buffer_capacity, NULL, GL_DYNAMIC_DRAW);
    generate_vertex_data();	//populate vertex buffer with data

    //Bind vertex buffers to vertex attribute objects
    glBindVertexBuffer(0, ogl_array_buffer_objects[AB_VERTEX], 0, vertex_attribute_position::getCapacity() + vertex_attribute_texcoord::getCapacity());

    glBindVertexArray(ogl_vertex_attribute_objects[VA_STARS]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, NULL);
    glBindVertexBuffer(0, ogl_array_buffer_objects[AB_VERTEX],
        ((2 * latitude_steps - 1) * longitude_steps + 2) * (vertex_attribute_position::getCapacity() + vertex_attribute_texcoord::getCapacity()),
        vertex_attribute_position::getCapacity() + vertex_attribute_sky_body_luminosity::getCapacity());

    glBindVertexArray(ogl_vertex_attribute_objects[VA_MOON]);
    glBindVertexBuffer(0, ogl_array_buffer_objects[AB_VERTEX],
        ((2 * latitude_steps - 1) * longitude_steps + 2) * (vertex_attribute_position::getCapacity() + vertex_attribute_texcoord::getCapacity()) +
        num_stars * (vertex_attribute_position::getCapacity() + vertex_attribute_sky_body_luminosity::getCapacity()),
        vertex_attribute_position::getCapacity() + vertex_attribute_sky_body_luminosity::getCapacity());

    glBindVertexArray(ogl_vertex_attribute_objects[VA_SUN]);
    glBindVertexBuffer(0, ogl_array_buffer_objects[AB_VERTEX],
        ((2 * latitude_steps - 1) * longitude_steps + 2) * (vertex_attribute_position::getCapacity() + vertex_attribute_texcoord::getCapacity()) +
        (num_stars + 1) * (vertex_attribute_position::getCapacity() + vertex_attribute_sky_body_luminosity::getCapacity()),
        vertex_attribute_position::getCapacity() + vertex_attribute_sky_body_luminosity::getCapacity());

    //Define clamping texture sampler
    clamping_sampler_ref_code = createTextureSampler("Skydome::clamping_sampler", SamplerMagnificationFilter::LINEAR, SamplerMinificationFilter::LINEAR_MIPMAP_NEAREST,
        SamplerWrapping{ SamplerWrappingMode::CLAMP_TO_EDGE, SamplerWrappingMode::CLAMP_TO_EDGE, SamplerWrappingMode::CLAMP_TO_EDGE });


    //Prepare sky dome rendering program
    if (!skydome_rendering_program_ref_code)
    {
        skydome_rendering_program_ref_code = createCompleteShaderProgram(skydome_rendering_program_name, { PipelineStage::VERTEX_SHADER, PipelineStage::FRAGMENT_SHADER });

        Shader sky_dome_vp{ ShaderProgram::getShaderBaseCatalog() + "Skydome.vp.glsl", ShaderType::VERTEX_SHADER, "skydome.vp.glsl" };
        Shader sky_dome_fp{ ShaderProgram::getShaderBaseCatalog() + "Skydome.fp.glsl", ShaderType::FRAGMENT_SHADER, "skydome.fp.glsl" };

        retrieveShaderProgram(skydome_rendering_program_ref_code)->addShader(sky_dome_vp);
        retrieveShaderProgram(skydome_rendering_program_ref_code)->addShader(sky_dome_fp);


        retrieveShaderProgram(skydome_rendering_program_ref_code)->bindVertexAttributeId("v4VertexPosition", vertex_attribute_position::getId());
        retrieveShaderProgram(skydome_rendering_program_ref_code)->bindVertexAttributeId("v2TexCoords", vertex_attribute_texcoord::getId());

        retrieveShaderProgram(skydome_rendering_program_ref_code)->link();

        //Set values for uniform variables that will not change throughout the life time of the object
        retrieveShaderProgram(skydome_rendering_program_ref_code)->assignUniformScalar("fPlanetRadius", atmospheric_scattering_constants::planet_radius);
        retrieveShaderProgram(skydome_rendering_program_ref_code)->assignUniformScalar("fLengthScale", atmospheric_scattering_constants::length_scale);
        retrieveShaderProgram(skydome_rendering_program_ref_code)->assignUniformScalar("fH0", atmospheric_scattering_constants::fH0);
    }


    //Prepare star field rendering program
    if (!star_field_rendering_program_ref_code)
    {
        star_field_rendering_program_ref_code = createCompleteShaderProgram(star_field_rendering_program_name, { PipelineStage::VERTEX_SHADER, PipelineStage::FRAGMENT_SHADER });

        Shader star_field_vp{ ShaderProgram::getShaderBaseCatalog() + "Stars.vp.glsl", ShaderType::VERTEX_SHADER, "stars.vp.glsl" };
        Shader star_field_fp{ ShaderProgram::getShaderBaseCatalog() + "Stars.fp.glsl", ShaderType::FRAGMENT_SHADER, "stars.fp.glsl" };

        retrieveShaderProgram(star_field_rendering_program_ref_code)->addShader(star_field_vp);
        retrieveShaderProgram(star_field_rendering_program_ref_code)->addShader(star_field_fp);

        retrieveShaderProgram(star_field_rendering_program_ref_code)->bindVertexAttributeId("v4StarLocation", vertex_attribute_position::getId());
        retrieveShaderProgram(star_field_rendering_program_ref_code)->bindVertexAttributeId("v4StarLuminosity", vertex_attribute_sky_body_luminosity::getId());

        retrieveShaderProgram(star_field_rendering_program_ref_code)->link();
    }


    //Prepare moon rendering program
    if (!moon_rendering_program_ref_code)
    {
        moon_rendering_program_ref_code = createCompleteShaderProgram(moon_rendering_program_name, { PipelineStage::VERTEX_SHADER, PipelineStage::FRAGMENT_SHADER });

        Shader moon_vp{ ShaderProgram::getShaderBaseCatalog() + "Moon.vp.glsl", ShaderType::VERTEX_SHADER, "moon.vp.glsl" };
        Shader moon_fp{ ShaderProgram::getShaderBaseCatalog() + "Moon.fp.glsl", ShaderType::FRAGMENT_SHADER, "moon.fp.glsl" };

        retrieveShaderProgram(moon_rendering_program_ref_code)->addShader(moon_vp);
        retrieveShaderProgram(moon_rendering_program_ref_code)->addShader(moon_fp);

        retrieveShaderProgram(moon_rendering_program_ref_code)->bindVertexAttributeId("v4MoonLocation", vertex_attribute_position::getId());
        retrieveShaderProgram(moon_rendering_program_ref_code)->bindVertexAttributeId("v4MoonLuminosity", vertex_attribute_sky_body_luminosity::getId());

        retrieveShaderProgram(moon_rendering_program_ref_code)->link();
    }

    //Prepare textures
    setTextureUnitOffset(0);
    in_scattering_sun.setStringName("Skydome::in_scattering_sun");
    in_scattering_sun.allocateStorage(1, 2, TextureSize{ longitude_steps, latitude_steps, 1 }, InternalPixelFormat::SIZED_FLOAT_RGBA32);
    in_scattering_moon.setStringName("Skydome::in_scattering_moon");
    in_scattering_moon.allocateStorage(1, 2, TextureSize{ longitude_steps, latitude_steps, 1 }, InternalPixelFormat::SIZED_FLOAT_RGBA32);

    //Generate Gaussian-shaped texture for star (this texture has hard-coded size of 64-by-64)
    float* star_texture_data = new float[64 * 64];
    for (int i = 0; i < 64; ++i)
    {
        for (int j = 0; j < 64; ++j)
        {
            float x = j * 6.0f / 63.0f - 3.0f;
            float y = i * 6.0f / 63.0f - 3.0f;

            star_texture_data[64 * i + j] = std::exp(-(x*x + y*y) / 2.0f);
        }
    }
    ImmutableTexture2D star_texture{ "Skydome::star_texture" };
    star_texture.allocateStorage(7, 1, TextureSize{ 64, 64, 1 }, InternalPixelFormat::SIZED_FLOAT_R16);
    star_texture.setMipmapLevelData(0, PixelLayout::RED, PixelDataType::FLOAT, star_texture_data);
    star_texture.generateMipmapLevels();
    delete[] star_texture_data;

    star_ref_code = registerTexture(star_texture, clamping_sampler_ref_code);

    //Load default textures for moon and star
    KTXTexture ktx_loader;
    ImmutableTexture2D sun_texture{ "Skydome::sun_texture" };
    ImmutableTexture2D moon_texture{ "Skydome::moon_texture" };

    ktx_loader.loadTexture(getTextureLookupPath() + "sun.ktx");
    sun_texture = TEXTURE_2D(ktx_loader.getContainedTexture());
    ktx_loader.releaseTexture();
    sun_ref_code = registerTexture(sun_texture, clamping_sampler_ref_code);

    ktx_loader.loadTexture(getTextureLookupPath() + "moon.ktx");
    moon_texture = TEXTURE_2D(ktx_loader.getContainedTexture());
    ktx_loader.releaseTexture();
    moon_ref_code = registerTexture(moon_texture, clamping_sampler_ref_code);
}

void Skydome::free_ogl_resources()
{
    if (ogl_vertex_attribute_objects[0])
        glDeleteVertexArrays(4, ogl_vertex_attribute_objects);

    if (ogl_array_buffer_objects[0])
        glDeleteBuffers(2, ogl_array_buffer_objects);
}

vec3 Skydome::celestial_to_direction(const vec2& celestial_location) const
{
    //Latitude is periodically mapped onto the range [-pi / 6, pi + pi / 6)
    //Longitude is periodically mapped onto the range [0, 2 * pi)
    vec2 corrected_celestial_location = celestial_location;

    corrected_celestial_location.x += pi / 6;
    corrected_celestial_location.x =
        (corrected_celestial_location.x >= 0 ? corrected_celestial_location.x - std::floor(corrected_celestial_location.x / (pi + pi / 3)) * (pi + pi / 3) :
        corrected_celestial_location.x + std::ceil(-corrected_celestial_location.x / (pi + pi / 3)) * (pi + pi / 3)) - pi / 6;

    corrected_celestial_location.y =
        corrected_celestial_location.y >= 0 ? corrected_celestial_location.y - std::floor(corrected_celestial_location.y / (2 * pi)) * 2 * pi :
        corrected_celestial_location.y + std::ceil(-corrected_celestial_location.y / (2 * pi)) * 2 * pi;

    vec3 direction_vector;
    direction_vector.x = std::cos(corrected_celestial_location.x) * std::cos(corrected_celestial_location.y);
    direction_vector.y = std::sin(corrected_celestial_location.x);
    direction_vector.z = -std::cos(corrected_celestial_location.x) * std::sin(corrected_celestial_location.y);

    return direction_vector;
}

vec2 Skydome::direction_to_celestial(const vec3& direction_vector) const
{
    vec2 celestial_location;
    vec2 horizontal_direction = vec2{ direction_vector.x, -direction_vector.z }.get_normalized();

    celestial_location.x = direction_vector.x >= 0 ? std::asin(direction_vector.y) : pi - std::asin(direction_vector.y);
    celestial_location.y = horizontal_direction.y >= 0 ? std::acos(horizontal_direction.x) : 2 * pi - std::acos(horizontal_direction.x);

    return celestial_location;
}

vec3 Skydome::celestial_to_cartesian(const vec2& celestial_location) const
{
    //Convert horizontal celestial coordinates used by default to equatorial celestial coordinates
    vec2 equatorial_location;
    equatorial_location.x = (pi - 2 * atmospheric_scattering_constants::horizon_angle) / pi * celestial_location.x + atmospheric_scattering_constants::horizon_angle;
    equatorial_location.y = celestial_location.y;

    //Convert equatorial celestial coordinates to default Cartesian coordinates
    vec3 cartesian_location;
    cartesian_location.y = atmospheric_scattering_constants::sky_sphere_radius * std::sin(equatorial_location.x) - atmospheric_scattering_constants::planet_radius;
    cartesian_location.x = atmospheric_scattering_constants::sky_sphere_radius * std::cos(equatorial_location.x) * std::cos(equatorial_location.y);
    cartesian_location.z = -atmospheric_scattering_constants::sky_sphere_radius * std::cos(equatorial_location.x) * std::sin(equatorial_location.y);

    return cartesian_location;
}

Skydome::Skydome(float daytime /* = 0.0f */, uint32_t num_night_sky_stars /* = 1000.0f */,
    uint32_t latitude_steps /* = 128 */, uint32_t longitude_steps /* = 128 */) :
    AbstractRenderableObject("Skydome"),
    radius{ 10000.0f }, num_stars{ num_night_sky_stars }, latitude_steps{ latitude_steps }, longitude_steps{ longitude_steps },
    sun_intensity{ 70.0f }, moon_intensity{ 0.5f }, moon_nominal_size{ 256.0f },
    num_vertices{ (2 * latitude_steps - 1) * longitude_steps + 2 + num_night_sky_stars + 2 },
    vertex_buffer_capacity{ ((2 * latitude_steps - 1) * longitude_steps + 2)*(vertex_attribute_position::getCapacity() + vertex_attribute_texcoord::getCapacity()) +
    (num_night_sky_stars + 2) * (vertex_attribute_position::getCapacity() + vertex_attribute_sky_body_luminosity::getCapacity()) },
    scintillation_parameter{ 0.75f, 1.0f }
{
    init_skydome();
    setDaytime(daytime);
    setMoonLocation(vec2{ pi / 6, pi / 12 });
}

Skydome::Skydome(std::string skydome_string_name, float radius,
    float daytime /* = 0.0f */, uint32_t num_night_sky_stars /* = 1000.0f */,
    uint32_t latitude_steps /* = 128 */, uint32_t longitude_steps /* = 128 */) :
    AbstractRenderableObject("Skydome", skydome_string_name),
    radius{ radius }, num_stars{ num_night_sky_stars },
    latitude_steps{ latitude_steps }, longitude_steps{ longitude_steps },
    sun_intensity{ 70.0f }, moon_intensity{ 0.5f }, moon_nominal_size{ 256.0f },
    num_vertices{ (2 * latitude_steps - 1) * longitude_steps + 2 + num_night_sky_stars + 2 },
    vertex_buffer_capacity{ ((2 * latitude_steps - 1) * longitude_steps + 2)*(vertex_attribute_position::getCapacity() + vertex_attribute_texcoord::getCapacity()) +
    (num_night_sky_stars + 2) * (vertex_attribute_position::getCapacity() + vertex_attribute_sky_body_luminosity::getCapacity()) },
    scintillation_parameter{ 0.75f, 1.0f }
{
    init_skydome();
    setDaytime(daytime);
    setMoonLocation(vec2{ pi / 6, pi / 12 });
}

Skydome::Skydome(const Skydome& other) :
AbstractRenderableObject(other),
AbstractRenderableObjectTextured(other),
ExtensionAggregator(other),
ref_counter{ other.ref_counter },
radius{ other.radius }, daytime{ other.daytime }, num_stars{ other.num_stars },
latitude_steps{ other.latitude_steps }, longitude_steps{ other.longitude_steps },
sun_direction{ other.sun_direction },
sun_intensity{ other.sun_intensity },
moon_direction{ other.moon_direction },
moon_intensity{ other.moon_intensity },
moon_nominal_size{ other.moon_nominal_size },
scattering_settings{ other.scattering_settings },
skydome_rendering_program_ref_code{ other.skydome_rendering_program_ref_code },
star_field_rendering_program_ref_code{ other.star_field_rendering_program_ref_code },
moon_rendering_program_ref_code{ other.moon_rendering_program_ref_code },
star_ref_code{ other.star_ref_code },
sun_ref_code{ other.sun_ref_code },
moon_ref_code{ other.moon_ref_code },
clamping_sampler_ref_code{ other.clamping_sampler_ref_code },
in_scattering_sun{ other.in_scattering_sun },
in_scattering_moon{ other.in_scattering_moon },
num_vertices{ other.num_vertices },
vertex_buffer_capacity{ other.vertex_buffer_capacity },
gl_context_state(other.gl_context_state),
scintillation_parameter{ 0.75f, 1.0f }
{
    //Initialize random generator
    init_random_generator();

    //Copy OpenGL objects
    ogl_vertex_attribute_objects[VA_SKYDOME] = other.ogl_vertex_attribute_objects[VA_SKYDOME];
    ogl_vertex_attribute_objects[VA_STARS] = other.ogl_vertex_attribute_objects[VA_STARS];

    ogl_array_buffer_objects[AB_INDEX] = other.ogl_array_buffer_objects[AB_INDEX];
    ogl_array_buffer_objects[AB_VERTEX] = other.ogl_array_buffer_objects[AB_VERTEX];

    //Increment reference counter
    ++(*ref_counter);
}

Skydome::~Skydome()
{
    //Decrement reference counter
    --(*ref_counter);

    //If number of references is 0, clear OpenGL and heap resources associated with the object
    if (!(*ref_counter))
    {
        free_ogl_resources();		//free OpengGL resources
        delete ref_counter;
    }
}

Skydome& Skydome::operator=(const Skydome& other)
{
    //Handle the special case of "assignment to itself"
    if (this == &other)
        return *this;

    //Increment reference counter of "other" object
    (*other.ref_counter)++;

    //Decrement reference counter of "this" object
    (*ref_counter)--;

    //If number of references to "this" object has reached 0, clear the resources associated with it
    if (!(*ref_counter))
    {
        free_ogl_resources();
        delete ref_counter;
    }

    //Copy object state
    //AbstractRenderableObject::operator=(other);
    AbstractRenderableObjectTextured::operator=(other);
    ExtensionAggregator::operator=(other);

    ref_counter = other.ref_counter;
    radius = other.radius;
    daytime = other.daytime;
    num_stars = other.num_stars;
    latitude_steps = other.latitude_steps;
    longitude_steps = other.longitude_steps;
    sun_direction = other.sun_direction;
    sun_intensity = other.sun_intensity;
    moon_direction = other.moon_direction;
    moon_intensity = other.moon_intensity;
    moon_nominal_size = other.moon_nominal_size;
    scattering_settings = other.scattering_settings;
    skydome_rendering_program_ref_code = other.skydome_rendering_program_ref_code;
    star_field_rendering_program_ref_code = other.star_field_rendering_program_ref_code;
    moon_rendering_program_ref_code = other.moon_rendering_program_ref_code;

    star_ref_code = other.star_ref_code;
    sun_ref_code = other.sun_ref_code;
    moon_ref_code = other.moon_ref_code;
    clamping_sampler_ref_code = other.clamping_sampler_ref_code;

    in_scattering_sun = other.in_scattering_sun;
    in_scattering_moon = other.in_scattering_moon;


    //*************************************Copy OpenGL objects****************************************
    ogl_vertex_attribute_objects[VA_SKYDOME] = other.ogl_vertex_attribute_objects[VA_SKYDOME];
    ogl_vertex_attribute_objects[VA_STARS] = other.ogl_vertex_attribute_objects[VA_STARS];
    ogl_vertex_attribute_objects[VA_MOON] = other.ogl_vertex_attribute_objects[VA_MOON];
    ogl_vertex_attribute_objects[VA_SUN] = other.ogl_vertex_attribute_objects[VA_SUN];

    ogl_array_buffer_objects[AB_INDEX] = other.ogl_array_buffer_objects[AB_INDEX];
    ogl_array_buffer_objects[AB_VERTEX] = other.ogl_array_buffer_objects[AB_VERTEX];
    //************************************************************************************************

    setDaytime(other.daytime);
    setMoonLocation(direction_to_celestial(other.moon_direction));

    current_rendering_pass = other.current_rendering_pass;
    gl_context_state = other.gl_context_state;

    return *this;
}



void Skydome::setRadius(float new_radius){ radius = new_radius; }

float Skydome::getRadius() const { return radius; }



void Skydome::setDaytime(float new_daytime)
{
    daytime = new_daytime >= 0 ? new_daytime - std::floor(new_daytime) : std::ceil(-new_daytime) + new_daytime;

    if (isDay())
        setSunLocation(vec2{ -pi / 6 + 2 * daytime * (pi + pi / 3), pi / 3 });
    else
        setSunLocation(vec2{ pi + pi / 6, pi / 3 });
}

float Skydome::getDaytime() const { return daytime; }

bool Skydome::isDay() const { return daytime < 0.5f; }

bool Skydome::isNight() const { return daytime >= 0.5f; }



void Skydome::applySunTexture(const ImmutableTexture2D& new_sun_texture)
{
    if (!sun_ref_code)
        sun_ref_code = registerTexture(new_sun_texture, clamping_sampler_ref_code);
    else
        updateTexture(sun_ref_code, new_sun_texture, clamping_sampler_ref_code);
}

void Skydome::applyMoonTexture(const ImmutableTexture2D& new_moon_texture)
{
    if (!moon_ref_code)
        moon_ref_code = registerTexture(new_moon_texture, clamping_sampler_ref_code);
    else
        updateTexture(moon_ref_code, new_moon_texture, clamping_sampler_ref_code);
}




void Skydome::setAtmosphereScatteringSettings(const LightScatteringSettings& scattering_settings)
{
    this->scattering_settings = scattering_settings;
}

LightScatteringSettings Skydome::getAtmosphericScatteringSettings() const { return scattering_settings; }




void Skydome::setSunLightIntensity(const vec3& intensity){ sun_intensity = intensity; }

vec3 Skydome::getSunLightIntensity() const { return sun_intensity; }

void Skydome::setSunLocation(const vec2& location){ sun_direction = celestial_to_direction(location); }

vec2 Skydome::getSunLocation() const { return direction_to_celestial(sun_direction); }

vec3 Skydome::getSunDirection() const
{
    return sun_direction;
}


void Skydome::setMoonLightIntensity(const vec3& intensity){ moon_intensity = intensity; }

vec3 Skydome::getMoonLightIntensity() const { return moon_intensity; }

void Skydome::setMoonLocation(const vec2& location)
{
    moon_direction = celestial_to_direction(location).get_normalized();

    vec3 v3MoonLocation = moon_direction.get_normalized();
    glBindBuffer(GL_ARRAY_BUFFER, ogl_array_buffer_objects[AB_VERTEX]);

    void *p_vertex_buffer = glMapBufferRange(GL_ARRAY_BUFFER,
        ((2 * latitude_steps - 1) * longitude_steps + 2) * (vertex_attribute_position::getCapacity() + vertex_attribute_texcoord::getCapacity()) +
        num_stars * (vertex_attribute_position::getCapacity() + vertex_attribute_sky_body_luminosity::getCapacity()),
        vertex_attribute_position::getCapacity() + vertex_attribute_sky_body_luminosity::getCapacity(),
        GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT);

    vertex_attribute_position::value_type* moon_location = static_cast<vertex_attribute_position::value_type*>(p_vertex_buffer);
    vertex_attribute_sky_body_luminosity::value_type* moon_color_modulation =
        reinterpret_cast<vertex_attribute_sky_body_luminosity::value_type*>(reinterpret_cast<char*>(moon_location)+vertex_attribute_position::getCapacity());

    moon_location[0] = v3MoonLocation.x;
    moon_location[1] = v3MoonLocation.y;
    moon_location[2] = v3MoonLocation.z;
    moon_location[3] = 1.0f;

    moon_color_modulation[0] = 1.0f;
    moon_color_modulation[1] = 1.0f;
    moon_color_modulation[2] = 1.0f;
    moon_color_modulation[3] = moon_nominal_size;		//size of moon

    glUnmapBuffer(GL_ARRAY_BUFFER);
}

vec2 Skydome::getMoonLocation() const { return direction_to_celestial(moon_direction); }

vec3 Skydome::getMoonDirection() const
{
    return moon_direction;
}

void Skydome::setNominalMoonSize(float size)
{
    moon_nominal_size = size;

    glBindBuffer(GL_ARRAY_BUFFER, ogl_array_buffer_objects[AB_VERTEX]);
    vertex_attribute_sky_body_luminosity::value_type* moon_size =
        static_cast<vertex_attribute_sky_body_luminosity::value_type*>(glMapBufferRange(GL_ARRAY_BUFFER,
        ((2 * latitude_steps - 1) * longitude_steps + 2) * (vertex_attribute_position::getCapacity() + vertex_attribute_texcoord::getCapacity()) +
        num_stars * (vertex_attribute_position::getCapacity() + vertex_attribute_sky_body_luminosity::getCapacity()) +
        vertex_attribute_position::getCapacity() + sizeof(vertex_attribute_sky_body_luminosity::value_type) * 3,
        sizeof(vertex_attribute_sky_body_luminosity::value_type), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT));
    *moon_size = moon_nominal_size;
    glUnmapBuffer(GL_ARRAY_BUFFER);
}

float Skydome::getNominalMoonSize() const { return moon_nominal_size; }




bool Skydome::supportsRenderingMode(uint32_t rendering_mode) const
{
    switch (rendering_mode)
    {
    case TW_RENDERING_MODE_DEFAULT:
    case TW_RENDERING_MODE_SILHOUETTE:
        return true;
    default:
        return false;
        break;
    }
}

void Skydome::configureViewProjectionTransform(const AbstractProjectingDevice& projecting_device)
{
    //Compute view transform with no location offset
    mat4 CenteredView = projecting_device.getViewTransform();
    CenteredView[0][3] = 0;
    CenteredView[1][3] = 0;
    CenteredView[2][3] = 0;

    //Modify projection transform, so that the far clipping plane is located infinitely far from the camera
    mat4 ProjectionTransform = projecting_device.getProjectionTransform();
    float near = projecting_device.getNearClipPlane();
    ProjectionTransform[2][2] = -1.0f;
    ProjectionTransform[2][3] = -2.0f * near;

    mat4  ModelViewProjection = ProjectionTransform * CenteredView;
    retrieveShaderProgram(skydome_rendering_program_ref_code)->assignUniformMatrix("m4MVP", ModelViewProjection);
    retrieveShaderProgram(star_field_rendering_program_ref_code)->assignUniformMatrix("m4MVP", ModelViewProjection);
    retrieveShaderProgram(moon_rendering_program_ref_code)->assignUniformMatrix("m4MVP", ModelViewProjection);

    //Set location of the viewer
    vec3 CameraLocation = projecting_device.getLocation() / (scattering_settings.scattering_to_far_world_ratio*projecting_device.getFarClipPlane()) +
        vec3{ 0.0f, atmospheric_scattering_constants::planet_radius, 0.0f };

    if (CameraLocation.norm() > atmospheric_scattering_constants::sky_sphere_radius)
        CameraLocation = CameraLocation.get_normalized()*atmospheric_scattering_constants::sky_sphere_radius;

    retrieveShaderProgram(skydome_rendering_program_ref_code)->assignUniformVector("v3CameraPosition", CameraLocation);
    retrieveShaderProgram(skydome_rendering_program_ref_code)->assignUniformScalar("fFarClipPlane", projecting_device.getFarClipPlane());
}

uint32_t Skydome::getNumberOfRenderingPasses(uint32_t rendering_mode) const
{
    //Skydome is rendered in three passes:
    //1st pass draws atmosphere with dynamic light scattering
    //2nd pass blends in the stars
    //3rd pass blends in the moon or sun depending on the day time setting
    return 3;
}

bool Skydome::configureRendering(AbstractRenderingDevice& render_target, uint32_t rendering_pass)
{
    //If requested render target is not yet active, activate it
    if (!render_target.isActive())
        render_target.makeActive();

    current_rendering_pass = rendering_pass;

    //If writes to the depth buffer were enabled before start of the rendering, disable them
    glGetBooleanv(GL_DEPTH_WRITEMASK, &gl_context_state.depth_mask_state);
    if (gl_context_state.depth_mask_state) glDepthMask(GL_FALSE);


    switch (rendering_pass)
    {
        //Sky dome rendering
    case 0:
        glBindVertexArray(ogl_vertex_attribute_objects[VA_SKYDOME]);


        gl_context_state.primitive_restart_enable_state = glIsEnabled(GL_PRIMITIVE_RESTART);
        if (!gl_context_state.primitive_restart_enable_state) glEnable(GL_PRIMITIVE_RESTART);
        glGetIntegerv(GL_PRIMITIVE_RESTART_INDEX, &gl_context_state.primitive_restart_index);
        glPrimitiveRestartIndex(0xFFFF);


        //Apply uniform variable values to the sky dome rendering shader program
        retrieveShaderProgram(skydome_rendering_program_ref_code)->assignUniformScalar("fSkydomeRadius", radius);

        retrieveShaderProgram(skydome_rendering_program_ref_code)->assignUniformVector("v3SunLightDirection", sun_direction);
        retrieveShaderProgram(skydome_rendering_program_ref_code)->assignUniformVector("v3SunLightIntensity", sun_intensity);
        retrieveShaderProgram(skydome_rendering_program_ref_code)->assignUniformVector("v3MoonLightDirection", moon_direction);
        retrieveShaderProgram(skydome_rendering_program_ref_code)->assignUniformVector("v3MoonLightIntensity", moon_intensity);

        retrieveShaderProgram(skydome_rendering_program_ref_code)->assignUniformVector("v3RayleighCoefficient",
            vec3{ 1.0f / std::pow(scattering_settings.red_wavelength, 4.0f),
            1.0f / std::pow(scattering_settings.green_wavelength, 4.0f),
            1.0f / std::pow(scattering_settings.blue_wavelength, 4.0f) } *scattering_settings.rayleigh_scale);

        retrieveShaderProgram(skydome_rendering_program_ref_code)->assignUniformScalar("fMieCoefficient", scattering_settings.mie_coefficient);
        retrieveShaderProgram(skydome_rendering_program_ref_code)->assignUniformScalar("fMiePhaseFunctionParameter", scattering_settings.mie_phase_function_param);


        COMPLETE_SHADER_PROGRAM_CAST(retrieveShaderProgram(skydome_rendering_program_ref_code)).activate();

        return true;

    case 1:
        glBindVertexArray(ogl_vertex_attribute_objects[VA_STARS]);


        gl_context_state.blend_enable_state = glIsEnabled(GL_BLEND);
        if (!gl_context_state.blend_enable_state) glEnable(GL_BLEND);
        glGetIntegerv(GL_BLEND_SRC_RGB, &gl_context_state.src_rgb);
        glGetIntegerv(GL_BLEND_SRC_ALPHA, &gl_context_state.src_alpha);
        glGetIntegerv(GL_BLEND_DST_RGB, &gl_context_state.dst_rgb);
        glGetIntegerv(GL_BLEND_DST_ALPHA, &gl_context_state.dst_alpha);
        glGetIntegerv(GL_BLEND_EQUATION_RGB, &gl_context_state.blend_eq_rgb);
        glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &gl_context_state.blend_eq_alpha);
        glBlendFuncSeparate(GL_ONE_MINUS_DST_ALPHA, GL_DST_ALPHA, GL_ZERO, GL_ONE);
        glBlendEquation(GL_FUNC_ADD);

        gl_context_state.program_point_size_enable_state = glIsEnabled(GL_PROGRAM_POINT_SIZE);
        if (!gl_context_state.program_point_size_enable_state) glEnable(GL_PROGRAM_POINT_SIZE);


        retrieveShaderProgram(star_field_rendering_program_ref_code)->assignUniformScalar("fSkydomeRadius", radius);
        retrieveShaderProgram(star_field_rendering_program_ref_code)->assignUniformScalar("fScintillationParameter", scintillation_parameter(default_random_generator));
        retrieveShaderProgram(star_field_rendering_program_ref_code)->assignUniformScalar("s2dStarTexture", getBindingUnit(star_ref_code));

        if (star_ref_code) bindTexture(star_ref_code);


        COMPLETE_SHADER_PROGRAM_CAST(retrieveShaderProgram(star_field_rendering_program_ref_code)).activate();

        return true;

    case 2:
        glBindVertexArray(ogl_vertex_attribute_objects[VA_MOON]);


        gl_context_state.blend_enable_state = glIsEnabled(GL_BLEND);
        if(!gl_context_state.blend_enable_state) glEnable(GL_BLEND);
        glGetIntegerv(GL_BLEND_SRC_RGB, &gl_context_state.src_rgb);
        glGetIntegerv(GL_BLEND_SRC_ALPHA, &gl_context_state.src_alpha);
        glGetIntegerv(GL_BLEND_DST_RGB, &gl_context_state.dst_rgb);
        glGetIntegerv(GL_BLEND_DST_ALPHA, &gl_context_state.dst_alpha);
        glGetIntegerv(GL_BLEND_EQUATION_RGB, &gl_context_state.blend_eq_rgb);
        glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &gl_context_state.blend_eq_alpha);
        glBlendFunc(GL_ONE_MINUS_DST_ALPHA, GL_DST_ALPHA);
        glBlendEquation(GL_FUNC_ADD);

        gl_context_state.program_point_size_enable_state = glIsEnabled(GL_PROGRAM_POINT_SIZE);
        if (!gl_context_state.program_point_size_enable_state) glEnable(GL_PROGRAM_POINT_SIZE);


        retrieveShaderProgram(moon_rendering_program_ref_code)->assignUniformScalar("fSkydomeRadius", radius);
        retrieveShaderProgram(moon_rendering_program_ref_code)->assignUniformScalar("s2dMoonTextureSampler", getBindingUnit(moon_ref_code));

        if (moon_ref_code) bindTexture(moon_ref_code);


        COMPLETE_SHADER_PROGRAM_CAST(retrieveShaderProgram(moon_rendering_program_ref_code)).activate();

        return true;

    default:
        return false;
    }
}

bool Skydome::render()
{
    switch (current_rendering_pass)
    {
    case 0:
    {
        ImageUnit in_scattering_sun_img, in_scattering_moon_img;
        in_scattering_sun_img.attachTexture(in_scattering_sun, 0, BufferAccess::WRITE, InternalPixelFormat::SIZED_FLOAT_RGBA32);
        in_scattering_moon_img.attachTexture(in_scattering_moon, 0, BufferAccess::WRITE, InternalPixelFormat::SIZED_FLOAT_RGBA32);
        retrieveShaderProgram(skydome_rendering_program_ref_code)->assignUniformScalar("in_scattering_sun", in_scattering_sun_img.getBinding());
        retrieveShaderProgram(skydome_rendering_program_ref_code)->assignUniformScalar("in_scattering_moon", in_scattering_moon_img.getBinding());

        const uint32_t base_sphere_num_indexes = 4 * (longitude_steps + 1) * (latitude_steps - 1) + 2 * (latitude_steps - 1) - 1;
        glDrawElements(GL_TRIANGLE_STRIP, base_sphere_num_indexes,
            GL_UNSIGNED_SHORT, reinterpret_cast<const GLvoid*>(0));
        glDrawElements(GL_TRIANGLE_FAN, 2 * (2 + longitude_steps) + 1,
            GL_UNSIGNED_SHORT, reinterpret_cast<const GLvoid*>(base_sphere_num_indexes * sizeof(GLushort)));
        return true;
    }

    case 1:
        glDrawArrays(GL_POINTS, 0, num_stars);
        return true;


    case 2:
        glDrawArrays(GL_POINTS, 0, 1);
        return true;

    default:
        return false;
    }

}

bool Skydome::configureRenderingFinalization()
{
    //If writes to the depth buffer were enabled before beginning of rendering, switch them on again
    if (gl_context_state.depth_mask_state) glDepthMask(GL_TRUE);

    switch (current_rendering_pass)
    {
    case 0:
        if (!gl_context_state.primitive_restart_enable_state) glDisable(GL_PRIMITIVE_RESTART);
        glPrimitiveRestartIndex(gl_context_state.primitive_restart_index);

        break;


    case 1:
    case 2:
        if (!gl_context_state.blend_enable_state) glDisable(GL_BLEND);
        glBlendFuncSeparate(gl_context_state.src_rgb, gl_context_state.dst_rgb, gl_context_state.src_alpha, gl_context_state.dst_alpha);
        glBlendEquationSeparate(gl_context_state.blend_eq_rgb, gl_context_state.blend_eq_alpha);

        if (!gl_context_state.program_point_size_enable_state) glDisable(GL_PROGRAM_POINT_SIZE);

        break;
    }

    return false;
}