#include "CylindricalSurface.h"

#include <complex>
#include <fstream>


#define pi 3.1415926535897932384626433832795f

#define AB_VERTEX 0		//offset to vertex array buffer
#define AB_INDEX  1		//offset to index array buffer


using namespace tiny_world;



typedef AbstractRenderableObjectExtensionAggregator<AbstractRenderableObjectLightEx, AbstractRenderableObjectHDRBloomEx> ExtensionAggregator;

const std::string CylindricalSurface::rendering_program0_name = "CylindricalSurface::rendering_program0";


void CylindricalSurface::applyScreenSize(const uvec2& screen_size)
{

}


void CylindricalSurface::init_cylindrical_surface()
{
    //Create OpenGL buffers
    glGenBuffers(2, ogl_buffers);

    //Create and setup vertex array object
    glGenVertexArrays(1, &ogl_vertex_array_object);
    glBindVertexArray(ogl_vertex_array_object);

    glEnableVertexAttribArray(vertex_attribute_position::getId());
    vertex_attribute_position::setVertexAttributeBufferLayout(0, 0);

    glEnableVertexAttribArray(vertex_attribute_texcoord::getId());
    vertex_attribute_texcoord::setVertexAttributeBufferLayout(vertex_attribute_position::getCapacity(), 0);



    //Create vertex buffer objects and populate them with data
    const uint32_t num_surface_vertices = 6 * (num_angular_base_nodes + 1) * num_length_base_nodes + 6 * 2 * (num_angular_base_nodes + 2);
    const uint32_t side_surface_vertices_offset = 6 * (num_angular_base_nodes + 1) * num_length_base_nodes;
    GLfloat* surface_vertices = new GLfloat[num_surface_vertices];

    for (unsigned int j = 0; j < num_length_base_nodes; ++j)
    {
        for (unsigned int i = 0; i <= num_angular_base_nodes; ++i)
        {
            //Vertex position
            surface_vertices[6 * ((num_angular_base_nodes + 1)*j + i) + 0] = i / (num_angular_base_nodes - 1.0f);
            surface_vertices[6 * ((num_angular_base_nodes + 1)*j + i) + 1] = j / (num_length_base_nodes - 1.0f);
            surface_vertices[6 * ((num_angular_base_nodes + 1)*j + i) + 2] = -0.5f + 1.0f / (num_length_base_nodes - 1)*j;
            surface_vertices[6 * ((num_angular_base_nodes + 1)*j + i) + 3] = 0.0f;

            //Vertex texture coordinates
            surface_vertices[6 * ((num_angular_base_nodes + 1)*j + i) + 4] = i / ((num_angular_base_nodes - 1) * texture_u_scale);
            surface_vertices[6 * ((num_angular_base_nodes + 1)*j + i) + 5] = j / ((num_length_base_nodes - 1) * texture_v_scale);
        }
    }


    //Position of the near face center (determined dynamically based on shape of concrete surface)
    //surface_vertices[side_surface_vertices_offset + 0] = 0;
    //surface_vertices[side_surface_vertices_offset + 1] = 0;
    //surface_vertices[side_surface_vertices_offset + 2] = 0.5f;
    //surface_vertices[side_surface_vertices_offset + 3] = 1.0f;

    //Texture coordinates of the near face center (determined dynamically based on shape of concrete surface)
    //surface_vertices[side_surface_vertices_offset + 4] = 0.5f;
    //surface_vertices[side_surface_vertices_offset + 5] = 0.5f;


    //Position of the far face center (determined dynamically based on shape of concrete surface)
    //surface_vertices[side_surface_vertices_offset + 6 * (num_angular_base_nodes + 1) + 0] = 0;
    //surface_vertices[side_surface_vertices_offset + 6 * (num_angular_base_nodes + 1) + 1] = 0;
    //surface_vertices[side_surface_vertices_offset + 6 * (num_angular_base_nodes + 1) + 2] = -0.5f;
    //surface_vertices[side_surface_vertices_offset + 6 * (num_angular_base_nodes + 1) + 3] = 1.0f;

    //Texture coordinates of the far face center (determined dynamically based on shape of concrete surface)
    //surface_vertices[side_surface_vertices_offset + 6 * (num_angular_base_nodes + 1) + 4] = 0.5f;
    //surface_vertices[side_surface_vertices_offset + 6 * (num_angular_base_nodes + 1) + 5] = 0.5f;


    for (unsigned int i = 1; i <= num_angular_base_nodes + 1; ++i)
    {
        //Near face vertex position
        surface_vertices[side_surface_vertices_offset + 6 * i + 0] = (i - 1) / (num_angular_base_nodes - 1.0f);
        surface_vertices[side_surface_vertices_offset + 6 * i + 1] = 1.0f;
        surface_vertices[side_surface_vertices_offset + 6 * i + 2] = 0.5f;
        surface_vertices[side_surface_vertices_offset + 6 * i + 3] = 0.0f;

        //Near face vertex texture coordinates (determined dynamically based on shape of concrete surface)
        //surface_vertices[side_surface_vertices_offset + 6 * i + 4] = 0.5f*std::cos(2 * pi * (i - 1) / num_angular_base_nodes) + 0.5f;
        //surface_vertices[side_surface_vertices_offset + 6 * i + 5] = 0.5f*std::sin(2 * pi * (i - 1) / num_angular_base_nodes) + 0.5f;


        //Far face vertex position (determined dynamically based on shape of concrete surface)
        surface_vertices[side_surface_vertices_offset + 6 * (num_angular_base_nodes + 2 + i) + 0] = (i - 1) / (num_angular_base_nodes - 1.0f);
        surface_vertices[side_surface_vertices_offset + 6 * (num_angular_base_nodes + 2 + i) + 1] = 0.0f;
        surface_vertices[side_surface_vertices_offset + 6 * (num_angular_base_nodes + 2 + i) + 2] = -0.5f;
        surface_vertices[side_surface_vertices_offset + 6 * (num_angular_base_nodes + 2 + i) + 3] = 0.0f;

        //Far face vertex texture coordinates (determined dynamically based on shape of concrete surface)
        //surface_vertices[side_surface_vertices_offset + 6 * (num_angular_base_nodes + 1 + i) + 4] = 0.5f*std::cos(2 * pi*(i - 1) / num_angular_base_nodes) + 0.5f;
        //surface_vertices[side_surface_vertices_offset + 6 * (num_angular_base_nodes + 1 + i) + 5] = 0.5f*std::sin(2 * pi*(i - 1) / num_angular_base_nodes) + 0.5f;
    }

    glBindBuffer(GL_ARRAY_BUFFER, ogl_buffers[AB_VERTEX]);
    glBufferData(GL_ARRAY_BUFFER, num_surface_vertices*sizeof(GLfloat), surface_vertices, GL_STATIC_DRAW);
    delete[] surface_vertices;
    glBindVertexBuffer(0, ogl_buffers[AB_VERTEX], 0, vertex_attribute_position::getCapacity() + vertex_attribute_texcoord::getCapacity());


    //Generate index data
    const uint32_t num_surface_indexes = 2 * (num_angular_base_nodes + 1)*(num_length_base_nodes - 1) + 2 * (num_angular_base_nodes + 2) + 1;
    GLuint* surface_indexes = new GLuint[num_surface_indexes];
    const uint32_t side_surface_indexes_offset = 2 * (num_angular_base_nodes + 1)*(num_length_base_nodes - 1);

    //Index data for the side surface
    for (unsigned int j = 0; j < num_length_base_nodes - 1; ++j)
    {
        surface_indexes[2 * j* (num_angular_base_nodes + 1) + 0] = (j + 1)*(num_angular_base_nodes + 1) + 0;
        surface_indexes[2 * j* (num_angular_base_nodes + 1) + 1] = j*(num_angular_base_nodes + 1) + 0;
        for (unsigned int i = 0; i < num_angular_base_nodes; ++i)
        {
            surface_indexes[2 * j* (num_angular_base_nodes + 1) + 2 * (i + 1) + 0] = (j + 1)*(num_angular_base_nodes + 1) + (i + 1);
            surface_indexes[2 * j* (num_angular_base_nodes + 1) + 2 * (i + 1) + 1] = j*(num_angular_base_nodes + 1) + (i + 1);
        }
    }

    //Index data for faces
    surface_indexes[side_surface_indexes_offset] = (num_angular_base_nodes + 1) * num_length_base_nodes;
    surface_indexes[side_surface_indexes_offset + num_angular_base_nodes + 2] = 0xFFFFFFFF;	//primitive restart index
    surface_indexes[side_surface_indexes_offset + num_angular_base_nodes + 3] = (num_angular_base_nodes + 1) * num_length_base_nodes + num_angular_base_nodes + 2;
    for (unsigned int i = 1; i <= num_angular_base_nodes + 1; ++i)
    {
        surface_indexes[side_surface_indexes_offset + i] =
            (num_angular_base_nodes + 1) * num_length_base_nodes + i;

        surface_indexes[side_surface_indexes_offset + num_angular_base_nodes + 3 + i] =
            (num_angular_base_nodes + 1) * num_length_base_nodes + 2 * num_angular_base_nodes + 4 - i;
    }

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ogl_buffers[AB_INDEX]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*num_surface_indexes, surface_indexes, GL_STATIC_DRAW);
    delete[] surface_indexes;


    //Setup sampler objects
    surface_map_sampler_ref_code = createTextureSampler("CylindricalSurface::surface_map_sampler", SamplerMagnificationFilter::LINEAR, SamplerMinificationFilter::LINEAR_MIPMAP_LINEAR,
        SamplerWrapping{ SamplerWrappingMode::REPEAT, SamplerWrappingMode::CLAMP_TO_EDGE, SamplerWrappingMode::CLAMP_TO_EDGE });


    //Setup shader programs
    if (!rendering_program0_ref_code)
    {
        std::string base_catalog = ShaderProgram::getShaderBaseCatalog();
        Shader vertex_shader{ base_catalog + "CylindricalSurface.vp.glsl", ShaderType::VERTEX_SHADER, "CylindricalSurface::rendering_program0::vertex_shader" };
        if (!vertex_shader)
        {
            set_error_state(true);
            std::string err_msg = std::string("Unable to compile vertex shader: ") + vertex_shader.getErrorString();
            set_error_string(err_msg.c_str());
            call_error_callback(err_msg.c_str());
            return;
        }

        Shader geometry_shader{ base_catalog + "CylindricalSurface.gp.glsl", ShaderType::GEOMETRY_SHADER, "CylindricalSurface::rendering_program0::geometry_shader" };
        if (!geometry_shader)
        {
            set_error_state(true);
            std::string err_msg = std::string("Unable to compile vertex shader: ") + geometry_shader.getErrorString();
            set_error_string(err_msg.c_str());
            call_error_callback(err_msg.c_str());
            return;
        }

        Shader fragment_shader{ base_catalog + "CylindricalSurface.fp.glsl", ShaderType::FRAGMENT_SHADER, "CylindricalSurface::rendering_program0::fragment_shader" };
        if (!fragment_shader)
        {
            set_error_state(true);
            std::string err_msg = std::string("Unable to compile vertex shader: ") + fragment_shader.getErrorString();
            set_error_string(err_msg.c_str());
            call_error_callback(err_msg.c_str());
            return;
        }

        rendering_program0_ref_code = createCompleteShaderProgram(rendering_program0_name,
        { PipelineStage::VERTEX_SHADER, PipelineStage::GEOMETRY_SHADER, PipelineStage::FRAGMENT_SHADER });
        retrieveShaderProgram(rendering_program0_ref_code)->addShader(vertex_shader);
        retrieveShaderProgram(rendering_program0_ref_code)->addShader(geometry_shader);
        retrieveShaderProgram(rendering_program0_ref_code)->addShader(fragment_shader);
        retrieveShaderProgram(rendering_program0_ref_code)->bindVertexAttributeId("v4Position", vertex_attribute_position::getId());
        retrieveShaderProgram(rendering_program0_ref_code)->bindVertexAttributeId("v2TexCoord", vertex_attribute_texcoord::getId());
        retrieveShaderProgram(rendering_program0_ref_code)->link();
        if (retrieveShaderProgram(rendering_program0_ref_code)->getErrorState())
        {
            set_error_state(true);
            std::string err_msg = std::string("Unable to link CylindrialSurface::rendering_program0: ") +
                retrieveShaderProgram(rendering_program0_ref_code)->getErrorString();
            set_error_string(err_msg.c_str());
            call_error_callback(err_msg.c_str());
            return;
        }
    }
}


void CylindricalSurface::process_surface_data(std::vector<vec2>& surface_map_data, uint32_t num_points_per_slice, uint32_t num_slices)
{
    //Check if surface map orientation should be changed
    vec3 v0{ surface_map_data[num_points_per_slice], -0.5f + 1.0f / (num_length_base_nodes - 1.0f) };
    vec3 v1{ surface_map_data[0], -0.5f };
    vec3 v2{ surface_map_data[num_points_per_slice + 1], -0.5f + 1.0f / (num_length_base_nodes - 1.0f) };
    float S = mat3{ v0, v1, v2 }.determinant();
    if (S < 0)
    {
        std::vector<vec2> reversed_surface_map_data;
        for (unsigned int j = 0; j < num_slices; ++j)
            for (unsigned int i = 0; i < num_points_per_slice; ++i)
                reversed_surface_map_data.push_back(surface_map_data[j*num_points_per_slice + num_points_per_slice - 1 - i]);
        surface_map_data = reversed_surface_map_data;
    }

    //Align surface's main axis with the world z-axis
    vec2 center{ 0.0f };	//vertically averaged mass center of each slice
    vec2 near_face_center{ 0.0f };	//mass center of the near-face of the surface
    vec2 far_face_center{ 0.0f };	//mass center of the far-face of the surface
    for (unsigned int j = 0; j < num_slices; ++j)
    {
        vec2 slice_center{ 0.0f };	//mass center of the current slice
        for (unsigned int i = 0; i < num_points_per_slice; ++i)
            slice_center = (slice_center * i + surface_map_data[j*num_points_per_slice + i]) / (i + 1);
        center = (center * j + slice_center) / (j + 1);

        if (j == 0) far_face_center = slice_center;
        if (j == num_slices - 1) near_face_center = slice_center;
    }

    //Find scaling factors of the texture coordinates applied to the surface's faces
    float near_face_scale = 0.0f;	//scaling factor of the near-face of the surface
    float far_face_scale = 0.0f;	//scaling factor of the far-face of the surface
    for (unsigned int i = 0; i < num_points_per_slice; ++i)
    {
        float current_vector_norm;

        current_vector_norm = (surface_map_data[(num_slices - 1)*num_points_per_slice + i] - near_face_center).norm();
        if (current_vector_norm > near_face_scale) near_face_scale = current_vector_norm;

        current_vector_norm = (surface_map_data[i] - far_face_center).norm();
        if (current_vector_norm > far_face_scale) far_face_scale = current_vector_norm;
    }
    near_face_scale *= 2;
    far_face_scale *= 2;


    //Define positions and texture coordinates for center points of the faces
    glBindBuffer(GL_ARRAY_BUFFER, ogl_buffers[AB_VERTEX]);
    GLfloat* face_vertex_data = static_cast<GLfloat*>(glMapBufferRange(GL_ARRAY_BUFFER, 6 * (num_angular_base_nodes + 1)*num_length_base_nodes*sizeof(GLfloat),
        6 * 2 * (num_angular_base_nodes + 2) * sizeof(GLfloat), GL_MAP_WRITE_BIT));

    //Position of center point of the near-face
    face_vertex_data[0] = near_face_center.x - center.x;
    face_vertex_data[1] = near_face_center.y - center.y;
    face_vertex_data[2] = 0.5f;
    face_vertex_data[3] = 1.0f;

    //Texture coordinates of center point of the near-face
    face_vertex_data[4] = 0.5f;
    face_vertex_data[5] = 0.5f;

    //Position of center point of the far-face
    face_vertex_data[6 * (num_angular_base_nodes + 2) + 0] = far_face_center.x - center.x;
    face_vertex_data[6 * (num_angular_base_nodes + 2) + 1] = far_face_center.y - center.y;
    face_vertex_data[6 * (num_angular_base_nodes + 2) + 2] = -0.5f;
    face_vertex_data[6 * (num_angular_base_nodes + 2) + 3] = 1.0f;

    //Texture coordinates of center point of the far-face
    face_vertex_data[6 * (num_angular_base_nodes + 2) + 4] = 0.5f;
    face_vertex_data[6 * (num_angular_base_nodes + 2) + 5] = 0.5f;


    //Define texture coordinates for edges of the faces
    for (unsigned int i = 0; i <= num_angular_base_nodes; ++i)
    {
        float ev_arg = i / (num_angular_base_nodes - 1.0f);
        ev_arg = (ev_arg - std::floor(ev_arg)) * 2 * pi;

        float nf_min_dist_left = 2 * pi, nf_min_dist_right = 2 * pi;
        float ff_min_dist_left = 2 * pi, ff_min_dist_right = 2 * pi;
        float nf_m1_arg, nf_p1_arg, ff_m1_arg, ff_p1_arg;
        unsigned int nf_m1, nf_p1, ff_m1, ff_p1;
        for (unsigned int j = 0; j < num_points_per_slice; ++j)
        {
            //Find interpolation nodes from within the near-face vertices
            vec2 v2CurrentDataPoint = (surface_map_data[(num_slices - 1)*num_points_per_slice + j] - near_face_center).get_normalized();
            float x = v2CurrentDataPoint.x;
            float y = v2CurrentDataPoint.y;

            float nfv_arg = static_cast<float>(y < 0)*pi + std::acos((y >= 0 ? 1 : -1) * x);
            float dist[3] = { nfv_arg - ev_arg, nfv_arg - 2 * pi - ev_arg, nfv_arg - ev_arg + 2 * pi };
            unsigned int mdist_idx =
                static_cast<unsigned int>(std::min_element(dist, dist + 3, [](float elem1, float elem2) -> bool {return std::abs(elem1) < std::abs(elem2); }) - &dist[0]);
            float mdist = std::abs(dist[mdist_idx]);

            //Equals 'true' if nfv_arg is greater then or equal to ev_arg in terms of "circular metrics"
            bool test = false;
            if (mdist_idx == 0 && dist[0] >= 0) test = true;
            if (mdist_idx == 1) nfv_arg -= 2 * pi;
            if (mdist_idx == 2 && dist[2] >= 0) test = true;

            if (!test && mdist < nf_min_dist_left)
            {
                nf_min_dist_left = mdist;
                nf_m1 = j;
                nf_m1_arg = nfv_arg;
            }

            if (test && mdist < nf_min_dist_right)
            {
                nf_min_dist_right = mdist;
                nf_p1 = j;
                nf_p1_arg = nfv_arg;
            }


            //Find interpolation nodes from within the far-face vertices
            v2CurrentDataPoint = (surface_map_data[j] - far_face_center).get_normalized();
            x = v2CurrentDataPoint.x;
            y = v2CurrentDataPoint.y;
            float ffv_arg = static_cast<float>(y < 0)*pi + std::acos((y >= 0 ? 1 : -1) * x);
            dist[0] = ffv_arg - ev_arg; dist[1] = ffv_arg - 2 * pi - ev_arg; dist[2] = ffv_arg - ev_arg + 2 * pi;
            mdist_idx = static_cast<unsigned int>(std::min_element(dist, dist + 3, [](float elem1, float elem2) -> bool {return std::abs(elem1) < std::abs(elem2); }) - &dist[0]);
            mdist = std::abs(dist[mdist_idx]);

            //Equals 'true' if nfv_arg is greater then or equal to ev_arg in terms of "circular metrics"
            test = false;
            if (mdist_idx == 0 && dist[0] >= 0) test = true;
            if (mdist_idx == 1) ffv_arg -= 2 * pi;
            if (mdist_idx == 2 && dist[2] >= 0) test = true;

            if (!test && mdist < ff_min_dist_left)
            {
                ff_min_dist_left = mdist;
                ff_m1 = j;
                ff_m1_arg = ffv_arg;
            }

            if (test && mdist < ff_min_dist_right)
            {
                ff_min_dist_right = mdist;
                ff_p1 = j;
                ff_p1_arg = ffv_arg;
            }
        }


        //Compute texture coordinates for the near-face
        vec2 v2InterpolatedVertex;
        if (nf_m1 != nf_p1)
        {
            float interpolation_coefficient;
            if (ev_arg < nf_m1_arg || ev_arg > nf_p1_arg)
                interpolation_coefficient = (ev_arg - 2*pi - nf_m1_arg) / (nf_p1_arg - nf_m1_arg);
            else
                interpolation_coefficient = (ev_arg - nf_m1_arg) / (nf_p1_arg - nf_m1_arg);

            v2InterpolatedVertex = surface_map_data[(num_slices - 1)*num_points_per_slice + nf_m1] - near_face_center +
                interpolation_coefficient *
                (surface_map_data[(num_slices - 1)*num_points_per_slice + nf_p1] - surface_map_data[(num_slices - 1)*num_points_per_slice + nf_m1]);
        }
        else
        {
            v2InterpolatedVertex = surface_map_data[(num_slices - 1)*num_points_per_slice + nf_m1] - near_face_center;
        }
        v2InterpolatedVertex /= near_face_scale;
        face_vertex_data[6 + 6 * i + 4] = v2InterpolatedVertex.x + 0.5f;
        face_vertex_data[6 + 6 * i + 5] = v2InterpolatedVertex.y + 0.5f;


        //Compute texture coordinates for the far-face
        if (ff_m1 != ff_p1)
        {
            float interpolation_coefficient;
            if (ev_arg < ff_m1_arg || ev_arg > ff_p1_arg)
                interpolation_coefficient = (ev_arg - 2 * pi - ff_m1_arg) / (ff_p1_arg - ff_m1_arg);
            else
                interpolation_coefficient = (ev_arg - ff_m1_arg) / (ff_p1_arg - ff_m1_arg);

            v2InterpolatedVertex = surface_map_data[ff_m1] - far_face_center +
                interpolation_coefficient*(surface_map_data[ff_p1] - surface_map_data[ff_m1]);
        }
        else
        {
            v2InterpolatedVertex = surface_map_data[ff_m1] - far_face_center;
        }
        v2InterpolatedVertex /= far_face_scale;
        face_vertex_data[6 + 6 * (num_angular_base_nodes + 2) + 6 * i + 4] = v2InterpolatedVertex.x + 0.5f;
        face_vertex_data[6 + 6 * (num_angular_base_nodes + 2) + 6 * i + 5] = v2InterpolatedVertex.y + 0.5f;
    }

    glUnmapBuffer(GL_ARRAY_BUFFER);


    //Align surface map data with its vertically averaged mass center
    for (unsigned int i = 0; i < num_points_per_slice* num_slices; ++i)
        surface_map_data[i] -= center;
}


CylindricalSurface::CylindricalSurface(uint32_t num_angular_base_nodes /* = 360 */, uint32_t num_length_base_nodes /* = 100 */) :
AbstractRenderableObject("CylindricalSurface"),
num_angular_base_nodes{ num_angular_base_nodes }, num_length_base_nodes{ num_length_base_nodes },
texture_u_scale{ 1.0f }, texture_v_scale{ 1.0f }, radius{ 1.0f }, length{ 10.0f }, p_render_target{ nullptr }, is_surface_map_defined{ false }
{
    init_cylindrical_surface();
}


CylindricalSurface::CylindricalSurface(const std::string& string_name, const std::string& source_file,
    uint32_t num_angular_base_nodes /* = 360 */, uint32_t num_length_base_nodes /* = 100 */) : AbstractRenderableObject("CylindricalSurface", string_name),
    num_angular_base_nodes{ num_angular_base_nodes }, num_length_base_nodes{ num_length_base_nodes },
    texture_u_scale{ 1.0f }, texture_v_scale{ 1.0f }, radius{ 1.0f }, length{ 10.0f }, p_render_target{ nullptr }, is_surface_map_defined{ false }
{
    init_cylindrical_surface();
    defineSurface(source_file);
}


CylindricalSurface::CylindricalSurface(const std::string& string_name, const std::vector<vec2>& source_data, uint32_t num_points_per_slice, uint32_t num_slices,
    uint32_t num_angular_base_nodes /* = 360 */, uint32_t num_length_base_nodes /* = 100 */) : AbstractRenderableObject("CylindricalSurface", string_name),
    num_angular_base_nodes{ num_angular_base_nodes }, num_length_base_nodes{ num_length_base_nodes },
    texture_u_scale{ 1.0f }, texture_v_scale{ 1.0f }, radius{ 1.0f }, length{ 10.0f }, p_render_target{ nullptr }, is_surface_map_defined{ false }
{
    init_cylindrical_surface();
    defineSurface(source_data, num_points_per_slice, num_slices);
}


CylindricalSurface::CylindricalSurface(const CylindricalSurface& other) :
AbstractRenderableObject(other), AbstractRenderableObjectTextured(other), ExtensionAggregator(other),
num_angular_base_nodes{ other.num_angular_base_nodes }, num_length_base_nodes{ other.num_length_base_nodes },
texture_u_scale{ other.texture_u_scale }, texture_v_scale{ other.texture_v_scale }, radius{ other.radius }, length{ other.length },

surface_map_reference_code{ other.surface_map_reference_code },
side_texture_reference_code{ other.side_texture_reference_code },
face_texture_reference_code{ other.face_texture_reference_code },
surface_map_sampler_ref_code{ other.surface_map_sampler_ref_code },

side_normal_map{ other.side_normal_map },
side_specular_map{ other.side_specular_map },
side_emission_map{ other.side_emission_map },
face_normal_map{ other.face_normal_map },
face_specular_map{ other.face_specular_map },
face_emission_map{ other.face_emission_map },

rendering_program0_ref_code{ other.rendering_program0_ref_code },

p_render_target{ other.p_render_target },
current_rendering_pass{ other.current_rendering_pass },
is_surface_map_defined{ other.is_surface_map_defined }
{
    //Generate OpenGL buffer object to store vertex and index data
    glGenBuffers(2, ogl_buffers);

    //Allocate space for the buffers
    const GLsizei vertex_data_size = sizeof(GLfloat)*(6 * (num_angular_base_nodes + 1)*num_length_base_nodes + 6 * 2 * (num_angular_base_nodes + 2));
    const GLsizei index_data_size = sizeof(GLuint)*(2 * (num_angular_base_nodes + 1)*(num_length_base_nodes - 1) + 2 * (num_angular_base_nodes + 2) + 1);
    glBindBuffer(GL_ARRAY_BUFFER, ogl_buffers[AB_VERTEX]);
    glBufferData(GL_ARRAY_BUFFER, vertex_data_size, NULL, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, ogl_buffers[AB_INDEX]);
    glBufferData(GL_ARRAY_BUFFER, index_data_size, NULL, GL_STATIC_DRAW);

    //Configure vertex attribute object
    glGenVertexArrays(1, &ogl_vertex_array_object);
    glBindVertexArray(ogl_vertex_array_object);
    glEnableVertexAttribArray(vertex_attribute_position::getId());
    vertex_attribute_position::setVertexAttributeBufferLayout(0, 0);
    glEnableVertexAttribArray(vertex_attribute_texcoord::getId());
    vertex_attribute_texcoord::setVertexAttributeBufferLayout(vertex_attribute_position::getCapacity(), 0);

    //Copy vertex data
    glBindBuffer(GL_COPY_READ_BUFFER, other.ogl_buffers[AB_VERTEX]);
    glBindBuffer(GL_COPY_WRITE_BUFFER, ogl_buffers[AB_VERTEX]);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, vertex_data_size);

    //Bind vertex buffer
    glBindVertexBuffer(0, ogl_buffers[AB_VERTEX], 0, vertex_attribute_position::getCapacity() + vertex_attribute_texcoord::getCapacity());

    //Copy index data
    glBindBuffer(GL_COPY_READ_BUFFER, other.ogl_buffers[AB_INDEX]);
    glBindBuffer(GL_COPY_WRITE_BUFFER, ogl_buffers[AB_INDEX]);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, index_data_size);
}


CylindricalSurface::CylindricalSurface(CylindricalSurface&& other) :
AbstractRenderableObject(std::move(other)), AbstractRenderableObjectTextured(std::move(other)), ExtensionAggregator(std::move(other)),
num_angular_base_nodes{ other.num_angular_base_nodes }, num_length_base_nodes{ other.num_length_base_nodes },
texture_u_scale{ other.texture_u_scale }, texture_v_scale{ other.texture_v_scale }, radius{ other.radius }, length{ other.length },

surface_map_reference_code{ std::move(other.surface_map_reference_code) },
side_texture_reference_code{ std::move(other.side_texture_reference_code) },
face_texture_reference_code{ std::move(other.face_texture_reference_code) },
surface_map_sampler_ref_code{ std::move(other.surface_map_sampler_ref_code) },

side_normal_map{ std::move(other.side_normal_map) },
side_specular_map{ std::move(other.side_specular_map) },
side_emission_map{ std::move(other.side_emission_map) },
face_normal_map{ std::move(other.face_normal_map) },
face_specular_map{ std::move(other.face_specular_map) },
face_emission_map{ std::move(other.face_emission_map) },

rendering_program0_ref_code{ std::move(other.rendering_program0_ref_code) },

p_render_target{ other.p_render_target },
current_rendering_pass{ other.current_rendering_pass },
is_surface_map_defined{ other.is_surface_map_defined }
{
    ogl_buffers[AB_VERTEX] = other.ogl_buffers[AB_VERTEX];
    ogl_buffers[AB_INDEX] = other.ogl_buffers[AB_INDEX];
    other.ogl_buffers[AB_VERTEX] = 0;
    other.ogl_buffers[AB_INDEX] = 0;

    ogl_vertex_array_object = other.ogl_vertex_array_object;
    other.ogl_vertex_array_object = 0;
}


CylindricalSurface::~CylindricalSurface()
{
    if (ogl_buffers[AB_VERTEX] && ogl_buffers[AB_INDEX]) glDeleteBuffers(2, ogl_buffers);
    if (ogl_vertex_array_object) glDeleteVertexArrays(1, &ogl_vertex_array_object);
}


float CylindricalSurface::getRadius() const { return radius; }

void CylindricalSurface::setRadius(float new_radius)
{
    radius = new_radius;
}


float CylindricalSurface::getLength() const { return length; }

void CylindricalSurface::setLength(float new_length)
{
    length = new_length;
}


void CylindricalSurface::setTextureScale(float u_scale, float v_scale)
{
    texture_u_scale = u_scale;
    texture_v_scale = v_scale;
}


CylindricalSurface& CylindricalSurface::operator=(const CylindricalSurface& other)
{
    //Account for the special case of "assignment to itself"
    if (this == &other)
        return *this;

    AbstractRenderableObject::operator=(other);
    AbstractRenderableObjectTextured::operator=(other);
    ExtensionAggregator::operator=(other);

    //Reallocate buffers if needed
    const GLsizei old_vertex_data_size = sizeof(GLfloat)*(6 * (num_angular_base_nodes + 1)*num_length_base_nodes + 6 * 2 * (num_angular_base_nodes + 2));
    const GLsizei old_index_data_size = sizeof(GLuint)*(2 * (num_angular_base_nodes + 1)*(num_length_base_nodes - 1) + 2 * (num_angular_base_nodes + 2) + 1);
    const GLsizei new_vertex_data_size = sizeof(GLfloat)*(6 * (other.num_angular_base_nodes + 1)*other.num_length_base_nodes + 6 * 2 * (other.num_angular_base_nodes + 2));
    const GLsizei new_index_data_size = sizeof(GLuint)*(2 * (other.num_angular_base_nodes + 1)*(other.num_length_base_nodes - 1) + 2 * (other.num_angular_base_nodes + 2) + 1);

    if (new_vertex_data_size > old_vertex_data_size)
    {
        glBindBuffer(GL_ARRAY_BUFFER, ogl_buffers[AB_VERTEX]);
        glBufferData(GL_ARRAY_BUFFER, new_vertex_data_size, NULL, GL_STATIC_DRAW);
    }
    if (new_index_data_size > old_index_data_size)
    {
        glBindBuffer(GL_ARRAY_BUFFER, ogl_buffers[AB_INDEX]);
        glBufferData(GL_ARRAY_BUFFER, new_index_data_size, NULL, GL_STATIC_DRAW);
    }

    //Copy vertex data
    glBindBuffer(GL_COPY_READ_BUFFER, other.ogl_buffers[AB_VERTEX]);
    glBindBuffer(GL_COPY_WRITE_BUFFER, ogl_buffers[AB_VERTEX]);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, new_vertex_data_size);

    //Copy index data
    glBindBuffer(GL_COPY_READ_BUFFER, other.ogl_buffers[AB_INDEX]);
    glBindBuffer(GL_COPY_WRITE_BUFFER, ogl_buffers[AB_INDEX]);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, new_index_data_size);

    //Copy object state
    num_angular_base_nodes = other.num_angular_base_nodes;
    num_length_base_nodes = other.num_length_base_nodes;
    texture_u_scale = other.texture_u_scale;
    texture_v_scale = other.texture_v_scale;
    float radius = other.radius;
    float length = other.length;

    surface_map_reference_code = other.surface_map_reference_code;
    side_texture_reference_code = other.side_texture_reference_code;
    face_texture_reference_code = other.face_texture_reference_code;
    surface_map_sampler_ref_code = other.surface_map_sampler_ref_code;

    side_normal_map = other.side_normal_map;
    side_specular_map = other.side_specular_map;
    side_emission_map = other.side_emission_map;
    face_normal_map = other.face_normal_map;
    face_specular_map = other.face_specular_map;
    face_emission_map = other.face_emission_map;

    rendering_program0_ref_code = other.rendering_program0_ref_code;
    p_render_target = other.p_render_target;
    current_rendering_pass = other.current_rendering_pass;
    is_surface_map_defined = other.is_surface_map_defined;

    return *this;
}


CylindricalSurface& CylindricalSurface::operator=(CylindricalSurface&& other)
{
    //Account for the special case of "assignment to itself"
    if (this == &other)
        return *this;

    AbstractRenderableObject::operator=(std::move(other));
    AbstractRenderableObjectTextured::operator=(std::move(other));
    ExtensionAggregator::operator=(std::move(other));

    //Free OpenGL resources owned by the object
    if (ogl_buffers[AB_VERTEX] && ogl_buffers[AB_INDEX]) glDeleteBuffers(2, ogl_buffers);
    if (ogl_vertex_array_object) glDeleteVertexArrays(1, &ogl_vertex_array_object);
    ogl_buffers[AB_VERTEX] = other.ogl_buffers[AB_VERTEX]; other.ogl_buffers[AB_VERTEX] = 0;
    ogl_buffers[AB_INDEX] = other.ogl_buffers[AB_INDEX]; other.ogl_buffers[AB_INDEX] = 0;
    ogl_vertex_array_object = other.ogl_vertex_array_object; other.ogl_vertex_array_object = 0;



    //Move object state
    num_angular_base_nodes = other.num_angular_base_nodes;
    num_length_base_nodes = other.num_length_base_nodes;
    texture_u_scale = other.texture_u_scale;
    texture_v_scale = other.texture_v_scale;
    float radius = other.radius;
    float length = other.length;

    surface_map_reference_code = std::move(other.surface_map_reference_code);
    side_texture_reference_code = std::move(other.side_texture_reference_code);
    face_texture_reference_code = std::move(other.face_texture_reference_code);
    surface_map_sampler_ref_code = std::move(other.surface_map_sampler_ref_code);

    side_normal_map = std::move(other.side_normal_map);
    side_specular_map = std::move(other.side_specular_map);
    side_emission_map = std::move(other.side_emission_map);
    face_normal_map = std::move(other.face_normal_map);
    face_specular_map = std::move(other.face_specular_map);
    face_emission_map = std::move(other.face_emission_map);

    rendering_program0_ref_code = std::move(other.rendering_program0_ref_code);

    p_render_target = other.p_render_target;
    current_rendering_pass = other.current_rendering_pass;
    is_surface_map_defined = other.is_surface_map_defined;

    return *this;
}


void CylindricalSurface::defineSurface(const std::string& source_file)
{
    std::ifstream input_data_stream{ source_file.c_str(), std::ios::in };
    if (!input_data_stream)
    {
        set_error_state(true);

        std::string error_message = "Could not read cylindrical surface map data from file \"" + source_file + "\". The file could not be open for reading.";
        set_error_string(error_message.c_str());
        call_error_callback(error_message.c_str());
        return;
    }


    std::vector<vec2> surface_map_data;
    unsigned int num_rows_parsed = 0;
    unsigned int num_1st_line_tokens = 0;	//number of tokens in the first line of the source file
    while (!input_data_stream.eof())
    {
        //First, get length of the line, which is currently being parsed
        std::ifstream::pos_type beginning_of_line = input_data_stream.tellg();
        input_data_stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        if (input_data_stream.eof()) input_data_stream.clear();	//if end-of-file has been reached, restore the input stream
        std::ifstream::pos_type end_of_line = input_data_stream.tellg();
        unsigned long long current_line_length = end_of_line - beginning_of_line + 1;
        input_data_stream.seekg(beginning_of_line);

        char* c_current_line = new char[static_cast<ptrdiff_t>(current_line_length)];
        input_data_stream.getline(c_current_line, current_line_length);

        std::string str_current_line{ c_current_line };
        delete[] c_current_line;

        //Eliminate non-printable delimiters that may appear in front and in the end of the current line
        size_t first_printable_character;
        if ((first_printable_character = str_current_line.find_first_not_of(" \t")) == std::string::npos) continue;
        size_t last_printable_character = str_current_line.find_last_not_of(" \t");
        str_current_line = str_current_line.substr(first_printable_character, last_printable_character - first_printable_character + 1);


        //Next, divide the currently processed line into tokens
        unsigned int num_tokens_parsed = 0;
        size_t token_start = 0;
        size_t token_end = 0;
        do
        {
            //Extract token
            token_end = str_current_line.find_first_of(" \t,", token_start);
            if (token_end == std::string::npos)	token_end = str_current_line.length();
            std::string token = str_current_line.substr(token_start, token_end - token_start);

            //Attempt to convert token into a floating point value
            try
            {
                float token_value = std::stof(token);
                if (num_rows_parsed % 2 == 0)
                    surface_map_data.push_back(vec2{ token_value, 0.0f });
                else
                {
                    surface_map_data[num_rows_parsed / 2 * num_1st_line_tokens + num_tokens_parsed].y = token_value;
                }
            }
            catch (std::invalid_argument e)
            {
                set_error_state(true);
                std::string err_msg = "Unable to read cylindrical surface data from file \"" + source_file + "\". The file is damaged or not formatted properly.";
                set_error_string(err_msg.c_str());
                call_error_callback(err_msg.c_str());
                return;
            }
            catch (std::out_of_range e)
            {
                set_error_state(true);
                std::string err_msg = "Unable to read cylindrical surface data from file \"" + source_file +
                    "\". The values in the file are out of the range representable by single-precision floating point variables";
                set_error_string(err_msg.c_str());
                call_error_callback(err_msg.c_str());
                return;
            }
            ++num_tokens_parsed;

            //Move towards the next token
            token_start = str_current_line.find_first_not_of(" \t", token_end);
            if (token_start != std::string::npos && str_current_line[token_start] == ',')
            {
                //It is allowed to separate tokens with a single comma delimiter
                ++token_start;
                token_start = str_current_line.find_first_not_of(" \t", token_start);
            }
        } while (token_start != std::string::npos);


        if (num_rows_parsed == 0)
            num_1st_line_tokens = num_tokens_parsed;
        else
        {
            if (num_1st_line_tokens != num_tokens_parsed)
            {
                set_error_state(true);
                std::string err_msg = "Unable to parse row " + std::to_string(num_rows_parsed + 1) + " in file \"" + source_file +
                    "\". The number of elements in this row (" + std::to_string(num_tokens_parsed) +
                    ") differs from the number of elements in the previous rows (" + std::to_string(num_1st_line_tokens) + ")";
                set_error_string(err_msg.c_str());
                call_error_callback(err_msg.c_str());
                return;
            }
        }

        num_rows_parsed++;
    }

    //Define texture coordinates for the faces and apply side surface alignment
    process_surface_data(surface_map_data, num_1st_line_tokens, num_rows_parsed / 2);

    //Construct texture object from the data parsed from file
    float* raw_surface_map_data = new float[num_rows_parsed * num_1st_line_tokens];
    for (unsigned int i = 0; i < num_rows_parsed / 2 * num_1st_line_tokens; ++i)
    {
        raw_surface_map_data[2 * i + 0] = surface_map_data[i].x;
        raw_surface_map_data[2 * i + 1] = surface_map_data[i].y;
    }
    ImmutableTexture2D surface_map_texture{ "CylindricalSurface::surface_map_texture" };
    surface_map_texture.allocateStorage(1, 1, TextureSize{ num_1st_line_tokens, num_rows_parsed / 2, 1 }, InternalPixelFormat::SIZED_FLOAT_RG32);
    surface_map_texture.setMipmapLevelData(0, PixelLayout::RG, PixelDataType::FLOAT, raw_surface_map_data);
    delete[] raw_surface_map_data;

    if (!surface_map_reference_code)
        surface_map_reference_code = registerTexture(surface_map_texture, surface_map_sampler_ref_code);
    else
        updateTexture(surface_map_reference_code, surface_map_texture, surface_map_sampler_ref_code);

    is_surface_map_defined = true;
}


void CylindricalSurface::defineSurface(const std::vector<vec2>& source_data, uint32_t num_points_per_slice, uint32_t num_slices)
{
    std::vector<vec2> aligned_source_data = source_data;

    //Define texture coordinates for the faces and apply side surface alignment
    process_surface_data(aligned_source_data, num_points_per_slice, num_slices);

    //Construct texture object from the given data table
    float* raw_surface_map_data = new float[num_points_per_slice*num_slices];
    for (unsigned int i = 0; i < num_points_per_slice*num_slices; ++i)
    {
        raw_surface_map_data[2 * i + 0] = aligned_source_data[i].x;
        raw_surface_map_data[2 * i + 1] = aligned_source_data[i].y;
    }
    ImmutableTexture2D surface_map_texture{ "CylindricalSurface::surface_map_texture" };
    surface_map_texture.allocateStorage(1, 1, TextureSize{ num_points_per_slice, num_slices, 1 }, InternalPixelFormat::SIZED_FLOAT_RG32);
    surface_map_texture.setMipmapLevelData(0, PixelLayout::RG, PixelDataType::FLOAT, raw_surface_map_data);
    delete[] raw_surface_map_data;

    if (!surface_map_reference_code)
        surface_map_reference_code = registerTexture(surface_map_texture, surface_map_sampler_ref_code);
    else
        updateTexture(surface_map_reference_code, surface_map_texture, surface_map_sampler_ref_code);

    is_surface_map_defined = true;
}


void CylindricalSurface::installTexture(const ImmutableTexture2D& side_surface_texture, const ImmutableTexture2D& face_texture)
{
    if (!side_texture_reference_code)
        side_texture_reference_code = registerTexture(side_surface_texture, surface_map_sampler_ref_code);
    else
        updateTexture(side_texture_reference_code, side_surface_texture, surface_map_sampler_ref_code);

    if (!face_texture_reference_code)
        face_texture_reference_code = registerTexture(face_texture);
    else
        updateTexture(face_texture_reference_code, face_texture);
}


void CylindricalSurface::applyNormalMapSourceTexture(const ImmutableTexture2D& side_surface_texture_NRM, const ImmutableTexture2D& face_texture_NRM)
{
    side_normal_map = side_surface_texture_NRM;
    face_normal_map = face_texture_NRM;
    ExtensionAggregator::applyNormalMapSourceTexture(side_normal_map);
}


void CylindricalSurface::applySpecularMapSourceTexture(const ImmutableTexture2D& side_surface_texture_SPEC, const ImmutableTexture2D& face_texture_SPEC)
{
    side_specular_map = side_surface_texture_SPEC;
    face_specular_map = face_texture_SPEC;
    ExtensionAggregator::applySpecularMapSourceTexture(side_specular_map);
}


void CylindricalSurface::applyEmissionMapSourceTexture(const ImmutableTexture2D& side_surface_texture_EMISSION, const ImmutableTexture2D& face_texture_EMISSION)
{
    side_emission_map = side_surface_texture_EMISSION;
    face_emission_map = face_texture_EMISSION;
    ExtensionAggregator::applyEmissionMapSourceTexture(side_emission_map);
}


bool CylindricalSurface::supportsRenderingMode(uint32_t rendering_mode) const
{
    switch (rendering_mode)
    {
    case TW_RENDERING_MODE_DEFAULT: return true;
    case  TW_RENDERING_MODE_SILHOUETTE: return true;
    default: return false;
    }
}


uint32_t CylindricalSurface::getNumberOfRenderingPasses(uint32_t rendering_mode) const
{
    switch (rendering_mode)
    {
    case TW_RENDERING_MODE_DEFAULT: return 2;

    case  TW_RENDERING_MODE_SILHOUETTE: return 2;

    default: return 0;
    }
}


bool CylindricalSurface::configureRendering(AbstractRenderingDevice& render_target, uint32_t rendering_pass)
{
    //Check if object has surface map loaded
    if (!is_surface_map_defined)
    {
        set_error_state(true);
        std::string err_msg = "Unable to render cylindrical surface: surface map is undefined";
        set_error_string(err_msg.c_str());
        call_error_callback(err_msg.c_str());
        return false;
    }

    //Activate rendering target
    if (!render_target.isActive())
        render_target.makeActive();

    p_render_target = &render_target;
    current_rendering_pass = rendering_pass;

    //Bind VAO
    glBindVertexArray(ogl_vertex_array_object);


    switch (getActiveRenderingMode())
    {
    case TW_RENDERING_MODE_DEFAULT:

        //Bind surface map
        if (surface_map_reference_code)
        {
            retrieveShaderProgram(rendering_program0_ref_code)->assignUniformScalar("surface_map", getBindingUnit(surface_map_reference_code));
            bindTexture(surface_map_reference_code);
        }

        //Activate rendering program
        COMPLETE_SHADER_PROGRAM_CAST(retrieveShaderProgram(rendering_program0_ref_code)).activate();

        switch (rendering_pass)
        {
        case 0:
            //Bind side surface texture
            if (side_texture_reference_code)
            {
                retrieveShaderProgram(rendering_program0_ref_code)->assignUniformScalar("surface_diffuse_texture", getBindingUnit(side_texture_reference_code));
                bindTexture(side_texture_reference_code);
            }

            //Bind normal, specular, and emission maps
            if (ExtensionAggregator::doesHaveNormalMap()) ExtensionAggregator::applyNormalMapSourceTexture(face_normal_map);
            if (ExtensionAggregator::doesHaveSpecularMap()) ExtensionAggregator::applySpecularMapSourceTexture(face_specular_map);
            if (ExtensionAggregator::doesHaveEmissionMap()) ExtensionAggregator::applyEmissionMapSourceTexture(face_emission_map);

            return true;

        case 1:
            //Configure context settings
            render_target.pushOpenGLContextSettings();
            render_target.setPrimitiveRestartEnableState(true);
            render_target.setPrimitiveRestartIndexValue(0xFFFFFFFF);
            render_target.applyOpenGLContextSettings();

            //Bind side surface texture
            if (face_texture_reference_code)
            {
                retrieveShaderProgram(rendering_program0_ref_code)->assignUniformScalar("surface_diffuse_texture", getBindingUnit(face_texture_reference_code));
                bindTexture(face_texture_reference_code);
            }

            //Bind normal, specular, and emission maps
            if (ExtensionAggregator::doesHaveNormalMap()) ExtensionAggregator::applyNormalMapSourceTexture(side_normal_map);
            if (ExtensionAggregator::doesHaveSpecularMap()) ExtensionAggregator::applySpecularMapSourceTexture(side_specular_map);
            if (ExtensionAggregator::doesHaveEmissionMap()) ExtensionAggregator::applyEmissionMapSourceTexture(side_emission_map);

            return true;
        }


    case TW_RENDERING_MODE_SILHOUETTE:
        return true;


    default:
        return false;
    }


}


void CylindricalSurface::configureViewProjectionTransform(const AbstractProjectingDevice& projecting_device)
{
    vec3 v3ScaleFactors = getObjectScale();
    v3ScaleFactors.x *= radius;
    v3ScaleFactors.y *= radius;
    v3ScaleFactors.z *= length;
    retrieveShaderProgram(rendering_program0_ref_code)->assignUniformVector("v3Scale", v3ScaleFactors);

    mat4 m4ScaleTransform{
        v3ScaleFactors.x, 0, 0, 0,
        0, v3ScaleFactors.y, 0, 0,
        0, 0, v3ScaleFactors.z, 0,
        0, 0, 0, 1 };

    mat4 m4ModelView = projecting_device.getViewTransform()*getObjectTransform()*m4ScaleTransform;

    retrieveShaderProgram(rendering_program0_ref_code)->assignUniformMatrix("m4ModelView", m4ModelView);
    retrieveShaderProgram(rendering_program0_ref_code)->assignUniformMatrix("m4Projection", projecting_device.getProjectionTransform());
}


bool CylindricalSurface::render()
{
    if (getErrorState()) return false;

    switch (current_rendering_pass)
    {
    case 0:
        glDrawElements(GL_TRIANGLE_STRIP, 2 * (num_angular_base_nodes + 1)*(num_length_base_nodes - 1), GL_UNSIGNED_INT, 0);
        return true;

    case 1:
        glDrawElements(GL_TRIANGLE_FAN, 2 * (num_angular_base_nodes + 2) + 1, GL_UNSIGNED_INT,
            reinterpret_cast<void*>(sizeof(GLuint) * 2 * (num_angular_base_nodes + 1)*(num_length_base_nodes - 1)));
        return true;

    default:
        return false;
    }

}


bool CylindricalSurface::configureRenderingFinalization()
{
    if (getErrorState()) return false;

    switch (current_rendering_pass)
    {
    case 0:
        return true;

    case 1:
        p_render_target->popOpenGLContextSettings();
        return true;

    default:
        return false;
    }
}