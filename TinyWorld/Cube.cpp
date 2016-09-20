#include "Cube.h"

using namespace tiny_world;

const std::string Cube::rendering_program_name = "Cube::rendering_program";

char Cube::vp_source[] =
"#version 430 core\n"
"uniform mat4 mvp;						//model-view-projection transform\n"
"in vec4 vertex_position;				//position of a vertex\n"
"in vec2 tex_coord;						//texture coordinate\n"
"out VertexData\n"
"{\n"
"	vec2 tex_coord;\n"
"}vs_out;\n"
"void main()\n"
"{\n"
"	gl_Position = mvp*vertex_position;\n"
"	vs_out.tex_coord = tex_coord;\n"
"}\n";

char Cube::fp_source[] =
"#version 430 core\n"
"layout(binding = 0) uniform sampler2D tex_sampler;\n"
"in VertexData\n"
"{\n"
"	vec2 tex_coord;\n"
"}vs_in;\n"
"out vec4 fColor;		//color of the output fragment\n"
"void main()\n"
"{\n"
"	fColor = texture(tex_sampler, vs_in.tex_coord);\n"
"}\n";


void Cube::applyScreenSize(const uvec2& screen_size)
{

}


void Cube::update_array_data()
{
	const uint32_t singe_vertex_size = vertex_attribute_position::getSize() + vertex_attribute_texcoord::getSize();
	GLfloat *buf = new GLfloat[singe_vertex_size * 36];
	for (int i = 0; i < 36; ++i)
	{
		buf[singe_vertex_size*i + 0] = this->vertices[i].x;
		buf[singe_vertex_size*i + 1] = this->vertices[i].y;
		buf[singe_vertex_size*i + 2] = this->vertices[i].z;
		buf[singe_vertex_size*i + 3] = this->vertices[i].w;

		buf[singe_vertex_size*i + 4] = tex_coords[i].x;
		buf[singe_vertex_size*i + 5] = tex_coords[i].y;
	}
	glBindBuffer(GL_ARRAY_BUFFER, ogl_array_buf_id);
	glBufferSubData(GL_ARRAY_BUFFER, 0,
		36 * (vertex_attribute_position::getCapacity() + vertex_attribute_texcoord::getCapacity()), buf);

	delete[] buf;
}

void Cube::setup_object()
{

	//Setup cube's vertices
	std::array<vec4, 8> vertices =
	{ vec4{ -0.5f, -0.5f, 0.5f }, vec4{ 0.5f, -0.5f, 0.5f }, vec4{ 0.5f, -0.5f, -0.5f }, vec4{ -0.5f, -0.5f, -0.5f },
	vec4{ -0.5f, 0.5f, 0.5f }, vec4{ 0.5f, 0.5f, 0.5f }, vec4{ 0.5f, 0.5f, -0.5f }, vec4{ -0.5f, 0.5f, -0.5f } };

	std::array<unsigned int, 36> indeces =
	{ 0, 3, 2, 2, 1, 0,
	7, 4, 5, 5, 6, 7,
	5, 1, 2, 2, 6, 5,
	3, 0, 4, 4, 7, 3,
	4, 0, 1, 1, 5, 4,
	2, 3, 7, 7, 6, 2 };

	for (int i = 0; i < 36; ++i)
		this->vertices[i] = vertices[indeces[i]];

	for (vec4& v : vertices)
		v = v*side_size;


	//Setup cube's texture coordinates
	tex_coords[0] = vec2{ 0.25f, 0.5f };
	tex_coords[1] = vec2{ 0.25f, 0.75f };
	tex_coords[2] = vec2{ 0.5f, 0.75f };
	tex_coords[3] = vec2{ 0.5f, 0.75f };
	tex_coords[4] = vec2{ 0.5f, 0.5f };
	tex_coords[5] = vec2{ 0.25f, 0.5f };
	
	tex_coords[7] = vec2{ 0.25f, 0.25f };
	tex_coords[6] = vec2{ 0.25f, 0.0f };
	tex_coords[8] = vec2{ 0.5f, 0.25f };
	tex_coords[9] = vec2{ 0.5f, 0.25f };
	tex_coords[10] = vec2{ 0.5f, 0.0f };
	tex_coords[11] = vec2{ 0.25f, 0.0f };

	tex_coords[12] = vec2{ 0.75f, 0.5f };
	tex_coords[13] = vec2{ 0.5f, 0.5f };
	tex_coords[14] = vec2{ 0.5f, 0.75f };
	tex_coords[15] = vec2{ 0.5f, 0.75f };
	tex_coords[16] = vec2{ 0.75f, 0.75f };
	tex_coords[17] = vec2{ 0.75f, 0.5f };
	
	tex_coords[18] = vec2{ 0.25f, 0.75f };
	tex_coords[19] = vec2{ 0.25f, 0.5f };
	tex_coords[20] = vec2{ 0.0f, 0.5f };
	tex_coords[21] = vec2{ 0.0f, 0.5f };
	tex_coords[22] = vec2{ 0.0f, 0.75f };
	tex_coords[23] = vec2{ 0.25f, 0.75f };

	tex_coords[24] = vec2{ 0.25f, 0.25f };
	tex_coords[25] = vec2{ 0.25f, 0.5f };
	tex_coords[26] = vec2{ 0.5f, 0.5f };
	tex_coords[27] = vec2{ 0.5f, 0.5f };
	tex_coords[28] = vec2{ 0.5f, 0.25f };
	tex_coords[29] = vec2{ 0.25f, 0.25f };

	tex_coords[30] = vec2{ 0.5f, 0.75f };
	tex_coords[31] = vec2{ 0.25f, 0.75f };
	tex_coords[32] = vec2{ 0.25f, 1.0f };
	tex_coords[33] = vec2{ 0.25f, 1.0f };
	tex_coords[34] = vec2{ 0.5f, 1.0f };
	tex_coords[35] = vec2{ 0.5f, 0.75f };

	//Setup cube's rendering program if it has not yet been initialized
	if (!rendering_program_ref_code)
	{
		rendering_program_ref_code = createCompleteShaderProgram(rendering_program_name, { PipelineStage::VERTEX_SHADER, PipelineStage::FRAGMENT_SHADER });

		retrieveShaderProgram(rendering_program_ref_code)->addShader(Shader{ std::make_pair(vp_source, std::strlen(vp_source)), ShaderType::VERTEX_SHADER, "cube_render_program_vp" });
		retrieveShaderProgram(rendering_program_ref_code)->addShader(Shader{ std::make_pair(fp_source, std::strlen(fp_source)), ShaderType::FRAGMENT_SHADER, "cube_render_program_fp" });

		retrieveShaderProgram(rendering_program_ref_code)->bindVertexAttributeId("vertex_position", vertex_attribute_position::getId());
		retrieveShaderProgram(rendering_program_ref_code)->bindVertexAttributeId("tex_coord", vertex_attribute_texcoord::getId());

		retrieveShaderProgram(rendering_program_ref_code)->link();
	}


	//Create vertex attribute object and setup vertex data
	glGenVertexArrays(1, &ogl_vertex_attribute_object_id);
	glBindVertexArray(ogl_vertex_attribute_object_id);
	glEnableVertexAttribArray(vertex_attribute_position::getId());
	glEnableVertexAttribArray(vertex_attribute_texcoord::getId());
	vertex_attribute_position::setVertexAttributeBufferLayout(0, 0);
	vertex_attribute_texcoord::setVertexAttributeBufferLayout(vertex_attribute_position::getCapacity(), 0);

	//Create and initialize array buffer
	glGenBuffers(1, &ogl_array_buf_id);
	glBindBuffer(GL_ARRAY_BUFFER, ogl_array_buf_id);
	glBufferData(GL_ARRAY_BUFFER, 36 * (vertex_attribute_position::getCapacity() + vertex_attribute_texcoord::getCapacity()),
		NULL, GL_STATIC_DRAW);

	update_array_data();
	glBindVertexBuffer(0, ogl_array_buf_id, 0,
		vertex_attribute_position::getCapacity() + vertex_attribute_texcoord::getCapacity());

	//Setup texture sampler
	main_sampler_ref_code = createTextureSampler("Cube::main_sampler", SamplerMagnificationFilter::LINEAR, SamplerMinificationFilter::LINEAR_MIPMAP_NEAREST,
		SamplerWrapping{ SamplerWrappingMode::CLAMP_TO_BORDER, SamplerWrappingMode::CLAMP_TO_BORDER, SamplerWrappingMode::CLAMP_TO_BORDER });
}

Cube::Cube() : AbstractRenderableObject("Cube"),
side_size{ 1.0f }
{
	setup_object();
}

Cube::Cube(std::string cube_string_name) : AbstractRenderableObject("Cube", cube_string_name), side_size{ 1.0f }
{
	setup_object();
}

Cube::Cube(std::string cube_string_name, float side_size) : AbstractRenderableObject("Cube", cube_string_name), side_size{ side_size }
{
	setup_object();
}

Cube::Cube(std::string cube_string_name, float side_size, vec3 location) :
AbstractRenderableObject("Cube", cube_string_name, location), side_size{ side_size }
{
	setup_object();
}

Cube::Cube(std::string cube_string_name, float side_size, vec3 location, float z_rot_angle, float y_rot_angle, float x_rot_angle) :
AbstractRenderableObject("Cube", cube_string_name, location, z_rot_angle, y_rot_angle, x_rot_angle), side_size{ side_size }
{
	setup_object();
}

Cube::Cube(const Cube& other) : AbstractRenderableObject(other), 
AbstractRenderableObjectTextured(other), AbstractRenderableObjectExtensionAggregator<>(other), 
rendering_program_ref_code{ other.rendering_program_ref_code }, side_size{ other.side_size }, main_texture_ref_code{ other.main_texture_ref_code },
main_sampler_ref_code{other.main_sampler_ref_code}
{
	setup_object();
}

Cube::Cube(Cube&& other) : AbstractRenderableObject(std::move(other)), 
AbstractRenderableObjectTextured(std::move(other)), AbstractRenderableObjectExtensionAggregator<>(std::move(other)), 
rendering_program_ref_code{ other.rendering_program_ref_code }, side_size{ other.side_size }, 
main_texture_ref_code{ other.main_texture_ref_code }, vertices(std::move(other.vertices)),
tex_coords(std::move(other.tex_coords)), main_sampler_ref_code{ std::move(other.main_sampler_ref_code) }
{
	ogl_array_buf_id = other.ogl_array_buf_id;
	other.ogl_array_buf_id = 0;

	ogl_vertex_attribute_object_id = other.ogl_vertex_attribute_object_id;
	other.ogl_vertex_attribute_object_id = 0;
}

Cube::~Cube()
{
	if (ogl_array_buf_id) glDeleteBuffers(1, &ogl_array_buf_id);
	if (ogl_vertex_attribute_object_id) glDeleteVertexArrays(1, &ogl_vertex_attribute_object_id);
}

Cube& Cube::operator=(const Cube& other)
{
	AbstractRenderableObject::operator=(other);
	AbstractRenderableObjectTextured::operator=(other);
	AbstractRenderableObjectExtensionAggregator<>::operator=(other);
	rendering_program_ref_code = other.rendering_program_ref_code;
	side_size = other.side_size;
	vertices = other.vertices;
	main_texture_ref_code = other.main_texture_ref_code;
	main_sampler_ref_code = other.main_sampler_ref_code;
	update_array_data();

	return *this;
}

Cube& Cube::operator=(Cube&& other)
{
	AbstractRenderableObject::operator=(std::move(other));
	AbstractRenderableObjectTextured::operator=(std::move(other));
	AbstractRenderableObjectExtensionAggregator<>::operator=(std::move(other));
	rendering_program_ref_code = other.rendering_program_ref_code;
	side_size = other.side_size;
	vertices = std::move(other.vertices);
	main_texture_ref_code = other.main_texture_ref_code;
	main_sampler_ref_code = other.main_sampler_ref_code;

	std::swap(ogl_array_buf_id, other.ogl_array_buf_id);
	std::swap(ogl_vertex_attribute_object_id, other.ogl_vertex_attribute_object_id);

	return *this;
}

float Cube::getSideSize() const { return side_size; }

void Cube::setSideSize(float new_side_size)
{
	for (vec4& v : vertices)
	{
		v = v *(new_side_size / side_size);
		v.w = 1.0f;
	}
	side_size = new_side_size;
	update_array_data();
}

void Cube::installTexture(const ImmutableTexture2D& _2d_texture)
{
	if (!main_texture_ref_code)
		main_texture_ref_code = registerTexture(_2d_texture, main_sampler_ref_code);
	else
		updateTexture(main_texture_ref_code, _2d_texture, main_sampler_ref_code);
}

bool Cube::supportsRenderingMode(uint32_t rendering_mode) const
{
	switch (rendering_mode)
	{
	case 0:
	case 1:
		return true;
	default:
		return false;
	}
}

void Cube::configureViewProjectionTransform(const AbstractProjectingDevice& projecting_device)
{
	mat4 mvp = projecting_device.getProjectionTransform() * projecting_device.getViewTransform() * getObjectTransform() * getObjectScaleTransform();
	retrieveShaderProgram(rendering_program_ref_code)->assignUniformMatrix("mvp", mvp);
}

uint32_t Cube::getNumberOfRenderingPasses(uint32_t rendering_mode) const 
{ 
	switch (rendering_mode)
	{
	case 0:
	case 1:
		return 1;
	default:
		return 0;
	}
}

bool Cube::configureRendering(AbstractRenderingDevice& render_target, uint32_t rendering_pass)
{
	//If requested render target is not yet active, activate it
	if (!render_target.isActive())
		render_target.makeActive();

	//Bind object's data buffer
	glBindVertexArray(ogl_vertex_attribute_object_id);

	COMPLETE_SHADER_PROGRAM_CAST(retrieveShaderProgram(rendering_program_ref_code)).activate();

	if (main_texture_ref_code.first != -1)
		bindTexture(main_texture_ref_code);

	return true;
}

bool Cube::render()
{
	switch (getActiveRenderingMode())
	{
	case 0:
		glDrawArrays(GL_TRIANGLES, 0, 36);
		break;
	case 1:
		glDrawArrays(GL_LINE_STRIP, 0, 36);
		break;
	default:
		break;
	}
	
	return true;
}

bool Cube::configureRenderingFinalization()
{
	return true;
}