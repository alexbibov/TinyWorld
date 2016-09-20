#include "FullscreenRectangle.h"

using namespace tiny_world;


const std::string FullscreenRectangle::fixed_vp_name = "fullscreen_rectangle_fixed_vertex_program";
const std::string FullscreenRectangle::fixed_fp_name = "fullscreen_rectangle_fixed_fragment_program";

std::array<vec4, FullscreenRectangle::num_vertices> FullscreenRectangle::vertices{vec4{ -1.0f / 2.0f, 1.0f / 2.0f, 0, 1 }, vec4{ -1.0f / 2.0f, -1.0f / 2.0f, 0, 1 },
vec4{ 1.0f / 2.0f, 1.0f / 2.0f, 0, 1 }, vec4{ 1.0f / 2.0f, -1.0f / 2.0f, 0, 1 }};

std::array<vec2, FullscreenRectangle::num_vertices> FullscreenRectangle::tex_coords{vec2{ 0.0f, 1.0f }, vec2{ 0.0f, 0.0f }, vec2{ 1.0f, 1.0f }, vec2{ 1.0f, 0.0f }};


void FullscreenRectangle::applyScreenSize(const uvec2& screen_size)
{
	//setDimensions(static_cast<float>(screen_size.x), static_cast<float>(screen_size.y));
}


void FullscreenRectangle::setup_shader_pipeline(uint32_t mode)
{
	//***********************************************************Setup shader pipeline***********************************************************
	//Attach user-defined geometry program to the shader pipeline if applicable
	if (p_user_defined_geometry_shader && mode == TW_RENDERING_MODE_FILTER_AND_DYNAMICS)
		shader_pipeline.attach(*p_user_defined_geometry_shader);

	//If no user-defined fragment shader was provided or simplified rendering mode is selected, use the default program
	if (!p_user_defined_fragment_shader || mode == TW_RENDERING_MODE_SIMPLIFIED)	
		shader_pipeline.attach(SEPARATE_SHADER_PROGRAM_CAST(retrieveShaderProgram(fixed_fp_ref_code)));
	else
		shader_pipeline.attach(*p_user_defined_fragment_shader);
}


void FullscreenRectangle::update_data()
{
	const uint32_t single_vertex_size = vertex_attribute_position::getSize() + vertex_attribute_texcoord::getSize();
	GLfloat *buf = new GLfloat[num_vertices * single_vertex_size];

	for (int i = 0; i < num_vertices; ++i)
	{
		buf[i * single_vertex_size] = vertices[i].x * width;
		buf[i * single_vertex_size + 1] = vertices[i].y * height;
		buf[i * single_vertex_size + 2] = vertices[i].z;
		buf[i * single_vertex_size + 3] = vertices[i].w;

		buf[i * single_vertex_size + 4] = tex_coords[i].x;
		buf[i * single_vertex_size + 5] = tex_coords[i].y;
	}

	glBindBuffer(GL_ARRAY_BUFFER, ogl_data_buf);
	glBufferSubData(GL_ARRAY_BUFFER, 0,
		num_vertices * (vertex_attribute_position::getCapacity() + vertex_attribute_texcoord::getCapacity()),
		buf);

	delete[] buf;
}


void FullscreenRectangle::setup_object()
{
	//**********************************************Configure fixed shader programs*********************************************

	char vp_source[] =		//define vertex shader source
		"#version 430 core\n"
		"in vec4 vertex_pos;\n"
		"in vec2 _tex_coord;\n"
		"uniform mat4 mvp;		//model-view-projection matrix affecting the rectangle\n"
		"out gl_PerVertex\n"
		"{\n"
		"	vec4 gl_Position;\n"
		"	float gl_PointSize;\n"
		"	float gl_ClipDistance[];\n"
		"};\n"
		"out vec2 tex_coord;\n"
		"void main()\n"
		"{\n"
		"gl_Position = mvp * vertex_pos;\n"
		"tex_coord = _tex_coord;\n"
		"}";


	char fp_source[] =	//define fragment shader source
		"#version 430 core\n"
		"in vec2 tex_coord;\n"
		"out vec4 fColor;\n"
		"uniform sampler2D source0;\n"
		"void main()\n"
		"{\n"
		"	fColor = texture(source0, tex_coord);\n"
		"}\n";

	if (!fixed_vp_ref_code)
	{
		fixed_vp_ref_code = createSeparateShaderProgram(fixed_vp_name, { PipelineStage::VERTEX_SHADER });
		retrieveShaderProgram(fixed_vp_ref_code)->addShader(Shader{ GLSLSourceCode{ vp_source, strlen(vp_source) }, ShaderType::VERTEX_SHADER, "fullscreen_rectangle_default_vertex_shader" });
		retrieveShaderProgram(fixed_vp_ref_code)->bindVertexAttributeId("vertex_pos", vertex_attribute_position::getId());
		retrieveShaderProgram(fixed_vp_ref_code)->bindVertexAttributeId("_tex_coord", vertex_attribute_texcoord::getId());
		retrieveShaderProgram(fixed_vp_ref_code)->link();
	}

	if (!fixed_fp_ref_code)
	{
		fixed_fp_ref_code = createSeparateShaderProgram(fixed_fp_name, { PipelineStage::FRAGMENT_SHADER });
		retrieveShaderProgram(fixed_fp_ref_code)->addShader(Shader{ GLSLSourceCode{ fp_source, strlen(fp_source) }, ShaderType::FRAGMENT_SHADER, "fullscreen_rectangle_default_fragment_shader" });
		retrieveShaderProgram(fixed_fp_ref_code)->link();
	}


	//Attach fixed vertex program to the shader pipeline
	shader_pipeline.attach(SEPARATE_SHADER_PROGRAM_CAST(retrieveShaderProgram(fixed_vp_ref_code)));

	//Configure the rest of the shader pipeline
	setup_shader_pipeline(getActiveRenderingMode());


	//************************************************Populate context with data********************************************
	//Create a vertex array object
	glGenVertexArrays(1, &ogl_vertex_array_object_id);
	glBindVertexArray(ogl_vertex_array_object_id);
	glEnableVertexAttribArray(vertex_attribute_position::getId());
	glEnableVertexAttribArray(vertex_attribute_texcoord::getId());
	vertex_attribute_position::setVertexAttributeBufferLayout(0, 0);
	vertex_attribute_texcoord::setVertexAttributeBufferLayout(vertex_attribute_position::getCapacity(), 0);

	//Create an array buffer object
	glGenBuffers(1, &ogl_data_buf);
	glBindBuffer(GL_ARRAY_BUFFER, ogl_data_buf);
	glBufferData(GL_ARRAY_BUFFER,
		(vertex_attribute_position::getCapacity() + vertex_attribute_texcoord::getCapacity()) * num_vertices,
		NULL,
		GL_STATIC_DRAW);

	//Populate the buffer with data
	update_data();
	glBindVertexBuffer(0, ogl_data_buf, 0, vertex_attribute_position::getCapacity() + vertex_attribute_texcoord::getCapacity());
}

FullscreenRectangle::FullscreenRectangle() : AbstractRenderableObject("FullscreenRectangle"),
width{ 1.0f }, height{ 1.0f }, p_projecting_device{ nullptr },
p_user_defined_geometry_shader{ nullptr }, p_user_defined_fragment_shader{ nullptr }
{
	setup_object();

	//Configure texture sampler object
	texture_sampler_ref_code = createTextureSampler("FullscreenRectangle::DefaultSampler", SamplerMagnificationFilter::NEAREST, SamplerMinificationFilter::NEAREST,
		SamplerWrapping{ SamplerWrappingMode::CLAMP_TO_EDGE,
		SamplerWrappingMode::CLAMP_TO_EDGE, SamplerWrappingMode::CLAMP_TO_EDGE });
}


FullscreenRectangle::FullscreenRectangle(float width, float height) : AbstractRenderableObject("FullscreenRectangle"),
width{ width }, height{ height }, p_projecting_device{ nullptr }, 
p_user_defined_geometry_shader{ nullptr }, p_user_defined_fragment_shader{ nullptr }
{
	setup_object();

	//Configure texture sampler object
	texture_sampler_ref_code = createTextureSampler("FullscreenRectangle::DefaultSampler", SamplerMagnificationFilter::NEAREST, SamplerMinificationFilter::NEAREST,
		SamplerWrapping{ SamplerWrappingMode::CLAMP_TO_EDGE,
		SamplerWrappingMode::CLAMP_TO_EDGE, SamplerWrappingMode::CLAMP_TO_EDGE });
}


FullscreenRectangle::FullscreenRectangle(const FullscreenRectangle& other) : 
AbstractRenderableObject(other), 
AbstractRenderableObjectTextured(other),
AbstractRenderableObjectExtensionAggregator<>(other),
width{ other.width }, height{ other.height }, 
p_projecting_device{ other.p_projecting_device }, 
fixed_vp_ref_code{ other.fixed_vp_ref_code },
fixed_fp_ref_code{ other.fixed_fp_ref_code },
p_user_defined_geometry_shader{ other.p_user_defined_geometry_shader },
geometry_shader_setup_func{ other.geometry_shader_setup_func }, 
p_user_defined_fragment_shader{ other.p_user_defined_fragment_shader },
fragment_shader_setup_func{ other.fragment_shader_setup_func },
texture_ref_code{ other.texture_ref_code },
texture_sampler_ref_code{ other.texture_sampler_ref_code }
{
	setup_object();
}


FullscreenRectangle::FullscreenRectangle(FullscreenRectangle&& other) : 
AbstractRenderableObject(std::move(other)), 
AbstractRenderableObjectTextured(std::move(other)),
AbstractRenderableObjectExtensionAggregator<>(std::move(other)),
width{ other.width }, height{ other.height },
p_projecting_device{ other.p_projecting_device }, 
fixed_vp_ref_code{ other.fixed_vp_ref_code }, 
fixed_fp_ref_code{ other.fixed_fp_ref_code }, 
shader_pipeline{ std::move(other.shader_pipeline) },
p_user_defined_geometry_shader{ other.p_user_defined_geometry_shader },
p_user_defined_fragment_shader{ other.p_user_defined_fragment_shader },
geometry_shader_setup_func{ std::move(other.geometry_shader_setup_func) }, 
fragment_shader_setup_func{ std::move(other.fragment_shader_setup_func) },
texture_ref_code{ other.texture_ref_code },
texture_sampler_ref_code{ std::move(other.texture_sampler_ref_code) },
ogl_vertex_array_object_id{ other.ogl_vertex_array_object_id }, ogl_data_buf{ other.ogl_data_buf }
{
	other.ogl_vertex_array_object_id = 0;
	other.ogl_data_buf = 0;
}


FullscreenRectangle& FullscreenRectangle::operator=(const FullscreenRectangle& other)
{
	//Account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	//Copy base settings
	AbstractRenderableObject::operator=(other);

	//Copy the rest of the settings
	AbstractRenderableObjectTextured::operator=(other);
	AbstractRenderableObjectExtensionAggregator<>::operator=(other);

	//Copy settings from the assignment source
	width = other.width;
	height = other.height;
	p_projecting_device = other.p_projecting_device;
	p_user_defined_geometry_shader = other.p_user_defined_geometry_shader;
	p_user_defined_fragment_shader = other.p_user_defined_fragment_shader;
	geometry_shader_setup_func = other.geometry_shader_setup_func;
	fragment_shader_setup_func = other.fragment_shader_setup_func;
	texture_ref_code = other.texture_ref_code;
	texture_sampler_ref_code = other.texture_sampler_ref_code;

	//Shader pipeline should not be assigned "as is", otherwise the object receiving the assignment
	//would end up referring to the shader programs belonging to the object being assigned
	//shader_pipeline = other.shader_pipeline;
	shader_pipeline.attach(SEPARATE_SHADER_PROGRAM_CAST(retrieveShaderProgram(fixed_vp_ref_code)));
	setup_shader_pipeline(getActiveRenderingMode());

	update_data();

	return *this;
}


FullscreenRectangle& FullscreenRectangle::operator=(FullscreenRectangle&& other)
{
	//Account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	//Move base settings
	AbstractRenderableObject::operator=(std::move(other));

	//Move the rest of the settings
	AbstractRenderableObjectTextured::operator=(std::move(other));
	AbstractRenderableObjectExtensionAggregator<>::operator=(std::move(other));

	//Move settings from the assignment source
	width = other.width;
	height = other.height;
	p_projecting_device = other.p_projecting_device;
	p_user_defined_geometry_shader = other.p_user_defined_geometry_shader;
	p_user_defined_fragment_shader = other.p_user_defined_fragment_shader;
	geometry_shader_setup_func = std::move(other.geometry_shader_setup_func);
	fragment_shader_setup_func = std::move(other.geometry_shader_setup_func);
	
	//Shader pipeline should not be assigned "as is", otherwise the object receiving the assignment
	//would end up referring to the shader programs belonging to the object being assigned
	//shader_pipeline = std::move(other.shader_pipeline);
	shader_pipeline.attach(SEPARATE_SHADER_PROGRAM_CAST(retrieveShaderProgram(fixed_vp_ref_code)));
	setup_shader_pipeline(getActiveRenderingMode());

	texture_ref_code = std::move(other.texture_ref_code);
	texture_sampler_ref_code = std::move(other.texture_sampler_ref_code);

	ogl_vertex_array_object_id = other.ogl_vertex_array_object_id;
	ogl_data_buf = other.ogl_data_buf;
	other.ogl_vertex_array_object_id = 0;
	other.ogl_data_buf = 0;

	return *this;
}


FullscreenRectangle::~FullscreenRectangle()
{
	if (ogl_vertex_array_object_id)
		glDeleteVertexArrays(1, &ogl_vertex_array_object_id);

	if (ogl_data_buf)
		glDeleteBuffers(1, &ogl_data_buf);
}


void FullscreenRectangle::setDimensions(float width, float height)
{
	this->width = width;
	this->height = height;
	update_data();
}


std::pair<float, float> FullscreenRectangle::getDimensions() const
{
	return std::make_pair(width, height);
}


void FullscreenRectangle::installTexture(const ImmutableTexture2D& _2d_texture)
{
	if (!texture_ref_code)
		texture_ref_code = registerTexture(_2d_texture, texture_sampler_ref_code);
	else
		updateTexture(texture_ref_code, _2d_texture, texture_sampler_ref_code);
}


void FullscreenRectangle::installSampler(const TextureSampler& sampler)
{
	*retrieveTextureSampler(texture_sampler_ref_code) = sampler;
}


void FullscreenRectangle::setDynamicEffect(const SeparateShaderProgram& geometry_shader, std::function<bool(const AbstractProjectingDevice&, const AbstractRenderingDevice&, int)> shader_setup_func)
{
	p_user_defined_geometry_shader = &geometry_shader;
	geometry_shader_setup_func = shader_setup_func;
}


void FullscreenRectangle::setFilterEffect(const SeparateShaderProgram& fragment_shader, std::function<bool(const AbstractProjectingDevice&, const AbstractRenderingDevice&, int)> shader_setup_func)
{
	p_user_defined_fragment_shader = &fragment_shader;
	fragment_shader_setup_func = shader_setup_func;
}


bool FullscreenRectangle::supportsRenderingMode(uint32_t rendering_mode) const
{
	//Three rendering modes are supported
	//mode 0: default rendering, user-defined filtering effect is enabled if provided
	//mode 1: dynamic effect mode, user defined dynamic rendering is enabled if provided. User-defined filtering effect is also enabled if provided.
	//mode 2: simplified rendering. Neither user-defined dynamic effect nor user-defined filtering effect is used.
	if (rendering_mode == TW_RENDERING_MODE_DEFAULT || 
		rendering_mode == TW_RENDERING_MODE_FILTER_AND_DYNAMICS || 
		rendering_mode == TW_RENDERING_MODE_SIMPLIFIED)
		return true;
	else
		return false;
}


void FullscreenRectangle::configureViewProjectionTransform(const AbstractProjectingDevice& projecting_device)
{
	p_projecting_device = &projecting_device;
	mat4 mvp = projecting_device.getProjectionTransform() * getObjectTransform() * getObjectScaleTransform();
	retrieveShaderProgram(fixed_vp_ref_code)->assignUniformMatrix("mvp", mvp);
}


uint32_t FullscreenRectangle::getNumberOfRenderingPasses(uint32_t rendering_mode) const
{
	//This object needs to only 1 rendering pass to get rendered properly
	return 1;
}


bool FullscreenRectangle::configureRendering(AbstractRenderingDevice& render_target, uint32_t rendering_pass)
{
	//If requested render target is not yet active, activate it
	if (!render_target.isActive())
		render_target.makeActive();

	if (rendering_pass > 0) return false;

	setup_shader_pipeline(getActiveRenderingMode());		//configure the shader pipeline

	glBindVertexArray(ogl_vertex_array_object_id);		//bind vertex array object to the context

	if (texture_ref_code)
		bindTexture(texture_ref_code);

	if (getActiveRenderingMode() != TW_RENDERING_MODE_SIMPLIFIED && p_user_defined_fragment_shader)
		const_cast<SeparateShaderProgram*>(p_user_defined_fragment_shader)->assignUniformScalar("source0", getBindingUnit(texture_ref_code));
	else
		retrieveShaderProgram(fixed_fp_ref_code)->assignUniformScalar("source0", getBindingUnit(texture_ref_code));


	//Call configuration callback functions to setup the user defined shader programs.
	//Note that callback functions should be called the last so that they can actually affect the state of the engine and their effect will
	//not get "overwritten" by object pre-rendering configuration routines

	if (getActiveRenderingMode() == TW_RENDERING_MODE_FILTER_AND_DYNAMICS && 
		p_user_defined_geometry_shader && !geometry_shader_setup_func(*p_projecting_device, render_target, getNumberOfTextures())) return false;


	if ((getActiveRenderingMode() == TW_RENDERING_MODE_FILTER_AND_DYNAMICS || getActiveRenderingMode() == TW_RENDERING_MODE_DEFAULT) &&
		p_user_defined_fragment_shader && !fragment_shader_setup_func(*p_projecting_device, render_target, getNumberOfTextures())) return false;

	return true;
}


bool FullscreenRectangle::render()
{
	shader_pipeline.bind();	//bind shader pipeline

	switch (getActiveRenderingMode())
	{
	case TW_RENDERING_MODE_DEFAULT:
		shader_pipeline.activate_program(retrieveShaderProgram(fixed_vp_ref_code)->getId());		//apply vertex program settings to the pipeline

		//apply fragment program settings to the pipeline
		if (p_user_defined_fragment_shader)		
			shader_pipeline.activate_program(p_user_defined_fragment_shader->getId());
		else
			shader_pipeline.activate_program(retrieveShaderProgram(fixed_fp_ref_code)->getId());

		break;

	case TW_RENDERING_MODE_FILTER_AND_DYNAMICS:
		//apply vertex program setting to the pipeline
		shader_pipeline.activate_program(retrieveShaderProgram(fixed_vp_ref_code)->getId());

		//apply geometry program settings to the pipeline
		if (p_user_defined_geometry_shader) 
			shader_pipeline.activate_program(p_user_defined_geometry_shader->getId());

		//apply fragment program settings to the pipeline
		if (p_user_defined_fragment_shader)
			shader_pipeline.activate_program(p_user_defined_fragment_shader->getId());
		else
			shader_pipeline.activate_program(retrieveShaderProgram(fixed_fp_ref_code)->getId());

		break;

	case TW_RENDERING_MODE_SIMPLIFIED:
		//apply vertex program settings to the pipeline
		shader_pipeline.activate_program(retrieveShaderProgram(fixed_vp_ref_code)->getId());

		//apply fragment program settings to the pipeline
		shader_pipeline.activate_program(retrieveShaderProgram(fixed_fp_ref_code)->getId());

		break;

	default:
		return false;
	}
	

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	return true;
}


bool FullscreenRectangle::configureRenderingFinalization()
{
	return true;
}

