#include "TextureSampler.h"

using namespace tiny_world;

bool SamplerWrapping::operator==(const SamplerWrapping& other) const
{
	return wrap_mode_s == other.wrap_mode_s && wrap_mode_t == other.wrap_mode_t && wrap_mode_r == other.wrap_mode_r;
}


void TextureSampler_core::init_sampler()
{
	min_filter = SamplerMinificationFilter::NEAREST;
	mag_filter = SamplerMagnificationFilter::NEAREST;
	boundary_resolution_mode.wrap_mode_r = SamplerWrappingMode::CLAMP_TO_EDGE;
	boundary_resolution_mode.wrap_mode_s = SamplerWrappingMode::CLAMP_TO_EDGE;
	boundary_resolution_mode.wrap_mode_t = SamplerWrappingMode::CLAMP_TO_EDGE;
	border_color = vec4{ 0.0f };

	glSamplerParameteri(ogl_sampler_id, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(min_filter));
	glSamplerParameteri(ogl_sampler_id, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(mag_filter));
	glSamplerParameteri(ogl_sampler_id, GL_TEXTURE_WRAP_R, static_cast<GLint>(boundary_resolution_mode.wrap_mode_r));
	glSamplerParameteri(ogl_sampler_id, GL_TEXTURE_WRAP_S, static_cast<GLint>(boundary_resolution_mode.wrap_mode_s));
	glSamplerParameteri(ogl_sampler_id, GL_TEXTURE_WRAP_T, static_cast<GLint>(boundary_resolution_mode.wrap_mode_t));
	glSamplerParameterfv(ogl_sampler_id, GL_TEXTURE_BORDER_COLOR, border_color.getDataAsArray());
}

TextureSampler_core::TextureSampler_core() : Entity("TextureSampler")
{
	glGenSamplers(1, &ogl_sampler_id);
	init_sampler();
}

TextureSampler_core::TextureSampler_core(const TextureSampler_core& other) : Entity(other),
min_filter(other.min_filter), mag_filter(other.mag_filter),
boundary_resolution_mode(other.boundary_resolution_mode), border_color{ other.border_color }
{
	glGenSamplers(1, &ogl_sampler_id);

	glSamplerParameteri(ogl_sampler_id, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(min_filter));
	glSamplerParameteri(ogl_sampler_id, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(mag_filter));
	glSamplerParameteri(ogl_sampler_id, GL_TEXTURE_WRAP_R, static_cast<GLint>(boundary_resolution_mode.wrap_mode_r));
	glSamplerParameteri(ogl_sampler_id, GL_TEXTURE_WRAP_S, static_cast<GLint>(boundary_resolution_mode.wrap_mode_s));
	glSamplerParameteri(ogl_sampler_id, GL_TEXTURE_WRAP_T, static_cast<GLint>(boundary_resolution_mode.wrap_mode_t));
	glSamplerParameterfv(ogl_sampler_id, GL_TEXTURE_BORDER_COLOR, border_color.getDataAsArray());
}

TextureSampler_core::TextureSampler_core(TextureSampler_core&& other) : Entity(std::move(other)),
boundary_resolution_mode(other.boundary_resolution_mode),
mag_filter(other.mag_filter), min_filter(other.min_filter), border_color{ other.border_color }
{
	ogl_sampler_id = other.ogl_sampler_id;
	other.ogl_sampler_id = 0;
}

TextureSampler_core::TextureSampler_core(const std::string& sampler_string_name) : 
Entity("TextureSampler", sampler_string_name)
{
	glGenSamplers(1, &ogl_sampler_id);
	init_sampler();
}

TextureSampler_core::~TextureSampler_core()
{
	if (ogl_sampler_id)
		glDeleteSamplers(1, &ogl_sampler_id);
}

void TextureSampler_core::setMinFilter(SamplerMinificationFilter min_filter)
{
	glSamplerParameteri(ogl_sampler_id, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(min_filter));
	this->min_filter = min_filter;
}

void TextureSampler_core::setMagFilter(SamplerMagnificationFilter mag_filter)
{
	glSamplerParameteri(ogl_sampler_id, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(mag_filter));
	this->mag_filter = mag_filter;
}

void TextureSampler_core::setWrapping(SamplerWrapping wrapping)
{
	glSamplerParameteri(ogl_sampler_id, GL_TEXTURE_WRAP_R, static_cast<GLint>(wrapping.wrap_mode_r));
	glSamplerParameteri(ogl_sampler_id, GL_TEXTURE_WRAP_S, static_cast<GLint>(wrapping.wrap_mode_s));
	glSamplerParameteri(ogl_sampler_id, GL_TEXTURE_WRAP_T, static_cast<GLint>(wrapping.wrap_mode_t));

	this->boundary_resolution_mode = wrapping;
}

void TextureSampler_core::setBorderColor(const vec4& border_color)
{
	this->border_color = border_color;
	glSamplerParameterfv(ogl_sampler_id, GL_TEXTURE_BORDER_COLOR, std::array < float, 4U > {{border_color.x, border_color.y, border_color.z, border_color.w}}.data());
}

SamplerMinificationFilter TextureSampler_core::getMinFilter() const { return min_filter; }

SamplerMagnificationFilter TextureSampler_core::getMagFilter() const { return mag_filter; }

SamplerWrapping TextureSampler_core::getWrapping() const { return boundary_resolution_mode; }

vec4 TextureSampler_core::getBorderColor() const { return border_color; }

GLuint TextureSampler_core::getOpenGLId() const { return ogl_sampler_id; }

TextureSampler_core& TextureSampler_core::operator=(const TextureSampler_core& other)
{
	if (this == &other)
		return *this;

	Entity::operator=(other);

	min_filter = other.min_filter;
	mag_filter = other.mag_filter;
	boundary_resolution_mode = other.boundary_resolution_mode;
	border_color = other.border_color;

	glSamplerParameteri(ogl_sampler_id, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(min_filter));
	glSamplerParameteri(ogl_sampler_id, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(mag_filter));
	glSamplerParameteri(ogl_sampler_id, GL_TEXTURE_WRAP_R, static_cast<GLint>(boundary_resolution_mode.wrap_mode_r));
	glSamplerParameteri(ogl_sampler_id, GL_TEXTURE_WRAP_S, static_cast<GLint>(boundary_resolution_mode.wrap_mode_s));
	glSamplerParameteri(ogl_sampler_id, GL_TEXTURE_WRAP_T, static_cast<GLint>(boundary_resolution_mode.wrap_mode_t));
	glSamplerParameterfv(ogl_sampler_id, GL_TEXTURE_BORDER_COLOR, border_color.getDataAsArray());

	return *this;
}

TextureSampler_core& TextureSampler_core::operator=(TextureSampler_core&& other)
{
	if (this == &other)
		return *this;

	Entity::operator=(std::move(other));

	std::swap(ogl_sampler_id, other.ogl_sampler_id);

	min_filter = other.min_filter;
	mag_filter = other.mag_filter;
	boundary_resolution_mode = other.boundary_resolution_mode;
	border_color = other.border_color;

	return *this;
}

bool TextureSampler_core::operator==(const TextureSampler_core& other) const
{
	return min_filter == other.min_filter && mag_filter == other.mag_filter &&
		boundary_resolution_mode == other.boundary_resolution_mode && border_color == other.border_color;
}


//*******************************************Sampler interface infrastructure***********************************************

TextureSampler::TextureSampler() : TextureSampler_core() {}

TextureSampler::TextureSampler(const TextureSampler& other) : TextureSampler_core(other) {}

TextureSampler::TextureSampler(TextureSampler&& other) : TextureSampler_core(std::move(other)) {}

TextureSampler& TextureSampler::operator=(const TextureSampler& other)
{
	TextureSampler_core::operator=(other);
	return *this;
}

TextureSampler& TextureSampler::operator=(TextureSampler&& other)
{
	TextureSampler_core::operator=(std::move(other));
	return *this;
}

TextureSampler::TextureSampler(const std::string& sampler_string_name) : TextureSampler_core(sampler_string_name) {}

TextureSampler::~TextureSampler() {}


GLuint TextureSampler::bind(GLuint texture_unit) const
{
	GLint ogl_current_texture_unit;
	glGetIntegerv(GL_ACTIVE_TEXTURE, &ogl_current_texture_unit);	//Get currently active texture unit

	if (ogl_current_texture_unit != GL_TEXTURE0 + texture_unit)
		glActiveTexture(GL_TEXTURE0 + texture_unit);	//Activate requested unit

	GLint ogl_current_sampler_in_requested_unit;
	glGetIntegerv(GL_SAMPLER_BINDING, &ogl_current_sampler_in_requested_unit);

	if (ogl_current_texture_unit != GL_TEXTURE0 + texture_unit)
		glActiveTexture(ogl_current_texture_unit);	//Restore previously active texture unit

	glBindSampler(texture_unit, getOpenGLId());	//Bind contained sampler to requested texture unit

	return ogl_current_sampler_in_requested_unit;
}