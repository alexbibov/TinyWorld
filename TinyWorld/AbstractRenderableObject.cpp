#include "AbstractRenderableObject.h"
#include <cmath>

using namespace tiny_world;




AbstractRenderableObject::ShaderProgramReferenceCode::ShaderProgramReferenceCode() : first{ -1 }, second{ -1 } {}

AbstractRenderableObject::ShaderProgramReferenceCode::ShaderProgramReferenceCode(int first, int second) : first{ first }, second{ second } {}

AbstractRenderableObject::ShaderProgramReferenceCode::operator bool() const { return first != -1 && second != -1; }




AbstractRenderableObject::ShaderProgramIterator::ShaderProgramIterator() : p_complete_programs(nullptr), p_separate_programs(nullptr), position(0) {}

AbstractRenderableObject::ShaderProgramIterator::ShaderProgramIterator(complete_shader_program_list& complete_rendering_programs,
	separate_shader_program_list& separate_rendering_programs, uint32_t position) :
	p_complete_programs(&complete_rendering_programs), p_separate_programs(&separate_rendering_programs), position(position) {}

AbstractRenderableObject::ShaderProgramIterator& AbstractRenderableObject::ShaderProgramIterator::operator++()
{
	++position;
	return *this;
}

AbstractRenderableObject::ShaderProgramIterator AbstractRenderableObject::ShaderProgramIterator::operator++(int)
{
	ShaderProgramIterator tmp(*this);
	position++;
	return tmp;
}

AbstractRenderableObject::ShaderProgramIterator& AbstractRenderableObject::ShaderProgramIterator::operator--()
{
	--position;
	return *this;
}

AbstractRenderableObject::ShaderProgramIterator AbstractRenderableObject::ShaderProgramIterator::operator--(int)
{
	ShaderProgramIterator tmp(*this);
	position--;
	return tmp;
}

ShaderProgram& AbstractRenderableObject::ShaderProgramIterator::operator*()
{
	if (position >= 0 && position < p_complete_programs->size())
		return p_complete_programs->at(position);

	if (position >= p_complete_programs->size() && position < p_separate_programs->size())
		return p_separate_programs->at(position - static_cast<uint32_t>(p_complete_programs->size()));

	throw std::out_of_range{ "ShaderProgramIterator out-of-range" };
}

ShaderProgram* AbstractRenderableObject::ShaderProgramIterator::operator->()
{
	if (position >= 0 && position < p_complete_programs->size())
		return &p_complete_programs->at(position);

	if (position >= p_complete_programs->size() && position < p_separate_programs->size())
		return &p_separate_programs->at(position);

	return nullptr;
}

bool AbstractRenderableObject::ShaderProgramIterator::operator==(const ShaderProgramIterator& other) const
{
	return p_complete_programs == other.p_complete_programs && p_separate_programs == other.p_separate_programs && position == other.position;
}

bool AbstractRenderableObject::ShaderProgramIterator::operator!=(const ShaderProgramIterator& other) const
{
	return !(*this == other);
}





AbstractRenderableObject::ShaderProgramIterator AbstractRenderableObject::ShaderProgramListBegin()
{
	return ShaderProgramIterator{ complete_rendering_programs, separate_rendering_programs, 0 };
}

AbstractRenderableObject::ShaderProgramIterator AbstractRenderableObject::ShaderProgramListEnd()
{
	return ShaderProgramIterator{ complete_rendering_programs, separate_rendering_programs, static_cast<uint32_t>(complete_rendering_programs.size() + separate_rendering_programs.size()) };
}

AbstractRenderableObject::AbstractRenderableObject() : Entity("AbstractRenderableObject"),
default_location{ 0 }, location{ 0 }, scale_factors{ 1.0f },
default_object_transform{ vec3{ 1, 0, 0 }, vec3{ 0, 1, 0 }, vec3{ 0, 0, 1 } }, object_transform{ vec3{ 1, 0, 0 }, vec3{ 0, 1, 0 }, vec3{ 0, 0, 1 } },
rendering_mode{ 0 }, screen_size{ 0 }
{

}

AbstractRenderableObject::AbstractRenderableObject(const std::string& renderable_object_class_string_name) :
Entity(renderable_object_class_string_name),
default_location{ 0 }, location{ 0 }, scale_factors{ 1.0f },
default_object_transform{ vec3{ 1, 0, 0 }, vec3{ 0, 1, 0 }, vec3{ 0, 0, 1 } }, object_transform{ vec3{ 1, 0, 0 }, vec3{ 0, 1, 0 }, vec3{ 0, 0, 1 } },
rendering_mode{ 0 }, screen_size{ 0 }
{

}

AbstractRenderableObject::AbstractRenderableObject(const std::string& renderable_object_class_string_name, const std::string& object_string_name) :
Entity(renderable_object_class_string_name, object_string_name),
default_location{ 0 }, location{ 0 }, scale_factors{ 1.0f },
default_object_transform{ vec3{ 1, 0, 0 }, vec3{ 0, 1, 0 }, vec3{ 0, 0, 1 } }, object_transform{ vec3{ 1, 0, 0 }, vec3{ 0, 1, 0 }, vec3{ 0, 0, 1 } },
rendering_mode{ 0 }, screen_size{ 0 }
{

}

AbstractRenderableObject::AbstractRenderableObject(const std::string& renderable_object_class_string_name, const std::string& object_string_name, const vec3& location) :
Entity(renderable_object_class_string_name, object_string_name),
default_location{ location }, location{ location }, scale_factors{ 1.0f },
default_object_transform{ vec3{ 1, 0, 0 }, vec3{ 0, 1, 0 }, vec3{ 0, 0, 1 } }, object_transform{ vec3{ 1, 0, 0 }, vec3{ 0, 1, 0 }, vec3{ 0, 0, 1 } },
rendering_mode{ 0 }, screen_size{ 0 }
{

}

AbstractRenderableObject::AbstractRenderableObject(const std::string& renderable_object_class_string_name, const std::string& object_string_name, const vec3& location,
	float x_rot_angle, float y_rot_angle, float z_rot_angle) :
	Entity(renderable_object_class_string_name, object_string_name),
	default_location{ location }, location{ location }, scale_factors{ 1.0f },
	default_object_transform{
	mat3{ vec3{ 1, 0, 0 }, vec3{ 0, std::cos(x_rot_angle), std::sin(x_rot_angle) }, vec3{ 0, -std::sin(x_rot_angle), std::cos(x_rot_angle) } }*
	mat3{ vec3{ std::cos(y_rot_angle), 0, -std::sin(y_rot_angle) }, vec3{ 0, 1, 0 }, vec3{ std::sin(y_rot_angle), 0, std::cos(y_rot_angle) } }*
	mat3{ vec3{ std::cos(z_rot_angle), std::sin(z_rot_angle), 0 }, vec3{ -std::sin(z_rot_angle), std::cos(z_rot_angle), 0 }, vec3{ 0, 0, 1 } }
	},
	object_transform{ vec3{ 1, 0, 0 }, vec3{ 0, 1, 0 }, vec3{ 0, 0, 1 } },
	rendering_mode{ 0 }, screen_size{ 0 }
{
	rotateX(x_rot_angle, RotationFrame::LOCAL);
	rotateY(y_rot_angle, RotationFrame::LOCAL);
	rotateZ(z_rot_angle, RotationFrame::LOCAL);

}

AbstractRenderableObject::AbstractRenderableObject(const AbstractRenderableObject& other) :
Entity(other),
default_location{ other.default_location }, location{ other.location }, scale_factors{ other.scale_factors },
default_object_transform{other.default_object_transform}, object_transform{ other.object_transform }, rendering_mode{ other.rendering_mode }, screen_size{ other.screen_size },
complete_rendering_programs(other.complete_rendering_programs),
separate_rendering_programs(other.separate_rendering_programs)
{

}

AbstractRenderableObject::AbstractRenderableObject(AbstractRenderableObject&& other) :
Entity(std::move(other)),
default_location{ other.default_location }, location{ std::move(other.location) },
scale_factors{ std::move(other.scale_factors) },
default_object_transform{std::move(other.default_object_transform)}, object_transform{ std::move(other.object_transform) },
rendering_mode{ other.rendering_mode }, screen_size{ std::move(other.screen_size) },
complete_rendering_programs(std::move(other.complete_rendering_programs)),
separate_rendering_programs(std::move(other.separate_rendering_programs))
{

}

AbstractRenderableObject::~AbstractRenderableObject()
{

}

AbstractRenderableObject& AbstractRenderableObject::operator=(const AbstractRenderableObject& other)
{
	if (this == &other)
		return *this;

	Entity::operator=(other);

	location = other.location;
	scale_factors = other.scale_factors;
	object_transform = other.object_transform;
	rendering_mode = other.rendering_mode;
	screen_size = other.screen_size;

	complete_rendering_programs = other.complete_rendering_programs;
	separate_rendering_programs = other.separate_rendering_programs;

	return *this;
}

AbstractRenderableObject& AbstractRenderableObject::operator=(AbstractRenderableObject&& other)
{
	if (this == &other)
		return *this;

	Entity::operator=(std::move(other));

	location = std::move(other.location);
	scale_factors = std::move(other.scale_factors);
	object_transform = std::move(other.object_transform);
	rendering_mode = other.rendering_mode;
	screen_size = std::move(other.screen_size);

	complete_rendering_programs = std::move(other.complete_rendering_programs);
	separate_rendering_programs = std::move(other.separate_rendering_programs);

	return *this;
}

AbstractRenderableObject::ShaderProgramReferenceCode AbstractRenderableObject::createCompleteShaderProgram(const std::string& program_string_name, std::initializer_list<PipelineStage> program_stages)
{
	uint32_t id = static_cast<uint32_t>(complete_rendering_programs.size());
	complete_rendering_programs.insert(std::move(std::make_pair(id, CompleteShaderProgram{ program_string_name })));
	ShaderProgramReferenceCode program_ref_code{ static_cast<int>(id), static_cast<int>(supported_shader_program_type::complete) };
	return injectExtension(program_ref_code, program_stages) ? program_ref_code : ShaderProgramReferenceCode{};
}

AbstractRenderableObject::ShaderProgramReferenceCode AbstractRenderableObject::createSeparateShaderProgram(const std::string& program_string_name, std::initializer_list<PipelineStage> program_stages)
{
	uint32_t id = static_cast<uint32_t>(separate_rendering_programs.size());
	separate_rendering_programs.insert(std::make_pair(id, SeparateShaderProgram{ program_string_name }));
	ShaderProgramReferenceCode program_ref_code{ static_cast<int>(id), static_cast<int>(supported_shader_program_type::separate) };
	return injectExtension(program_ref_code, program_stages) ? program_ref_code : ShaderProgramReferenceCode{};
}

ShaderProgram* AbstractRenderableObject::retrieveShaderProgram(const ShaderProgramReferenceCode& shader_program_ref_code)
{
	switch (static_cast<supported_shader_program_type>(shader_program_ref_code.second))
	{
	case supported_shader_program_type::complete:
		return shader_program_ref_code.first >= 0 && shader_program_ref_code.first < static_cast<int>(complete_rendering_programs.size()) ? &complete_rendering_programs.at(shader_program_ref_code.first) : nullptr;

	case supported_shader_program_type::separate:
		return shader_program_ref_code.first >= 0 && shader_program_ref_code.first < static_cast<int>(separate_rendering_programs.size()) ? &separate_rendering_programs.at(shader_program_ref_code.first) : nullptr;

	default:
		return nullptr;
	}
}

bool AbstractRenderableObject::updateShaderProgram(const ShaderProgramReferenceCode& shader_program_ref_code, const ShaderProgram& new_shader_program)
{
	if (new_shader_program.isSeparate())
	{
		if (shader_program_ref_code.first >= static_cast<int>(separate_rendering_programs.size()) || shader_program_ref_code.first < 0 ||
			static_cast<supported_shader_program_type>(shader_program_ref_code.second) != supported_shader_program_type::separate)
			return false;

		separate_rendering_programs.at(shader_program_ref_code.first) = dynamic_cast<const SeparateShaderProgram&>(new_shader_program);
		return true;
	}
	else
	{
		if (shader_program_ref_code.first >= static_cast<int>(complete_rendering_programs.size()) || shader_program_ref_code.first < 0 ||
			static_cast<supported_shader_program_type>(shader_program_ref_code.second) != supported_shader_program_type::complete)
			return false;

		complete_rendering_programs.at(shader_program_ref_code.first) = dynamic_cast<const CompleteShaderProgram&>(new_shader_program);
		return true;
	}
}

vec3 AbstractRenderableObject::getLocation() const { return location; }

mat4 AbstractRenderableObject::getObjectTransform() const
{
	mat4 combined_transform{ object_transform[0][0], object_transform[1][0], object_transform[2][0], 0,
		object_transform[0][1], object_transform[1][1], object_transform[2][1], 0,
		object_transform[0][2], object_transform[1][2], object_transform[2][2], 0,
		location[0], location[1], location[2], 1 };

	return combined_transform;
}

mat4 AbstractRenderableObject::getObjectScaleTransform() const
{
	return mat4{ scale_factors.x, 0.0f, 0.0f, 0.0f,
		0.0f, scale_factors.y, 0.0f, 0.0f,
		0.0f, 0.0f, scale_factors.z, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f };
}

vec3 AbstractRenderableObject::getObjectScale() const{ return scale_factors; }

void AbstractRenderableObject::setLocation(const vec3& new_location)
{
	location = new_location;
}

void AbstractRenderableObject::rotateX(float angle, RotationFrame frame)
{
	float c = std::cos(angle);
	float s = std::sin(angle);
	mat3 rx{ vec3{ 1, 0, 0 }, vec3{ 0, c, s }, vec3{ 0, -s, c } };

	switch (frame)
	{
	case RotationFrame::LOCAL:
		object_transform = object_transform * rx;
		break;

	case RotationFrame::GLOBAL:
		object_transform = rx * object_transform;
		break;
	}
}

void AbstractRenderableObject::rotateY(float angle, RotationFrame frame)
{
	float c = std::cos(angle);
	float s = std::sin(angle);
	mat3 ry = mat3{ vec3{ c, 0, -s }, vec3{ 0, 1, 0 }, vec3{ s, 0, c } };

	switch (frame)
	{
	case RotationFrame::LOCAL:
		object_transform = object_transform * ry;
		break;

	case RotationFrame::GLOBAL:
		object_transform = ry * object_transform;
		break;
	}
}

void AbstractRenderableObject::rotateZ(float angle, RotationFrame frame)
{
	float c = std::cos(angle);
	float s = std::sin(angle);
	mat3 rz = mat3{ vec3{ c, s, 0 }, vec3{ -s, c, 0 }, vec3{ 0, 0, 1 } };

	switch (frame)
	{
	case RotationFrame::LOCAL:
		object_transform = object_transform * rz;
		break;

	case RotationFrame::GLOBAL:
		object_transform = rz * object_transform;
		break;
	}
}

void AbstractRenderableObject::rotate(const vec3& axis, float angle, RotationFrame frame)
{
	float c = std::cos(angle);
	float s = std::sin(angle);
	mat3 raxis = mat3{ vec3{ c + (1 - c)*axis.x*axis.x, (1 - c)*axis.x*axis.y + s*axis.z, (1 - c)*axis.x*axis.z - s*axis.y },
		vec3{ (1 - c)*axis.x*axis.y - s*axis.z, c + (1 - c)*axis.y*axis.y, (1 - c)*axis.y*axis.z + s*axis.x },
		vec3{ (1 - c)*axis.x*axis.z + s*axis.y, (1 - c)*axis.y*axis.z - s*axis.x, c + (1 - c)*axis.z*axis.z } };

	switch (frame)
	{
	case RotationFrame::LOCAL:
		object_transform = object_transform * raxis;
		break;

	case RotationFrame::GLOBAL:
		object_transform = raxis * object_transform;
		break;
	}
}

void AbstractRenderableObject::translate(const vec3& translation)
{
	location = location + translation;
}

void AbstractRenderableObject::scale(float x_scale_factor, float y_scale_factor, float z_scale_factor)
{
	scale_factors = vec3{ scale_factors.x * x_scale_factor, scale_factors.y * y_scale_factor, scale_factors.z * z_scale_factor };
}

void AbstractRenderableObject::scale(const vec3& new_scale_factors)
{
	scale_factors.x *= new_scale_factors.x;
	scale_factors.y *= new_scale_factors.y;
	scale_factors.z *= new_scale_factors.z;
}

void AbstractRenderableObject::apply3DTransform(const mat3& _3d_transform_matrix, RotationFrame frame)
{
	switch (frame)
	{
	case RotationFrame::LOCAL:
		object_transform = object_transform*_3d_transform_matrix;
		break;

	case RotationFrame::GLOBAL:
		object_transform = _3d_transform_matrix*object_transform;
		break;
	}
}

void AbstractRenderableObject::applyRotation(const quaternion& q, RotationFrame frame)
{
	switch (frame)
	{
	case RotationFrame::LOCAL:
		object_transform = object_transform * q.getRotationMatrix();
		break;
	case RotationFrame::GLOBAL:
		object_transform = q.getRotationMatrix() * object_transform;
		break;
	}
}

void AbstractRenderableObject::resetObjectTransform()
{
	location = default_location;
	object_transform = default_object_transform;
}

void AbstractRenderableObject::resetObjectRotation()
{
	object_transform = default_object_transform;
}

void AbstractRenderableObject::resetObjectLocation()
{
	location = default_location;
}


uint32_t AbstractRenderableObject::selectRenderingMode(uint32_t new_rendering_mode)
{
	uint32_t rv = rendering_mode;
	rendering_mode = new_rendering_mode;
	return rv;
}


uint32_t AbstractRenderableObject::getActiveRenderingMode() const { return rendering_mode; }


void AbstractRenderableObject::setScreenSize(const uvec2& screen_size)
{
	this->screen_size = screen_size;
	applyScreenSize(screen_size);
}


void AbstractRenderableObject::setScreenSize(uint32_t width, uint32_t height)
{
	screen_size.x = width;
	screen_size.y = height;
	applyScreenSize(screen_size);
}


uvec2 AbstractRenderableObject::getScreenSize() const { return screen_size; }


bool AbstractRenderableObject::prepareRendering(AbstractRenderingDevice& render_target, uint32_t rendering_pass)
{
	applyExtension();
	return configureRendering(render_target, rendering_pass);
}


void AbstractRenderableObject::applyViewProjectionTransform(const AbstractProjectingDevice& projecting_device)
{
	applyViewerTransform(projecting_device);
	configureViewProjectionTransform(projecting_device);
}


bool AbstractRenderableObject::finalizeRendering()
{
	if (!configureRenderingFinalization()) return false;
	releaseExtension();
	return true;
}




AbstractRenderableObjectTextured::TextureReferenceCode::TextureReferenceCode() : first{ -1 }, second{ -1 } {}
AbstractRenderableObjectTextured::TextureReferenceCode::TextureReferenceCode(int first, int second) : first{ first }, second{ second } {}
AbstractRenderableObjectTextured::TextureReferenceCode::operator bool() const { return first >= 0 && second >= 0 && second <= 3; }


AbstractRenderableObjectTextured::TextureSamplerReferenceCode::TextureSamplerReferenceCode() : first{ -1 } {}
AbstractRenderableObjectTextured::TextureSamplerReferenceCode::TextureSamplerReferenceCode(int first) : first{ first } {}
AbstractRenderableObjectTextured::TextureSamplerReferenceCode::operator bool() const { return first >= 0; }
int AbstractRenderableObjectTextured::TextureSamplerReferenceCode::getFirst() const { return first; }




TextureUnitBlock* AbstractRenderableObjectTextured::p_texture_unit_block = nullptr;
std::string AbstractRenderableObjectTextured::texture_lookup_path = "tw_textures/";

TextureUnitBlock* AbstractRenderableObjectTextured::getTextureUnitBlockPointer() { return AbstractRenderableObjectTextured::p_texture_unit_block; }

void AbstractRenderableObjectTextured::defineTextureUnitBlockGlobalPointer(TextureUnitBlock* p_texture_unit_block)
{
	AbstractRenderableObjectTextured::p_texture_unit_block = p_texture_unit_block;
}

void AbstractRenderableObjectTextured::defineTextureLookupPath(const std::string& new_texture_lookup_path)
{
	AbstractRenderableObjectTextured::texture_lookup_path = new_texture_lookup_path;
}

std::string AbstractRenderableObjectTextured::getTextureLookupPath() { return AbstractRenderableObjectTextured::texture_lookup_path; }

AbstractRenderableObjectTextured::supported_texture_type AbstractRenderableObjectTextured::getTextureType(const Texture& texture)
{
	//If the texture has 6 faces, then this is a cubemap
	if (texture.getNumberOfFaces() == 6)
		return supported_texture_type::cubemap;

	//if the texture has 1 face and its dimension is equal to 1, then it is an immutable 1D texture
	if (texture.getNumberOfFaces() == 1 && texture.getDimension() == TextureDimension::_1D)
		return supported_texture_type::_1d_texture;

	//If the texture has dimension of 2 and 1 face, then it is an immutable 2D texture
	if (texture.getNumberOfFaces() == 1 && texture.getDimension() == TextureDimension::_2D)
		return supported_texture_type::_2d_texture;

	//if the texture has dimension of 3 and 1 face, then it is an immutable 3D texture
	if (texture.getNumberOfFaces() == 1 && texture.getDimension() == TextureDimension::_3D)
		return supported_texture_type::_3d_texture;

	//if the texture yields 'true' on isBufferTexture inquiry, then this is a buffer texture
	if (texture.isBufferTexture())
		return supported_texture_type::buffer_texture;

	return supported_texture_type::unsupported;
}

AbstractRenderableObjectTextured::AbstractRenderableObjectTextured() :
texture_unit_offset{ 0 }, texture_unit_counter{ texture_unit_offset }
{
	default_sampler_ref_code = createTextureSampler("AbstractRenderableObjectTextured::default_texture_sampler");
}

AbstractRenderableObjectTextured::AbstractRenderableObjectTextured(const AbstractRenderableObjectTextured& other) :
_1d_textures(other._1d_textures), _2d_textures(other._2d_textures), _3d_textures(other._3d_textures),
cubemap_textures(other.cubemap_textures), buffer_textures(other.buffer_textures),
samplers(other.samplers), texture_unit_offset{ other.texture_unit_offset }, texture_unit_counter{ other.texture_unit_counter },
default_sampler_ref_code(other.default_sampler_ref_code)
{

}

AbstractRenderableObjectTextured::AbstractRenderableObjectTextured(AbstractRenderableObjectTextured&& other) :
_1d_textures( std::move(other._1d_textures)), _2d_textures(std::move(other._2d_textures)), _3d_textures(std::move(other._3d_textures)),
cubemap_textures(std::move(other.cubemap_textures)), buffer_textures(std::move(other.buffer_textures)),
samplers(std::move(other.samplers)), texture_unit_offset{ other.texture_unit_offset }, texture_unit_counter{ other.texture_unit_counter },
default_sampler_ref_code{ std::move(other.default_sampler_ref_code) }
{

}

AbstractRenderableObjectTextured& AbstractRenderableObjectTextured::operator=(const AbstractRenderableObjectTextured& other)
{
	//Account for the case of "assignment to itself"
	if (this == &other)
		return *this;

	_1d_textures = other._1d_textures;
	_2d_textures = other._2d_textures;
	_3d_textures = other._3d_textures;
	cubemap_textures = other.cubemap_textures;
	buffer_textures = other.buffer_textures;
	samplers = other.samplers;
	default_sampler_ref_code = other.default_sampler_ref_code;


	texture_unit_offset = other.texture_unit_offset;
	texture_unit_counter = other.texture_unit_counter;

	return *this;
}

AbstractRenderableObjectTextured& AbstractRenderableObjectTextured::operator=(AbstractRenderableObjectTextured&& other)
{
	//Account for the case of "assignment to itself"
	if (this == &other)
		return *this;

	_1d_textures = std::move(other._1d_textures);
	_2d_textures = std::move(other._2d_textures);
	_3d_textures = std::move(other._3d_textures);
	cubemap_textures = std::move(other.cubemap_textures);
	buffer_textures = std::move(other.buffer_textures);
	samplers = std::move(other.samplers);
	default_sampler_ref_code = std::move(other.default_sampler_ref_code);


	texture_unit_offset = other.texture_unit_offset;
	texture_unit_counter = other.texture_unit_counter;

	return *this;
}

AbstractRenderableObjectTextured::~AbstractRenderableObjectTextured()
{

}

AbstractRenderableObjectTextured::TextureReferenceCode AbstractRenderableObjectTextured::registerTexture(const Texture& texture,
	TextureSamplerReferenceCode sampler_reference_code /* = TextureSamplerReferenceCode */)
{
	return registerTexture(texture, texture_unit_counter++, sampler_reference_code);
}

AbstractRenderableObjectTextured::TextureReferenceCode AbstractRenderableObjectTextured::registerTexture(const Texture& texture, uint32_t unit_binding_block,
	TextureSamplerReferenceCode sampler_reference_code /* = TextureSamplerReferenceCode */)
{
	//Determine the type of provided texture

	switch (getTextureType(texture))
	{
	case supported_texture_type::cubemap:
		cubemap_textures.push_back(texture_entry < ImmutableTextureCubeMap > {unit_binding_block,
			dynamic_cast<const ImmutableTextureCubeMap&>(texture), sampler_reference_code ? sampler_reference_code : default_sampler_ref_code});
		return TextureReferenceCode{ static_cast<int>(cubemap_textures.size()) - 1,
			static_cast<int>(supported_texture_type::cubemap) };

	case supported_texture_type::_1d_texture:
		_1d_textures.push_back(texture_entry < ImmutableTexture1D > {unit_binding_block,
			dynamic_cast<const ImmutableTexture1D&>(texture), sampler_reference_code ? sampler_reference_code : default_sampler_ref_code});
		return TextureReferenceCode{ static_cast<int>(_1d_textures.size()) - 1,
			static_cast<int>(supported_texture_type::_1d_texture) };

	case supported_texture_type::_2d_texture:
		_2d_textures.push_back(texture_entry < ImmutableTexture2D > {unit_binding_block,
			dynamic_cast<const ImmutableTexture2D&>(texture), sampler_reference_code ? sampler_reference_code : default_sampler_ref_code});
		return TextureReferenceCode{ static_cast<int>(_2d_textures.size()) - 1,
			static_cast<int>(supported_texture_type::_2d_texture) };

	case supported_texture_type::_3d_texture:
		_3d_textures.push_back(texture_entry < ImmutableTexture3D > {unit_binding_block,
			dynamic_cast<const ImmutableTexture3D&>(texture), sampler_reference_code ? sampler_reference_code : default_sampler_ref_code});
		return TextureReferenceCode{ static_cast<int>(_3d_textures.size()) - 1,
			static_cast<int>(supported_texture_type::_3d_texture) };

	case supported_texture_type::buffer_texture:
		buffer_textures.push_back(texture_entry < BufferTexture > {unit_binding_block,
			dynamic_cast<const BufferTexture&>(texture), sampler_reference_code ? sampler_reference_code : default_sampler_ref_code});
		return TextureReferenceCode{ static_cast<int>(buffer_textures.size()) - 1,
			static_cast<int>(supported_texture_type::buffer_texture) };

	default:
		return TextureReferenceCode{ -1, -1 };
	}
}

bool AbstractRenderableObjectTextured::updateTexture(TextureReferenceCode& texture_reference_code, const Texture& new_texture,
	TextureSamplerReferenceCode sampler_reference_code /* = TextureSamplerReferenceCode */)
{
	//check if the texture reference code for which the update operation has been requested is valid
	if (!texture_reference_code) return false;


	//check if the texture being replaced differs in type from the replacement texture
	supported_texture_type old_texture_type = static_cast<supported_texture_type>(texture_reference_code.second);
	supported_texture_type new_texture_type = getTextureType(new_texture);

	if (old_texture_type != new_texture_type)
	{
		int target_texture_unit;
		TextureSamplerReferenceCode old_sampler_ref_code;

		//Identify  texture binding unit and the sampler used by the texture being replaced
		switch (old_texture_type)
		{
		case supported_texture_type::cubemap:
			target_texture_unit = cubemap_textures[texture_reference_code.first].first;
			old_sampler_ref_code = cubemap_textures[texture_reference_code.first].third;
			break;

		case supported_texture_type::_1d_texture:
			target_texture_unit = _1d_textures[texture_reference_code.first].first;
			old_sampler_ref_code = _1d_textures[texture_reference_code.first].third;
			break;

		case supported_texture_type::_2d_texture:
			target_texture_unit = _2d_textures[texture_reference_code.first].first;
			old_sampler_ref_code = _2d_textures[texture_reference_code.first].third;
			break;

		case supported_texture_type::_3d_texture:
			target_texture_unit = _3d_textures[texture_reference_code.first].first;
			old_sampler_ref_code = _3d_textures[texture_reference_code.first].third;
			break;

		case supported_texture_type::buffer_texture:
			target_texture_unit = buffer_textures[texture_reference_code.first].first;
			old_sampler_ref_code = buffer_textures[texture_reference_code.first].third;
			break;

		default: return false;
		}


		//Check if the binding unit used by the texture being replaced is already in use by some other texture that has the same type as the replacement texture
		switch (new_texture_type)
		{
		case supported_texture_type::cubemap:
		{
			cubemap_texture_list::iterator target_texture_entry;
			if ((target_texture_entry = std::find_if(cubemap_textures.begin(), cubemap_textures.end(),
				[target_texture_unit](cubemap_texture_list::value_type elem) -> bool{return elem.first == target_texture_unit; })) != cubemap_textures.end())
			{
				target_texture_entry->second = dynamic_cast<const ImmutableTextureCubeMap&>(new_texture);
				target_texture_entry->third = sampler_reference_code ? sampler_reference_code : old_sampler_ref_code;
				texture_reference_code.first = static_cast<int>(target_texture_entry - cubemap_textures.begin());
			}
			else
			{
				cubemap_textures.push_back(texture_entry < ImmutableTextureCubeMap > {static_cast<uint32_t>(target_texture_unit),
					dynamic_cast<const ImmutableTextureCubeMap&>(new_texture), sampler_reference_code ? sampler_reference_code : old_sampler_ref_code});
				texture_reference_code.first = static_cast<int>(cubemap_textures.size() - 1);
			}
			texture_reference_code.second = static_cast<int>(supported_texture_type::cubemap);
			return true;
		}

		case supported_texture_type::_1d_texture:
		{
			_1d_texture_list::iterator target_texture_entry;
			if ((target_texture_entry = std::find_if(_1d_textures.begin(), _1d_textures.end(),
				[target_texture_unit](_1d_texture_list::value_type elem) -> bool{return elem.first == target_texture_unit; })) != _1d_textures.end())
			{
				target_texture_entry->second = dynamic_cast<const ImmutableTexture1D&>(new_texture);
				target_texture_entry->third = sampler_reference_code ? sampler_reference_code : old_sampler_ref_code;
				texture_reference_code.first = static_cast<int>(target_texture_entry - _1d_textures.begin());
			}
			else
			{
				_1d_textures.push_back(texture_entry < ImmutableTexture1D > {static_cast<uint32_t>(target_texture_unit),
					dynamic_cast<const ImmutableTexture1D&>(new_texture), sampler_reference_code ? sampler_reference_code : old_sampler_ref_code});
				texture_reference_code.first = static_cast<int>(_1d_textures.size() - 1);
			}
			texture_reference_code.second = static_cast<int>(supported_texture_type::_1d_texture);
			return true;
		}

		case supported_texture_type::_2d_texture:
		{
			_2d_texture_list::iterator target_texture_entry;
			if ((target_texture_entry = std::find_if(_2d_textures.begin(), _2d_textures.end(),
				[target_texture_unit](_2d_texture_list::value_type elem) -> bool{return elem.first == target_texture_unit; })) != _2d_textures.end())
			{
				target_texture_entry->second = dynamic_cast<const ImmutableTexture2D&>(new_texture);
				target_texture_entry->third = sampler_reference_code ? sampler_reference_code : old_sampler_ref_code;
				texture_reference_code.first = static_cast<int>(target_texture_entry - _2d_textures.begin());
			}
			else
			{
				_2d_textures.push_back(texture_entry < ImmutableTexture2D > {static_cast<uint32_t>(target_texture_unit),
					dynamic_cast<const ImmutableTexture2D&>(new_texture), sampler_reference_code ? sampler_reference_code : old_sampler_ref_code});
				texture_reference_code.first = static_cast<int>(_2d_textures.size() - 1);
			}
			texture_reference_code.second = static_cast<int>(supported_texture_type::_2d_texture);
			return true;
		}

		case supported_texture_type::_3d_texture:
		{
			_3d_texture_list::iterator target_texture_entry;
			if ((target_texture_entry = std::find_if(_3d_textures.begin(), _3d_textures.end(),
				[target_texture_unit](_3d_texture_list::value_type elem) -> bool{return elem.first == target_texture_unit; })) != _3d_textures.end())
			{
				target_texture_entry->second = dynamic_cast<const ImmutableTexture3D&>(new_texture);
				target_texture_entry->third = sampler_reference_code ? sampler_reference_code : old_sampler_ref_code;
				texture_reference_code.first = static_cast<int>(target_texture_entry - _3d_textures.begin());
			}
			else
			{
				_3d_textures.push_back(texture_entry < ImmutableTexture3D > {static_cast<uint32_t>(target_texture_unit),
					dynamic_cast<const ImmutableTexture3D&>(new_texture), sampler_reference_code ? sampler_reference_code : old_sampler_ref_code});
				texture_reference_code.first = static_cast<int>(_3d_textures.size() - 1);
			}
			texture_reference_code.second = static_cast<int>(supported_texture_type::_3d_texture);
			return true;
		}

		case supported_texture_type::buffer_texture:
		{
			buffer_texture_list::iterator target_texture_entry;
			if ((target_texture_entry = std::find_if(buffer_textures.begin(), buffer_textures.end(),
				[target_texture_unit](buffer_texture_list::value_type elem) -> bool{return elem.first == target_texture_unit; })) != buffer_textures.end())
			{
				target_texture_entry->second = dynamic_cast<const BufferTexture&>(new_texture);
				target_texture_entry->third = sampler_reference_code ? sampler_reference_code : old_sampler_ref_code;
				texture_reference_code.first = static_cast<int>(target_texture_entry - buffer_textures.begin());
			}
			else
			{
				buffer_textures.push_back(texture_entry < BufferTexture > {static_cast<uint32_t>(target_texture_unit),
					dynamic_cast<const BufferTexture&>(new_texture), sampler_reference_code ? sampler_reference_code : old_sampler_ref_code});
				texture_reference_code.first = static_cast<int>(buffer_textures.size() - 1);
			}
			texture_reference_code.second = static_cast<int>(supported_texture_type::buffer_texture);
			return true;
		}

		default: return false;
		}
	}
	else
	{
		switch (new_texture_type)
		{
		case supported_texture_type::cubemap:
			cubemap_textures[texture_reference_code.first].second = dynamic_cast<const ImmutableTextureCubeMap&>(new_texture);
			if (sampler_reference_code)
				cubemap_textures[texture_reference_code.first].third = sampler_reference_code;
			return true;

		case supported_texture_type::_1d_texture:
			_1d_textures[texture_reference_code.first].second = dynamic_cast<const ImmutableTexture1D&>(new_texture);
			if (sampler_reference_code)
				_1d_textures[texture_reference_code.first].third = sampler_reference_code;
			return true;

		case supported_texture_type::_2d_texture:
			_2d_textures[texture_reference_code.first].second = dynamic_cast<const ImmutableTexture2D&>(new_texture);
			if (sampler_reference_code)
				_2d_textures[texture_reference_code.first].third = sampler_reference_code;
			return true;

		case supported_texture_type::_3d_texture:
			_3d_textures[texture_reference_code.first].second = dynamic_cast<const ImmutableTexture3D&>(new_texture);
			if (sampler_reference_code)
				_3d_textures[texture_reference_code.first].third = sampler_reference_code;
			return true;

		case supported_texture_type::buffer_texture:
			buffer_textures[texture_reference_code.first].second = dynamic_cast<const BufferTexture&>(new_texture);
			if (sampler_reference_code)
				buffer_textures[texture_reference_code.first].third = sampler_reference_code;
			return true;

		default: return false;
		}
	}
}

bool AbstractRenderableObjectTextured::updateSampler(const TextureReferenceCode& texture_reference_code, TextureSamplerReferenceCode sampler_reference_code /* = TextureSamplerReferenceCode */)
{
	if (!texture_reference_code) return false;

	supported_texture_type texture_type = static_cast<supported_texture_type>(texture_reference_code.second);

	switch (texture_type)
	{
	case supported_texture_type::cubemap:
		cubemap_textures[texture_reference_code.first].third = sampler_reference_code;
		return true;

	case supported_texture_type::_1d_texture:
		_1d_textures[texture_reference_code.first].third = sampler_reference_code;
		return true;

	case supported_texture_type::_2d_texture:
		_2d_textures[texture_reference_code.first].third = sampler_reference_code;
		return true;

	case supported_texture_type::_3d_texture:
		_3d_textures[texture_reference_code.first].third = sampler_reference_code;
		return true;

	case supported_texture_type::buffer_texture:
		buffer_textures[texture_reference_code.first].third = sampler_reference_code;
		return true;

	case supported_texture_type::unsupported:
	default:
		return false;
	}
}

int AbstractRenderableObjectTextured::getBindingUnit(TextureReferenceCode reference_code) const
{
	switch (static_cast<supported_texture_type>(reference_code.second))
	{
	case supported_texture_type::cubemap:
		if (reference_code.first >= static_cast<int>(cubemap_textures.size()))
			return -1;
		return cubemap_textures[reference_code.first].first;

	case supported_texture_type::_1d_texture:
		if (reference_code.first >= static_cast<int>(_1d_textures.size()))
			return -1;
		return _1d_textures[reference_code.first].first;

	case  supported_texture_type::_2d_texture:
		if (reference_code.first >= static_cast<int>(_2d_textures.size()))
			return -1;
		return _2d_textures[reference_code.first].first;

	case supported_texture_type::_3d_texture:
		if (reference_code.first >= static_cast<int>(_3d_textures.size()))
			return -1;
		return _3d_textures[reference_code.first].first;

	case supported_texture_type::buffer_texture:
		if (reference_code.first >= static_cast<int>(buffer_textures.size()))
			return -1;
		return buffer_textures[reference_code.first].first;

	default:
		return -1;
	}
}

AbstractRenderableObjectTextured::TextureSamplerReferenceCode AbstractRenderableObjectTextured::createTextureSampler(const std::string& string_name, SamplerMagnificationFilter magnification_filter /* = SamplerMagnificationFilter::LINEAR */,
	SamplerMinificationFilter minification_filter /* = SamplerMinificationFilter::LINEAR_MIPMAP_NEAREST */, SamplerWrapping wrapping /* = SamplerWrapping */, const vec4& v4BorderColor /* = vec4(0.0f) */)
{
	samplers.push_back(TextureSampler{ string_name });
	std::vector<TextureSampler>::size_type offset = samplers.size() - 1;
	samplers[offset].setMagFilter(magnification_filter);
	samplers[offset].setMinFilter(minification_filter);
	samplers[offset].setWrapping(wrapping);
	samplers[offset].setBorderColor(v4BorderColor);

	return TextureSamplerReferenceCode{ static_cast<int>(offset) };
}

TextureSampler* AbstractRenderableObjectTextured::retrieveTextureSampler(TextureSamplerReferenceCode sampler_reference_code)
{
	return &samplers[sampler_reference_code.getFirst()];
}

void AbstractRenderableObjectTextured::setTextureUnitOffset(uint32_t offset)
{
	texture_unit_offset = offset;
}

void AbstractRenderableObjectTextured::bindTextures() const
{
	std::for_each(cubemap_textures.begin(), cubemap_textures.end(),
		[this](const texture_entry<ImmutableTextureCubeMap> list_entry) -> void
	{
		getTextureUnitBlockPointer()->switchActiveTextureUnit(list_entry.first);
		getTextureUnitBlockPointer()->bindSampler(samplers[list_entry.third.getFirst()]);
		getTextureUnitBlockPointer()->bindTexture(list_entry.second);
	});

	std::for_each(_1d_textures.begin(), _1d_textures.end(),
		[this](const texture_entry<ImmutableTexture1D> list_entry) -> void
	{
		getTextureUnitBlockPointer()->switchActiveTextureUnit(list_entry.first);
		getTextureUnitBlockPointer()->bindSampler(samplers[list_entry.third.getFirst()]);
		getTextureUnitBlockPointer()->bindTexture(list_entry.second);
	});

	std::for_each(_2d_textures.begin(), _2d_textures.end(),
		[this](const texture_entry<ImmutableTexture2D> list_entry) -> void
	{
		getTextureUnitBlockPointer()->switchActiveTextureUnit(list_entry.first);
		getTextureUnitBlockPointer()->bindSampler(samplers[list_entry.third.getFirst()]);
		getTextureUnitBlockPointer()->bindTexture(list_entry.second);
	});

	std::for_each(_3d_textures.begin(), _3d_textures.end(),
		[this](const texture_entry<ImmutableTexture3D> list_entry) -> void
	{
		getTextureUnitBlockPointer()->switchActiveTextureUnit(list_entry.first);
		getTextureUnitBlockPointer()->bindSampler(samplers[list_entry.third.getFirst()]);
		getTextureUnitBlockPointer()->bindTexture(list_entry.second);
	});

	std::for_each(buffer_textures.begin(), buffer_textures.end(),
		[this](const texture_entry<BufferTexture> list_entry) -> void
	{
		getTextureUnitBlockPointer()->switchActiveTextureUnit(list_entry.first);
		getTextureUnitBlockPointer()->bindSampler(samplers[list_entry.third.getFirst()]);
		getTextureUnitBlockPointer()->bindTexture(list_entry.second);
	});
}

bool AbstractRenderableObjectTextured::bindTexture(TextureReferenceCode reference_code) const
{
	switch (static_cast<supported_texture_type>(reference_code.second))
	{
	case supported_texture_type::cubemap:
		if (reference_code.first >= static_cast<int>(cubemap_textures.size()))
			return false;
		getTextureUnitBlockPointer()->switchActiveTextureUnit(cubemap_textures[reference_code.first].first);
		getTextureUnitBlockPointer()->bindSampler(samplers[cubemap_textures[reference_code.first].third.getFirst()]);
		getTextureUnitBlockPointer()->bindTexture(cubemap_textures[reference_code.first].second);
		return true;

	case supported_texture_type::_1d_texture:
		if (reference_code.first >= static_cast<int>(_1d_textures.size()))
			return false;
		getTextureUnitBlockPointer()->switchActiveTextureUnit(_1d_textures[reference_code.first].first);
		getTextureUnitBlockPointer()->bindSampler(samplers[_1d_textures[reference_code.first].third.getFirst()]);
		getTextureUnitBlockPointer()->bindTexture(_1d_textures[reference_code.first].second);
		return true;

	case supported_texture_type::_2d_texture:
		if (reference_code.first >= static_cast<int>(_2d_textures.size()))
			return false;
		getTextureUnitBlockPointer()->switchActiveTextureUnit(_2d_textures[reference_code.first].first);
		getTextureUnitBlockPointer()->bindSampler(samplers[_2d_textures[reference_code.first].third.getFirst()]);
		getTextureUnitBlockPointer()->bindTexture(_2d_textures[reference_code.first].second);
		return true;

	case supported_texture_type::_3d_texture:
		if (reference_code.first >= static_cast<int>(_3d_textures.size()))
			return false;
		getTextureUnitBlockPointer()->switchActiveTextureUnit(_3d_textures[reference_code.first].first);
		getTextureUnitBlockPointer()->bindSampler(samplers[_3d_textures[reference_code.first].third.getFirst()]);
		getTextureUnitBlockPointer()->bindTexture(_3d_textures[reference_code.first].second);
		return true;

	case supported_texture_type::buffer_texture:
		if (reference_code.first >= static_cast<int>(buffer_textures.size()))
			return false;
		getTextureUnitBlockPointer()->switchActiveTextureUnit(buffer_textures[reference_code.first].first);
		getTextureUnitBlockPointer()->bindSampler(samplers[buffer_textures[reference_code.first].third.getFirst()]);
		getTextureUnitBlockPointer()->bindTexture(buffer_textures[reference_code.first].second);
		return true;

	default: return false;
	}
}

uint32_t AbstractRenderableObjectTextured::getNumberOfCubemaps() const { return static_cast<uint32_t>(cubemap_textures.size()); }

uint32_t AbstractRenderableObjectTextured::getNumberOf1DTextures() const { return static_cast<uint32_t>(_1d_textures.size()); };

uint32_t AbstractRenderableObjectTextured::getNumberOf2DTextures() const { return static_cast<uint32_t>(_2d_textures.size()); };

uint32_t AbstractRenderableObjectTextured::getNumberOf3DTextures() const { return static_cast<uint32_t>(_3d_textures.size()); };

uint32_t AbstractRenderableObjectTextured::getNumberOfBufferTextures() const { return static_cast<uint32_t>(buffer_textures.size()); }

uint32_t AbstractRenderableObjectTextured::getNumberOfTextures() const
{
	return static_cast<uint32_t>(cubemap_textures.size() + _1d_textures.size() + _2d_textures.size() + _3d_textures.size() + buffer_textures.size());
}