#include "Texture.h"

using namespace tiny_world;




bool TextureSize::operator==(const TextureSize& other) const
{
	return width == other.width && height == other.height && depth == other.depth;
}

bool TextureSize::operator!=(const TextureSize& other) const{ return !(*this == other); }




void Texture::incrementReferenceCounter()
{
	if (p_texture_details) ++p_texture_details->ref_counter;
}

void Texture::decrementReferenceCounter()
{
	if (p_texture_details) --p_texture_details->ref_counter;
}

uint32_t Texture::getReferenceCount() const
{
	if (p_texture_details) return p_texture_details->ref_counter;
	else
	{
		set_error_state(true);
		std::string err_msg = "Unable to retrieve reference count for the texture " + getStringName() + ". The texture has not been initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return TW_INVALID_RETURN_VALUE;
	}
}


GLenum Texture::getOpenGLStorageInternalFormat() const
{
	if (p_texture_details) return p_texture_details->ogl_internal_texel_storage_format;
	else
	{
		set_error_state(true);
		std::string err_msg = "Unable to retrieve internal texel storage format employed by the texture " + getStringName() + ". The texture has not been initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return TW_INVALID_RETURN_VALUE;
	}
}

GLuint Texture::getOpenGLId() const
{
	if(isInitialized()) return ogl_texture_id;
	else
	{
		set_error_state(true);
		std::string err_msg = "Unable to retrieve OpenGL identifier of the texture " + getStringName() + ". The texture has not been initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return TW_INVALID_RETURN_VALUE;
	}
}

void Texture::initialize(const TextureSize& texture_size, const TextureSize& storage_size, GLenum storage_format)
{
	if (!p_texture_details)
	{
		glGenTextures(1, &ogl_texture_id);
		p_texture_details = new TextureDetails;
		p_texture_details->ref_counter = 1U;
	}

	p_texture_details->texture_size = texture_size;
	p_texture_details->storage_size = storage_size;
	p_texture_details->ogl_internal_texel_storage_format = storage_format;
}

TextureSlot Texture::getBindingSlot() const 
{
	if (!isInitialized())
	{
		set_error_state(true);
		std::string err_msg = "Unable to query binding slot used by the texture " + getStringName() + ". The texture has not been initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return TextureSlot::TEXTURE_UNKNOWN;
	}

	return static_cast<TextureSlot>(getOpenGLTextureBinding().gl_texture_target); 
}


Texture::Texture(const std::string& texture_class_string_name) : Entity{ texture_class_string_name },
ogl_texture_id{ 0 }, p_texture_details{ nullptr }
{

}

Texture::Texture(const std::string& texture_class_string_name, const std::string& texture_string_name) : Entity{ texture_class_string_name, texture_string_name },
ogl_texture_id{ 0 }, p_texture_details{ nullptr }
{

}

Texture::Texture(const Texture& other) : Entity{ other }, ogl_texture_id{ other.ogl_texture_id }, p_texture_details{ other.p_texture_details }
{
	if (p_texture_details) ++p_texture_details->ref_counter;
}

Texture::Texture(Texture&& other) : Entity{ std::move(other) }, ogl_texture_id{ other.ogl_texture_id }, p_texture_details{ other.p_texture_details }
{
	if (p_texture_details) ++p_texture_details->ref_counter;
}


Texture::~Texture()
{
	if (p_texture_details && !(--p_texture_details->ref_counter))
	{
		glDeleteTextures(1, &ogl_texture_id);
		delete p_texture_details;
	}
}


Texture& Texture::operator=(const Texture& other)
{
	if (this == &other)
		return *this;

	Entity::operator=(other);

	if (p_texture_details && !(--p_texture_details->ref_counter))
	{
		glDeleteTextures(1, &ogl_texture_id);
		delete p_texture_details;
	}

	ogl_texture_id = other.ogl_texture_id;
	p_texture_details = other.p_texture_details;
	if (p_texture_details) ++(p_texture_details->ref_counter);

	return *this;
}


Texture& Texture::operator=(Texture&& other)
{
	if (this == &other)
		return *this;

	Entity::operator=(std::move(other));

	std::swap(ogl_texture_id, other.ogl_texture_id);
	std::swap(p_texture_details, other.p_texture_details);

	return *this;
}


bool Texture::operator==(const Texture& other) const { return ogl_texture_id == other.ogl_texture_id; }

bool Texture::operator!=(const Texture& other) const{ return ogl_texture_id != other.ogl_texture_id; }


bool Texture::isInitialized() const{ return p_texture_details != nullptr; }

PixelFormatTraits Texture::getStorageFormatTraits() const
{
	if (p_texture_details)
		return PixelFormatTraits{ p_texture_details->ogl_internal_texel_storage_format };
	else
	{
		set_error_state(true);
		std::string err_msg = "Unable to retrieve pixel format traits for the texture " + getStringName() + ". The texture has not been initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return PixelFormatTraits{ 0xFFFFFFFF };
	}
}

TextureSize Texture::getTextureSize() const
{
	if (p_texture_details)
		return p_texture_details->texture_size;
	else
	{
		set_error_state(true);
		std::string err_msg = "Unable to retrieve size of the texture " + getStringName() + ". The texture has not been initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return TextureSize{ 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF };
	}
}

StorageSize Texture::getStorageSize() const
{
	if (p_texture_details)
		return p_texture_details->storage_size;
	else
	{
		set_error_state(true);
		std::string err_msg = "Unable to retrieve storage size of the texture " + getStringName() + ". The texture has not been initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return StorageSize{ 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF };
	}
}