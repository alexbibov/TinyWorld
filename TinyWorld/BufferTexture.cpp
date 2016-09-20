#include "BufferTexture.h"

using namespace tiny_world;




//Implementation of the core part of the infrastructure

Texture::TextureBinding BufferTexture_Core::getOpenGLTextureBinding() const
{
	return Texture::TextureBinding{ GL_TEXTURE_BUFFER, GL_TEXTURE_BUFFER_BINDING };
}

uint32_t BufferTexture_Core::bind() const
{
	GLint rv;
	glGetIntegerv(GL_TEXTURE_BUFFER_BINDING, &rv);

	if (isInitialized())
	{
		glBindTexture(GL_TEXTURE_BUFFER, getOpenGLId());
		return static_cast<GLuint>(rv);
	}
	else
	{
		set_error_state(true);
		std::string err_msg = "Unable to bind buffer texture " + getStringName() + ". The texture has to be initialized first";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return static_cast<GLuint>(rv);
	}
}


BufferTexture_Core::BufferTexture_Core() : Texture("BufferTexture"), p_shared_data{ nullptr }, is_buffer_provided_by_user{ false }
{

}

BufferTexture_Core::BufferTexture_Core(const std::string& texture_string_name) : Texture("BufferTexture", texture_string_name), p_shared_data{ nullptr }, is_buffer_provided_by_user{ false }
{

}

BufferTexture_Core::BufferTexture_Core(const BufferTexture_Core& other) : Texture(other), p_shared_data(other.p_shared_data), is_buffer_provided_by_user{ other.is_buffer_provided_by_user }
{
	
}

BufferTexture_Core::BufferTexture_Core(BufferTexture_Core&& other) : Texture(std::move(other)), p_shared_data(other.p_shared_data), is_buffer_provided_by_user{ other.is_buffer_provided_by_user }
{
	
}

BufferTexture_Core::~BufferTexture_Core()
{
	if (p_shared_data && !getReferenceCount()) delete p_shared_data;
}

BufferTexture_Core& BufferTexture_Core::operator=(const BufferTexture_Core& other)
{
	//Account for the possibility of assignment to itself
	if (this == &other)
		return *this;

	Texture::operator=(other);
	if (p_shared_data && !getReferenceCount()) delete p_shared_data;

	p_shared_data = other.p_shared_data;
	is_buffer_provided_by_user = other.is_buffer_provided_by_user;

	return *this;
}

BufferTexture_Core& BufferTexture_Core::operator=(BufferTexture_Core&& other)
{
	//Account for the special case of assignment to itself
	if (this == &other)
		return *this;

	Texture::operator=(std::move(other));
	std::swap(p_shared_data, other.p_shared_data);
	is_buffer_provided_by_user = other.is_buffer_provided_by_user;

	return *this;
}

uint32_t BufferTexture_Core::getNumberOfArrayLayers() const{ return 1; }

uint32_t BufferTexture_Core::getNumberOfMipmapLevels() const { return 1; }

uint32_t BufferTexture_Core::getNumberOfFaces() const { return 0; }

uint32_t BufferTexture_Core::getNumberOfSamples() const { return 0; }

TextureDimension BufferTexture_Core::getDimension() const { return TextureDimension::_1D; }

bool BufferTexture_Core::isArrayTexture() const { return false; }

bool BufferTexture_Core::isCompressed() const { return false; }

bool BufferTexture_Core::isBufferTexture() const { return true; }

Texture* BufferTexture_Core::clone() const { return new BufferTexture_Core{ *this }; }

uint32_t BufferTexture_Core::getTextureBufferSize() const
{ 
	if (isInitialized()) return p_shared_data->texture_buffer->getSize();
	else
	{
		set_error_state(true);
		std::string err_msg = "Unable to retrieve the size of the buffer employed by the buffer texture " + getStringName() + ". The texture has not been initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return TW_INVALID_RETURN_VALUE;
	}
}

uint32_t BufferTexture_Core::getTexelCount() const
{
	if (isInitialized()) return p_shared_data->texture_buffer->getSize() / (getStorageFormatTraits().getMinimalStorageSize() / 8);
	else
	{
		set_error_state(true);
		std::string err_msg = "Unable to retrieve the number of texels that could be contained in the buffer texture " + getStringName() + ". The texture has not been initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return TW_INVALID_RETURN_VALUE;
	}
}

void BufferTexture_Core::allocateStorage(size_t texture_size, BufferTextureInternalPixelFormat internal_format)
{
	bool is_buffer_object_replaced = false;

	if (!p_shared_data)
	{
		p_shared_data = new BufferTextureSharedDetails{ SharedBuffer{ BufferBindingTarget::TEXTURE, BufferUsage::COPY, BufferUsageFrequency::STATIC } };
		is_buffer_object_replaced = true;
	}
		

	if (is_buffer_provided_by_user)
	{
		p_shared_data->texture_buffer = SharedBuffer{ BufferBindingTarget::TEXTURE, BufferUsage::COPY, BufferUsageFrequency::STATIC };
		is_buffer_provided_by_user = false;
		is_buffer_object_replaced = true;
	}
	p_shared_data->texture_buffer->allocate(texture_size);
	p_shared_data->internal_format = internal_format;

	uint32_t texel_count = texture_size / (PixelFormatTraits{ internal_format }.getMinimalStorageSize() / 8);
	Texture::initialize(TextureSize{ texel_count, 1, 1 }, StorageSize{ texel_count, 1, 1 }, static_cast<GLenum>(internal_format));

	if (is_buffer_object_replaced)
	{

		GLint currently_bound_buffer;
		glGetIntegerv(GL_TEXTURE_BUFFER_BINDING, &currently_bound_buffer);

		glBindTexture(GL_TEXTURE_BUFFER, getOpenGLId());
		glTexBuffer(GL_TEXTURE_BUFFER, static_cast<GLenum>(internal_format), p_shared_data->texture_buffer->ogl_buffer_id);

		glBindTexture(GL_TEXTURE_BUFFER, currently_bound_buffer);
	}
}

void BufferTexture_Core::setSubData(ptrdiff_t data_chunk_offset, size_t data_chunk_size, const void* data)
{
	if (!isInitialized())
	{
		set_error_state(true);
		std::string err_msg = "Unable to update data chunk [" + std::to_string(data_chunk_offset) + ", " +
			std::to_string(data_chunk_offset + data_chunk_size) + ") in buffer texture " + getStringName() +
			". The texture has to be initialized first";
		set_error_string(err_msg);
		call_error_callback(err_msg);
	}
	else
	{
		p_shared_data->texture_buffer->setSubData(data_chunk_offset, data_chunk_size, data);
	}
}

void* BufferTexture_Core::map(BufferTextureAccessPolicy access_policy) const
{
	if (!isInitialized())
	{
		set_error_state(true);
		std::string err_msg = "Unable to map buffer texture " + getStringName() + " onto the client's address space. " +
			"The texture needs to be initialized first";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return nullptr;
	}
	return p_shared_data->texture_buffer->map(access_policy);
}

void* BufferTexture_Core::map(ptrdiff_t range_offset, size_t range_size, BufferTextureRangeAccessPolicy access_bits) const
{
	if (!isInitialized())
	{
		set_error_state(true);
		std::string err_msg = "Unable to map data range of buffer texture " + getStringName() + " onto the client's address space. " +
			"The texture needs to be initialized first";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return nullptr;
	}
	return p_shared_data->texture_buffer->map(range_offset, range_size, access_bits);
}

void BufferTexture_Core::unmap() const
{
	if (!isInitialized())
	{
		set_error_state(true);
		std::string err_msg = "Unable to unmap the buffer texture " + getStringName() + " from the client's address space. The texture has not been initialized.";
		set_error_string(err_msg);
		call_error_callback(err_msg);
	}
	else
		p_shared_data->texture_buffer->unmap();
}

void BufferTexture_Core::attachBuffer(const SharedBuffer& shared_buffer, BufferTextureInternalPixelFormat internal_format)
{
	if (!p_shared_data) p_shared_data = new BufferTextureSharedDetails;

	p_shared_data->texture_buffer = shared_buffer;
	p_shared_data->internal_format = internal_format;

	uint32_t texel_count = shared_buffer->getSize() / (PixelFormatTraits{ internal_format }.getMinimalStorageSize() / 8);
	Texture::initialize(TextureSize{ texel_count, 1, 1 }, TextureSize{ texel_count, 1, 1 }, static_cast<GLenum>(internal_format));

	is_buffer_provided_by_user = true;

	glBindTexture(GL_TEXTURE_BUFFER, getOpenGLId());
	glTexBuffer(GL_TEXTURE_BUFFER, static_cast<GLenum>(internal_format), p_shared_data->texture_buffer->ogl_buffer_id);
}

bool BufferTexture_Core::copyTexelData(const BufferTexture_Core& destination_texture) const
{
	if (p_shared_data->texture_buffer->buffer_size != destination_texture.p_shared_data->texture_buffer->buffer_size ||
		p_shared_data->internal_format != destination_texture.p_shared_data->internal_format)
		return false;

	*destination_texture.p_shared_data->texture_buffer = *p_shared_data->texture_buffer;
	return true;
}




//Implementation of the interface part of the infrastructure

uint32_t BufferTexture::bind() const{ return::BufferTexture_Core::bind(); }

void BufferTexture::attachToFBO(FramebufferAttachmentPoint attachment) const
{
	if (!isInitialized())
	{
		set_error_state(true);
		const char* err_msg = "Texture must be initialized before it could be attached to  a framebuffer";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	glFramebufferTexture(GL_DRAW_FRAMEBUFFER, static_cast<GLenum>(attachment), getOpenGLId(), 0);
}

void BufferTexture::attachToImageUnit(uint32_t image_unit, BufferAccess access, BufferTextureInternalPixelFormat format) const
{
	//Check if the internal format and the access format belong to the same storage class
	PixelFormatTraits internal_format_traits = getStorageFormatTraits();
	PixelFormatTraits access_format_traits{ format };

	if (internal_format_traits.getStorageClass() == PixelStorageClass::unknown)
	{
		set_error_state(true);
		std::string err_msg = "Unable to determine storage class of internal storage format of buffer texture " +
			getStringName() + ". The specified internal storage format may be unsupported";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	if (access_format_traits.getStorageClass() == PixelStorageClass::unknown)
	{
		set_error_state(true);
		std::string err_msg = "Unable to determine storage class of the requested access format while attaching buffer texture " +
			getStringName() + " to an image unit. The specified access format may be unsupported";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	if (internal_format_traits.getStorageClass() != access_format_traits.getStorageClass())
	{
		set_error_state(true);
		std::string err_msg = "Internal storage format of buffer texture " + getStringName() + " has storage class different from the storage class "
			"of the access format requested while attaching the texture to an image unit. Both the access format and the internal storage format of "
			"the texture must have same storage class to be compatible";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}


	glBindImageTexture(static_cast<GLuint>(image_unit), getOpenGLId(), 0, GL_FALSE, 0, static_cast<GLenum>(access), static_cast<GLenum>(format));
}

BufferTexture::BufferTexture() : BufferTexture_Core{} {}

BufferTexture::BufferTexture(const std::string& texture_string_name) : BufferTexture_Core{ texture_string_name } {}