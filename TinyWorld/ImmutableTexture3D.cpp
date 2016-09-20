#include "ImmutableTexture3D.h"

#include <algorithm>

using namespace tiny_world;


ImmutableTexture::TextureBinding ImmutableTexture3D::perform_allocation()
{
	TextureBinding rv;
	rv.gl_texture_target = GL_TEXTURE_3D;
	rv.gl_texture_binding = GL_TEXTURE_BINDING_3D;

	GLint current_texture_id;
	glGetIntegerv(rv.gl_texture_binding, &current_texture_id);	//Get OpenGL identifier of the texture currently bound to GL_TEXTURE_3D binding slot
	glBindTexture(rv.gl_texture_target, getOpenGLId());		//Bind "this" texture to the context

	//Allocate storage for "this" texture
	TextureSize texture_size = getTextureSize();
	glTexStorage3D(rv.gl_texture_target, getNumberOfMipmapLevels(), getOpenGLStorageInternalFormat(), 
		texture_size.width, texture_size.height, texture_size.depth);

	//Restore the old binding
	glBindTexture(rv.gl_texture_target, current_texture_id);

	return rv;
}


TextureDimension ImmutableTexture3D::query_dimension() const { return TextureDimension::_3D; }


ImmutableTexture3D::ImmutableTexture3D() : ImmutableTexture("ImmutableTexture3D") {}

ImmutableTexture3D::ImmutableTexture3D(const std::string& texture_string_name) : ImmutableTexture("ImmutableTexture3D", texture_string_name) {}


void ImmutableTexture3D::setMipmapLevelData(uint32_t mipmap_level, PixelLayout pixel_layout, PixelDataType pixel_component_type, const void* data)
{
	if (!isInitialized())
	{
		set_error_state(true);
		std::string err_msg = "Unable to assign new data to 3D texture \"" + getStringName() + "\": the texture was not initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
	}

	//Get current texture binding and bind contained texture to the context
	GLint current_texture_id = bind();
	TextureSize storage_size = getStorageSize();

	uint32_t width = std::max<uint32_t>(1, storage_size.width >> mipmap_level);
	uint32_t height = std::max<uint32_t>(1, storage_size.height >> mipmap_level);
	uint32_t depth = std::max<uint32_t>(1, storage_size.depth >> mipmap_level);

	glTexSubImage3D(getOpenGLTextureBinding().gl_texture_target, mipmap_level, 0, 0, 0, width, height, depth,
		static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_component_type), static_cast<const GLvoid*>(data));

	//Restore the old texture binding
	glBindTexture(getOpenGLTextureBinding().gl_texture_target, current_texture_id);
}


void ImmutableTexture3D::setMipmapLevelData(uint32_t mipmap_level, InternalPixelFormatCompressed compressed_storage_format, size_t compressed_data_size, const void* data)
{
	if (!isInitialized())
	{
		set_error_state(true);
		std::string err_msg = "Unable to assign new data to 3D texture \"" + getStringName() + "\": the texture was not initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
	}

	//Get current texture binding and bind contained texture to the context
	GLint current_texture_id = bind();
	TextureSize storage_size = getStorageSize();

	uint32_t width = std::max<uint32_t>(1, storage_size.width >> mipmap_level);
	uint32_t height = std::max<uint32_t>(1, storage_size.height >> mipmap_level);
	uint32_t depth = std::max<uint32_t>(1, storage_size.depth >> mipmap_level);

	glCompressedTexSubImage3D(getOpenGLTextureBinding().gl_texture_target, mipmap_level, 0, 0, 0, width, height, depth,
		static_cast<GLenum>(compressed_storage_format), static_cast<GLsizei>(compressed_data_size), static_cast<const GLvoid*>(data));

	//Restore old binding
	glBindTexture(getOpenGLTextureBinding().gl_texture_target, current_texture_id);
}


void ImmutableTexture3D::getMipmapLevelImageData(uint32_t mipmap_level, PixelLayout pixel_layout, PixelDataType pixel_component_type, TextureSize* image_size, void* data) const
{
	if (!isInitialized())
	{
		set_error_state(true);
		std::string err_msg = "Can not retrieve data from 3D texture \"" + getStringName() + "\": the texture was not initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	//Get current texture binding and bind contained texture to the context
	GLint current_texture_id = bind();

	if (image_size)
	{
		GLint img_width, img_height, img_depth;
		glGetTexParameteriv(GL_TEXTURE_WIDTH, mipmap_level, &img_width);
		glGetTexParameteriv(GL_TEXTURE_HEIGHT, mipmap_level, &img_height);
		glGetTexParameteriv(GL_TEXTURE_DEPTH, mipmap_level, &img_depth);

		image_size->width = img_width;
		image_size->height = img_height;
		image_size->depth = img_depth;
	}

	if (data)
		glGetTexImage(getOpenGLTextureBinding().gl_texture_target, mipmap_level,
		static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_component_type), static_cast<GLvoid*>(data));

	//Restore the old binding
	glBindTexture(getOpenGLTextureBinding().gl_texture_target, current_texture_id);
}


void ImmutableTexture3D::getMipmapLevelImageData(uint32_t mipmap_level, size_t* compressed_data_size, InternalPixelFormatCompressed* compressed_storage_format, TextureSize* image_size, void* data) const
{
	if (!isInitialized())
	{
		set_error_state(true);
		std::string err_msg = "Can not retrieve data from 3D texture \"" + getStringName() + "\": the texture was not initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	if (!isCompressed())
	{
		set_error_state(true);
		std::string err_msg = "Can not retrieve compressed image data from 3D texture \"" + getStringName() + "\": the texture does not use compression";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	//Retrieve the current texture binding and bind contained texture to the active context
	GLint gl_current_texture_id = bind();

	if (compressed_data_size)
	{
		GLint compressed_image_size;
		glGetTexLevelParameteriv(getOpenGLTextureBinding().gl_texture_target, mipmap_level, GL_TEXTURE_COMPRESSED_IMAGE_SIZE, &compressed_image_size);
		*compressed_data_size = static_cast<size_t>(compressed_image_size);
	}

	if (compressed_storage_format)
	{
		GLint compressed_internal_format;
		glGetTexLevelParameteriv(getOpenGLTextureBinding().gl_texture_target, mipmap_level, GL_TEXTURE_INTERNAL_FORMAT, &compressed_internal_format);
		*compressed_storage_format = static_cast<InternalPixelFormatCompressed>(compressed_internal_format);
	}

	if (image_size)
	{
		GLint img_width, img_height, img_depth;
		glGetTexLevelParameteriv(getOpenGLTextureBinding().gl_texture_target, mipmap_level, GL_TEXTURE_WIDTH, &img_width);
		glGetTexLevelParameteriv(getOpenGLTextureBinding().gl_texture_target, mipmap_level, GL_TEXTURE_HEIGHT, &img_height);
		glGetTexLevelParameteriv(getOpenGLTextureBinding().gl_texture_target, mipmap_level, GL_TEXTURE_DEPTH, &img_depth);

		image_size->width = img_width;
		image_size->height = img_height;
		image_size->depth = img_depth;
	}

	if (data)
		glGetCompressedTexImage(getOpenGLTextureBinding().gl_texture_target, mipmap_level, static_cast<GLvoid*>(data));


	//Restore the old binding
	glBindTexture(getOpenGLTextureBinding().gl_texture_target, gl_current_texture_id);
}


Texture* ImmutableTexture3D::clone() const
{
	return new ImmutableTexture3D{ *this };
}


bool ImmutableTexture3D::copyTexelData(uint32_t source_mipmap_level, uint32_t source_offset_x, uint32_t source_offset_y, uint32_t source_offset_z,
	const ImmutableTexture3D& destination_texture, uint32_t destination_mipmap_level, uint32_t destination_offset_x, uint32_t destination_offset_y, uint32_t destination_offset_z,
	uint32_t copy_buffer_width, uint32_t copy_buffer_height, uint32_t copy_buffer_depth) const
{
	if (getOpenGLStorageInternalFormat() != destination_texture.getOpenGLStorageInternalFormat()) return false;

	glCopyImageSubData(getOpenGLId(), GL_TEXTURE_3D, source_mipmap_level, source_offset_x, source_offset_y, source_offset_z,
		destination_texture.getOpenGLId(), GL_TEXTURE_3D, destination_mipmap_level, destination_offset_x, destination_offset_y, destination_offset_z,
		copy_buffer_width, copy_buffer_height, copy_buffer_depth);
	return true;
}