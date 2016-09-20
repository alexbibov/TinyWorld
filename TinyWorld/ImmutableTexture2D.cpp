#include "ImmutableTexture2D.h"
#include "VectorTypes.h"

#include <algorithm>

using namespace tiny_world;


ImmutableTexture::TextureBinding ImmutableTexture2D::perform_allocation()
{	
	TextureBinding rv;
	rv.gl_texture_target = isArrayTexture() ? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D;
	rv.gl_texture_binding = isArrayTexture() ? GL_TEXTURE_BINDING_2D_ARRAY : GL_TEXTURE_BINDING_2D;

	GLint current_texture_id;
	glGetIntegerv(rv.gl_texture_binding, &current_texture_id);	//save id of the texture currently bound to GL_TEXTURE_2D target
	glBindTexture(rv.gl_texture_target, getOpenGLId());

	TextureSize texture_size = getTextureSize();
	if (isArrayTexture())
		glTexStorage3D(GL_TEXTURE_2D_ARRAY, getNumberOfMipmapLevels(), getOpenGLStorageInternalFormat(),
		texture_size.width, texture_size.height, getNumberOfArrayLayers());
	else
		glTexStorage2D(GL_TEXTURE_2D, getNumberOfMipmapLevels(), getOpenGLStorageInternalFormat(),
		texture_size.width, texture_size.height);

	glBindTexture(rv.gl_texture_target, current_texture_id);	//restore old binding

	return rv;
}


TextureDimension ImmutableTexture2D::query_dimension() const { return TextureDimension::_2D; }


ImmutableTexture2D::ImmutableTexture2D() : ImmutableTexture("ImmutableTexture2D") {}

ImmutableTexture2D::ImmutableTexture2D(const std::string& texture_string_name) : ImmutableTexture("ImmutableTexture2D", texture_string_name) {}


void ImmutableTexture2D::setMipmapLevelData(uint32_t mipmap_level, PixelLayout pixel_layout, PixelDataType pixel_component_type, const void *data)
{
	setMipmapLevelLayerData(mipmap_level, 0, pixel_layout, pixel_component_type, data);
}


void ImmutableTexture2D::setMipmapLevelData(uint32_t mipmap_level, InternalPixelFormatCompressed compressed_storage_format, size_t compressed_data_size, const void *data)
{
	setMipmapLevelLayerData(mipmap_level, 0, compressed_storage_format, compressed_data_size, data);
}


void ImmutableTexture2D::setMipmapLevelLayerData(uint32_t mipmap_level, uint32_t array_layer, PixelLayout pixel_layout, 
	PixelDataType pixel_component_type, const void *data)
{
	setMipmapLevelMultiLayersData(mipmap_level, array_layer, 1, pixel_layout, pixel_component_type, data);
}


void ImmutableTexture2D::setMipmapLevelLayerData(uint32_t mipmap_level, uint32_t array_layer, InternalPixelFormatCompressed compressed_storage_format, 
	size_t compressed_data_size, const void *data)
{
	setMipmapLevelMultiLayersData(mipmap_level, array_layer, 1, compressed_storage_format, compressed_data_size, data);
}


void ImmutableTexture2D::setMipmapLevelMultiLayersData(uint32_t mipmap_level, uint32_t start_array_layer, uint32_t number_of_array_layers, PixelLayout pixel_layout, PixelDataType pixel_component_type, const void* data)
{
	if (!isInitialized())
	{
		set_error_state(true);
		std::string err_msg = "Unable to set new data for requested array layers " + std::to_string(start_array_layer) + "-" + std::to_string(start_array_layer + number_of_array_layers - 1) +
			" of mipmap level " + std::to_string(mipmap_level) + " of 2D texture \"" + getStringName() + "\": the texture was not initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	if (start_array_layer + number_of_array_layers - 1 >= getNumberOfArrayLayers())
	{
		set_error_state(true);
		std::string err_msg = "Unable to set new data for requested array layers " + std::to_string(start_array_layer) + "-" + std::to_string(start_array_layer + number_of_array_layers - 1) +
			"of mipmap level " + std::to_string(mipmap_level) + " of 2D texture \"" + getStringName() + "\": indexes " +
			std::to_string(getNumberOfArrayLayers()) + "-" + std::to_string(start_array_layer + number_of_array_layers - 1) + " are out of range";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	//Retrieve the old texture binding and bind current texture to the context
	TextureSize texture_size = getTextureSize();
	GLuint gl_old_texture_id = bind();

	uint32_t width = std::max<uint32_t>(1, texture_size.width >> mipmap_level);
	uint32_t height = std::max<uint32_t>(1, texture_size.height >> mipmap_level);

	if (isArrayTexture())
		glTexSubImage3D(GL_TEXTURE_2D_ARRAY, mipmap_level, 0, 0, start_array_layer, width, height, number_of_array_layers,
		static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_component_type), static_cast<const GLvoid*>(data));
	else
		glTexSubImage2D(GL_TEXTURE_2D, mipmap_level, 0, 0, width, height, 
		static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_component_type), static_cast<const GLvoid*>(data));

	//Restore the old texture binding
	glBindTexture(getOpenGLTextureBinding().gl_texture_target, gl_old_texture_id);
}


void ImmutableTexture2D::setMipmapLevelMultiLayersData(uint32_t mipmap_level, uint32_t start_array_layer, uint32_t number_of_array_layers, InternalPixelFormatCompressed compressed_storage_format, size_t compressed_data_size, const void* data)
{
	if (!isInitialized())
	{
		set_error_state(true);
		std::string err_msg = "Unable to set new data for requested array layers " + std::to_string(start_array_layer) + "-" + std::to_string(start_array_layer + number_of_array_layers - 1) +
			" of mipmap level " + std::to_string(mipmap_level) + " of 2D texture \"" + getStringName() + "\": the texture was not initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	if (start_array_layer + number_of_array_layers - 1 >= getNumberOfArrayLayers())
	{
		set_error_state(true);
		std::string err_msg = "Unable to set new data for requested array layers " + std::to_string(start_array_layer) + "-" + std::to_string(start_array_layer + number_of_array_layers - 1) +
			"of mipmap level " + std::to_string(mipmap_level) + " of 2D texture \"" + getStringName() + "\": indexes " +
			std::to_string(getNumberOfArrayLayers()) + "-" + std::to_string(start_array_layer + number_of_array_layers - 1) + " are out of range";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	//Retrieve the old texture binding and bind current texture to the context
	TextureSize texture_size = getTextureSize();
	GLuint gl_old_texture_id = bind();

	uint32_t width = std::max<uint32_t>(1, texture_size.width >> mipmap_level);
	uint32_t height = std::max<uint32_t>(1, texture_size.height >> mipmap_level);

	if (isArrayTexture())
		glCompressedTexSubImage3D(GL_TEXTURE_2D_ARRAY, mipmap_level, 0, 0, start_array_layer, width, height, number_of_array_layers,
		static_cast<GLenum>(compressed_storage_format), static_cast<GLsizei>(compressed_data_size), static_cast<const GLvoid*>(data));
	else
		glCompressedTexSubImage2D(GL_TEXTURE_2D, mipmap_level, 0, 0, width, height,
		static_cast<GLenum>(compressed_storage_format), static_cast<GLsizei>(compressed_data_size), static_cast<const GLvoid*>(data));

	//Restore the old texture binding
	glBindTexture(getOpenGLTextureBinding().gl_texture_target, gl_old_texture_id);
}


void ImmutableTexture2D::getMipmapLevelImageData(uint32_t mipmap_level, PixelLayout pixel_layout, PixelDataType pixel_component_type, TextureSize *image_size, void *data) const
{
	if (!isInitialized())
	{
		set_error_state(true);
		std::string err_msg = "Unable to retrieve data from 2D texture \"" + getStringName() + "\": the texture has not been initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	GLuint ogl_old_texture_id = bind();		//Bind current texture to its target

	if (image_size)
	{
		GLint img_width, img_height;
		glGetTexLevelParameteriv(getOpenGLTextureBinding().gl_texture_target, mipmap_level, GL_TEXTURE_WIDTH, &img_width);
		glGetTexLevelParameteriv(getOpenGLTextureBinding().gl_texture_target, mipmap_level, GL_TEXTURE_HEIGHT, &img_height);
		image_size->width = img_width;
		image_size->height = img_height;
		image_size->depth = 0;	//this is a 2D-texture, i.e. it has no depth
	}
	
	if (data)
		glGetTexImage(getOpenGLTextureBinding().gl_texture_target, mipmap_level, 
		static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_component_type), static_cast<GLvoid*>(data));

	glBindTexture(getOpenGLTextureBinding().gl_texture_target, ogl_old_texture_id);		//Restore old binding
}

void ImmutableTexture2D::getMipmapLevelImageData(uint32_t mipmap_level, size_t *compressed_data_size, InternalPixelFormatCompressed *compressed_storage_format, TextureSize *image_size, void *data) const
{
	if (!isInitialized())
	{
		set_error_state(true);
		std::string err_msg = "Unable to retrieve data from 2D texture \"" + getStringName() + "\": the texture has not been initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	if (!isCompressed())
	{
		set_error_state(true);
		std::string err_msg = "Unable to retrieve raw compressed data from 2D texture \"" + getStringName() + "\": the texture does not use compression";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	GLuint ogl_old_texture_id = bind();		//Bind current texture to its target

	if (compressed_data_size)
	{
		GLint compressed_img_size = 0;
		glGetTexLevelParameteriv(getOpenGLTextureBinding().gl_texture_target, mipmap_level, GL_TEXTURE_COMPRESSED_IMAGE_SIZE, &compressed_img_size);
		*compressed_data_size = static_cast<size_t>(compressed_img_size);
	}
		
	if (compressed_storage_format)
	{
		GLint compressed_internal_format = 0;
		glGetTexLevelParameteriv(getOpenGLTextureBinding().gl_texture_target, mipmap_level, GL_TEXTURE_INTERNAL_FORMAT, &compressed_internal_format);
		*compressed_storage_format = static_cast<InternalPixelFormatCompressed>(compressed_internal_format);
	}

	if (image_size)
	{
		GLint width, height;
		glGetTexLevelParameteriv(getOpenGLTextureBinding().gl_texture_target, mipmap_level, GL_TEXTURE_WIDTH, &width);
		glGetTexLevelParameteriv(getOpenGLTextureBinding().gl_texture_target, mipmap_level, GL_TEXTURE_HEIGHT, &height);
		image_size->width = width;
		image_size->height = height;
		image_size->depth = 0;	//this is a 2D-texture, i.e. it has no depth
	}

	if (data)
		glGetCompressedTexImage(getOpenGLTextureBinding().gl_texture_target, mipmap_level, static_cast<GLvoid*>(data));

	glBindTexture(getOpenGLTextureBinding().gl_texture_target, ogl_old_texture_id);		//Restore old binding
}

Texture* ImmutableTexture2D::clone() const
{
	return new ImmutableTexture2D(*this);
}

ImmutableTexture2D ImmutableTexture2D::combine(const ImmutableTexture2D& other) const
{
	//Check that all textures being combined have consistent dimensions
	if (getTextureSize() != other.getTextureSize())
	{
		set_error_state(true);
		const char* err_msg = "Unable to combine 2D texture objects into an array: the textures being combined are having inconsistent dimensions";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return *this;
	}


	//Check that all textures being combines have same number of mipmap levels
	if (getNumberOfMipmapLevels() != other.getNumberOfMipmapLevels())
	{
		set_error_state(true);
		const char* err_msg = "Unable to combine 2D texture objects into an array: the textures being combined are having different number of mipmap levels";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return *this;
	}

	
	//Check that all textures being combined have same internal storage format
	if (getOpenGLStorageInternalFormat() != other.getOpenGLStorageInternalFormat())
	{
		set_error_state(true);
		const char* err_msg = "Unable to combine 2D texture objects into an array: the textures being combined must use the same internal pixel storage format";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return *this;
	}

	ImmutableTexture2D combined_texture{ getStringName() + "_" + other.getStringName() + "_combined" };
	if (isCompressed())
		combined_texture.allocateStorage(getNumberOfMipmapLevels(), getNumberOfArrayLayers() + other.getNumberOfArrayLayers(),
		getTextureSize(), static_cast<InternalPixelFormatCompressed>(getOpenGLStorageInternalFormat()));
	else
		combined_texture.allocateStorage(getNumberOfMipmapLevels(), getNumberOfArrayLayers() + other.getNumberOfArrayLayers(),
		getTextureSize(), static_cast<InternalPixelFormat>(getOpenGLStorageInternalFormat()));

	//Get currently active pack alignment
	GLint pixel_storage_pack_alignment;	//pack alignment of the texture
	glGetIntegerv(GL_PACK_ALIGNMENT, &pixel_storage_pack_alignment);

	//Change unpack alignment settings
	GLint pixel_storage_unpack_alignment;	//unpack alignment of the texture
	glGetIntegerv(GL_UNPACK_ALIGNMENT, &pixel_storage_unpack_alignment);
	glPixelStorei(GL_UNPACK_ALIGNMENT, pixel_storage_pack_alignment);

	//Determine optimal pixel storage format, type, and size that should be used to retrieve data from the textures being combined
	PixelFormatTraits pixel_format_traits = getStorageFormatTraits();
	const short pixel_size = pixel_format_traits.getOptimalStorageSize();

	for (int mipmap_level = 0; mipmap_level < static_cast<int>(getNumberOfMipmapLevels()); ++mipmap_level)
	{
		uint32_t mipmap_level_width = std::max<uint32_t>(1, getTextureSize().width >> mipmap_level);
		uint32_t mipmap_level_height = std::max<uint32_t>(1, getTextureSize().height >> mipmap_level);


		void* img_data = nullptr;
		if (isCompressed())
		{
			//Retrieve size of the first texture being combined
			size_t _1st_component_size;
			getMipmapLevelImageData(mipmap_level, &_1st_component_size, nullptr, nullptr, nullptr);

			//Retrieve size of the second texture being combined
			size_t _2nd_component_size;
			other.getMipmapLevelImageData(mipmap_level, &_2nd_component_size, nullptr, nullptr, nullptr);

			img_data = malloc(_1st_component_size + _2nd_component_size);

			getMipmapLevelImageData(mipmap_level, nullptr, nullptr, nullptr, img_data);
			other.getMipmapLevelImageData(mipmap_level, nullptr, nullptr, nullptr, static_cast<char*>(img_data)+_1st_component_size);

			combined_texture.setMipmapLevelMultiLayersData(mipmap_level, 0, getNumberOfArrayLayers() + other.getNumberOfArrayLayers(),
				static_cast<InternalPixelFormatCompressed>(getOpenGLStorageInternalFormat()), _1st_component_size + _2nd_component_size, img_data);
		}
		else
		{
			const uint32_t row_padding = pixel_storage_pack_alignment - ((pixel_size * mipmap_level_width) % pixel_storage_pack_alignment ?
				(pixel_size * mipmap_level_width) % pixel_storage_pack_alignment : pixel_storage_pack_alignment);

			unsigned int storage_size =
				(pixel_size * mipmap_level_width + row_padding) * mipmap_level_height *
				(getNumberOfArrayLayers() + other.getNumberOfArrayLayers());
			unsigned int layer_shift_size = (pixel_size * mipmap_level_width + row_padding) * mipmap_level_height;

			img_data = malloc(storage_size);

			getMipmapLevelImageData(mipmap_level, pixel_format_traits.getPixelLayout(), pixel_format_traits.getOptimalStorageType(), nullptr, img_data);
			other.getMipmapLevelImageData(mipmap_level, pixel_format_traits.getPixelLayout(), pixel_format_traits.getOptimalStorageType(), nullptr,
				static_cast<char*>(img_data)+layer_shift_size * getNumberOfArrayLayers());

			combined_texture.setMipmapLevelMultiLayersData(mipmap_level, 0, getNumberOfArrayLayers() + other.getNumberOfArrayLayers(),
				pixel_format_traits.getPixelLayout(), pixel_format_traits.getOptimalStorageType(), img_data);
		}
		free(img_data);
	}
	glPixelStorei(GL_UNPACK_ALIGNMENT, pixel_storage_unpack_alignment);	//restore old unpack alignment

	return combined_texture;
}


bool ImmutableTexture2D::copyTexelData(uint32_t source_mipmap_level, uint32_t source_first_array_layer, uint32_t source_offset_x, uint32_t source_offset_y,
	const ImmutableTexture2D& destination_texture, uint32_t destination_mipmap_level, uint32_t destination_first_array_layer, uint32_t destination_offset_x, uint32_t destination_offset_y,
	uint32_t copy_buffer_width, uint32_t copy_buffer_height, uint32_t num_array_layers_to_copy) const
{
	if (getOpenGLStorageInternalFormat() != destination_texture.getOpenGLStorageInternalFormat()) return false;

	glCopyImageSubData(getOpenGLId(), isArrayTexture() ? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D, source_mipmap_level, source_offset_x, source_offset_y, source_first_array_layer,
		destination_texture.getOpenGLId(), destination_texture.isArrayTexture() ? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D, destination_mipmap_level, destination_offset_x, destination_offset_y, destination_first_array_layer,
		copy_buffer_width, copy_buffer_height, num_array_layers_to_copy);
	return true;
}