#include "ImmutableTexture1D.h"

#include <algorithm>

using namespace tiny_world;


ImmutableTexture::TextureBinding ImmutableTexture1D::perform_allocation()
{
	TextureBinding rv;
	rv.gl_texture_target = isArrayTexture() ? GL_TEXTURE_1D_ARRAY : GL_TEXTURE_1D;
	rv.gl_texture_binding = isArrayTexture() ? GL_TEXTURE_BINDING_1D_ARRAY : GL_TEXTURE_BINDING_1D;

	//Retrieve identifier of the OpenGL texture currently bound to the context and bind "this" texture instead
	GLint current_texture_id;
	glGetIntegerv(rv.gl_texture_binding, &current_texture_id);
	glBindTexture(rv.gl_texture_target, getOpenGLId());

	//Perform allocation of the texture storage
	TextureSize texture_size = getTextureSize();
	if (isArrayTexture())
		glTexStorage2D(GL_TEXTURE_1D_ARRAY, getNumberOfMipmapLevels(), getOpenGLStorageInternalFormat(), texture_size.width, getNumberOfArrayLayers());
	else
		glTexStorage1D(GL_TEXTURE_1D, getNumberOfMipmapLevels(), getOpenGLStorageInternalFormat(), texture_size.width);

	//Restore the old texture binding
	glBindTexture(rv.gl_texture_target, current_texture_id);

	return rv;
}


TextureDimension ImmutableTexture1D::query_dimension() const { return TextureDimension::_1D; }


ImmutableTexture1D::ImmutableTexture1D() : ImmutableTexture("ImmutableTexture1D") {}

ImmutableTexture1D::ImmutableTexture1D(const std::string& texture_string_name) : ImmutableTexture("ImmutableTexture1D", texture_string_name) {}


void ImmutableTexture1D::setMipmapLevelData(uint32_t mipmap_level, PixelLayout pixel_layout, PixelDataType pixel_component_type, const void* data)
{
	setMipmapLevelLayerData(mipmap_level, 0, pixel_layout, pixel_component_type, data);
}


void ImmutableTexture1D::setMipmapLevelData(uint32_t mipmap_level, InternalPixelFormatCompressed compressed_storage_format, size_t compressed_data_size, const void* data)
{
	setMipmapLevelLayerData(mipmap_level, 0, compressed_storage_format, compressed_data_size, data);
}


void ImmutableTexture1D::setMipmapLevelLayerData(uint32_t mipmap_level, uint32_t array_layer, PixelLayout pixel_layout, PixelDataType pixel_component_type, const void* data)
{
	setMipmapLevelMultiLayersData(mipmap_level, array_layer, 1, pixel_layout, pixel_component_type, data);
}


void ImmutableTexture1D::setMipmapLevelLayerData(uint32_t mipmap_level, uint32_t array_layer, InternalPixelFormatCompressed compressed_storage_format, size_t compressed_data_size, const void* data)
{
	setMipmapLevelMultiLayersData(mipmap_level, array_layer, 1, compressed_storage_format, compressed_data_size, data);
}


void ImmutableTexture1D::setMipmapLevelMultiLayersData(uint32_t mipmap_level, uint32_t start_array_layer, uint32_t number_of_array_layers, PixelLayout pixel_layout, PixelDataType pixel_component_type, const void* data)
{
	if (!isInitialized())
	{
		set_error_state(true);
		std::string err_msg = "Unable to set new data for requested array layers " + std::to_string(start_array_layer) + "-" + std::to_string(start_array_layer + number_of_array_layers - 1) +
			" of mipmap level " + std::to_string(mipmap_level) + " of 1D texture \"" + getStringName() + "\": the texture was not initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	if (start_array_layer + number_of_array_layers - 1 >= getNumberOfArrayLayers())
	{
		set_error_state(true);
		std::string err_msg = "Unable to set new data for requested array layers " + std::to_string(start_array_layer) + "-" + std::to_string(start_array_layer + number_of_array_layers - 1) +
			"of mipmap level " + std::to_string(mipmap_level) + " of 1D texture \"" + getStringName() +"\": indexes " + 
			std::to_string(getNumberOfArrayLayers()) + "-" + std::to_string(start_array_layer + number_of_array_layers - 1) + " are out of range";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	//Retrieve the old texture binding and bind current texture to the context
	TextureSize texture_size = getTextureSize();
	GLuint gl_old_texture_id = bind();

	uint32_t width = std::max<uint32_t>(1, texture_size.width >> mipmap_level);

	if (isArrayTexture())
		glTexSubImage2D(GL_TEXTURE_1D_ARRAY, mipmap_level, 0, start_array_layer, width, number_of_array_layers,
		static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_component_type), static_cast<const GLvoid*>(data));
	else
		glTexSubImage1D(GL_TEXTURE_1D, mipmap_level, 0, width,
		static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_component_type), static_cast<const GLvoid*>(data));

	//Restore the old texture binding
	glBindTexture(getOpenGLTextureBinding().gl_texture_target, gl_old_texture_id);
}


void ImmutableTexture1D::setMipmapLevelMultiLayersData(uint32_t mipmap_level, uint32_t start_array_layer, uint32_t number_of_array_layers, InternalPixelFormatCompressed compressed_storage_format, size_t compressed_data_size, const void* data)
{
	if (!isInitialized())
	{
		set_error_state(true);
		std::string err_msg = "Unable to set new data for requested array layers " + std::to_string(start_array_layer) + "-" + std::to_string(start_array_layer + number_of_array_layers - 1) +
			" of mipmap level " + std::to_string(mipmap_level) + " of 1D texture \"" + getStringName() + "\": the texture was not initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	if (start_array_layer + number_of_array_layers - 1 >= getNumberOfArrayLayers())
	{
		set_error_state(true);
		std::string err_msg = "Unable to set new data for requested array layers " + std::to_string(start_array_layer) + "-" + std::to_string(start_array_layer + number_of_array_layers - 1) +
			"of mipmap level " + std::to_string(mipmap_level) + " of 1D texture \"" + getStringName() + "\": indexes " +
			std::to_string(getNumberOfArrayLayers()) + "-" + std::to_string(start_array_layer + number_of_array_layers - 1) + " are out of range";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	//Retrieve the old texture binding and bind current texture to the context
	TextureSize texture_size = getTextureSize();
	GLuint gl_old_texture_id = bind();

	uint32_t width = std::max<uint32_t>(1, texture_size.width >> mipmap_level);

	if (isArrayTexture())
		glCompressedTexSubImage2D(GL_TEXTURE_1D_ARRAY, mipmap_level, 0, start_array_layer, width, number_of_array_layers,
		static_cast<GLenum>(compressed_storage_format), static_cast<GLsizei>(compressed_data_size), static_cast<const GLvoid*>(data));
	else
		glCompressedTexSubImage1D(GL_TEXTURE_1D, mipmap_level, 0, width,
		static_cast<GLenum>(compressed_storage_format), static_cast<GLsizei>(compressed_data_size), static_cast<const GLvoid*>(data));

	//Restore the old binding
	glBindTexture(getOpenGLTextureBinding().gl_texture_target, getOpenGLId());
}


void ImmutableTexture1D::getMipmapLevelImageData(uint32_t mipmap_level, PixelLayout pixel_layout, PixelDataType pixel_component_type, TextureSize *image_size, void* data) const
{
	if (!isInitialized())
	{
		set_error_state(true);
		std::string err_msg = "Unable to retrieve data from 1D texture \"" + getStringName() + "\": the texture has not been initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	//Retrieve raw OpenGL identifier of the currently bound texture and bind "this" texture to the context instead
	GLuint ogl_texture_id = bind();

	if (image_size)
	{
		GLint img_width;
		glGetTexLevelParameteriv(getOpenGLTextureBinding().gl_texture_target, mipmap_level, GL_TEXTURE_WIDTH, &img_width);
		image_size->width = img_width;
		image_size->height = 0;		//this is a 1D texture, i.e. it does not define concepts of height and depth
		image_size->depth = 0;
	}

	if (data)
		glGetTexImage(getOpenGLTextureBinding().gl_texture_target, mipmap_level, 
		static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_component_type), static_cast<GLvoid*>(data));

	//Restore the old binding
	glBindTexture(getOpenGLTextureBinding().gl_texture_target, ogl_texture_id);
}


void ImmutableTexture1D::getMipmapLevelImageData(uint32_t mipmap_level, size_t* compressed_data_size, InternalPixelFormatCompressed* compressed_storage_format, TextureSize* image_size, void* data) const
{
	if (!isInitialized())
	{
		set_error_state(true);
		std::string err_msg = "Unable to retrieve data from 1D texture \"" + getStringName() + "\": the texture has not been initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	if (!isCompressed())
	{
		set_error_state(true);
		std::string err_msg = "Unable to retrieve raw compressed data from 1D texture \"" + getStringName() +"\": the texture does not use compression";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	//Retrieve raw OpenGL identifier of the currently bound texture and bind "this" texture to the context instead
	GLuint ogl_texture_id = bind();

	if (compressed_data_size)
	{
		GLint size;
		glGetTexLevelParameteriv(getOpenGLTextureBinding().gl_texture_target, mipmap_level, GL_TEXTURE_COMPRESSED_IMAGE_SIZE, &size);
		*compressed_data_size = static_cast<size_t>(size);
	}

	if (compressed_storage_format)
	{
		GLint storage_format;
		glGetTexLevelParameteriv(getOpenGLTextureBinding().gl_texture_target, mipmap_level, GL_TEXTURE_INTERNAL_FORMAT, &storage_format);
		*compressed_storage_format = static_cast<InternalPixelFormatCompressed>(storage_format);
	}

	if (image_size)
	{
		GLint width;
		glGetTexLevelParameteriv(getOpenGLTextureBinding().gl_texture_target, mipmap_level, GL_TEXTURE_WIDTH, &width);
		image_size->width = width;
		image_size->height = 0;
		image_size->depth = 0;
	}

	if (data)
		glGetCompressedTexImage(getOpenGLTextureBinding().gl_texture_target, mipmap_level, static_cast<GLvoid*>(data));

	//Restore the old binding
	glBindTexture(getOpenGLTextureBinding().gl_texture_target, ogl_texture_id);
}


Texture* ImmutableTexture1D::clone() const { return new ImmutableTexture1D{ *this }; }


ImmutableTexture1D ImmutableTexture1D::combine(const ImmutableTexture1D& other) const
{
	//Check that all textures being combined are having consistent dimensions
	if (getTextureSize() != other.getTextureSize())
	{
		set_error_state(true);
		const char* err_msg = "Unable to combine 1D texture objects into an array: the textures being combined are having inconsistent dimensions";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return *this;
	}
	
	//Check that all textures being combined are having same number of LOD-levels
	if (getNumberOfMipmapLevels() != other.getNumberOfMipmapLevels())
	{
		set_error_state(true);
		const char* err_msg = "Unable to combine 1D texture objects into an array: the textures being combined are having different number of mipmap levels";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return *this;
	}

	//Check that both texture objects being combined are using the same internal pixel storage format
	if (getOpenGLStorageInternalFormat() != other.getOpenGLStorageInternalFormat())
	{
		set_error_state(true);
		const char* err_msg = "Unable to combine 1D texture objects into an array: the textures being combined must use the same internal pixel storage format";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return *this;
	}


	//Allocate storage for combined texture
	ImmutableTexture1D combined_texture{ getStringName() + "_" + other.getStringName() + "_combined" };
	if (isCompressed())
		combined_texture.allocateStorage(getNumberOfMipmapLevels(), getNumberOfArrayLayers() + other.getNumberOfArrayLayers(), getTextureSize(),
		static_cast<InternalPixelFormatCompressed>(getOpenGLStorageInternalFormat()));
	else
		combined_texture.allocateStorage(getNumberOfMipmapLevels(), getNumberOfArrayLayers() + other.getNumberOfArrayLayers(), getTextureSize(),
		static_cast<InternalPixelFormat>(getOpenGLStorageInternalFormat()));


	//Get currently active texture pack alignment
	GLint pixel_storage_pack_alignment;
	glGetIntegerv(GL_PACK_ALIGNMENT, &pixel_storage_pack_alignment);

	//Retrieve texture unpack alignment setting currently in use and change it so that it corresponds to the packing alignment
	GLint pixel_storage_unpack_alignment;
	glGetIntegerv(GL_UNPACK_ALIGNMENT, &pixel_storage_unpack_alignment);
	glPixelStorei(GL_UNPACK_ALIGNMENT, pixel_storage_pack_alignment);


	PixelFormatTraits pixel_format_traits{ getOpenGLStorageInternalFormat() };
	const short pixel_size = pixel_format_traits.getOptimalStorageSize();

	for (int mipmap_level = 0; mipmap_level < static_cast<int>(getNumberOfMipmapLevels()); ++mipmap_level)
	{
		uint32_t mipmap_level_width = std::max<uint32_t>(1, getTextureSize().width >> mipmap_level);


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
			const uint32_t row_padding = pixel_storage_pack_alignment - (pixel_size * mipmap_level_width % pixel_storage_pack_alignment ?
				pixel_size * mipmap_level_width % pixel_storage_pack_alignment : pixel_storage_pack_alignment);

			const uint32_t layer_shift = pixel_size * mipmap_level_width + row_padding;
			const uint32_t storage_size = layer_shift * (getNumberOfArrayLayers() + other.getNumberOfArrayLayers());

			img_data = malloc(storage_size);

			getMipmapLevelImageData(mipmap_level, pixel_format_traits.getPixelLayout(), pixel_format_traits.getOptimalStorageType(), nullptr, img_data);
			other.getMipmapLevelImageData(mipmap_level, pixel_format_traits.getPixelLayout(), pixel_format_traits.getOptimalStorageType(), nullptr,
				static_cast<char*>(img_data)+layer_shift*getNumberOfArrayLayers());

			combined_texture.setMipmapLevelMultiLayersData(mipmap_level, 0, getNumberOfArrayLayers() + other.getNumberOfArrayLayers(),
				pixel_format_traits.getPixelLayout(), pixel_format_traits.getOptimalStorageType(), img_data);  
		}
		free(img_data);
	}

	//Restore the old texture unpack alignment setting
	glPixelStorei(GL_UNPACK_ALIGNMENT, pixel_storage_unpack_alignment);

	return combined_texture;
}


bool ImmutableTexture1D::copyTexelData(uint32_t source_mipmap_level, uint32_t source_first_array_layer, uint32_t source_offset,
	const ImmutableTexture1D& destination_texture, uint32_t destination_mipmap_level, uint32_t destination_first_array_layer, uint32_t destination_offset, uint32_t copy_buffer_width, uint32_t num_array_layers_to_copy) const
{
	if (getOpenGLStorageInternalFormat() != destination_texture.getOpenGLStorageInternalFormat()) return false;

	glCopyImageSubData(getOpenGLId(), isArrayTexture() ? GL_TEXTURE_1D_ARRAY : GL_TEXTURE_1D, source_mipmap_level, source_offset, 0, source_first_array_layer,
		destination_texture.getOpenGLId(), destination_texture.isArrayTexture() ? GL_TEXTURE_1D_ARRAY : GL_TEXTURE_1D, destination_mipmap_level, destination_offset, 0, destination_first_array_layer,
		copy_buffer_width, 1, num_array_layers_to_copy);

	return true;
}