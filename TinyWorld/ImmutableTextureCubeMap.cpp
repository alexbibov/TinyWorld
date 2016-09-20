#include "ImmutableTextureCubeMap.h"

#include <algorithm>

using namespace tiny_world;

ImmutableTextureCubeMap::ImmutableTextureCubeMap() : ImmutableTexture("ImmutableTextureCubemap") {}


ImmutableTextureCubeMap::ImmutableTextureCubeMap(const std::string& texture_string_name) : 
ImmutableTexture("ImmutableTextureCubemap", texture_string_name) {}


ImmutableTextureCubeMap::ImmutableTextureCubeMap(const ImmutableTexture2D& positive_x, const ImmutableTexture2D& negative_x,
	const ImmutableTexture2D& positive_y, const ImmutableTexture2D& negative_y,
	const ImmutableTexture2D& positive_z, const ImmutableTexture2D& negative_z) : ImmutableTexture("ImmutableTextureCubemap")
{
	feed_data_from_2d_textures(std::vector<const ImmutableTexture2D>({ positive_x, negative_x, positive_y, negative_y, positive_z, negative_z }));
}

ImmutableTextureCubeMap::ImmutableTextureCubeMap(const std::vector<const ImmutableTexture2D>& _2d_textures) : ImmutableTexture("ImmutableTextureCubemap")
{
	feed_data_from_2d_textures(_2d_textures);
}


ImmutableTexture::TextureBinding ImmutableTextureCubeMap::perform_allocation()
{
	TextureBinding rv;
	rv.gl_texture_target = isArrayTexture() ? GL_TEXTURE_CUBE_MAP_ARRAY : GL_TEXTURE_CUBE_MAP;
	rv.gl_texture_binding = isArrayTexture() ? GL_TEXTURE_BINDING_CUBE_MAP_ARRAY : GL_TEXTURE_BINDING_CUBE_MAP;

	GLint ogl_current_texture_id;
	glGetIntegerv(rv.gl_texture_binding, &ogl_current_texture_id);
	glBindTexture(rv.gl_texture_target, getOpenGLId());
	
	TextureSize texture_size = getTextureSize();

	if (isArrayTexture())
	{
		glTexStorage3D(GL_TEXTURE_CUBE_MAP_ARRAY, getNumberOfMipmapLevels(), getOpenGLStorageInternalFormat(),
			texture_size.width, texture_size.height, 6*getNumberOfArrayLayers());
	}
	else
		glTexStorage2D(GL_TEXTURE_CUBE_MAP, getNumberOfMipmapLevels(), getOpenGLStorageInternalFormat(),
		texture_size.width, texture_size.height);

	glBindTexture(rv.gl_texture_target, ogl_current_texture_id);		//Restore old binding

	return rv;
}

TextureDimension ImmutableTextureCubeMap::query_dimension() const { return TextureDimension::_2D; }


bool ImmutableTextureCubeMap::feed_data_from_2d_textures(const std::vector<const ImmutableTexture2D>& _2d_textures)
{
	//Get number of 2D textures that is enough to create an array of fully defined cubemaps
	uint32_t num_cubemaps = static_cast<uint32_t>(_2d_textures.size()) / 6;	//number of resulting cubemaps
	uint32_t num_textures = 6 * num_cubemaps;	   //number of textures that participate in construction of the cubemaps
	std::vector<const ImmutableTexture2D>::const_iterator begin = _2d_textures.begin();
	std::vector<const ImmutableTexture2D>::const_iterator end = _2d_textures.begin() + num_textures;

	//Check if all textures are having same dimensions
	TextureSize reference_texture_size = _2d_textures[0].getTextureSize();
	if (std::find_if(begin, end,
		[&reference_texture_size](const ImmutableTexture2D texture) -> bool
	{
		TextureSize current_texture_size = texture.getTextureSize();
		return current_texture_size.width != reference_texture_size.width ||
			current_texture_size.height != reference_texture_size.height;
	}) != end)
	{
		set_error_state(true);
		const char* err_msg = "Unable to construct cubemap texture from the given 2D textures: all 2D textures participating in construction of a cubemap must be of same size";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return false;
	}

	//Check if all textures are having same number of LOD-levels
	uint32_t reference_number_of_mipmaps = _2d_textures[0].getNumberOfMipmapLevels();
	if (std::find_if(begin, end,
		[reference_number_of_mipmaps](const ImmutableTexture2D texture) -> bool
	{
		return texture.getNumberOfMipmapLevels() != reference_number_of_mipmaps;
	}) != end)
	{
		set_error_state(true);
		const char* err_msg = "Unable to construct cubemap texture from the given 2D textures: all 2D textures participating in construction of a cubemap must have same number of mipmap-levels";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return false;
	}

	//Check if all textures are using the same internal pixel storage format
	GLenum reference_internal_pixel_storage_format = _2d_textures[0].getOpenGLStorageInternalFormat();
	if (std::find_if(begin, end,
		[reference_internal_pixel_storage_format](const ImmutableTexture2D texture) -> bool
	{
		return texture.getOpenGLStorageInternalFormat() != reference_internal_pixel_storage_format;
	}) != end)
	{
		set_error_state(true);
		const char* err_msg = "Unable to construct cubemap texture from the given 2D textures: all 2D textures participating in construction of a cubemap must use the same internal pixel storage format";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return false;
	}

	//Compute number of array layers in the resulting cubemap
	size_t resulting_cubemap_layers = 0;
	for (std::vector<const ImmutableTexture2D>::const_iterator texture_group = begin; texture_group < end; texture_group += 6)
	{
		//Check if all textures in the current texture group are having same number of array layers
		uint32_t reference_number_of_array_layers = texture_group->getNumberOfArrayLayers();
		if (std::find_if(texture_group, texture_group + 6,
			[reference_number_of_array_layers](const ImmutableTexture2D& texture) -> bool
		{
			return texture.getNumberOfArrayLayers() != reference_number_of_array_layers;
		}) != texture_group + 6)
		{
			set_error_state(true);
			const char* err_msg = "Unable to construct cubemap texture from the given 2D textures: all 2D textures participating in construction of a cubemap are divided into groups of 6 textures "
				"with all textures contained within the same group required to have same number of texture array layers";
			set_error_string(err_msg);
			call_error_callback(err_msg);
			return false;
		}

		resulting_cubemap_layers += reference_number_of_array_layers;
	}
	    
	//Allocate storage for the new cubemap texture object (note: allocation is done differently for compressed and non-compressed textures)
	if (isCompressed())
		allocateStorage(reference_number_of_mipmaps, resulting_cubemap_layers, reference_texture_size,
		static_cast<InternalPixelFormatCompressed>(reference_internal_pixel_storage_format));
	else
		allocateStorage(reference_number_of_mipmaps, resulting_cubemap_layers, reference_texture_size,
		static_cast<InternalPixelFormat>(reference_internal_pixel_storage_format));

	//Generate name for the cubemap texture
	std::string texture_name = "";
	std::for_each(begin, end, [&texture_name](const ImmutableTexture2D texture) -> void{ texture_name += texture.getStringName() + "_"; });
	texture_name += "combined_cubemap";



	//Get currently active pixel storage pack alignment settings
	GLint pixel_storage_pack_alignment;
	glGetIntegerv(GL_PACK_ALIGNMENT, &pixel_storage_pack_alignment);

	//Change pixel unpack alignment settings so that it corresponds to the pack alignment
	GLint pixel_storage_unpack_alignment;
	glGetIntegerv(GL_UNPACK_ALIGNMENT, &pixel_storage_unpack_alignment);
	glPixelStorei(GL_UNPACK_ALIGNMENT, pixel_storage_pack_alignment);

	//Construct format traits of internal storage format used by the source 2D textures
	PixelFormatTraits pixel_format_traits = _2d_textures[0].getStorageFormatTraits();
	const short pixel_size = pixel_format_traits.getOptimalStorageSize();	//returns minimal number of bytes that is enough to represent a single pixel for the given internal storage format


	for (int mipmap_level = 0; mipmap_level < static_cast<int>(reference_number_of_mipmaps); ++mipmap_level)
	{
		uint32_t mipmap_level_width = std::max(1U, reference_texture_size.width >> mipmap_level);
		uint32_t mipmap_level_height = std::max(1U, reference_texture_size.height >> mipmap_level);

		//Compute padding in each row of uncompressed texture data written to the client memory
		const uint32_t row_padding = pixel_storage_pack_alignment - ((pixel_size*mipmap_level_width) % pixel_storage_pack_alignment ?
			(pixel_size*mipmap_level_width) % pixel_storage_pack_alignment : pixel_storage_pack_alignment);

		void* img_data = nullptr;
		size_t image_data_size = 0;
		for (std::vector<const ImmutableTexture2D>::const_iterator texture_group = begin; texture_group < end; texture_group += 6)
		{
			//All textures in the same cubemap assembly group are having same number of array layers and are using the same pixel storage format. Therefore their size in bytes is same
			size_t proxy_buffer_size;
			if (isCompressed())
				texture_group->getMipmapLevelImageData(mipmap_level, &proxy_buffer_size, nullptr, nullptr, nullptr);
			else
				proxy_buffer_size = 6 * (pixel_size*mipmap_level_width + row_padding)*mipmap_level_height*texture_group->getNumberOfArrayLayers();
			image_data_size += 6 * proxy_buffer_size;
		}


		img_data = malloc(image_data_size);
		size_t texture_group_shift = 0;
		for (std::vector<const ImmutableTexture2D>::const_iterator texture_group = begin; texture_group < end; texture_group += 6)
		{
			size_t proxy_buffer_size;
			void* proxy_buffer[6];

			//All textures in the same cubemap assembly group are having same number of array layers and are using the same compression format. Therefore their size in bytes is same
			if (isCompressed())
				texture_group->getMipmapLevelImageData(mipmap_level, &proxy_buffer_size, nullptr, nullptr, nullptr);
			else
				proxy_buffer_size = (pixel_size*mipmap_level_width + row_padding)*mipmap_level_height*texture_group->getNumberOfArrayLayers();

			for (int face = 0; face < 6; ++face)
			{
				proxy_buffer[face] = malloc(proxy_buffer_size);
				(texture_group + face)->getMipmapLevelImageData(mipmap_level, nullptr, nullptr, nullptr, proxy_buffer[face]);
			}


			for (uint32_t layer = 0; layer < texture_group->getNumberOfArrayLayers(); ++layer)
			{
				size_t layer_shift = proxy_buffer_size / texture_group->getNumberOfArrayLayers();

				for (int face = 0; face < 6; ++face)
					memcpy(static_cast<char*>(img_data)+texture_group_shift + (6 * layer + face)*layer_shift,
					static_cast<char*>(proxy_buffer[face]) + layer*layer_shift, layer_shift);
			}

			for (int face = 0; face < 6; ++face)
				free(proxy_buffer[face]);
			texture_group_shift += 6 * proxy_buffer_size;
		}

		if (isCompressed())
			setMipmapLevelMultiLayerFacesData(mipmap_level, 0, resulting_cubemap_layers * 6, static_cast<InternalPixelFormatCompressed>(reference_internal_pixel_storage_format), image_data_size, img_data);
		else
			setMipmapLevelMultiLayerFacesData(mipmap_level, 0, resulting_cubemap_layers * 6, pixel_format_traits.getPixelLayout(), pixel_format_traits.getOptimalStorageType(), img_data);

		free(img_data);
	}
	
	//Restore the old settings of unpacking alignment
	glPixelStorei(GL_UNPACK_ALIGNMENT, pixel_storage_unpack_alignment);

	return true;
}


void ImmutableTextureCubeMap::setMipmapLevelData(uint32_t mipmap_level, PixelLayout pixel_layout, PixelDataType pixel_component_type,
	const void* positive_x, const void* negative_x, const void* positive_y, const void* negative_y, const void* positive_z, const void* negative_z)
{
	setMipmapLevelLayerData(mipmap_level, 0, pixel_layout, pixel_component_type, positive_x, negative_x, positive_y, negative_y, positive_z, negative_z);
}


void ImmutableTextureCubeMap::setMipmapLevelData(uint32_t mipmap_level, InternalPixelFormatCompressed compressed_data_type, size_t compressed_data_size,
	const void* positive_x, const void* negative_x, const void* positive_y, const void* negative_y, const void* positive_z, const void* negative_z)
{
	setMipmapLevelLayerData(mipmap_level, 0, compressed_data_type, compressed_data_size, positive_x, negative_x, positive_y, negative_y, positive_z, negative_z);
}


void ImmutableTextureCubeMap::setMipmapLevelData(uint32_t mipmap_level, CubemapFace face, PixelLayout pixel_layout, PixelDataType pixel_component_type, const void* data)
{
	setMipmapLevelLayerData(mipmap_level, 0, face, pixel_layout, pixel_component_type, data);
}


void ImmutableTextureCubeMap::setMipmapLevelData(uint32_t mipmap_level, CubemapFace face, InternalPixelFormatCompressed compressed_data_type, size_t compressed_data_size, const void* data)
{
	setMipmapLevelLayerData(mipmap_level, 0, face, compressed_data_type, compressed_data_size, data);
}


void ImmutableTextureCubeMap::setMipmapLevelLayerData(uint32_t mipmap_level, uint32_t array_layer,
	PixelLayout pixel_layout, PixelDataType pixel_component_type,
	const void* positive_x, const void* negative_x, const void* positive_y, const void* negative_y, const void* positive_z, const void* negative_z)
{
	setMipmapLevelMultiLayersData(mipmap_level, array_layer, 1, pixel_layout, pixel_component_type, positive_x, negative_x, positive_y, negative_y, positive_z, negative_z);
}


void ImmutableTextureCubeMap::setMipmapLevelLayerData(uint32_t mipmap_level, uint32_t array_layer, 
	InternalPixelFormatCompressed compressed_data_type, size_t compressed_data_size, 
	const void* positive_x, const void* negative_x, const void* positive_y, const void* negative_y, const void* positive_z, const void* negative_z)
{
	setMipmapLevelMultiLayersData(mipmap_level, array_layer, 1, compressed_data_type, compressed_data_size, positive_x, negative_x, positive_y, negative_y, positive_z, negative_z);
}


void ImmutableTextureCubeMap::setMipmapLevelLayerData(uint32_t mipmap_level, uint32_t array_layer, CubemapFace face,
	PixelLayout pixel_layout, PixelDataType pixel_component_type, const void* data)
{
	setMipmapLevelMultiLayersData(mipmap_level, array_layer, 1, face, pixel_layout, pixel_component_type, data);
}


void ImmutableTextureCubeMap::setMipmapLevelLayerData(uint32_t mipmap_level, uint32_t array_layer, CubemapFace face, 
	InternalPixelFormatCompressed compressed_data_type, size_t compressed_data_size, const void* data)
{
	setMipmapLevelMultiLayersData(mipmap_level, array_layer, 1, face, compressed_data_type, compressed_data_size, data);
}


void ImmutableTextureCubeMap::setMipmapLevelMultiLayersData(uint32_t mipmap_level, uint32_t start_array_layer, uint32_t number_of_array_layers, CubemapFace face, PixelLayout pixel_layout, PixelDataType pixel_component_type, const void* data)
{
	switch (face)
	{
	case CubemapFace::POSITIVE_X:
		setMipmapLevelMultiLayersData(mipmap_level, start_array_layer, number_of_array_layers, pixel_layout, pixel_component_type, data, nullptr, nullptr, nullptr, nullptr, nullptr);
		break;
	case CubemapFace::NEGATIVE_X:
		setMipmapLevelMultiLayersData(mipmap_level, start_array_layer, number_of_array_layers, pixel_layout, pixel_component_type, nullptr, data, nullptr, nullptr, nullptr, nullptr);
		break;
	case CubemapFace::POSITIVE_Y:
		setMipmapLevelMultiLayersData(mipmap_level, start_array_layer, number_of_array_layers, pixel_layout, pixel_component_type, nullptr, nullptr, data, nullptr, nullptr, nullptr);
		break;
	case CubemapFace::NEGATIVE_Y:
		setMipmapLevelMultiLayersData(mipmap_level, start_array_layer, number_of_array_layers, pixel_layout, pixel_component_type, nullptr, nullptr, nullptr, data, nullptr, nullptr);
		break;
	case CubemapFace::POSITIVE_Z:
		setMipmapLevelMultiLayersData(mipmap_level, start_array_layer, number_of_array_layers, pixel_layout, pixel_component_type, nullptr, nullptr, nullptr, nullptr, data, nullptr);
		break;
	case CubemapFace::NEGATIVE_Z:
		setMipmapLevelMultiLayersData(mipmap_level, start_array_layer, number_of_array_layers, pixel_layout, pixel_component_type, nullptr, nullptr, nullptr, nullptr, nullptr, data);
		break;
	}
}


void ImmutableTextureCubeMap::setMipmapLevelMultiLayersData(uint32_t mipmap_level, uint32_t start_array_layer, uint32_t number_of_array_layers, CubemapFace face, InternalPixelFormatCompressed compressed_data_format, size_t compressed_data_size, const void* data)
{
	switch (face)
	{
	case CubemapFace::POSITIVE_X:
		setMipmapLevelMultiLayersData(mipmap_level, start_array_layer, number_of_array_layers, compressed_data_format, compressed_data_size, data, nullptr, nullptr, nullptr, nullptr, nullptr);
		break;
	case CubemapFace::NEGATIVE_X:
		setMipmapLevelMultiLayersData(mipmap_level, start_array_layer, number_of_array_layers, compressed_data_format, compressed_data_size, nullptr, data, nullptr, nullptr, nullptr, nullptr);
		break;
	case CubemapFace::POSITIVE_Y:
		setMipmapLevelMultiLayersData(mipmap_level, start_array_layer, number_of_array_layers, compressed_data_format, compressed_data_size, nullptr, nullptr, data, nullptr, nullptr, nullptr);
		break;
	case CubemapFace::NEGATIVE_Y:
		setMipmapLevelMultiLayersData(mipmap_level, start_array_layer, number_of_array_layers, compressed_data_format, compressed_data_size, nullptr, nullptr, nullptr, data, nullptr, nullptr);
		break;
	case CubemapFace::POSITIVE_Z:
		setMipmapLevelMultiLayersData(mipmap_level, start_array_layer, number_of_array_layers, compressed_data_format, compressed_data_size, nullptr, nullptr, nullptr, nullptr, data, nullptr);
		break;
	case CubemapFace::NEGATIVE_Z:
		setMipmapLevelMultiLayersData(mipmap_level, start_array_layer, number_of_array_layers, compressed_data_format, compressed_data_size, nullptr, nullptr, nullptr, nullptr, nullptr, data);
		break;
	}
}


void ImmutableTextureCubeMap::setMipmapLevelMultiLayersData(uint32_t mipmap_level, uint32_t start_array_layer, uint32_t number_of_array_layers, PixelLayout pixel_layout, PixelDataType pixel_component_type, 
	const void* positive_x, const void* negative_x, const void* positive_y, const void* negative_y, const void* positive_z, const void* negative_z)
{
	if (!isInitialized())
	{
		set_error_state(true);
		std::string err_msg = "Unable to set new data for requested array layers " + std::to_string(start_array_layer) + "-" + std::to_string(start_array_layer + number_of_array_layers - 1) +
			" of mipmap level " + std::to_string(mipmap_level) + " of cubemap texture \"" + getStringName() + "\" for faces " +
			(positive_x ? "" : "+X ") + (negative_x ? "" : "-X ") + (positive_y ? "" : "+Y ") + (negative_y ? "" : "-Y ") + (positive_z ? "" : "+Z ") + (negative_z ? "" : "-Z ") +
			". The texture was not initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	if (start_array_layer + number_of_array_layers - 1 >= getNumberOfArrayLayers())
	{
		set_error_state(true);
		std::string err_msg = "Unable to set new data for requested array layers " + std::to_string(start_array_layer) + "-" + std::to_string(start_array_layer + number_of_array_layers - 1) +
			"of mipmap level " + std::to_string(mipmap_level) + " of cubemap texture \"" + getStringName() + "\" for faces " +
			(positive_x ? "" : "+X ") + (negative_x ? "" : "-X ") + (positive_y ? "" : "+Y ") + (negative_y ? "" : "-Y ") + (positive_z ? "" : "+Z ") + (negative_z ? "" : "-Z ") +
			". Indexes " + std::to_string(getNumberOfArrayLayers()) + "-" + std::to_string(start_array_layer + number_of_array_layers - 1) + " are out of range";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	//Retrieve the old texture binding and bind current texture to the context
	TextureSize texture_size = getTextureSize();
	GLuint gl_old_texture_id = bind();

	uint32_t width = std::max(1U, texture_size.width >> mipmap_level);
	uint32_t height = std::max(1U, texture_size.height >> mipmap_level);


	if (isArrayTexture())
	{
		GLint unpack_alignment;
		glGetIntegerv(GL_UNPACK_ALIGNMENT, &unpack_alignment);
		unsigned short pixel_storage_size = PixelDataTraits{ pixel_layout, pixel_component_type }.getPixelStorageSize();
		uint32_t row_padding = unpack_alignment - (width*pixel_storage_size%unpack_alignment ? width*pixel_storage_size%unpack_alignment : unpack_alignment);
		size_t stride = (pixel_storage_size*width + row_padding)*height;

		for (uint32_t layer = 0; layer < number_of_array_layers; ++layer)
		{
			if (positive_x)
				glTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, mipmap_level, 0, 0, (start_array_layer + layer) * 6 + 0, width, height, 1,
				static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_component_type), static_cast<const char*>(positive_x)+stride*layer);

			if (negative_x)
				glTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, mipmap_level, 0, 0, (start_array_layer + layer) * 6 + 1, width, height, 1,
				static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_component_type), static_cast<const char*>(negative_x)+stride*layer);

			if (positive_y)
				glTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, mipmap_level, 0, 0, (start_array_layer + layer) * 6 + 2, width, height, 1,
				static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_component_type), static_cast<const char*>(positive_y)+stride*layer);

			if (negative_y)
				glTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, mipmap_level, 0, 0, (start_array_layer + layer) * 6 + 3, width, height, 1,
				static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_component_type), static_cast<const char*>(negative_y)+stride*layer);

			if (positive_z)
				glTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, mipmap_level, 0, 0, (start_array_layer + layer) * 6 + 4, width, height, 1,
				static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_component_type), static_cast<const char*>(positive_z)+stride*layer);

			if (negative_z)
				glTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, mipmap_level, 0, 0, (start_array_layer + layer) * 6 + 5, width, height, 1,
				static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_component_type), static_cast<const char*>(negative_z)+stride*layer);
		}
	}
	else
	{
		if (positive_x)
			glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, mipmap_level, 0, 0, width, height, static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_component_type), positive_x);

		if (negative_x)
			glTexSubImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, mipmap_level, 0, 0, width, height, static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_component_type), negative_x);

		if (positive_y)
			glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, mipmap_level, 0, 0, width, height, static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_component_type), positive_y);

		if (negative_y)
			glTexSubImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, mipmap_level, 0, 0, width, height, static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_component_type), negative_y);

		if (positive_z)
			glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, mipmap_level, 0, 0, width, height, static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_component_type), positive_z);

		if (negative_z)
			glTexSubImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, mipmap_level, 0, 0, width, height, static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_component_type), negative_z);
	}

	//Restore the old texture binding
	glBindTexture(getOpenGLTextureBinding().gl_texture_target, gl_old_texture_id);
}


void ImmutableTextureCubeMap::setMipmapLevelMultiLayersData(uint32_t mipmap_level, uint32_t start_array_layer, uint32_t number_of_array_layers, InternalPixelFormatCompressed compressed_data_format, size_t compressed_data_size, 
	const void* positive_x, const void* negative_x, const void* positive_y, const void* negative_y, const void* positive_z, const void* negative_z)
{
	if (!isInitialized())
	{
		set_error_state(true);
		std::string err_msg = "Unable to set new data for requested array layers " + std::to_string(start_array_layer) + "-" + std::to_string(start_array_layer + number_of_array_layers - 1) +
			" of mipmap level " + std::to_string(mipmap_level) + " of cubemap texture \"" + getStringName() + "\" for faces " +
			(positive_x ? "" : "+X ") + (negative_x ? "" : "-X ") + (positive_y ? "" : "+Y ") + (negative_y ? "" : "-Y ") + (positive_z ? "" : "+Z ") + (negative_z ? "" : "-Z ") +
			". The texture was not initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	if (start_array_layer + number_of_array_layers - 1 >= getNumberOfArrayLayers())
	{
		set_error_state(true);
		std::string err_msg = "Unable to set new data for requested array layers " + std::to_string(start_array_layer) + "-" + std::to_string(start_array_layer + number_of_array_layers - 1) +
			"of mipmap level " + std::to_string(mipmap_level) + " of cubemap texture \"" + getStringName() + "\" for faces " +
			(positive_x ? "" : "+X ") + (negative_x ? "" : "-X ") + (positive_y ? "" : "+Y ") + (negative_y ? "" : "-Y ") + (positive_z ? "" : "+Z ") + (negative_z ? "" : "-Z ") +
			". Indexes " + std::to_string(getNumberOfArrayLayers()) + "-" + std::to_string(start_array_layer + number_of_array_layers - 1) + " are out of range";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	//Retrieve the old texture binding and bind current texture to the context
	TextureSize texture_size = getTextureSize();
	GLuint gl_old_texture_id = bind();

	uint32_t width = std::max(1U, texture_size.width >> mipmap_level);
	uint32_t height = std::max(1U, texture_size.height >> mipmap_level);


	if (isArrayTexture())
	{
		size_t stride = compressed_data_size / number_of_array_layers;

		for (uint32_t layer = 0; layer < number_of_array_layers; ++layer)
		{
			if (positive_x)
				glCompressedTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, mipmap_level, 0, 0, (start_array_layer + layer) * 6 + 0, width, height, 1,
				static_cast<GLenum>(compressed_data_format), stride, static_cast<const char*>(positive_x)+stride*layer);

			if (negative_x)
				glCompressedTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, mipmap_level, 0, 0, (start_array_layer + layer) * 6 + 1, width, height, 1,
				static_cast<GLenum>(compressed_data_format), stride, static_cast<const char*>(negative_x)+stride*layer);

			if (positive_y)
				glCompressedTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, mipmap_level, 0, 0, (start_array_layer + layer) * 6 + 2, width, height, 1,
				static_cast<GLenum>(compressed_data_format), stride, static_cast<const char*>(positive_y)+stride*layer);

			if (negative_y)
				glCompressedTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, mipmap_level, 0, 0, (start_array_layer + layer) * 6 + 3, width, height, 1,
				static_cast<GLenum>(compressed_data_format), stride, static_cast<const char*>(negative_y)+stride*layer);

			if (positive_z)
				glCompressedTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, mipmap_level, 0, 0, (start_array_layer + layer) * 6 + 4, width, height, 1,
				static_cast<GLenum>(compressed_data_format), stride, static_cast<const char*>(positive_z)+stride*layer);

			if (negative_z)
				glCompressedTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, mipmap_level, 0, 0, (start_array_layer + layer) * 6 + 5, width, height, 1,
				static_cast<GLenum>(compressed_data_format), stride, static_cast<const char*>(negative_z)+stride*layer);
		}
	}
	else
	{
		if (positive_x)
			glCompressedTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, mipmap_level, 0, 0, width, height, static_cast<GLenum>(compressed_data_format), compressed_data_size, positive_x);

		if (negative_x)
			glCompressedTexSubImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, mipmap_level, 0, 0, width, height, static_cast<GLenum>(compressed_data_format), compressed_data_size, negative_x);

		if (positive_y)
			glCompressedTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, mipmap_level, 0, 0, width, height, static_cast<GLenum>(compressed_data_format), compressed_data_size, positive_y);

		if (negative_y)
			glCompressedTexSubImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, mipmap_level, 0, 0, width, height, static_cast<GLenum>(compressed_data_format), compressed_data_size, negative_y);

		if (positive_z)
			glCompressedTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, mipmap_level, 0, 0, width, height, static_cast<GLenum>(compressed_data_format), compressed_data_size, positive_z);

		if (negative_z)
			glCompressedTexSubImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, mipmap_level, 0, 0, width, height, static_cast<GLenum>(compressed_data_format), compressed_data_size, negative_z);
	}

	//Restore the old texture binding
	glBindTexture(getOpenGLTextureBinding().gl_texture_target, gl_old_texture_id);
}


void ImmutableTextureCubeMap::setMipmapLevelMultiLayerFacesData(uint32_t mipmap_level, uint32_t start_layer_face, uint32_t number_of_layer_faces, PixelLayout pixel_layout, PixelDataType pixel_component_type, const void* data)
{
	if (!isInitialized())
	{
		set_error_state(true);
		std::string err_msg = "Unable to assign new data for requested layer-faces " + std::to_string(start_layer_face) + "-" + std::to_string(start_layer_face + number_of_layer_faces - 1) +
			"of cubemap texture \"" + getStringName() + "\": the texture was not initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	if (start_layer_face + number_of_layer_faces - 1 >= 6*getNumberOfArrayLayers())
	{
		set_error_state(true);
		std::string err_msg = "Unable to assign new data for requested layer-faces " + std::to_string(start_layer_face) + "-" + std::to_string(start_layer_face + number_of_layer_faces - 1) +
			"of cubemap texture \"" + getStringName() + "\". Indexes " + std::to_string(6 * getNumberOfArrayLayers()) + "-" + std::to_string(start_layer_face + number_of_layer_faces - 1) +
			" are out of range";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	//Retrieve the old texture binding and bind current texture to the context
	TextureSize texture_size = getTextureSize();
	GLuint gl_old_texture_id = bind();

	uint32_t width = std::max(1U, texture_size.width >> mipmap_level);
	uint32_t height = std::max(1U, texture_size.height >> mipmap_level);


	if (isArrayTexture())
		glTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, mipmap_level, 0, 0, start_layer_face, width, height, number_of_layer_faces,
		static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_component_type), data);
	else
	{
		GLint unpack_padding;
		glGetIntegerv(GL_UNPACK_ALIGNMENT, &unpack_padding);
		unsigned short pixel_storage_size = PixelDataTraits{ pixel_layout, pixel_component_type }.getPixelStorageSize();
		uint32_t row_padding = unpack_padding - (width*pixel_storage_size%unpack_padding ? width*pixel_storage_size%unpack_padding : unpack_padding);
		size_t stride = (width*pixel_storage_size + row_padding)*height;

		for (int face = 0; face < 6; ++face)
			glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, mipmap_level, 0, 0, width, height,
			static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_component_type), static_cast<const char*>(data)+face*stride);
	}


	//Restore the old texture binding
	glBindTexture(getOpenGLTextureBinding().gl_texture_target, gl_old_texture_id);
}


void ImmutableTextureCubeMap::setMipmapLevelMultiLayerFacesData(uint32_t mipmap_level, uint32_t start_layer_face, uint32_t number_of_layer_faces, InternalPixelFormatCompressed compressed_data_format, size_t compressed_data_size, const void* data)
{
	if (!isInitialized())
	{
		set_error_state(true);
		std::string err_msg = "Unable to assign new data for requested layer-faces " + std::to_string(start_layer_face) + "-" + std::to_string(start_layer_face + number_of_layer_faces - 1) +
			"of cubemap texture \"" + getStringName() + "\": the texture was not initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	if (start_layer_face + number_of_layer_faces - 1 >= 6 * getNumberOfArrayLayers())
	{
		set_error_state(true);
		std::string err_msg = "Unable to assign new data for requested layer-faces " + std::to_string(start_layer_face) + "-" + std::to_string(start_layer_face + number_of_layer_faces - 1) +
			"of cubemap texture \"" + getStringName() + "\". Indexes " + std::to_string(6 * getNumberOfArrayLayers()) + "-" + std::to_string(start_layer_face + number_of_layer_faces - 1) +
			" are out of range";
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
		glCompressedTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, mipmap_level, 0, 0, start_layer_face, width, height, number_of_layer_faces,
		static_cast<GLenum>(compressed_data_format), static_cast<GLsizei>(compressed_data_size), data);
	else
	{
		size_t stride = compressed_data_size / 6;

		for (int face = 0; face < 6; ++face)
			glCompressedTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, mipmap_level, 0, 0, width, height,
			static_cast<GLenum>(compressed_data_format), static_cast<GLsizei>(compressed_data_size), static_cast<const char*>(data)+face*stride);
	}


	//Restore the old texture binding
	glBindTexture(getOpenGLTextureBinding().gl_texture_target, gl_old_texture_id);
}


void ImmutableTextureCubeMap::getMipmapLevelImageData(GLint mipmap_level, CubemapFace face, PixelLayout pixel_layout, PixelDataType pixel_component_type, TextureSize *img_size, void *data) const
{
	if (!isInitialized())
	{
		set_error_state(true);
		std::string err_msg = "Unable to retrieve data from cubemap texture \"" + getStringName() + "\": the texture has not been initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	GLuint gl_current_texture_id = bind();

	GLint width, height;
	GLenum texture_target = isArrayTexture() ? GL_TEXTURE_CUBE_MAP_ARRAY : static_cast<GLenum>(face);
	glGetTexLevelParameteriv(texture_target, mipmap_level, GL_TEXTURE_WIDTH, &width);
	glGetTexLevelParameteriv(texture_target, mipmap_level, GL_TEXTURE_HEIGHT, &height);
	
	
	if (img_size)
	{
		img_size->width = width;
		img_size->height = height;
		img_size->depth = 0;	//Cubemap textures are 2D-textures composed of six rectangular images corresponding to faces of a cube. They do not have depth, therefore it defaults to 0.
	}

	if (data)
	{
		if (isArrayTexture())
		{
			GLenum internal_storage_format = getOpenGLStorageInternalFormat();

			GLint pack_padding; glGetIntegerv(GL_PACK_ALIGNMENT, &pack_padding);
			unsigned short pixel_size = PixelDataTraits{ pixel_layout, pixel_component_type }.getPixelStorageSize();
			uint32_t row_padding = pack_padding - (width*pixel_size%pack_padding ? width*pixel_size%pack_padding : pack_padding);

			size_t proxy_buffer_size = (width*pixel_size + row_padding)*height*getNumberOfArrayLayers() * 6;
			void* proxy_buffer = malloc(proxy_buffer_size);
			glGetTexImage(GL_TEXTURE_CUBE_MAP_ARRAY, mipmap_level, static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_component_type), proxy_buffer);
			size_t layer_shift = (width*pixel_size + row_padding)*height;
			
			for (uint32_t layer = 0; layer < getNumberOfArrayLayers(); ++layer)
				memcpy(static_cast<char*>(data)+layer_shift*layer,
				static_cast<char*>(proxy_buffer)+(6 * layer + static_cast<GLenum>(face)-GL_TEXTURE_CUBE_MAP_POSITIVE_X)*layer_shift, layer_shift);
			free(proxy_buffer);
		}
		else
			glGetTexImage(static_cast<GLenum>(face), mipmap_level, static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_component_type), data);
	}
	
	glBindTexture(getOpenGLTextureBinding().gl_texture_target, gl_current_texture_id);	//Restore the old binding
}

void ImmutableTextureCubeMap::getMipmapLevelImageData(GLint mipmap_level, CubemapFace face, 
	size_t *compressed_data_size, InternalPixelFormatCompressed *compressed_format, TextureSize *img_size, void *data) const
{
	if (!isInitialized())
	{
		set_error_state(true);
		std::string err_msg = "Unable to retrieve data from cubemap texture \"" + getStringName() + "\": the texture has not been initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	if (!isCompressed())
	{
		set_error_state(true);
		std::string err_msg = "Unable to retrieve raw compressed data from cubemap texture \"" + getStringName() + "\": the texture does not use compression";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}


	GLuint gl_current_texture_id = bind();

	GLenum texture_target = isArrayTexture() ? GL_TEXTURE_CUBE_MAP_ARRAY : static_cast<GLenum>(face);
	GLint ogl_compressed_img_size; glGetTexLevelParameteriv(texture_target, mipmap_level, GL_TEXTURE_COMPRESSED_IMAGE_SIZE, &ogl_compressed_img_size);

	if (compressed_data_size)
		*compressed_data_size = isArrayTexture() ? static_cast<size_t>(ogl_compressed_img_size) / 6 : static_cast<size_t>(ogl_compressed_img_size);
	
	if (compressed_format)
	{
		GLint ogl_compressed_img_format;
		glGetTexLevelParameteriv(texture_target, mipmap_level, GL_TEXTURE_INTERNAL_FORMAT, &ogl_compressed_img_format);
		*compressed_format = static_cast<InternalPixelFormatCompressed>(ogl_compressed_img_format);
	}
		
	if (img_size)
	{
		GLint width, height;
		glGetTexLevelParameteriv(texture_target, mipmap_level, GL_TEXTURE_WIDTH, &width);
		glGetTexLevelParameteriv(texture_target, mipmap_level, GL_TEXTURE_HEIGHT, &height);

		img_size->width = width;
		img_size->height = height;
		img_size->depth = 0;	//Cubemaps are essentially 2D textures, so they do not have depth, which defaults to 0
	}

	if (data)
	{
		if (isArrayTexture())
		{
			void* proxy_buffer = malloc(ogl_compressed_img_size);
			glGetCompressedTexImage(GL_TEXTURE_CUBE_MAP_ARRAY, mipmap_level, proxy_buffer);
			size_t layer_shift = ogl_compressed_img_size / 6 / getNumberOfArrayLayers();

			for (uint32_t layer = 0; layer < getNumberOfArrayLayers(); ++layer)
				memcpy(static_cast<char*>(data)+layer*layer_shift,
				static_cast<char*>(proxy_buffer)+(6 * layer + static_cast<GLenum>(face)-GL_TEXTURE_CUBE_MAP_POSITIVE_X)*layer_shift, layer_shift);
			free(proxy_buffer);
		}
		else
			glGetCompressedTexImage(static_cast<GLenum>(face), mipmap_level, static_cast<GLvoid*>(data));
	}
		

	glBindTexture(getOpenGLTextureBinding().gl_texture_target, gl_current_texture_id);		//Restore old binding
}

Texture* ImmutableTextureCubeMap::clone() const
{
	return new ImmutableTextureCubeMap(*this);
}


ImmutableTextureCubeMap ImmutableTextureCubeMap::combine(const ImmutableTextureCubeMap& other) const
{
	//Check that both cubemaps being combined are having the same size of faces
	if (getTextureSize() != other.getTextureSize())
	{
		set_error_state(true);
		const char* err_msg = "Unable to combine two cubemap texture objects into cubemap array: cubemaps being combined must have faces of same size";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return *this;
	}

	//Check that both cubemaps being combined are having same number of mipmaps
	if (getNumberOfMipmapLevels() != other.getNumberOfMipmapLevels())
	{
		set_error_state(true);
		const char* err_msg = "Unable to combine two cubemap texture objects into cubemap array: cubemaps being combined must have the same number of mipmap-levels";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return *this;
	}

	//Check that both cubemaps being combined are using the same internal pixel storage format
	if (getOpenGLStorageInternalFormat() != other.getOpenGLStorageInternalFormat())
	{
		set_error_state(true);
		const char* err_msg = "Unable to combine two cubemap texture objects into cubemap array: cubemaps being combined must use the same internal pixel storage format";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return *this;
	}


	//Retrieve currently active pixel storage pack alignment setting
	GLint pixel_storage_pack_alignment;
	glGetIntegerv(GL_PACK_ALIGNMENT, &pixel_storage_pack_alignment);

	//Change pixel storage unpack alignment so that it coincides with pixel storage pack alignment
	GLint pixel_storage_unpack_alignment;
	glGetIntegerv(GL_UNPACK_ALIGNMENT, &pixel_storage_unpack_alignment);
	glPixelStorei(GL_UNPACK_ALIGNMENT, pixel_storage_pack_alignment);


	//Retrieve format traits of the internal pixel storage format used by the cubemap texture objects being combined
	PixelFormatTraits pixel_format_traits{ getOpenGLStorageInternalFormat() };
	const short pixel_size = pixel_format_traits.getOptimalStorageSize();


	//Construct new cubemap texture object
	ImmutableTextureCubeMap combined_cubemap{ getStringName() + "_" + other.getStringName() + "_combined" };

	if (isCompressed())
		combined_cubemap.allocateStorage(getNumberOfMipmapLevels(), getNumberOfArrayLayers() + other.getNumberOfArrayLayers(), getTextureSize(),
		static_cast<InternalPixelFormatCompressed>(getOpenGLStorageInternalFormat()));
	else
		combined_cubemap.allocateStorage(getNumberOfMipmapLevels(), getNumberOfArrayLayers() + other.getNumberOfArrayLayers(), getTextureSize(),
		static_cast<InternalPixelFormat>(getOpenGLStorageInternalFormat()));


	for (int mipmap_level = 0; mipmap_level < static_cast<int>(getNumberOfMipmapLevels()); ++mipmap_level)
	{
		uint32_t mipmap_level_width = std::max(1U, getTextureSize().width >> mipmap_level);
		uint32_t mipmap_level_height = std::max(1U, getTextureSize().height >> mipmap_level);

		void* img_data = nullptr;
		if (isCompressed())
		{
			//Get size of the first cubemap object
			//Note: all face of compressed cubemap objects must have the same dimensions and must use the same internal compression format. Hence, sizes of all 6 faces should be the same
			size_t _1st_component_compressed_face_size;
			getMipmapLevelImageData(mipmap_level, CubemapFace::POSITIVE_X, &_1st_component_compressed_face_size, nullptr, nullptr, nullptr);


			//Get size of the second cubemap object
			//Note: all face of compressed cubemap objects must have the same dimensions and must use the same internal compression format. Hence, sizes of all 6 faces should be the same
			size_t _2nd_component_compressed_face_size;
			other.getMipmapLevelImageData(mipmap_level, CubemapFace::POSITIVE_X, &_2nd_component_compressed_face_size, nullptr, nullptr, nullptr);

			//Allocate storage for auxiliary buffer
			const size_t storage_size = (_1st_component_compressed_face_size + _2nd_component_compressed_face_size) * 6;

			img_data = malloc(storage_size);

			//Gather texture data for the positive-X face of the cubemap
			getMipmapLevelImageData(mipmap_level, CubemapFace::POSITIVE_X, nullptr, nullptr, nullptr,
				img_data);
			
			other.getMipmapLevelImageData(mipmap_level, CubemapFace::POSITIVE_X, nullptr, nullptr, nullptr,
				static_cast<char*>(img_data)+_1st_component_compressed_face_size);

			//Gather texture data for the negative-X face of the cubemap
			getMipmapLevelImageData(mipmap_level, CubemapFace::NEGATIVE_X, nullptr, nullptr, nullptr,
				static_cast<char*>(img_data)+(_1st_component_compressed_face_size + _2nd_component_compressed_face_size));

			other.getMipmapLevelImageData(mipmap_level, CubemapFace::NEGATIVE_X, nullptr, nullptr, nullptr,
				static_cast<char*>(img_data)+2 * _1st_component_compressed_face_size + _2nd_component_compressed_face_size);

			//Gather texture data for the positive-Y face of the cubemap
			getMipmapLevelImageData(mipmap_level, CubemapFace::POSITIVE_Y, nullptr, nullptr, nullptr,
				static_cast<char*>(img_data)+2 * (_1st_component_compressed_face_size + _2nd_component_compressed_face_size));

			other.getMipmapLevelImageData(mipmap_level, CubemapFace::POSITIVE_Y, nullptr, nullptr, nullptr,
				static_cast<char*>(img_data)+3 * _1st_component_compressed_face_size + 2 * _2nd_component_compressed_face_size);

			//Gather texture data for the negative-Y face of the cubemap
			getMipmapLevelImageData(mipmap_level, CubemapFace::NEGATIVE_Y, nullptr, nullptr, nullptr,
				static_cast<char*>(img_data)+3 * (_1st_component_compressed_face_size + _2nd_component_compressed_face_size));

			other.getMipmapLevelImageData(mipmap_level, CubemapFace::NEGATIVE_Y, nullptr, nullptr, nullptr,
				static_cast<char*>(img_data)+4 * _1st_component_compressed_face_size + 3 * _2nd_component_compressed_face_size);

			//Gather texture data for the positive-Z face of the cubemap
			getMipmapLevelImageData(mipmap_level, CubemapFace::POSITIVE_Z, nullptr, nullptr, nullptr,
				static_cast<char*>(img_data)+4 * (_1st_component_compressed_face_size + _2nd_component_compressed_face_size));

			other.getMipmapLevelImageData(mipmap_level, CubemapFace::POSITIVE_Z, nullptr, nullptr, nullptr,
				static_cast<char*>(img_data)+5 * _1st_component_compressed_face_size + 6 * _2nd_component_compressed_face_size);

			//Gather texture data for the negative-Z face of the cubemap
			getMipmapLevelImageData(mipmap_level, CubemapFace::NEGATIVE_Z, nullptr, nullptr, nullptr,
				static_cast<char*>(img_data)+5 * (_1st_component_compressed_face_size + _2nd_component_compressed_face_size));

			other.getMipmapLevelImageData(mipmap_level, CubemapFace::NEGATIVE_Z, nullptr, nullptr, nullptr,
				static_cast<char*>(img_data)+6 * _1st_component_compressed_face_size + 5 * _2nd_component_compressed_face_size);


			//Define data for combined cubemap object
			combined_cubemap.setMipmapLevelMultiLayersData(mipmap_level, 0, getNumberOfArrayLayers() + other.getNumberOfArrayLayers(),
				static_cast<InternalPixelFormatCompressed>(getOpenGLStorageInternalFormat()),
				_1st_component_compressed_face_size + _2nd_component_compressed_face_size,
				img_data,
				static_cast<char*>(img_data)+(_1st_component_compressed_face_size + _2nd_component_compressed_face_size),
				static_cast<char*>(img_data)+2 * (_1st_component_compressed_face_size + _2nd_component_compressed_face_size),
				static_cast<char*>(img_data)+3 * (_1st_component_compressed_face_size + _2nd_component_compressed_face_size),
				static_cast<char*>(img_data)+4 * (_1st_component_compressed_face_size + _2nd_component_compressed_face_size),
				static_cast<char*>(img_data)+5 * (_1st_component_compressed_face_size + _2nd_component_compressed_face_size));
		}
		else
		{
			//Compute per-row pack padding applied when texture data is returned to client memory
			const uint32_t row_padding = pixel_storage_pack_alignment - (pixel_size * mipmap_level_width % pixel_storage_pack_alignment ?
				pixel_size * mipmap_level_width % pixel_storage_pack_alignment : pixel_storage_pack_alignment);

			//Compute size in bytes of a single texture array layer of the current mipmap level
			const size_t layer_shift = (pixel_size * mipmap_level_width + row_padding) * mipmap_level_height;

			//Compute size in bytes of the full storage for a single cubemap face
			const size_t storage_size = layer_shift * (getNumberOfArrayLayers() + other.getNumberOfArrayLayers());

			//Allocate auxiliary buffer for the image data
			img_data = malloc(6*storage_size);

			//Gather texture data for the positive-X face of the combined cubemap
			getMipmapLevelImageData(mipmap_level, static_cast<CubemapFace>(GL_TEXTURE_CUBE_MAP_POSITIVE_X),
				pixel_format_traits.getPixelLayout(), pixel_format_traits.getOptimalStorageType(), nullptr, 
				img_data);

			other.getMipmapLevelImageData(mipmap_level, static_cast<CubemapFace>(GL_TEXTURE_CUBE_MAP_POSITIVE_X),
				pixel_format_traits.getPixelLayout(), pixel_format_traits.getOptimalStorageType(), nullptr,
				static_cast<char*>(img_data)+layer_shift*getNumberOfArrayLayers());

			//Gather texture data for the negative-X face of the combined cubemap
			getMipmapLevelImageData(mipmap_level, static_cast<CubemapFace>(GL_TEXTURE_CUBE_MAP_NEGATIVE_X),
				pixel_format_traits.getPixelLayout(), pixel_format_traits.getOptimalStorageType(), nullptr,
				static_cast<char*>(img_data)+layer_shift*(getNumberOfArrayLayers() + other.getNumberOfArrayLayers()));

			other.getMipmapLevelImageData(mipmap_level, static_cast<CubemapFace>(GL_TEXTURE_CUBE_MAP_NEGATIVE_X),
				pixel_format_traits.getPixelLayout(), pixel_format_traits.getOptimalStorageType(), nullptr,
				static_cast<char*>(img_data)+layer_shift*(2 * getNumberOfArrayLayers() + other.getNumberOfArrayLayers()));

			//Gather texture data for the positive-Y face of the combined cubemap
			getMipmapLevelImageData(mipmap_level, static_cast<CubemapFace>(GL_TEXTURE_CUBE_MAP_POSITIVE_Y),
				pixel_format_traits.getPixelLayout(), pixel_format_traits.getOptimalStorageType(), nullptr,
				static_cast<char*>(img_data)+2*layer_shift*(getNumberOfArrayLayers() + other.getNumberOfArrayLayers()));

			other.getMipmapLevelImageData(mipmap_level, static_cast<CubemapFace>(GL_TEXTURE_CUBE_MAP_POSITIVE_Y),
				pixel_format_traits.getPixelLayout(), pixel_format_traits.getOptimalStorageType(), nullptr,
				static_cast<char*>(img_data)+layer_shift*(3 * getNumberOfArrayLayers() + 2 * other.getNumberOfArrayLayers()));

			//Gather texture data for the negative-Y face of the combined cubemap
			getMipmapLevelImageData(mipmap_level, static_cast<CubemapFace>(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y),
				pixel_format_traits.getPixelLayout(), pixel_format_traits.getOptimalStorageType(), nullptr,
				static_cast<char*>(img_data)+3 * layer_shift*(getNumberOfArrayLayers() + other.getNumberOfArrayLayers()));

			other.getMipmapLevelImageData(mipmap_level, static_cast<CubemapFace>(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y),
				pixel_format_traits.getPixelLayout(), pixel_format_traits.getOptimalStorageType(), nullptr,
				static_cast<char*>(img_data)+layer_shift*(4 * getNumberOfArrayLayers() + 3 * other.getNumberOfArrayLayers()));
			
			//Gather texture data for the positive-Z face of the combined cubemap
			getMipmapLevelImageData(mipmap_level, static_cast<CubemapFace>(GL_TEXTURE_CUBE_MAP_POSITIVE_Z),
				pixel_format_traits.getPixelLayout(), pixel_format_traits.getOptimalStorageType(), nullptr,
				static_cast<char*>(img_data)+4 * layer_shift*(getNumberOfArrayLayers() + other.getNumberOfArrayLayers()));

			other.getMipmapLevelImageData(mipmap_level, static_cast<CubemapFace>(GL_TEXTURE_CUBE_MAP_POSITIVE_Z),
				pixel_format_traits.getPixelLayout(), pixel_format_traits.getOptimalStorageType(), nullptr,
				static_cast<char*>(img_data)+layer_shift*(5 * getNumberOfArrayLayers() + 4 * other.getNumberOfArrayLayers()));

			//Gather texture data for the negative-Z face of the combined cubemap
			getMipmapLevelImageData(mipmap_level, static_cast<CubemapFace>(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z),
				pixel_format_traits.getPixelLayout(), pixel_format_traits.getOptimalStorageType(), nullptr,
				static_cast<char*>(img_data)+5 * layer_shift*(getNumberOfArrayLayers() + other.getNumberOfArrayLayers()));

			other.getMipmapLevelImageData(mipmap_level, static_cast<CubemapFace>(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z),
				pixel_format_traits.getPixelLayout(), pixel_format_traits.getOptimalStorageType(), nullptr,
				static_cast<char*>(img_data)+layer_shift*(6 * getNumberOfArrayLayers() + 5 * other.getNumberOfArrayLayers()));

			//Set face information for resulting combined cubemap
			combined_cubemap.setMipmapLevelMultiLayersData(mipmap_level, 0, getNumberOfArrayLayers() + other.getNumberOfArrayLayers(),
				pixel_format_traits.getPixelLayout(), pixel_format_traits.getOptimalStorageType(),
				img_data, static_cast<char*>(img_data)+storage_size, static_cast<char*>(img_data)+2 * storage_size, static_cast<char*>(img_data)+3 * storage_size,
				static_cast<char*>(img_data)+4 * storage_size, static_cast<char*>(img_data)+5 * storage_size);
		}
		free(img_data);
		
	}

	//Restore the old unpack alignment setting
	glPixelStorei(GL_UNPACK_ALIGNMENT, pixel_storage_unpack_alignment);

	return combined_cubemap;
}


bool ImmutableTextureCubeMap::copyTexelData(uint32_t source_mipmap_level, uint32_t source_first_array_layer, uint32_t source_offset_x, uint32_t source_offset_y,
	const ImmutableTextureCubeMap& destination_texture, uint32_t destination_mipmap_level, uint32_t destination_first_array_layer, uint32_t destination_offset_x, uint32_t destination_offset_y,
	uint32_t copy_buffer_width, uint32_t copy_buffer_height, uint32_t num_array_layers_to_copy) const
{
	if (getOpenGLStorageInternalFormat() != destination_texture.getOpenGLStorageInternalFormat()) return false;

	glCopyImageSubData(getOpenGLId(), isArrayTexture() ? GL_TEXTURE_CUBE_MAP_ARRAY : GL_TEXTURE_CUBE_MAP, source_mipmap_level, source_offset_x, source_offset_y, source_first_array_layer,
		destination_texture.getOpenGLId(), destination_texture.isArrayTexture() ? GL_TEXTURE_CUBE_MAP_ARRAY : GL_TEXTURE_CUBE_MAP, destination_mipmap_level, destination_offset_x, destination_offset_y, destination_first_array_layer,
		copy_buffer_width, copy_buffer_height, num_array_layers_to_copy);
	return true;
}