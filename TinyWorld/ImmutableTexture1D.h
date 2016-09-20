#ifndef TW__IMMUTABLE_TEXTURE_1D__

#include "ImmutableTexture.h"

namespace tiny_world
{
	class ImmutableTexture1D final : public ImmutableTexture
	{
		friend class ImmutableTexture2D;
	private:
		TextureBinding perform_allocation() override;
		TextureDimension query_dimension() const override;

	public:
		//The following functions are intended to access and alter texture data

		//Texture data "setters":

		//Initiates pixel transfer operation that supplies new uncompressed texture data for the base (i.e. zero) array layer of the given mipmap level of the texture object
		void setMipmapLevelData(uint32_t mipmap_level, PixelLayout pixel_layout, PixelDataType pixel_component_type, const void* data);

		//Initiates pixel transfer operation that supplies new texture data represented in a compressed texture format for the base (i.e. zero) array layer of the given mipmap level of the texture object
		void setMipmapLevelData(uint32_t mipmap_level, InternalPixelFormatCompressed compressed_storage_format, size_t compressed_data_size, const void* data);

		//Initiates pixel transfer operation that supplies new uncompressed texture data for the given array layer of the given mipmap level of the texture object
		void setMipmapLevelLayerData(uint32_t mipmap_level, uint32_t array_layer, PixelLayout pixel_layout, PixelDataType pixel_component_type, const void* data);

		//Initiates pixel transfer operation that supplies new texture data represented in a compressed texture format for the given array layer of the given mipmap level of the texture object
		void setMipmapLevelLayerData(uint32_t mipmap_level, uint32_t array_layer, InternalPixelFormatCompressed compressed_storage_format, size_t compressed_data_size, const void* data);

		//Initiates pixel transfer operation that supplies uncompressed texture data to multiple array layers of the given mipmap level of the texture object within a single call
		void setMipmapLevelMultiLayersData(uint32_t mipmap_level, uint32_t start_array_layer, uint32_t number_of_array_layers, PixelLayout pixel_layout, PixelDataType pixel_component_type, const void* data);

		//Initiates pixel transfer operation that supplies new texture data represented using compressed texture format to multiple array layers of the given mipmap level of 
		//the texture object within a single function call
		void setMipmapLevelMultiLayersData(uint32_t mipmap_level, uint32_t start_array_layer, uint32_t number_of_array_layers, InternalPixelFormatCompressed compressed_storage_format, size_t compressed_data_size, const void* data);


		//Texture data "getters":

		//Initiates pixel transfer operation that extract texture data from the given mipmap level of the texture object
		void getMipmapLevelImageData(uint32_t mipmap_level, PixelLayout pixel_layout, PixelDataType pixel_component_type, TextureSize *image_size, void* data) const;

		//Initiates pixel transfer operation that extract texture data in a raw compressed texture format from the given mipmap level of the texture object. This version of the function can only be called for textures that use
		//compressed internal pixel storage format. Calling this function for a texture that does not use compression results in an undefined behavior
		void getMipmapLevelImageData(uint32_t mipmap_level, size_t* compressed_data_size, InternalPixelFormatCompressed* compressed_storage_format, TextureSize* image_size, void* data) const;


		//Basic constructor/destructor infrastructure

		ImmutableTexture1D();	//default constructor
		explicit ImmutableTexture1D(const std::string& texture_string_name);	//Initializes texture using user-defied string name


		//Miscellaneous
		
		//Creates full copy of the texture object on heap memory and returns pointer to the newly created object using generic texture access interface
		Texture* clone() const override;

		//Combines two 1D textures (or texture arrays) into a single 1D texture array
		ImmutableTexture1D combine(const ImmutableTexture1D& other) const;

		//Copies texel data contained in this 1D texture to the given destination 1D texture or 1D texture array. Note that the texel data can be copied only between two textures having same internal texel format.
		//The function returns 'true' on success and 'false' on failure. No error data besides the return value is generated in case of failure.
		bool copyTexelData(uint32_t source_mipmap_level, uint32_t source_first_array_layer, uint32_t source_offset,
			const ImmutableTexture1D& destination_texture, uint32_t destination_mipmap_level, uint32_t destination_first_array_layer, uint32_t destination_offset, 
			uint32_t copy_buffer_width, uint32_t num_array_layers_to_copy) const;
	};
}

#define TW__IMMUTABLE_TEXTURE_1D__
#endif