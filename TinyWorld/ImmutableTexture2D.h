#ifndef TW__IMMUTABLE_TEXTURE_2D__

#include "ImmutableTexture.h"

namespace tiny_world{


	class ImmutableTextureCubeMap;	//Forward declaration of the class implementing cubemap texture objects. This declaration is needed to establish friendship relationship between the classes


	class ImmutableTexture2D final : public ImmutableTexture
	{
		friend class ImmutableTextureCubeMap;
	private:
		TextureBinding perform_allocation() override;		//texture storage allocator
		TextureDimension query_dimension() const override;
	
	public:
		//Functions to access and alter texture data
		
		//Texture data "setters"

		//Initiates pixel transfer operation, which sets data provided in an uncompressed format for the  0-layer of the given mipmap level of contained texture
		void setMipmapLevelData(uint32_t mipmap_level, PixelLayout pixel_layout, PixelDataType pixel_component_type, const void* data);

		//Initiates pixel transfer operation, which sets data provided in a supported compressed format for the 0-layer of the given mipmap level of contained texture
		void setMipmapLevelData(uint32_t mipmap_level, InternalPixelFormatCompressed compressed_storage_format, size_t compressed_data_size, const void* data);
		
		//Initiates pixel transfer operation, which sets data provided in an uncompressed format for the given layer of the given mipmap level of contained texture
		void setMipmapLevelLayerData(uint32_t mipmap_level, uint32_t array_layer, PixelLayout pixel_layout, PixelDataType pixel_component_type, const void* data);

		//Initiates pixel transfer operation, which sets data provided in a supported compressed format for the given layer of the given mipmap level of contained texture
		void setMipmapLevelLayerData(uint32_t mipmap_level, uint32_t array_layer, InternalPixelFormatCompressed compressed_storage_format, size_t compressed_data_size, const void* data);

		//Initiates pixel transfer operation, which  assigns new texture data for multiple array layers of the given mipmap level of the texture. This allows to update several layers of the 
		//texture using only one function call resulting in better efficiency of the final code
		void setMipmapLevelMultiLayersData(uint32_t mipmap_level, uint32_t start_array_layer, uint32_t number_of_array_layers, PixelLayout pixel_layout, PixelDataType pixel_component_type, const void* data);

		//Initiates pixel transfer operation, which assigns new texture data represented using a compressed texture format for multiple array layers of the given mipmap level of the texture.
		//This allows to update data for several texture layers within a single function call resulting in better efficiency of the final code
		void setMipmapLevelMultiLayersData(uint32_t mipmap_level, uint32_t start_array_layer, uint32_t number_of_array_layers, InternalPixelFormatCompressed compressed_storage_format, size_t compressed_data_size, const void* data);
		

		//Texture data "getters"

		//Initializes pixel transfer operation, which extracts image data from the texture object and converts it into representation specified by requested pixel layout and data component type. This function can not extract individual
		//layers of the given mipmap level of an array texture in an efficient way. Hence, the function extract data for the FULL mipmap level, which means that all array layers are extracted at once.
		void getMipmapLevelImageData(uint32_t mipmap_level, PixelLayout pixel_layout, PixelDataType pixel_component_type, TextureSize* image_size, void* data) const;	
		
		//Extracts raw compressed data from a texture object that uses compression. Calling this function on an uncompressed texture object will put it to an erroneous state. This function can not extract individual
		//layers of the given mipmap level of an array texture in an efficient way. Hence, the function extract data for the FULL mipmap level, which means that all array layers are extracted at once.
		void getMipmapLevelImageData(uint32_t mipmap_level, size_t* compressed_data_size, InternalPixelFormatCompressed* compressed_storage_format, TextureSize* image_size, void* data) const;

		//Constructor infrastructure

		ImmutableTexture2D();	//Default constructor
		explicit ImmutableTexture2D(const std::string& texture_string_name);	//allows initialization using user-defined string name associated with the texture

		//Miscellaneous
		Texture* clone() const override;

		//Combines two 2D immutable textures (or even arrays) into a single 2D texture array. The function would only work if all the textures 
		//that are part of combination have same internal storage format, same size, and same number of mipmap levels. 
		//Otherwise the object will get to an erroneous state and behavior of the function is undefined. 
		ImmutableTexture2D combine(const ImmutableTexture2D& other) const;

		//Copies texel data contained in this 2D texture to the given destination 2D texture or 2D texture array. Note that the texel data can be copied only between two textures having same internal texel format.
		//The function returns 'true' on success and 'false' on failure. No error data besides the return value is generated in case of failure.
		bool copyTexelData(uint32_t source_mipmap_level, uint32_t source_first_array_layer, uint32_t source_offset_x, uint32_t source_offset_y,
			const ImmutableTexture2D& destination_texture, uint32_t destination_mipmap_level, uint32_t destination_first_array_layer, uint32_t destination_offset_x, uint32_t destination_offset_y,
			uint32_t copy_buffer_width, uint32_t copy_buffer_height, uint32_t num_array_layers_to_copy) const;
	};

}

#define TW__IMMUTABLE_TEXTURE_2D__
#endif