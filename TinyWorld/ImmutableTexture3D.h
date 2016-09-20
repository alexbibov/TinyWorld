#ifndef TW__IMMUTABLE_TEXTURE_3D__

#include "ImmutableTexture.h"


namespace tiny_world
{

	class ImmutableTexture3D final : public ImmutableTexture
	{
	private:
		TextureBinding perform_allocation() override;
		TextureDimension query_dimension() const override;

	public:

		//Below are the functions used to access and alter texture data


		//Texture data "setters":

		//Initiates pixel transfer operation that assigns data represented in an uncompressed format to the given mipmap level of contained texture object
		void setMipmapLevelData(uint32_t mipmap_level, PixelLayout pixel_layout, PixelDataType pixel_component_type, const void* data);

		//Initiates pixel transfer operation that assigns data represented in a compressed texture format to the given mipmap level of contained texture object
		void setMipmapLevelData(uint32_t mipmap_level, InternalPixelFormatCompressed compressed_storage_format, size_t compressed_data_size, const void* data);


		//Texture data "getters":

		//Initiates pixel transfer operation that extracts data from the given mipmap level of contained texture object, performs decompression if necessary and returns 
		//the data using requested pixel layout and data component type. 
		void getMipmapLevelImageData(uint32_t mipmap_level, PixelLayout pixel_layout, PixelDataType pixel_component_type, TextureSize* image_size, void* data) const;

		//Initiates pixel transfer operation that extracts raw compressed data from the given mipmap level of contained texture object. This function performs no background compression, which means that
		//it can only be called for compressed texture objects. Doing otherwise will put the object to an erroneous state.
		void getMipmapLevelImageData(uint32_t mipmap_level, size_t* compressed_data_size, InternalPixelFormatCompressed* compressed_storage_format, TextureSize* image_size, void* data) const;


		//Basic constructor/destructor infrastructure:

		ImmutableTexture3D();	//Default constructor
		explicit ImmutableTexture3D(const std::string& texture_string_name);	//Creates 3D-texture weakly identified by the given string name


		//Miscellaneous functions:

		Texture* clone() const override;	//clones "this" texture object and returns its fully equivalent copy to the caller


		//Copies texel data from this 3D texture to the given destination 3D texture. Note that the texel data can be copied only between two textures having same internal texel format.
		//The function returns 'true' on success and 'false' on failure. No error data besides the return value is generated in case of failure.
		bool copyTexelData(uint32_t source_mipmap_level, uint32_t source_offset_x, uint32_t source_offset_y, uint32_t source_offset_z,
			const ImmutableTexture3D& destination_texture, uint32_t destination_mipmap_level, uint32_t destination_offset_x, uint32_t destination_offset_y, uint32_t destination_offset_z,
			uint32_t copy_buffer_width, uint32_t copy_buffer_height, uint32_t copy_buffer_depth) const;
	};

}


#define TW__IMMUTABLE_TEXTURE_3D__
#endif