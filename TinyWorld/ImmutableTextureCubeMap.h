#ifndef TW__IMMUTABLE_TEXTURE_CUBE__

#include "ImmutableTexture.h"
#include "ImmutableTexture2D.h"

#include <vector>

namespace tiny_world{

	enum class CubemapFace : GLenum{
		POSITIVE_X = GL_TEXTURE_CUBE_MAP_POSITIVE_X,
		NEGATIVE_X = GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
		POSITIVE_Y = GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
		NEGATIVE_Y = GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
		POSITIVE_Z = GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
		NEGATIVE_Z = GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
	};

	class ImmutableTextureCubeMap final : public ImmutableTexture
	{
	private:
		virtual TextureBinding perform_allocation() override;
		virtual TextureDimension query_dimension() const override;

		//Fills cubemap texture with data from an array of 2D textures. If length of the array is not a multiple of 6, the redundant textures are ignored. Returns 'true' on success,
		bool feed_data_from_2d_textures(const std::vector<const ImmutableTexture2D>& _2d_textures);

	public:
		//Functions to access and alter texture data

		//Texture data "setters":

		//Initiates pixel transfer operation, which assigns new texture data represented in an uncompressed texture format to the base (i.e. zero) layer of the given mipmap-level of
		//contained cubemap texture object. New data is assigned only to the faces, for which valid pointers have been provided. For instance, if positive_x = nullptr and 
		//negative_z = 0, then the data will be updated only for negative-x, positive-y, negative-y, and positive-z cubemap faces. If some of the pointers refer to an invalid block
		//of memory, then behavior of this function is not defined.
		void setMipmapLevelData(uint32_t mipmap_level, PixelLayout pixel_layout, PixelDataType pixel_component_type,
			const void* positive_x, const void* negative_x, const void* positive_y, const void* negative_y, const void* positive_z, const void* negative_z);


		//Initiates pixel transfer operation, which assigns new texture data represented using a compressed texture format to the base (i.e. zero) array layer of the given mipmap level of
		//contained cubemap texture object. New data is assigned only to the faces, for which valid pointers have been provided. For instance, if positive_x = nullptr and 
		//negative_z = 0, then the data will be updated only for negative-x, positive-y, negative-y, and positive-z cubemap faces. If some of the pointers refer to an invalid block
		//of memory, then behavior of this function is not defined.
		void setMipmapLevelData(uint32_t mipmap_level, InternalPixelFormatCompressed compressed_data_format, size_t compressed_data_size,
			const void* positive_x, const void* negative_x, const void* positive_y, const void* negative_y, const void* positive_z, const void* negative_z);


		//Initiates pixel transfer operation, which assigns new uncompressed texture data to the given FACE of the base (i.e. zero) array layer of the given mipmap level of
		//contained cubemap texture object.
		void setMipmapLevelData(uint32_t mipmap_level, CubemapFace face, PixelLayout pixel_layout, PixelDataType pixel_component_type, const void* data);


		//Initiates pixel transfer operation, which assigns new texture data represented using COMPRESSED texture format to the given FACE of the base (i.e. zero) array layer of
		//the given mipmap level of contained cubemap texture object
		void setMipmapLevelData(uint32_t mipmap_level, CubemapFace face, InternalPixelFormatCompressed compressed_data_format, size_t compressed_data_size, const void* data);


		//Initiates pixel transfer operation, which assigns uncompressed texture data to the specified layer of the given mipmap level of contained cubemap texture object. 
		//New data is fed only to those faces of the specified cubemap layer, for which the corresponding pointer has a value that does not get reduced to zero. For instance, if
		//positive_x = nullptr and negative_z = 0, the change will be applied only to negative-x, negative-y, positive-y, and positive-z faces of the cubemap. If some of the pointers
		//refer to an invalid memory block, the result of the operation is undefined.
		void setMipmapLevelLayerData(uint32_t mipmap_level, uint32_t array_layer, PixelLayout pixel_layout, PixelDataType pixel_component_type, 
			const void* positive_x, const void* negative_x, const void* positive_y, const void* negative_y, const void* positive_z, const void* negative_z);


		//Initiates pixel transfer operation, which assigns texture data to the specified layer of the given mipmap level of contained cubemap object using compressed texture format.
		//New data is fed only to those faces of the specified cubemap layer, for which the corresponding pointer has a value that does not get reduced to zero. For instance, if
		//positive_x = nullptr and negative_z = 0, the change will be applied only to negative-x, negative-y, positive-y, and positive-z faces of the cubemap. If some of the pointers
		//refer to an invalid memory block, the result of the operation is undefined.
		void setMipmapLevelLayerData(uint32_t mipmap_level, uint32_t array_layer, InternalPixelFormatCompressed compressed_data_format, size_t compressed_data_size,
			const void* positive_x, const void* negative_x, const void* positive_y, const void* negative_y, const void* positive_z, const void* negative_z);
		
		
		//Initiates pixel transfer operation, which assigns new uncompressed texture data to the given FACE of the specified layer in the given mipmap level of contained cubemap texture object
		void setMipmapLevelLayerData(uint32_t mipmap_level, uint32_t array_layer, CubemapFace face, PixelLayout pixel_layout, PixelDataType pixel_component_type, const void* data);


		//Initiates pixel transfer operation, which assigns new texture data represented using compressed texture format to the given FACE of the specified array layer of the given mipmap level of contained cubemap texture object
		void setMipmapLevelLayerData(uint32_t mipmap_level, uint32_t array_layer, CubemapFace face, InternalPixelFormatCompressed compressed_data_format, size_t compressed_data_size, const void* data);
		

		//Initiates pixel transfer operation, which updates multiple layers of the given face of the cubemap in the given mipmap level. 
		void setMipmapLevelMultiLayersData(uint32_t mipmap_level, uint32_t start_array_layer, uint32_t number_of_array_layers, CubemapFace face, PixelLayout pixel_layout, PixelDataType pixel_component_type, const void* data);


		//Initiates pixel transfer operation, which assigns new data represented using compressed texture format to multiple layers of the given face of the cubemap in the given mipmap level
		void setMipmapLevelMultiLayersData(uint32_t mipmap_level, uint32_t start_array_layer, uint32_t number_of_array_layers, CubemapFace face, InternalPixelFormatCompressed compressed_data_format, size_t compressed_data_size, const void* data);


		//Initiates pixel transfer operation, which updates multiple array layers of requested faces of the cube map in the given mipmap level. The data is updated only for the faces, 
		//for which the source is provided by a pointer that can not be reduced to zero, i.e.  if positive_x = nullptr and negative_z = 0, the update will be applied only to the faces -X, +Y, -Y, and +Z.
		void setMipmapLevelMultiLayersData(uint32_t mipmap_level, uint32_t start_array_layer, uint32_t number_of_array_layers, PixelLayout pixel_layout, PixelDataType pixel_component_type,
			const void* positive_x, const void* negative_x, const void* positive_y, const void* negative_y, const void* positive_z, const void* negative_z);


		//Initiates pixel transfer operation, which assigns new data represented using compressed texture format to multiple array layers of requested faces of the cube map in the given mipmap level. 
		//The data gets updated only for the faces, for which the data source is supplied using pointer that can not be reduced to zero. For example, if positive_x = nullptr and negative_z = 0, the data will be updated only for the faces -X, +Y, -Y, and +Z.
		//Here "compressed_data_size" defines number of compressed image bytes in each of the cubemap face image data arrays (positive_x, ..., negative_z)
		void setMipmapLevelMultiLayersData(uint32_t mipmap_level, uint32_t start_array_layer, uint32_t number_of_array_layers, InternalPixelFormatCompressed compressed_data_format, size_t compressed_data_size,
			const void* positive_x, const void* negative_x, const void* positive_y, const void* negative_y, const void* positive_z, const void* negative_z); 


		//Initiates pixel transfer operation, which assigns new data to multiple layer-faces of the given mipmap level  of the cubemap object. The layer faces are sorted as +X, -X, +Y, -Y, +Z, -Z. Hence, the layer-face
		//located at address start_address+6*I+3 represents cubemap face -Y of texture array layer with zero-based index I. This is a convenience function, which allows to update data in cubemap faces using the standard OpenGL ordering
		void setMipmapLevelMultiLayerFacesData(uint32_t mipmap_level, uint32_t start_layer_face, uint32_t number_of_layer_faces, PixelLayout pixel_layout, PixelDataType pixel_component_type, const void* data);


		//Initiates pixel transfer operation, which assigns new texture data represented using a compressed texture format to multiple layer-faces of the given mipmap level of cubemap texture object. 
		//The layer faces are ordered as +X, -X, +Y, -Y, +Z, -Z. Hence, the layer-face located at address start_address+6*I+3 represents cubemap face -Y of texture array layer with zero-based index I.
		//This is a convenience function that allows to update multiple cubemap faces by a single function call following standard OpenGL face ordering applied to array cubemaps. Note that "compessed_data_size" must define
		//total number of bytes of compressed image data located at the starting address pointed by "data"
		void setMipmapLevelMultiLayerFacesData(uint32_t mipmap_level, uint32_t start_layer_face, uint32_t number_of_layer_faces, InternalPixelFormatCompressed compressed_data_format, size_t compressed_data_size, const void* data);


		//Texture data "getters":

		//Initiates pixel transfer operation, which extracts texture data from contained cubemap object and returns it to the caller using requested pixel format. This function can not efficiently extract data for individual layers of
		//array cubemaps. This means that returned data contains information from ALL array layers packed together. For instance, when asked to extract data from positive-Y cubemap face, the function returns data for this face
		//from all cubemap layers for the given mipmap-level.
		void getMipmapLevelImageData(GLint mipmap_level, CubemapFace face, PixelLayout pixel_layout, PixelDataType pixel_component_type, TextureSize* img_size, void* data) const;

		//Initiates pixel transfer operation, which extracts texture data using raw compressed texture format from contained cubemap object and returns it to the caller. This function can not efficiently extract data for individual layers of
		//array cubemaps. This means that returned data contains information from ALL array layers packed together. For instance, when asked to extract data from positive-Y cubemap face, the function returns data for this face
		//from all cubemap layers for the given mipmap-level.
		void getMipmapLevelImageData(GLint mipmap_level, CubemapFace face, size_t* compressed_data_size, InternalPixelFormatCompressed* compressed_format, TextureSize* img_size, void* data) const;

		//Constructor infrastructure
		
		ImmutableTextureCubeMap();		//Default constructor
		explicit ImmutableTextureCubeMap(const std::string& texture_string_name);		//Allows initialization of contained texture using string name for further identification purposes

		//Allows to initialize a new non-array cubemap texture using data from six 2D-textures for the faces. The data for all mipmap-levels are taken from provided 
		//2D-textures meaning that all of them should be having same number of LOD-levels
		ImmutableTextureCubeMap(const ImmutableTexture2D& positive_x, const ImmutableTexture2D& negative_x,
			const ImmutableTexture2D& positive_y, const ImmutableTexture2D& negative_y,
			const ImmutableTexture2D& positive_z, const ImmutableTexture2D& negative_z);

		//Allows to initialize a new array or non-array cubemap texture using data from a vector of 2D-textures to populate the faces. If length of supplied vector is not a multiple of 6, 
		//no error is generated and reminding textures do not participate in construction of the cubemap. The data for mipmap levels are taken from the source 2D textures, which means that all these textures
		//should have same number of LOD-levels for successful construction
		ImmutableTextureCubeMap(const std::vector<const ImmutableTexture2D>& _2d_textures);

		//Miscellaneous

		//Clones the cubemap on the heap and returns its full copy as a pointer to the base interface
		Texture* clone() const override;


		//Allows to combine two cubemap texture objects into a single cubemap array
		ImmutableTextureCubeMap combine(const ImmutableTextureCubeMap& other) const;


		//Copies texel data to the given destination cubemap texture or cubemap texture array. Note that the texel data can be copied only between two textures having same internal texel format.
		//The function returns 'true' on success and 'false' on failure. No error data besides the return value is generated in case of failure.
		bool copyTexelData(uint32_t source_mipmap_level, uint32_t source_first_array_layer, uint32_t source_offset_x, uint32_t source_offset_y,
			const ImmutableTextureCubeMap& destination_texture, uint32_t destination_mipmap_level, uint32_t destination_first_array_layer, uint32_t destination_offset_x, uint32_t destination_offset_y,
			uint32_t copy_buffer_width, uint32_t copy_buffer_height, uint32_t num_array_layers_to_copy) const;
	};

}


#define TW__IMMUTABLE_TEXTURE_CUBE__
#endif