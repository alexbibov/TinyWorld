#ifndef TW__IMMUTABLE_TEXTURE__

#include "Texture.h"


namespace tiny_world{

	//Possible packing and unpacking alignments of the texture storage
	enum class TextureStorageAlignment : GLint
	{
		_1BYTE = 1,
		_2BYTE = 2,
		_4BYTE = 4, 
		_8BYTE = 8
	};



	//Implements core functionality of immutable texture objects. This class is not used neither by the end-user, nor by the other related objects

	class ImmutableTexture;		//Forward declaration of immutable texture

	class ImmutableTexture_Core : public Texture
	{
	private:
		//Internal variables describing the object's state

		TextureBinding texture_binding;			//Structure containing information about OpenGL target and binding state corresponding to the contained texture

		TextureDimension texture_dimension;			//Dimension of the contained texture. Does not count array layers.
		TextureDimension storage_dimension;			//Dimension of storage required by contained texture. Basicly, this defines which version of glTexStorage*() must be called.

		uint32_t num_array_layers;				//Number of array layers present in contained texture object
		uint32_t num_mipmap_levels;				//Number of mipmap levels of the contained texture object
		uint32_t num_faces;						//Number of texture faces. Equals 1 for all textures except cubemaps. Equals 6 for cubemaps.

		TextureStorageAlignment pack_padding;					//texture pack padding
		TextureStorageAlignment unpack_padding;				//texture unpack padding

		void init_immutable_texture();			//Performs pre-initialization particulars
		inline void allocateStorage(uint32_t num_mipmap_levels, uint32_t num_array_layers, TextureSize texture_size, GLenum internal_storage_format);	//applies texture settings needed for allocation of the storage

	protected:
		TextureBinding getOpenGLTextureBinding() const override;	//Returns OpenGL texture binding target and the corresponding GL_TEXTURE_BINDING_* constant
		GLuint bind() const override;	//Binds texture to the corresponding target. Returns OpenGL id of the texture previously bound to this target

		virtual TextureBinding perform_allocation() = 0;	//Performs actual allocation of texture memory. This function is called implicitly by allocateStorage(), which ensures that provided storage and texture parameters are correctly initialized
		virtual TextureDimension query_dimension() const = 0;	//Queries dimension of contained texture. Must be implemented by an inherited object.

	public:
		//Getter infrastructure

		uint32_t getNumberOfArrayLayers() const override;	//Returns number of array layers stored in contained texture object. If texture object is not an array texture, the function returns 1.
		uint32_t getNumberOfMipmapLevels() const override;	//Returns number of mipmap levels in the texture object. If mip-mapping is not used, returns 1.
		uint32_t getNumberOfFaces() const override;	//Returns number of texture faces
		uint32_t getNumberOfSamples() const override;	//Returns the number of samples used by the texture
		TextureDimension getDimension() const override;	//Returns dimension of contained texture
		bool isArrayTexture() const override;	//Returns 'true' if num_array_layers > 1
		bool isCompressed()	  const override;	//Returns 'true' if contained texture is compressed
		bool isBufferTexture() const override;	//Always returns 'false' for immutable textures


		TextureStorageAlignment getPackPadding() const;				//Returns currently active pack padding value associated with the texture alias
		TextureStorageAlignment getUnpackPadding() const;				//Returns currently active unpack padding value associated with the texture alias


		//Helper functions
		void generateMipmapLevels();	//Automatically generates mipmap levels for contained texture. This function has no effect if the object is in an erroneous state.

		//Allocates storage for an uncompressed texture. This function has no effect if the texture has already been initialized or the object is in an erroneous state.
		void allocateStorage(uint32_t num_mipmap_levels, uint32_t num_array_layers, TextureSize texture_size, InternalPixelFormat internal_format);

		//Allocates storage for a compressed texture. This function has no effect if the texture object has already been initialized or persists in an erroneous state.
		void allocateStorage(uint32_t num_mipmap_levels, uint32_t num_array_layers, TextureSize texture_size, InternalPixelFormatCompressed internal_format_compressed);

		//Sets new pack padding value associated with this texture alias. The function returns previously active pack padding value.
		//Note: pack padding value is associated with texture alias, not with the texture object as it simply defines how texture data is returned to the client memory, when the texture is accessed using this alias.
		TextureStorageAlignment setPackPadding(TextureStorageAlignment new_pack_padding);

		//Sets new unpack padding value associated with this texture alias. The function returns previously active unpack padding value.
		//Note: unpack padding value is associated with texture alias, not with the texture object as it simply defines how data read from the client memory should be interpret by the texture object, when
		//the texture is being accessed using this alias.
		TextureStorageAlignment setUnpackPadding(TextureStorageAlignment new_unpack_padding);


		//Allows to copy texel data between textures of any (possibly different) derived types. The possibility to perform copy of texel data is limited to the case where the source and the destination textures are
		//both having same pixel formats. If either source or destination texture (or both) are array textures, then their slices are accessed via parameters *_offset_z and copy_buffer_depth. For cube-map textures
		//access rules are based on face-layer array representation, where the faces are listed in order "positive-x, negative-x, positive-y, negative-y, positive-z, negative-z". For example, the positive-z face of 6-th
		//array layer of a cube-map texture will be accessed similar to a normal array layer of a 2D array texture with layer offset 6*6+4(=40th array layer).
		//The function returns 'true' on success or false on failure. In case of failure, no other error data besides the return value is generated
		bool copyTexelData(uint32_t source_mipmap_level, uint32_t source_offset_x, uint32_t source_offset_y, uint32_t source_offset_z,
			const ImmutableTexture& destination_texture, uint32_t destination_mipmap_level, uint32_t destination_offset_x, uint32_t destination_offset_y, uint32_t destination_offset_z,
			uint32_t copy_buffer_width, uint32_t copy_buffer_height, uint32_t copy_buffer_depth) const;
									

	protected:
		//Constructor-destructor infrastructure
		explicit ImmutableTexture_Core(const std::string& texture_class_string_name);			//Default constructor: initializes texture with the given string name of the entity class to which it belongs
		ImmutableTexture_Core(const std::string& texture_class_string_name, const std::string& texture_string_name);		//Initializes texture using provided string names of the texture itself and of the entity class to which it belongs
	};
	



	//Implements interface infrastructure of immutable textures. All immutable texture implementations are inherited from this class, which provides a common interface to all of them
	class ImmutableTexture : public ImmutableTexture_Core{
		friend class AbstractRenderingDevice;
		friend class Framebuffer;
		friend class ImageUnit;

	protected:
		GLuint bind() const override;	//Binds texture to the corresponding target. Returns OpenGL id of the texture previously bound to this target

		void attachToFBO(FramebufferAttachmentPoint attachment, uint32_t mipmap_level) const;	//attaches mipmap level the texture to currently active framebuffer (i.e. the FBO bound to GL_DRAW_FRAMEBUFER target of the context) 
		
		//attaches a layer of mipmap level of an array texture to currently active framebuffer (i.e. the FBO currently bound to GL_DRAW_FRAMEBUFFER target of the context). 
		//If the texture is not an array or cube-map texture, parameter attachment_layer must be equal to 0. Here the meaning of attachment_layer differs from that of the texture array layer. It is considered that the cube-map textures
		//are also "layered" textures with layers represented by the cube-map texture faces in the order as follows: +x, -x, +y, -y, +z, -z. In addition, cube-map array texture has its layers filled by the series of 6-face sets.
		//More precisely, for a cube-map array texture, the layers are ordered as follows:
		//+x(0), -x(0), +y(0), -y(0), +z(0), -z(0), +x(1), -x(1), +y(1), -y(1), +z(1), -z(1), ...
		//Here the index in parentheses denotes the cube-map array texture layer.
		void attachToFBO(FramebufferAttachmentPoint attachment, uint32_t attachment_layer, uint32_t mipmap_level) const;


		//Attaches whole mipmap-level of the texture to the specified image unit. If texture is an array texture then all its layers of the given mipmap-level will be attached to the image unit.
		//Note that compressed textures can not be attached to image units.
		void attachToImageUnit(uint32_t image_unit, uint32_t mipmap_level, BufferAccess access, InternalPixelFormat format) const;

		//Attaches image unit to the specified layer of the given mipmap-level of the texture object. Note that image units can not be attached to compressed texture.
		//If texture to which an image unit is getting attached is not an array texture, then value of "layer" must be zero.
		void attachToImageUnit(uint32_t image_unit, uint32_t mipmap_level, uint32_t layer, BufferAccess access, InternalPixelFormat format) const;


		//ImmutableTexture can only be accessed through its children: explicit instantiation is not allowed

		explicit ImmutableTexture(const std::string& texture_class_string_name);
		ImmutableTexture(const std::string& texture_class_string_name, const std::string& texture_string_name);
	};



}

#define TW__IMMUTABLE_TEXTURE__
#endif