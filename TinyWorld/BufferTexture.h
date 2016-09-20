//Implements buffer texture objects. Note that buffer textures are different from immutable textures in the sense that
//they are not "aliased" meaning that each buffer texture object owns  separate texture data and when objects are copy-initialized 
//or copy-assigned the data between them does not get shared, but a deep-copy is performed instead

#ifndef TW__BUFFER_TEXTURE__

#include "Texture.h"
#include "SharedBuffer.h"


namespace tiny_world
{
	using BufferTextureAccessPolicy = BufferAccess;
	using BufferTextureRangeAccessPolicy = BufferRangeAccess;


	//Implements core part of the buffer texture infrastructure
	class BufferTexture_Core : public Texture
	{
	private:
		//Describes parameters shared between alias objects of a buffer texture
		struct BufferTextureSharedDetails
		{
			SharedBuffer texture_buffer;	//pointer to the buffer object owned by the texture
			BufferTextureInternalPixelFormat internal_format;	//internal pixel format used by the texture storage
		};

		BufferTextureSharedDetails* p_shared_data;	//data shared between alias objects of the buffer texture
		bool is_buffer_provided_by_user;	//equals 'true' if the buffer for the texture has been provided by the client, equals 'false' otherwise

	protected:
		TextureBinding getOpenGLTextureBinding() const override;		//returns OpenGL texture binding target and the corresponding GL_TEXTURE_BINDING_* constant
		GLuint bind() const override;	//binds texture to the corresponding texture binding target. Returns the OpenGL id of the texture previously bound to this target

	public:
		//Default initialization
		BufferTexture_Core();

		//Creates buffer texture with the given string name
		explicit BufferTexture_Core(const std::string& texture_string_name);

		//Copy-initialization constructor
		BufferTexture_Core(const BufferTexture_Core& other);

		//Move-initialization constructor
		BufferTexture_Core(BufferTexture_Core&& other);

		//Standard destructor
		~BufferTexture_Core();

		//Copy-assignment operator
		BufferTexture_Core& operator=(const BufferTexture_Core& other);

		//Move-assignment operator
		BufferTexture_Core& operator=(BufferTexture_Core&& other);


		uint32_t getNumberOfArrayLayers() const override;	//Returns the number of array layers stored in the texture object. If the texture object is not an array texture, the function returns 1.
		uint32_t getNumberOfMipmapLevels() const override;	//Returns the number of mipmap levels in the texture object. Returns 1 if mipmaps are not in use.
		uint32_t getNumberOfFaces() const override;	//Returns the number of texture faces. Always returns 1 for textures that are not either cubemaps or cubemap arrays
		uint32_t getNumberOfSamples() const override;	//Returns the number of samples used by the texture. For non multi-sample textures the returned value of this function should always be 1.
		TextureDimension getDimension() const override;	//Returns dimension of the texture
		bool isArrayTexture() const override;	//Returns 'true' if the texture is an array texture
		bool isCompressed() const override;	//Returns 'true' if the texture uses a compressed storage format
		bool isBufferTexture() const override;	//Returns 'true' if the texture is a buffer texture (i.e. uses a buffer object for its storage). Returns 'false' otherwise
		Texture* clone() const override;	//Creates a copy of the texture object on heap memory. The caller is responsible for deleting the created copy manually when there is no more need in it.

		//if the texture has been initialized returns the size in bytes of its storage buffer. Otherwise returns TW_INVALID_RETURN_VALUE and puts the buffer texture alias object into an erroneous state
		uint32_t getTextureBufferSize() const;

		//Returns the largest number of texels that could be stored into the buffer texture as suggested by the size of the buffer and by the internal texture format, which was provided on the texture storage allocation.
		//If this function is called before the texture has been initialized, it will put the buffer texture alias object into an erroneous state and will return TW_INVALID_RETURN_VALUE
		uint32_t getTexelCount() const;

		//Allocates storage for the buffer texture object using requested internal pixel format representation of the data. Note that parameter 
		//'texture_size' defines the raw texture storage size in bytes, but not the number of texels as suggested by the given internal format.
		//This function can be called repetitively effectively reinitializing the storage employed by the buffer texture. 
		//This change gets reflected in all alias objects referring to the buffer texture for which the storage is having been reinitialized 
		void allocateStorage(size_t texture_size, BufferTextureInternalPixelFormat internal_format);

		//Copies user-defined data to the given range of the buffer assigned to the buffer texture object. The function will generate an error if called before the textured has been initialized
		void setSubData(ptrdiff_t data_chunk_offset, size_t data_chunk_size, const void* data);

		//Maps buffer assigned to the buffer texture object onto the address space of the client in accordance with requested access policy. Note that 
		//actual manipulation with the data is allowed to break this policy, but this could lead to significant performance penalties depending on the OpenGL 
		//implementation provided by the graphics driver
		void* map(BufferTextureAccessPolicy access_policy) const;

		//Maps a part of the buffer assigned to the buffer texture object onto the address space of the client given the range access policy. Note, that this
		//range access policy does not limit the ways by which the data could actually be manipulated by the client. However, breaking this policy may lead 
		//to significant performance degradation depending on the OpenGL implementation provided by the graphics driver
		void* map(ptrdiff_t range_offset, size_t range_size, BufferTextureRangeAccessPolicy access_bits) const;

		//Unmaps the buffer owned by the buffer texture (or the previously mapped buffer range) from the client space
		void unmap() const;

		//Attaches a user defined buffer to the texture. Note this function has similar effect to allocateStorage(...). In other words, it initializes the texture and if
		//the texture has already been initialized by calling allocateStorage(...) this function will have no effect. Similarly, allocateStorage(...) will have no effect after
		//calling this function. The buffer object supplied to the input of this function is encouraged to declare BufferBindingTarget::TEXTURE as its binding target, though
		//this is not an obligatory requirement (although, following this rule may increase performance on some video drivers). The 'internal_format' declares the format, which
		//the texture should use to interpret the data contained in the supplied buffer
		void attachBuffer(const SharedBuffer& shared_buffer, BufferTextureInternalPixelFormat internal_format);

		//Copies all contents of the texture into the destination texture. Note that this function will fail if the source and destination textures are having different formats or own
		//buffers of different sizes. The function returns 'true' on success and 'false' on failure
		bool copyTexelData(const BufferTexture_Core& destination_texture) const;
	};


	//Implements interface part of the buffer texture infrastructure
	class BufferTexture final : public BufferTexture_Core
	{
		friend class AbstractRenderingDevice;
		friend class Framebuffer;
		friend class ImageUnit;

	protected:
		GLuint bind() const override;	//binds texture to the corresponding texture binding target. Returns the OpenGL id of the texture previously bound to this target

		//Attaches the buffer texture to the currently active framebuffer object (i.e. the FBO currently bound to GL_DRAW_FRAMEBUFFER binding point)
		void attachToFBO(FramebufferAttachmentPoint attachment) const;

		//Attaches the buffer texture to the given image unit
		void attachToImageUnit(uint32_t image_unit, BufferAccess access, BufferTextureInternalPixelFormat format) const;

	public:
		BufferTexture();	//default initialization
		explicit BufferTexture(const std::string& texture_string_name);
	};
}

#define TW__BUFFER_TEXTURE__
#endif