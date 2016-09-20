#ifndef TW__TEXTURE__

#include <GL/glew.h>

#ifndef GLEW_STATIC
#define GLEW_STATIC
#endif

#include <GLFW/glfw3.h>
#include <string>
#include <cinttypes>

#include "Entity.h"
#include "PixelTraits.h"
#include "Misc.h"


namespace tiny_world
{
	//Describes texture type-slots
	enum class TextureSlot : GLenum{
		TEXTURE_UNKNOWN = 0xFFFFFFFF,
		TEXTURE_1D = GL_TEXTURE_1D,
		TEXTURE_1D_ARRAY = GL_TEXTURE_1D_ARRAY,
		TEXTURE_2D = GL_TEXTURE_2D,
		TEXTURE_2D_ARRAY = GL_TEXTURE_2D_ARRAY,
		TEXTURE_3D = GL_TEXTURE_3D,
		TEXTURE_CUBE_MAP = GL_TEXTURE_CUBE_MAP,
		TEXTURE_CUBE_MAP_ARRAY = GL_TEXTURE_CUBE_MAP_ARRAY,
		TEXTURE_BUFFER = GL_TEXTURE_BUFFER,
		TEXTURE_2D_MULTISAMPLE = GL_TEXTURE_2D_MULTISAMPLE,
		TEXTURE_2D_MULTISAMPLE_ARRAY = GL_TEXTURE_2D_MULTISAMPLE_ARRAY
	};


	enum class TextureDimension : uint32_t {
		_1D = 1, _2D = 2, _3D = 3
	};


	struct TextureSize{
		uint32_t width;
		uint32_t height;
		uint32_t depth;
		bool operator ==(const TextureSize& other) const;
		bool operator !=(const TextureSize& other) const;
	};

	typedef TextureSize StorageSize;


	enum class FramebufferAttachmentPoint : GLenum
	{
		STENCIL = GL_STENCIL_ATTACHMENT,
		DEPTH = GL_DEPTH_ATTACHMENT,
		STENCIL_DEPTH = GL_DEPTH_STENCIL_ATTACHMENT,
		COLOR0 = GL_COLOR_ATTACHMENT0,
		COLOR1 = GL_COLOR_ATTACHMENT1,
		COLOR2 = GL_COLOR_ATTACHMENT2,
		COLOR3 = GL_COLOR_ATTACHMENT3,
		COLOR4 = GL_COLOR_ATTACHMENT4,
		COLOR5 = GL_COLOR_ATTACHMENT5,
		COLOR6 = GL_COLOR_ATTACHMENT6,
		COLOR7 = GL_COLOR_ATTACHMENT7,
		COLOR8 = GL_COLOR_ATTACHMENT8,
		COLOR9 = GL_COLOR_ATTACHMENT9,
		COLOR10 = GL_COLOR_ATTACHMENT10,
		COLOR11 = GL_COLOR_ATTACHMENT11,
		COLOR12 = GL_COLOR_ATTACHMENT12,
		COLOR13 = GL_COLOR_ATTACHMENT13,
		COLOR14 = GL_COLOR_ATTACHMENT14,
		COLOR15 = GL_COLOR_ATTACHMENT15
	};




	//Implements the most abstract properties that all textures possess
	class Texture : public Entity
	{
		friend class TextureUnitBlock;
	protected:
		//Defines texture binding target and the corresponding GL_TEXTURE_BINDING_* constant
		struct TextureBinding{
			GLenum gl_texture_target;
			GLenum gl_texture_binding;
		};

	private:
		//Describes data shared by all alias objects of the texture
		struct TextureDetails
		{
			TextureSize texture_size;					//Size of the contained texture. Does not take into account layers in array texture objects.
			StorageSize storage_size;					//Size of the storage of contained texture. Accounts for the layer dimension in array textures.
			GLenum ogl_internal_texel_storage_format;	//Internal texel storage format employed by the texture
			uint32_t ref_counter;					//Number of alias objects referring to the texture
		};

		GLuint ogl_texture_id;	//OpenGL identifier of the texture
		TextureDetails* p_texture_details;	//data shared between alias objects of the texture

	protected:
		void incrementReferenceCounter();	//if the texture has been initialized increments internal reference counter of the texture. Otherwise has no effect
		void decrementReferenceCounter();	//If the texture has been initialized decrements internal reference counter of the texture. Otherwise has no effect
		
		//If the texture has been initialized returns the current value of the internal reference counter of the texture. If texture has not been initialized returns TW_INVALID_RETURN_VALUE and puts the object into an erroneous state
		uint32_t getReferenceCount() const;


		//if the texture has been initialized returns constant identifying the format used for internal storage of contained texture. If the texture has not been initialized returns TW_INVALID_RETURN_VALUE and puts the object into an erroneous state
		GLenum getOpenGLStorageInternalFormat() const;

		//returns OpenGL identifier of the texture. If the texture has not been initialized returns TW_INVALID_RETURN_VALUE and puts the object into an erroneous state
		GLuint getOpenGLId() const;
		
		//initializes the texture. This function must be called by derived classes when allocating storage for the texture. It is allowed to call this function more than once which enables technical possibility to alter texture storage format and size.
		//The values provided on the input of this function will be shared by all alias objects referring to the texture
		void initialize(const TextureSize& texture_size, const TextureSize& storage_size, GLenum storage_format);

		TextureSlot getBindingSlot() const;	//returns the binding slot to which the texture should be bound. The function puts the object in an erroneous state if it is invoked before the texture has been initialized


		virtual TextureBinding getOpenGLTextureBinding() const = 0;		//returns OpenGL texture binding target and the corresponding GL_TEXTURE_BINDING_* constant
		virtual GLuint bind() const = 0;	//binds texture to the corresponding texture binding target. Returns the OpenGL id of the texture previously bound to this target


		Texture(const std::string& texture_class_string_name);	//Default initialization (requires derived types to provide a string name for the class of the texture they implement)
		Texture(const std::string& texture_class_string_name, const std::string& texture_string_name);	//Initializes texture using the given string name of the class to which it belongs and a string name the texture itself
		Texture(const Texture& other);	//Copy initialization
		Texture(Texture&& other);	//Move initialization

	public:
		bool isInitialized() const;	//Returns 'true' if the texture has been properly initialized, returns 'false' otherwise
		PixelFormatTraits getStorageFormatTraits() const;	//Returns the traits object associated with the internal texture storage format. If the texture has not yet been initialized puts the object into an erroneous state and returns an undefined value
		
		//Returns the dimensions of the texture not counting the extra layer dimension in array textures (e.g. a 2D array texture will only define the 'width' and 'height' values in the returned structure, but not the 'depth' value)
		//If the texture has not been initialized the function puts the object into an erroneous state and returns an undefined value
		TextureSize getTextureSize() const;

		//Returns the size of the storage employed by the texture. Accounts for the extra layer dimension in array textures.
		//For instance a 2D array texture will naturally define its 'width' and 'height' dimensions, but will also store the corresponding number of its layers in the 'depth' filed of the structure returned by this function.
		//If the texture has not been initialized the function puts the object into an erroneous state and returns an undefined value
		StorageSize getStorageSize() const;


		virtual uint32_t getNumberOfArrayLayers() const = 0;	//Returns the number of array layers stored in the texture object. If the texture object is not an array texture, the function returns 1.
		virtual uint32_t getNumberOfMipmapLevels() const = 0;	//Returns the number of mipmap levels in the texture object. Returns 1 if mipmaps are not in use.
		virtual uint32_t getNumberOfFaces() const = 0;	//Returns the number of texture faces. Always returns 1 for textures that are not either cubemaps or cubemap arrays
		virtual uint32_t getNumberOfSamples() const = 0;	//Returns the number of samples used by the texture. For non multi-sample textures the returned value of this function should always be 1.
		virtual TextureDimension getDimension() const = 0;	//Returns dimension of the texture
		virtual bool isArrayTexture() const = 0;	//Returns 'true' if the texture is an array texture
		virtual bool isCompressed() const = 0;	//Returns 'true' if the texture uses a compressed storage format
		virtual bool isBufferTexture() const = 0;	//Returns 'true' if the texture is a buffer texture (i.e. uses a buffer object for its storage). Returns 'false' otherwise
		virtual Texture* clone() const = 0;	//Creates a copy of the texture object on heap memory. The caller is responsible for deleting the created copy manually when there is no more need in it.

		bool operator==(const Texture& other) const;	//Compares two texture aliases. Yields 'true' if both aliases point to the same OpenGL texture object. Yields 'false' otherwise
		bool operator!=(const Texture& other) const;	//Compares two texture aliases. Yields 'true' if the aliases refer to different OpenGL texture objects. Yields 'false' otherwise


		Texture& operator=(const Texture& other);	//Copy-assignment operator overloading
		Texture& operator=(Texture&& other);	//Move-assignment operator overloading

		~Texture();	//destructor
	};
}

#define TW__TEXTURE__
#endif