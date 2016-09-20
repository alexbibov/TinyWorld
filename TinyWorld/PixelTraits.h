#ifndef TW__PIXEL_TRAITS__

#include <GL/glew.h>

#include <utility>

namespace tiny_world
{
	//Describes various formats that are used for internal pixel storage
	enum class InternalPixelFormat : GLenum
	{
		BASE_DEPTH = GL_DEPTH_COMPONENT,
		BASE_DEPTH_STENCIL = GL_DEPTH_STENCIL,
		BASE_R = GL_RED,
		BASE_RG = GL_RG,
		BASE_RGB = GL_RGB,
		BASE_RGBA = GL_RGBA,

		SIZED_R8 = GL_R8,
		SIZED_R8_SNORM = GL_R8_SNORM,
		SIZED_R16 = GL_R16,
		SIZED_R16_SNORM = GL_R16_SNORM,
		SIZED_RG8 = GL_RG8,
		SIZED_RG8_SNORM = GL_RG8_SNORM,
		SIZED_RG16 = GL_RG16,
		SIZED_RG16_SNORM = GL_RG16_SNORM,
		SIZED_R3_G3_B2 = GL_R3_G3_B2,
		SIZED_RGB4 = GL_RGB4,
		SIZED_RGB5 = GL_RGB5,
		SIZED_RGB565 = GL_RGB565,
		SIZED_RGB8 = GL_RGB8,
		SIZED_RGB8_SNORM = GL_RGB8_SNORM,
		SIZED_RGB10 = GL_RGB10,
		SIZED_RGB12 = GL_RGB12,
		SIZED_RGB16 = GL_RGB16,
		SIZED_RGB16_SNORM = GL_RGB16_SNORM,
		SIZED_RGBA2 = GL_RGBA2,
		SIZED_RGBA4 = GL_RGBA4,
		SIZED_RGB5_A1 = GL_RGB5_A1,
		SIZED_RGBA8 = GL_RGBA8,
		SIZED_RGBA8_SNORM = GL_RGBA8_SNORM,
		SIZED_RGB10_A2 = GL_RGB10_A2,
		SIZED_RGB10_A2UI = GL_RGB10_A2UI,
		SIZED_RGBA12 = GL_RGBA12,
		SIZED_RGBA16 = GL_RGBA16,
		SIZED_RGBA16_SNORM = GL_RGBA16_SNORM,
		SIZED_SRGB8 = GL_SRGB8,
		SIZED_SRGB8_ALPHA8 = GL_SRGB8_ALPHA8,
		SIZED_FLOAT_R16 = GL_R16F,
		SIZED_FLOAT_RG16 = GL_RG16F,
		SIZED_FLOAT_RGB16 = GL_RGB16F,
		SIZED_FLOAT_RGBA16 = GL_RGBA16F,
		SIZED_FLOAT_R32 = GL_R32F,
		SIZED_FLOAT_RG32 = GL_RG32F,
		SIZED_FLOAT_RGB32 = GL_RGB32F,
		SIZED_FLOAT_RGBA32 = GL_RGBA32F,
		SIZED_FLOAT_R11_G11_B10 = GL_R11F_G11F_B10F,
		SIZED_FLOAT_RGB9_E5 = GL_RGB9_E5,
		SIZED_FLOAT_DEPTH32 = GL_DEPTH_COMPONENT32F,
		SIZED_FLOAT_DEPTH32_STENCIL8 = GL_DEPTH32F_STENCIL8,

		SIZED_INT_R8 = GL_R8I,
		SIZED_UINT_R8 = GL_R8UI,
		SIZED_INT_R16 = GL_R16I,
		SIZED_UINT_R16 = GL_R16UI,
		SIZED_INT_R32 = GL_R32I,
		SIZED_UINT_R32 = GL_R32UI,
		SIZED_INT_RG8 = GL_RG8I,
		SIZED_UINT_RG8 = GL_RG8UI,
		SIZED_INT_RG16 = GL_RG16I,
		SIZED_UINT_RG16 = GL_RG16UI,
		SIZED_INT_RG32 = GL_RG32I,
		SIZED_UINT_RG32 = GL_RG32UI,
		SIZED_INT_RGB8 = GL_RGB8I,
		SIZED_UINT_RGB8 = GL_RGB8UI,
		SIZED_INT_RGB16 = GL_RGB16I,
		SIZED_UINT_RGB16 = GL_RGB16UI,
		SIZED_INT_RGB32 = GL_RGB32I,
		SIZED_UINT_RGB32 = GL_RGB32UI,
		SIZED_INT_RGBA8 = GL_RGBA8I,
		SIZED_UINT_RGBA8 = GL_RGBA8UI,
		SIZED_INT_RGBA16 = GL_RGBA16I,
		SIZED_UINT_RGBA16 = GL_RGBA16UI,
		SIZED_INT_RGBA32 = GL_RGBA32I,
		SIZED_UINT_RGBA32 = GL_RGBA32UI,

		SIZED_DEPTH16 = GL_DEPTH_COMPONENT16,
		SIZED_DEPTH24 = GL_DEPTH_COMPONENT24,
		SIZED_DEPTH32 = GL_DEPTH_COMPONENT32,
		SIZED_DEPTH24_STENCIL8 = GL_DEPTH24_STENCIL8,
		SIZED_STENCIL1 = GL_STENCIL_INDEX1,
		SIZED_STENCIL4 = GL_STENCIL_INDEX4,
		SIZED_STENCIL8 = GL_STENCIL_INDEX8,
		SIZED_STENCIL16 = GL_STENCIL_INDEX16,
	};

	//Describes various compressed formats that can be used for internal pixel storage at OpenGL side
	enum class InternalPixelFormatCompressed
	{
		COMPRESSED_BASE_R = GL_COMPRESSED_RED,
		COMPRESSED_BASE_RG = GL_COMPRESSED_RG,
		COMPRESSED_BASE_RGB = GL_COMPRESSED_RGB,
		COMPRESSED_BASE_RGBA = GL_COMPRESSED_RGBA,
		COMPRESSED_BASE_SRGB = GL_COMPRESSED_SRGB,
		COMPRESSED_BASE_SRGB_ALPHA = GL_COMPRESSED_SRGB_ALPHA,
		COMPRESSED_R_RGTC1 = GL_COMPRESSED_RED_RGTC1,
		COMPRESSED_SIGNED_R_RGTC1 = GL_COMPRESSED_SIGNED_RED_RGTC1,
		COMPRESSED_RG_RGTC2 = GL_COMPRESSED_RG_RGTC2,
		COMPRESSED_SIGNED_RG_RGTC2 = GL_COMPRESSED_SIGNED_RG_RGTC2,
		COMPRESSED_RGBA_BPTC_UNORM = GL_COMPRESSED_RGBA_BPTC_UNORM,
		COMPRESSED_SRGB_ALPHA_BPTC_UNORM = GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM,
		COMPRESSED_RGB_BPTC_SIGNED_FLOAT = GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT,
		COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT = GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT,
		COMPRESSED_RGB8_ETC2 = GL_COMPRESSED_RGB8_ETC2,
		COMPRESSED_SRGB8_ETC2 = GL_COMPRESSED_SRGB8_ETC2,
		COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2 = GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2,
		COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2 = GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2,
		COMPRESSED_RGBA8_ETC2_EAC = GL_COMPRESSED_RGBA8_ETC2_EAC,
		COMPRESSED_SRGB8_ALPHA8_ETC2_EAC = GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC,
		COMPRESSED_R11_EAC = GL_COMPRESSED_R11_EAC,
		COMPRESSED_SIGNED_R11_EAC = GL_COMPRESSED_SIGNED_R11_EAC,
		COMPRESSED_RG11_EAC = GL_COMPRESSED_RG11_EAC,
		COMPRESSED_SIGNED_RG11_EAC = GL_COMPRESSED_SIGNED_RG11_EAC
	};

	//Enumerates storage classes of various color pixel storage formats (depth, stencil, compressed and some of the packed storage formats all belong to "unknown" storage class)
	enum class PixelStorageClass
	{
		_4x32, _4x16, _4x8, _3x32, _3x16, _3x8,
		a, b, _2x32, _2x16, _2x8, _1x32, _1x16, _1x8, unknown
	};

	//Describes various pixel formats that determine what pixel data gets stored and how the storage should be organized
	enum class PixelLayout : GLenum
	{
		STENCIL = GL_STENCIL_INDEX,
		DEPTH = GL_DEPTH_COMPONENT,
		DEPTH_STENCIL = GL_DEPTH_STENCIL,
		RED = GL_RED,
		GREEN = GL_GREEN,
		BLUE = GL_BLUE,
		RG = GL_RG,
		RGB = GL_RGB,
		RGBA = GL_RGBA,
		BGR = GL_BGR,
		BGRA = GL_BGRA,
		INTEGER_RED = GL_RED_INTEGER,
		INTEGER_GREEN = GL_GREEN_INTEGER,
		INTEGER_BLUE = GL_BLUE_INTEGER,
		INTEGER_RG = GL_RG_INTEGER,
		INTEGER_RGB = GL_RGB_INTEGER,
		INTEGER_RGBA = GL_RGBA_INTEGER,
		INTEGER_BGR = GL_BGR_INTEGER,
		INTEGER_BGRA = GL_BGRA_INTEGER
	};

	//Describes pixel formats supported by glReadPixels(...)
	enum class PixelReadLayout : GLenum
	{
		STENCIL = GL_STENCIL_INDEX,
		DEPTH = GL_DEPTH_COMPONENT,
		DEPTH_STENCIL = GL_DEPTH_STENCIL,
		RED = GL_RED,
		GREEN = GL_GREEN,
		BLUE = GL_BLUE,
		RGB = GL_RGB,
		BGR = GL_BGR,
		RGBA = GL_RGBA,
		BGRA = GL_BGRA
	};

	//Describes various packed and unpacked pixel storage types
	enum class PixelDataType : GLenum
	{
		UBYTE = GL_UNSIGNED_BYTE,
		UBYTE_3_3_2 = GL_UNSIGNED_BYTE_3_3_2,
		UBYTE_2_3_3_R = GL_UNSIGNED_BYTE_2_3_3_REV,
		BYTE = GL_BYTE,

		USHORT = GL_UNSIGNED_SHORT,
		USHORT_5_6_5 = GL_UNSIGNED_SHORT_5_6_5,
		USHORT_5_6_5_R = GL_UNSIGNED_SHORT_5_6_5_REV,
		USHORT_4_4_4_4 = GL_UNSIGNED_SHORT_4_4_4_4,
		USHORT_4_4_4_4_R = GL_UNSIGNED_SHORT_4_4_4_4_REV,
		USHORT_5_5_5_1 = GL_UNSIGNED_SHORT_5_5_5_1,
		USHORT_1_5_5_5_R = GL_UNSIGNED_SHORT_1_5_5_5_REV,
		SHORT = GL_SHORT,

		UINT = GL_UNSIGNED_INT,
		UINT_10_10_10_2 = GL_UNSIGNED_INT_10_10_10_2,
		UINT_2_10_10_10_R = GL_UNSIGNED_INT_2_10_10_10_REV,
		UINT_10F_11F_11F_R = GL_UNSIGNED_INT_10F_11F_11F_REV,
		UINT_5_9_9_9_R = GL_UNSIGNED_INT_5_9_9_9_REV,
		UINT_24_8 = GL_UNSIGNED_INT_24_8,
		INT = GL_INT,

		FLOAT_32_UINT_24_8_R = GL_FLOAT_32_UNSIGNED_INT_24_8_REV,
		FLOAT = GL_FLOAT
	};

	//Describes various internal pixel storage formats that could be used with buffer textures
	enum class BufferTextureInternalPixelFormat : GLenum
	{
		SIZED_R8 = GL_R8,
		SIZED_R16 = GL_R16,
		SIZED_FLOAT_R16 = GL_R16F,
		SIZED_FLOAT_R32 = GL_R32F,
		SIZED_INT_R8 = GL_R8I, 
		SIZED_INT_R16 = GL_R16I,
		SIZED_INT_R32 = GL_R32I,
		SIZED_UINT_R8 = GL_R8UI,
		SIZED_UINT_R16 = GL_R16UI,
		SIZED_UINT_R32 = GL_R32UI,

		SIZED_RG8 = GL_RG8,
		SIZED_RG16 = GL_RG16,
		SIZED_FLOAT_RG16 = GL_RG16F,
		SIZED_FLOAT_RG32 = GL_RG32F,
		SIZED_INT_RG8 = GL_RG8I,
		SIZED_INT_RG16 = GL_RG16I,
		SIZED_INT_RG32 = GL_RG32I,
		SIZED_UINT_RG8 = GL_RG8UI,
		SIZED_UINT_RG16 = GL_RG16UI,
		SIZED_UINT_RG32 = GL_RG32UI,

		SIZED_FLOAT_RGB32 = GL_RGB32F,
		SIZED_INT_RGB32 = GL_RGB32I,
		SIZED_UINT_RGB32 = GL_RGB32UI,

		SIZED_RGBA8 = GL_RGBA8,
		SIZED_RGBA16 = GL_RGBA16,
		SIZED_FLOAT_RGBA16 = GL_RGBA16F,
		SIZED_FLOAT_RGBA32 = GL_RGBA32F,
		SIZED_INT_RGBA8 = GL_RGBA8I,
		SIZED_INT_RGBA16 = GL_RGBA16I,
		SIZED_INT_RGBA32 = GL_RGBA32I,
		SIZED_UINT_RGBA8 = GL_RGBA8UI,
		SIZED_UINT_RGBA16 = GL_RGBA16UI,
		SIZED_UINT_RGBA32 = GL_RGBA32UI,
	};

	//Buffer texture access levels
	enum class BufferAccess : GLenum
	{
		READ = GL_READ_ONLY, WRITE = GL_WRITE_ONLY, READ_WRITE = GL_READ_WRITE
	};



	class PixelFormatTraits
	{
	private:
		GLenum ogl_format;	//storage format represented by an OpenGL enumeration value (see OpenGL 4.3 standard description for further details on accepted values)
		unsigned short num_of_components;		//number of stored components
		bool is_color;	//equals 'true' if storage format is a color format
		bool is_depth;	//equals 'true' if storage format is a depth format
		bool is_stencil;	//equals 'true' if storage format is a stencil format
		bool is_recognized;	//equals 'true' if storage format has been successfully recognized
		bool is_compressed;	//equals 'true' if storage format is compressed
		bool is_signed_normalized;	//equals 'true' if storage format is "signed normalized"
		bool is_float_format;	//equals 'true' if internal storage format is floating point
		bool is_integer_format;	//equals 'true' if internal storage format is integer
		bool is_signed;			//equals 'true' if underlying pixel data type is signed
		PixelLayout pixel_layout;	//layout describing data contained by each pixel
		PixelDataType optimal_storage_type;	//the 'optimal' type, which can be used to store pixel data in the given format
		unsigned short optimal_storage_size;	//size in bytes of a single pixel stored using the 'optimal' storage type

		short r_size;		//size of the red component of the storage represented in bits. Equals to -1 if undefined.
		short g_size;		//size of the green component of the storage represented in bits. Equals to -1 if undefined.
		short b_size;		//size of the blue component of the storage represented in bits. Equals to -1 if undefined.
		short a_size;		//size of the alpha component of the storage represented in bits. Equals to -1 if undefined.

		short depth_size;		//size of the depth component of the storage represented in bits. Equals to -1 if undefined.
		short stencil_size;		//size of the stencil component of the storage represented in bits. Equals to -1 if undefined.

		bool is_compatible_with_buffer_textures;	//equals 'true' if the format can be used by buffer texture storage. Equals 'false' otherwise.
		short minimal_storage_size;	//minimal size in bits of the storage needed to accommodate a single pixel encoded using the given internal storage format. If this value cannot be determined equals to -1.

		void ConstructSizedFormatTraits(InternalPixelFormat internal_format);		//Constructs traits object for a sized pixel format
		void ConstructCompressedFormatTraits(InternalPixelFormatCompressed internal_format_compressed);	//Constructs traits object for a compressed pixel format

	public:
		PixelFormatTraits(InternalPixelFormat internal_format);
		PixelFormatTraits(InternalPixelFormatCompressed internal_format_compressed);
		PixelFormatTraits(BufferTextureInternalPixelFormat buffer_texture_internal_format);
		PixelFormatTraits(GLenum ogl_internal_format);

		unsigned short getNumberOfTexelComponents() const;	//returns number of components contained in each texel
		bool isColor() const;	//returns 'true' if storage format is a color format
		bool isDepth() const;	//returns 'true' if storage format is a depth format
		bool isStencil() const;	//returns 'true' if storage format is a stencil format
		bool isRecognized() const;	//returns 'true' if internal storage format is known to the engine. Returns 'false' otherwise. The known formats are those explicitly listed in OpenGL 4.5 specification document
		bool isCompressed() const;	//returns 'true' for compressed storage formats
		bool isSignedNormalized() const;	//returns 'true' for internal storage formats that are normalized to the range of [-1, 1] in GLSL
		bool isSigned() const;	//returns 'true' if pixel format is encoded by a signed data format
		bool isFloat() const;	//returns 'true' for floating point storage formats
		bool isInteger() const;	//returns 'true' for integer storage formats
		bool isBufferTextureCompatible() const;	//returns 'true' if the format can be used for the storage with buffer textures. Returns 'false' otherwise

		short getRedBits() const;	//amount of bits used for the red channel. Equals to -1 if cannot be determined for requested storage format
		short getGreenBits() const;	//amount of bits used for the green channel. Equals to -1 if cannot be determined for requested storage format
		short getBlueBits() const;	//amount of bits used for the blue channel. Equals to -1 if cannot be determined for requested storage format
		short getAlphaBits() const;	//amount of bits used for the alpha channel. Equals to -1 if cannot be determined for requested storage format

		short getDepthBits() const;	//amount of bits used to store depth information. Equals to -1 if cannot be determined for requested storage format
		short getStencilBits() const;	//amount of bits used to store stencil information. Equals to -1 if cannot be determined for requested storage format

		PixelLayout getPixelLayout() const;	//returns layout of the given pixel storage format
		PixelDataType getOptimalStorageType() const;	//returns optimal storage type for the given pixel format
		unsigned short getOptimalStorageSize() const;	//returns storage size of a single pixel assuming that the pixel is represented using the optimal storage type
		
		//Returns minimal size of the storage in bits needed to contain a single pixel encoded using the internal storage format supplied during initialization of the traits object. 
		//Returns -1 if this value cannot be determined for the given storage format
		short getMinimalStorageSize() const;
		
		PixelStorageClass getStorageClass() const;	//returns storage class corresponding to the given pixel storage format
		GLenum getOpenGLFormatEnumerationValue() const;		//returns OpenGL enumeration value corresponding to the pixel storage format for which the traits object has been constructed


		bool operator ==(const PixelFormatTraits& other) const;
		bool operator !=(const PixelFormatTraits& other) const;
	};




	class PixelDataTraits
	{
	private:
		GLenum pixel_layout;	//a value from the OpenGL enumeration describing layout of the pixel data
		GLenum pixel_data_type;	//a value from the OpenGL enumeration describing pixel component data type

		unsigned short num_of_components;	//number of components in pixel
		bool is_color;	//equals 'true' if pixel data is a color data
		bool is_depth;	//equals 'true' if pixel data is intended for usage in the depth buffer
		bool is_stencil;	//equals 'true' if pixel data is intended for usage in the stencil buffer
		bool is_recognized;	//equals 'true' if both the layout of the pixel data and the pixel component data type have been recognized by the traits object
		bool is_integer_format;	//equals 'true' if the pixel data is described using one of the integer pixel layouts
		bool is_signed;	//equals 'true' if pixel component data type is signed
		bool has_internal_representation;	//equals 'true' if the pixel layout has corresponding (uncompressed) internal storage format
		bool supports_compression;	//equals 'true' if the pixel layout supports compression

		InternalPixelFormat optimal_internal_format;	//"optimal" uncompressed internal storage format corresponding to the given pixel data layout and pixel data component type
		InternalPixelFormatCompressed optimal_internal_format_compressed;	//"optimal" compressed internal storage format corresponding to the given pixel layout and pixel data component type

		unsigned short storage_size;	//size in bytes occupied by a single pixel with given layout and component type

		short r_size;	//size of the red pixel component represented in bits. Equals to -1 if undefined
		short g_size;	//size of the green pixel component represented in bits. Equals to -1 if undefined
		short b_size;	//size of the blue pixel component represented in bits. Equals to -1 if undefined
		short a_size;	//size of the alpha pixel component represented in bits. Equals to -1 if undefined

		short depth_size;	//size of the depth pixel component represented in bits. Equals to -1 if undefined
		short stencil_size;	//size of the stencil pixel components represented in bits. Equals to -1 if undefined

		void ConstructTraits();	//constructs traits data for the given layout and the component type

	public:
		PixelDataTraits(PixelLayout pixel_layout, PixelDataType pixel_data_type);
		PixelDataTraits(GLenum pixel_layout, GLenum pixel_data_type);

		unsigned short getNumberOfComponents() const;	//returns number of components contained in each texel
		bool isColor() const;	//returns 'true' if pixel data layout represents color data
		bool isDepth() const;	//returns 'true' if pixel data layout represents depth data
		bool isStencil() const;	//returns 'true' if pixel data layout represents stencil data
		bool isRecognized() const;	//returns 'true' if pixel data layout and pixel component data type both have been recognized. The known layouts and data types are those explicitly listed in OpenGL 4.5 specification document
		bool isInteger() const;	//returns 'true' for integer pixel layouts
		bool isSigned() const;	//returns 'true' if pixel component data type is signed
		bool hasInternalRepresentation() const;	//equals 'true' if pixel layout has corresponding (uncompressed) internal representation
		bool supportsCompression() const;	//returns 'true' if the pixel layout supports compression

		short getRedBits() const;	//amount of bits used by the red channel of the given pixel layout. Equals to -1 if cannot be determined for requested pixel layout
		short getGreenBits() const;	//amount of bits used by the green channel of the given pixel layout. Equals to -1 if cannot be determined for requested pixel layout
		short getBlueBits() const;	//amount of bits used by the blue channel of the given pixel layout. Equals to -1 if cannot be determined for requested pixel layout
		short getAlphaBits() const;	//amount of bits used by the alpha channel of the given pixel layout. Equals to -1 if cannot be determined for requested pixel layout

		short getDepthBits() const;	//amount of depth bits in the given pixel layout. Equals to -1 if cannot be determined for requested pixel layout
		short getStencilBits() const;	//amount of stencil bits in the given pixel layout. Equals to -1 if cannot be determined for requested pixel layout

		unsigned short getPixelStorageSize() const;	//returns size in bytes occupied by a single pixel having the given data layout and component data type

		//Returns pair with the first element containing a raw value from OpenGL enumeration describing pixel data layout, and the second element containing a value from the OpenGL enumeration describing pixel component data type.
		std::pair<GLenum, GLenum> getOpenGLEnumerationLayoutAndComponentType() const;

		//Returns "optimal" uncompressed internal storage format to be used with the given pixel data layout and pixel component data type. Note that not all pixel data layouts have corresponding internal storage formats.
		//If pixel data layout used to construct the pixel data traits object does not have corresponding internal storage format the return value of this function is undefined
		void getOptimalUncompressedInternalStorageFormat(InternalPixelFormat* p_optimal_internal_format) const;

		//Retrieves "optimal" compressed internal storage format to be used with the given pixel data layout and pixel component data type.
		//Note that compression is only supported by some color pixel layouts. If the pixel data traits object have been constructed for a non-color pixel layout or the pixel layout does not support compression the return value is undefined
		void getOptimalCompressedInternalStorageFormat(InternalPixelFormatCompressed* p_optimal_internal_format_compressed) const;

		bool operator==(const PixelDataTraits& other) const;
		bool operator!=(const PixelDataTraits& other) const;
	};
}

#define TW__PIXEL_TRAITS__
#endif