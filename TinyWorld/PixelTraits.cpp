#include "PixelTraits.h"

using namespace tiny_world;



void PixelFormatTraits::ConstructSizedFormatTraits(InternalPixelFormat internal_format)
{
	//Set default values for internal storage format parameters
	ogl_format = static_cast<GLenum>(internal_format);
	is_color = false;
	is_depth = false;
	is_stencil = false;
	is_recognized = true;	//by default we assume that the format has been successfully recognized
	is_compressed = false;
	is_signed_normalized = false;
	is_float_format = false;
	is_integer_format = false;
	is_signed = false;
	r_size = 0;
	g_size = 0;
	b_size = 0;
	a_size = 0;
	depth_size = 0;
	stencil_size = 0;
	pixel_layout = PixelLayout::RGBA;
	optimal_storage_type = PixelDataType::UBYTE;
	optimal_storage_size = 4;
	is_compatible_with_buffer_textures = false;
	minimal_storage_size = 32;

	switch (internal_format)
	{
	case InternalPixelFormat::BASE_DEPTH:
		num_of_components = 1;
		is_depth = true;
		r_size = -1;
		g_size = -1;
		b_size = -1;
		a_size = -1;
		depth_size = -1;
		stencil_size = -1;
		pixel_layout = PixelLayout::DEPTH;
		optimal_storage_type = PixelDataType::USHORT;
		optimal_storage_size = 2;
		minimal_storage_size = -1;
		break;
	case InternalPixelFormat::BASE_DEPTH_STENCIL:
		num_of_components = 2;
		is_depth = true;
		is_stencil = true;
		r_size = -1;
		g_size = -1;
		b_size = -1;
		a_size = -1;
		depth_size = -1;
		stencil_size = -1;
		pixel_layout = PixelLayout::DEPTH_STENCIL;
		optimal_storage_type = PixelDataType::USHORT;
		optimal_storage_size = 4;
		minimal_storage_size = -1;
		break;
	case InternalPixelFormat::BASE_R:
		num_of_components = 1;
		is_color = true;
		r_size = -1;
		g_size = -1;
		b_size = -1;
		a_size = -1;
		depth_size = -1;
		stencil_size = -1;
		pixel_layout = PixelLayout::RED;
		optimal_storage_size = 1;
		minimal_storage_size = -1;
		break;
	case InternalPixelFormat::BASE_RG:
		num_of_components = 2;
		is_color = true;
		r_size = -1;
		g_size = -1;
		b_size = -1;
		a_size = -1;
		depth_size = -1;
		stencil_size = -1;
		pixel_layout = PixelLayout::RG;
		optimal_storage_size = 2;
		minimal_storage_size = -1;
		break;
	case InternalPixelFormat::BASE_RGB:
		num_of_components = 3;
		is_color = true;
		r_size = -1;
		g_size = -1;
		b_size = -1;
		a_size = -1;
		depth_size = -1;
		stencil_size = -1;
		pixel_layout = PixelLayout::RGB;
		optimal_storage_size = 3;
		minimal_storage_size = -1;
		break;
	case InternalPixelFormat::BASE_RGBA:
		num_of_components = 4;
		is_color = true;
		r_size = -1;
		g_size = -1;
		b_size = -1;
		a_size = -1;
		depth_size = -1;
		stencil_size = -1;
		minimal_storage_size = -1;
		break;



	case InternalPixelFormat::SIZED_R8:
		num_of_components = 1;
		is_color = true;
		r_size = 8;
		pixel_layout = PixelLayout::RED;
		optimal_storage_size = 1;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 8;
		break;
	case InternalPixelFormat::SIZED_R8_SNORM:
		num_of_components = 1;
		is_color = true;
		is_signed_normalized = true;
		is_signed = true;
		r_size = 8;
		pixel_layout = PixelLayout::RED;
		optimal_storage_type = PixelDataType::BYTE;
		optimal_storage_size = 1;
		minimal_storage_size = 8;
		break;
	case InternalPixelFormat::SIZED_R16:
		num_of_components = 1;
		is_color = true;
		r_size = 16;
		pixel_layout = PixelLayout::RED;
		optimal_storage_type = PixelDataType::USHORT;
		optimal_storage_size = 2;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 16;
		break;
	case InternalPixelFormat::SIZED_R16_SNORM:
		num_of_components = 1;
		is_color = true;
		is_signed_normalized = true;
		is_signed = true;
		r_size = 16;
		pixel_layout = PixelLayout::RED;
		optimal_storage_type = PixelDataType::SHORT;
		optimal_storage_size = 2;
		minimal_storage_size = 16;
		break;
	case InternalPixelFormat::SIZED_RG8:
		num_of_components = 2;
		is_color = true;
		r_size = 8;
		g_size = 8;
		pixel_layout = PixelLayout::RG;
		optimal_storage_size = 2;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 16;
		break;
	case InternalPixelFormat::SIZED_RG8_SNORM:
		num_of_components = 2;
		is_color = true;
		is_signed_normalized = true;
		is_signed = true;
		r_size = 8;
		g_size = 8;
		pixel_layout = PixelLayout::RG;
		optimal_storage_type = PixelDataType::BYTE;
		optimal_storage_size = 2;
		minimal_storage_size = 16;
		break;
	case InternalPixelFormat::SIZED_RG16:
		num_of_components = 2;
		is_color = true;
		r_size = 16;
		g_size = 16;
		pixel_layout = PixelLayout::RG;
		optimal_storage_type = PixelDataType::USHORT;
		optimal_storage_size = 4;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 32;
		break;
	case InternalPixelFormat::SIZED_RG16_SNORM:
		num_of_components = 2;
		is_color = true;
		is_signed_normalized = true;
		is_signed = true;
		r_size = 16;
		g_size = 16;
		pixel_layout = PixelLayout::RG;
		optimal_storage_type = PixelDataType::SHORT;
		optimal_storage_size = 4;
		minimal_storage_size = 32;
		break;
	case InternalPixelFormat::SIZED_R3_G3_B2:
		num_of_components = 3;
		is_color = true;
		r_size = 3;
		g_size = 3;
		b_size = 2;
		pixel_layout = PixelLayout::RGB;
		optimal_storage_type = PixelDataType::UBYTE_3_3_2;
		optimal_storage_size = 1;
		minimal_storage_size = 8;
		break;
	case InternalPixelFormat::SIZED_RGB4:
		num_of_components = 3;
		is_color = true;
		r_size = 4;
		g_size = 4;
		b_size = 4;
		pixel_layout = PixelLayout::RGB;
		optimal_storage_size = 3;
		minimal_storage_size = 12;
		break;
	case InternalPixelFormat::SIZED_RGB5:
		num_of_components = 3;
		is_color = true;
		r_size = 5;
		g_size = 5;
		b_size = 5;
		pixel_layout = PixelLayout::RGB;
		optimal_storage_size = 3;
		minimal_storage_size = 15;
		break;
	case InternalPixelFormat::SIZED_RGB565:
		num_of_components = 3;
		is_color = true;
		r_size = 5;
		g_size = 6;
		b_size = 5;
		pixel_layout = PixelLayout::RGB;
		optimal_storage_type = PixelDataType::USHORT_5_6_5;
		optimal_storage_size = 2;
		minimal_storage_size = 16;
		break;
	case InternalPixelFormat::SIZED_RGB8:
		num_of_components = 3;
		is_color = true;
		r_size = 8;
		g_size = 8;
		b_size = 8;
		pixel_layout = PixelLayout::RGB;
		optimal_storage_size = 3;
		minimal_storage_size = 24;
		break;
	case InternalPixelFormat::SIZED_RGB8_SNORM:
		num_of_components = 3;
		is_color = true;
		is_signed_normalized = true;
		is_signed = true;
		r_size = 8;
		g_size = 8;
		b_size = 8;
		pixel_layout = PixelLayout::RGB;
		optimal_storage_type = PixelDataType::BYTE;
		optimal_storage_size = 3;
		minimal_storage_size = 24;
		break;
	case InternalPixelFormat::SIZED_RGB10:
		num_of_components = 3;
		is_color = true;
		r_size = 10;
		g_size = 10;
		b_size = 10;
		pixel_layout = PixelLayout::RGB;
		optimal_storage_type = PixelDataType::USHORT;
		optimal_storage_size = 6;
		minimal_storage_size = 30;
		break;
	case InternalPixelFormat::SIZED_RGB12:
		num_of_components = 3;
		is_color = true;
		r_size = 12;
		g_size = 12;
		b_size = 12;
		pixel_layout = PixelLayout::RGB;
		optimal_storage_type = PixelDataType::USHORT;
		optimal_storage_size = 6;
		minimal_storage_size = 36;
		break;
	case InternalPixelFormat::SIZED_RGB16:
		num_of_components = 3;
		is_color = true;
		r_size = 16;
		g_size = 16;
		b_size = 16;
		pixel_layout = PixelLayout::RGB;
		optimal_storage_type = PixelDataType::USHORT;
		optimal_storage_size = 6;
		minimal_storage_size = 48;
		break;
	case InternalPixelFormat::SIZED_RGB16_SNORM:
		num_of_components = 3;
		is_color = true;
		is_signed_normalized = true;
		is_signed = true;
		r_size = 16;
		g_size = 16;
		b_size = 16;
		pixel_layout = PixelLayout::RGB;
		optimal_storage_type = PixelDataType::SHORT;
		optimal_storage_size = 6;
		minimal_storage_size = 48;
		break;
	case InternalPixelFormat::SIZED_RGBA2:
		num_of_components = 4;
		is_color = true;
		r_size = 2;
		g_size = 2;
		b_size = 2;
		a_size = 2;
		pixel_layout = PixelLayout::RGBA;
		optimal_storage_size = 4;
		minimal_storage_size = 8;
		break;
	case InternalPixelFormat::SIZED_RGBA4:
		num_of_components = 4;
		is_color = true;
		r_size = 4;
		g_size = 4;
		b_size = 4;
		a_size = 4;
		pixel_layout = PixelLayout::RGBA;
		optimal_storage_type = PixelDataType::USHORT_4_4_4_4;
		optimal_storage_size = 2;
		minimal_storage_size = 16;
		break;
	case InternalPixelFormat::SIZED_RGB5_A1:
		num_of_components = 4;
		is_color = true;
		r_size = 5;
		g_size = 5;
		b_size = 5;
		a_size = 1;
		pixel_layout = PixelLayout::RGBA;
		optimal_storage_type = PixelDataType::USHORT_5_5_5_1;
		optimal_storage_size = 2;
		minimal_storage_size = 16;
		break;
	case InternalPixelFormat::SIZED_RGBA8:
		num_of_components = 4;
		is_color = true;
		r_size = 8;
		g_size = 8;
		b_size = 8;
		a_size = 8;
		pixel_layout = PixelLayout::RGBA;
		optimal_storage_size = 4;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 32;
		break;
	case InternalPixelFormat::SIZED_RGBA8_SNORM:
		num_of_components = 4;
		is_color = true;
		is_signed_normalized = true;
		is_signed = true;
		r_size = 8;
		g_size = 8;
		b_size = 8;
		a_size = 8;
		pixel_layout = PixelLayout::RGBA;
		optimal_storage_type = PixelDataType::BYTE;
		optimal_storage_size = 4;
		minimal_storage_size = 32;
		break;
	case InternalPixelFormat::SIZED_RGB10_A2:
		num_of_components = 4;
		is_color = true;
		r_size = 10;
		g_size = 10;
		b_size = 10;
		a_size = 2;
		pixel_layout = PixelLayout::RGBA;
		optimal_storage_type = PixelDataType::UINT_10_10_10_2;
		optimal_storage_size = 4;
		minimal_storage_size = 32;
		break;
	case InternalPixelFormat::SIZED_RGB10_A2UI:
		num_of_components = 4;
		is_color = true;
		r_size = 10;
		g_size = 10;
		b_size = 10;
		a_size = 2;
		pixel_layout = PixelLayout::RGBA;
		optimal_storage_type = PixelDataType::UINT_10_10_10_2;
		optimal_storage_size = 4;
		minimal_storage_size = 32;
		break;
	case InternalPixelFormat::SIZED_RGBA12:
		num_of_components = 4;
		is_color = true;
		r_size = 12;
		g_size = 12;
		b_size = 12;
		a_size = 12;
		pixel_layout = PixelLayout::RGBA;
		optimal_storage_type = PixelDataType::USHORT;
		optimal_storage_size = 8;
		minimal_storage_size = 48;
		break;
	case InternalPixelFormat::SIZED_RGBA16:
		num_of_components = 4;
		is_color = true;
		r_size = 16;
		g_size = 16;
		b_size = 16;
		a_size = 16;
		pixel_layout = PixelLayout::RGBA;
		optimal_storage_type = PixelDataType::USHORT;
		optimal_storage_size = 8;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 64;
		break;
	case InternalPixelFormat::SIZED_RGBA16_SNORM:
		num_of_components = 4;
		is_color = true;
		is_signed_normalized = true;
		is_signed = true;
		r_size = 16;
		g_size = 16;
		b_size = 16;
		a_size = 16;
		pixel_layout = PixelLayout::RGBA;
		optimal_storage_type = PixelDataType::SHORT;
		optimal_storage_size = 8;
		minimal_storage_size = 64;
		break;
	case InternalPixelFormat::SIZED_SRGB8:
		num_of_components = 3;
		is_color = true;
		r_size = 8;
		g_size = 8;
		b_size = 8;
		pixel_layout = PixelLayout::RGB;
		optimal_storage_type = PixelDataType::UBYTE;
		optimal_storage_size = 3;
		minimal_storage_size = 24;
		break;
	case InternalPixelFormat::SIZED_SRGB8_ALPHA8:
		num_of_components = 4;
		is_color = true;
		r_size = 8;
		g_size = 8;
		b_size = 8;
		a_size = 8;
		pixel_layout = PixelLayout::RGBA;
		optimal_storage_type = PixelDataType::UBYTE;
		optimal_storage_size = 4;
		minimal_storage_size = 32;
		break;
	case InternalPixelFormat::SIZED_FLOAT_R16:
		num_of_components = 1;
		is_color = true;
		is_float_format = true;
		is_signed = true;
		r_size = 16;
		pixel_layout = PixelLayout::RED;
		optimal_storage_type = PixelDataType::FLOAT;
		optimal_storage_size = 4;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 16;
		break;
	case InternalPixelFormat::SIZED_FLOAT_RG16:
		num_of_components = 2;
		is_color = true;
		is_float_format = true;
		is_signed = true;
		r_size = 16;
		g_size = 16;
		pixel_layout = PixelLayout::RG;
		optimal_storage_type = PixelDataType::FLOAT;
		optimal_storage_size = 8;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 32;
		break;
	case InternalPixelFormat::SIZED_FLOAT_RGB16:
		num_of_components = 3;
		is_color = true;
		is_float_format = true;
		is_signed = true;
		r_size = 16;
		g_size = 16;
		b_size = 16;
		pixel_layout = PixelLayout::RGB;
		optimal_storage_type = PixelDataType::FLOAT;
		optimal_storage_size = 12;
		minimal_storage_size = 48;
		break;
	case InternalPixelFormat::SIZED_FLOAT_RGBA16:
		num_of_components = 4;
		is_color = true;
		is_float_format = true;
		is_signed = true;
		r_size = 16;
		g_size = 16;
		b_size = 16;
		a_size = 16;
		pixel_layout = PixelLayout::RGBA;
		optimal_storage_type = PixelDataType::FLOAT;
		optimal_storage_size = 16;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 64;
		break;
	case InternalPixelFormat::SIZED_FLOAT_R32:
		num_of_components = 1;
		is_color = true;
		is_float_format = true;
		is_signed = true;
		r_size = 32;
		pixel_layout = PixelLayout::RED;
		optimal_storage_type = PixelDataType::FLOAT;
		optimal_storage_size = 4;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 32;
		break;
	case InternalPixelFormat::SIZED_FLOAT_RG32:
		num_of_components = 2;
		is_color = true;
		is_float_format = true;
		is_signed = true;
		r_size = 32;
		g_size = 32;
		pixel_layout = PixelLayout::RG;
		optimal_storage_type = PixelDataType::FLOAT;
		optimal_storage_size = 8;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 64;
		break;
	case InternalPixelFormat::SIZED_FLOAT_RGB32:
		num_of_components = 3;
		is_color = true;
		is_float_format = true;
		is_signed = true;
		r_size = 32;
		g_size = 32;
		b_size = 32;
		pixel_layout = PixelLayout::RGB;
		optimal_storage_type = PixelDataType::FLOAT;
		optimal_storage_size = 12;
		minimal_storage_size = 96;
		break;
	case InternalPixelFormat::SIZED_FLOAT_RGBA32:
		num_of_components = 4;
		is_color = true;
		is_float_format = true;
		is_signed = true;
		r_size = 32;
		g_size = 32;
		b_size = 32;
		a_size = 32;
		pixel_layout = PixelLayout::RGBA;
		optimal_storage_type = PixelDataType::FLOAT;
		optimal_storage_size = 16;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 128;
		break;
	case InternalPixelFormat::SIZED_FLOAT_R11_G11_B10:
		num_of_components = 3;
		is_color = true;
		is_float_format = true;
		r_size = 11;
		g_size = 11;
		b_size = 10;
		pixel_layout = PixelLayout::RGB;
		optimal_storage_type = PixelDataType::UINT_10F_11F_11F_R;
		optimal_storage_size = 4;
		minimal_storage_size = 32;
		break;
	case InternalPixelFormat::SIZED_FLOAT_RGB9_E5:
		num_of_components = 3;
		is_color = true;
		is_float_format = true;
		r_size = 14;
		g_size = 14;
		b_size = 14;
		pixel_layout = PixelLayout::RGB;
		optimal_storage_type = PixelDataType::UINT_5_9_9_9_R;
		optimal_storage_size = 4;
		minimal_storage_size = 32;
		break;
	case InternalPixelFormat::SIZED_INT_R8:
		num_of_components = 1;
		is_color = true;
		is_integer_format = true;
		is_signed = true;
		r_size = 8;
		pixel_layout = PixelLayout::INTEGER_RED;
		optimal_storage_type = PixelDataType::BYTE;
		optimal_storage_size = 1;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 8;
		break;
	case InternalPixelFormat::SIZED_UINT_R8:
		num_of_components = 1;
		is_color = true;
		is_integer_format = true;
		r_size = 8;
		pixel_layout = PixelLayout::INTEGER_RED;
		optimal_storage_type = PixelDataType::UBYTE;
		optimal_storage_size = 1;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 8;
		break;
	case InternalPixelFormat::SIZED_INT_R16:
		num_of_components = 1;
		is_color = true;
		is_integer_format = true;
		is_signed = true;
		r_size = 16;
		pixel_layout = PixelLayout::INTEGER_RED;
		optimal_storage_type = PixelDataType::SHORT;
		optimal_storage_size = 2;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 16;
		break;
	case InternalPixelFormat::SIZED_UINT_R16:
		num_of_components = 1;
		is_color = true;
		is_integer_format = true;
		r_size = 16;
		pixel_layout = PixelLayout::INTEGER_RED;
		optimal_storage_type = PixelDataType::USHORT;
		optimal_storage_size = 2;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 16;
		break;
	case InternalPixelFormat::SIZED_INT_R32:
		num_of_components = 1;
		is_color = true;
		is_integer_format = true;
		is_signed = true;
		r_size = 32;
		pixel_layout = PixelLayout::INTEGER_RED;
		optimal_storage_type = PixelDataType::INT;
		optimal_storage_size = 4;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 32;
		break;
	case InternalPixelFormat::SIZED_UINT_R32:
		num_of_components = 1;
		is_color = true;
		is_integer_format = true;
		r_size = 32;
		pixel_layout = PixelLayout::INTEGER_RED;
		optimal_storage_type = PixelDataType::UINT;
		optimal_storage_size = 4;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 32;
		break;
	case InternalPixelFormat::SIZED_INT_RG8:
		num_of_components = 2;
		is_color = true;
		is_integer_format = true;
		is_signed = true;
		r_size = 8;
		g_size = 8;
		pixel_layout = PixelLayout::INTEGER_RG;
		optimal_storage_type = PixelDataType::BYTE;
		optimal_storage_size = 2;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 16;
		break;
	case InternalPixelFormat::SIZED_UINT_RG8:
		num_of_components = 2;
		is_color = true;
		is_integer_format = true;
		r_size = 8;
		g_size = 8;
		pixel_layout = PixelLayout::INTEGER_RG;
		optimal_storage_type = PixelDataType::UBYTE;
		optimal_storage_size = 2;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 16;
		break;
	case InternalPixelFormat::SIZED_INT_RG16:
		num_of_components = 2;
		is_color = true;
		is_integer_format = true;
		is_signed = true;
		r_size = 16;
		g_size = 16;
		pixel_layout = PixelLayout::INTEGER_RG;
		optimal_storage_type = PixelDataType::SHORT;
		optimal_storage_size = 4;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 32;
		break;
	case InternalPixelFormat::SIZED_UINT_RG16:
		num_of_components = 2;
		is_color = true;
		is_integer_format = true;
		r_size = 16;
		g_size = 16;
		pixel_layout = PixelLayout::INTEGER_RG;
		optimal_storage_type = PixelDataType::USHORT;
		optimal_storage_size = 4;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 32;
		break;
	case InternalPixelFormat::SIZED_INT_RG32:
		num_of_components = 2;
		is_color = true;
		is_integer_format = true;
		is_signed = true;
		r_size = 32;
		g_size = 32;
		pixel_layout = PixelLayout::INTEGER_RG;
		optimal_storage_type = PixelDataType::INT;
		optimal_storage_size = 8;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 64;
		break;
	case InternalPixelFormat::SIZED_UINT_RG32:
		num_of_components = 2;
		is_color = true;
		is_integer_format = true;
		r_size = 32;
		g_size = 32;
		pixel_layout = PixelLayout::INTEGER_RG;
		optimal_storage_type = PixelDataType::UINT;
		optimal_storage_size = 8;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 64;
		break;
	case InternalPixelFormat::SIZED_INT_RGB8:
		num_of_components = 3;
		is_color = true;
		is_integer_format = true;
		is_signed = true;
		r_size = 8;
		g_size = 8;
		b_size = 8;
		pixel_layout = PixelLayout::INTEGER_RGB;
		optimal_storage_type = PixelDataType::BYTE;
		optimal_storage_size = 3;
		minimal_storage_size = 24;
		break;
	case InternalPixelFormat::SIZED_UINT_RGB8:
		num_of_components = 3;
		is_color = true;
		is_integer_format = true;
		r_size = 8;
		g_size = 8;
		b_size = 8;
		pixel_layout = PixelLayout::INTEGER_RGB;
		optimal_storage_type = PixelDataType::UBYTE;
		optimal_storage_size = 3;
		minimal_storage_size = 24;
		break;
	case InternalPixelFormat::SIZED_INT_RGB16:
		num_of_components = 3;
		is_color = true;
		is_integer_format = true;
		is_signed = true;
		r_size = 16;
		g_size = 16;
		b_size = 16;
		pixel_layout = PixelLayout::INTEGER_RGB;
		optimal_storage_type = PixelDataType::SHORT;
		optimal_storage_size = 6;
		minimal_storage_size = 48;
		break;
	case InternalPixelFormat::SIZED_UINT_RGB16:
		num_of_components = 3;
		is_color = true;
		is_integer_format = true;
		r_size = 16;
		g_size = 16;
		b_size = 16;
		pixel_layout = PixelLayout::INTEGER_RGB;
		optimal_storage_type = PixelDataType::USHORT;
		optimal_storage_size = 6;
		minimal_storage_size = 48;
		break;
	case InternalPixelFormat::SIZED_INT_RGB32:
		num_of_components = 3;
		is_color = true;
		is_integer_format = true;
		is_signed = true;
		r_size = 32;
		g_size = 32;
		b_size = 32;
		pixel_layout = PixelLayout::INTEGER_RGB;
		optimal_storage_type = PixelDataType::INT;
		optimal_storage_size = 12;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 96;
		break;
	case InternalPixelFormat::SIZED_UINT_RGB32:
		num_of_components = 3;
		is_color = true;
		is_integer_format = true;
		r_size = 32;
		g_size = 32;
		b_size = 32;
		pixel_layout = PixelLayout::INTEGER_RGB;
		optimal_storage_type = PixelDataType::UINT;
		optimal_storage_size = 12;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 96;
		break;
	case InternalPixelFormat::SIZED_INT_RGBA8:
		num_of_components = 4;
		is_color = true;
		is_integer_format = true;
		is_signed = true;
		r_size = 8;
		g_size = 8;
		b_size = 8;
		a_size = 8;
		pixel_layout = PixelLayout::INTEGER_RGBA;
		optimal_storage_type = PixelDataType::BYTE;
		optimal_storage_size = 4;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 32;
		break;
	case InternalPixelFormat::SIZED_UINT_RGBA8:
		num_of_components = 4;
		is_color = true;
		is_integer_format = true;
		r_size = 8;
		g_size = 8;
		b_size = 8;
		a_size = 8;
		pixel_layout = PixelLayout::INTEGER_RGBA;
		optimal_storage_type = PixelDataType::UBYTE;
		optimal_storage_size = 4;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 32;
		break;
	case InternalPixelFormat::SIZED_INT_RGBA16:
		num_of_components = 4;
		is_color = true;
		is_integer_format = true;
		is_signed = true;
		r_size = 16;
		g_size = 16;
		b_size = 16;
		a_size = 16;
		pixel_layout = PixelLayout::INTEGER_RGBA;
		optimal_storage_type = PixelDataType::SHORT;
		optimal_storage_size = 8;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 64;
		break;
	case InternalPixelFormat::SIZED_UINT_RGBA16:
		num_of_components = 4;
		is_color = true;
		is_integer_format = true;
		r_size = 16;
		g_size = 16;
		b_size = 16;
		a_size = 16;
		pixel_layout = PixelLayout::INTEGER_RGBA;
		optimal_storage_type = PixelDataType::USHORT;
		optimal_storage_size = 8;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 64;
		break;
	case InternalPixelFormat::SIZED_INT_RGBA32:
		num_of_components = 4;
		is_color = true;
		is_integer_format = true;
		is_signed = true;
		r_size = 32;
		g_size = 32;
		b_size = 32;
		a_size = 32;
		pixel_layout = PixelLayout::INTEGER_RGBA;
		optimal_storage_type = PixelDataType::INT;
		optimal_storage_size = 16;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 128;
		break;
	case InternalPixelFormat::SIZED_UINT_RGBA32:
		num_of_components = 4;
		is_color = true;
		is_integer_format = true;
		r_size = 32;
		g_size = 32;
		b_size = 32;
		a_size = 32;
		pixel_layout = PixelLayout::INTEGER_RGBA;
		optimal_storage_type = PixelDataType::UINT;
		optimal_storage_size = 16;
		is_compatible_with_buffer_textures = true;
		minimal_storage_size = 128;
		break;
	case InternalPixelFormat::SIZED_DEPTH16:
		num_of_components = 1;
		is_depth = true;
		depth_size = 16;
		pixel_layout = PixelLayout::DEPTH;
		optimal_storage_type = PixelDataType::USHORT;
		optimal_storage_size = 2;
		minimal_storage_size = 16;
		break;
	case InternalPixelFormat::SIZED_DEPTH24:
		num_of_components = 1;
		is_depth = true;
		depth_size = 24;
		pixel_layout = PixelLayout::DEPTH;
		optimal_storage_type = PixelDataType::UINT;
		optimal_storage_size = 4;
		minimal_storage_size = 24;
		break;
	case InternalPixelFormat::SIZED_DEPTH32:
		num_of_components = 1;
		is_depth = true;
		depth_size = 32;
		pixel_layout = PixelLayout::DEPTH;
		optimal_storage_type = PixelDataType::UINT;
		optimal_storage_size = 4;
		minimal_storage_size = 32;
		break;
	case InternalPixelFormat::SIZED_FLOAT_DEPTH32:
		num_of_components = 1;
		is_depth = true;
		is_float_format = true;
		is_signed = true;
		depth_size = 32;
		pixel_layout = PixelLayout::DEPTH;
		optimal_storage_type = PixelDataType::FLOAT;
		optimal_storage_size = 4;
		minimal_storage_size = 32;
		break;
	case InternalPixelFormat::SIZED_DEPTH24_STENCIL8:
		num_of_components = 2;
		is_depth = true;
		is_stencil = true;
		depth_size = 24;
		stencil_size = 8;
		pixel_layout = PixelLayout::DEPTH_STENCIL;
		optimal_storage_type = PixelDataType::UINT_24_8;
		optimal_storage_size = 4;
		minimal_storage_size = 32;
		break;
	case InternalPixelFormat::SIZED_FLOAT_DEPTH32_STENCIL8:
		num_of_components = 2;
		is_depth = true;
		is_stencil = true;
		is_float_format = true;
		depth_size = 32;
		stencil_size = 8;
		pixel_layout = PixelLayout::DEPTH_STENCIL;
		optimal_storage_type = PixelDataType::FLOAT_32_UINT_24_8_R;
		optimal_storage_size = 8;
		minimal_storage_size = 40;
		break;
	case InternalPixelFormat::SIZED_STENCIL1:
		num_of_components = 1;
		is_stencil = true;
		stencil_size = 1;
		pixel_layout = PixelLayout::STENCIL;
		optimal_storage_size = 1;
		minimal_storage_size = 1;
		break;
	case InternalPixelFormat::SIZED_STENCIL4:
		num_of_components = 1;
		is_stencil = true;
		stencil_size = 4;
		pixel_layout = PixelLayout::STENCIL;
		optimal_storage_size = 1;
		minimal_storage_size = 4;
		break;
	case InternalPixelFormat::SIZED_STENCIL8:
		num_of_components = 1;
		is_stencil = true;
		stencil_size = 8;
		pixel_layout = PixelLayout::STENCIL;
		optimal_storage_size = 1;
		minimal_storage_size = 8;
		break;
	case InternalPixelFormat::SIZED_STENCIL16:
		num_of_components = 1;
		is_stencil = true;
		stencil_size = 16;
		pixel_layout = PixelLayout::STENCIL;
		optimal_storage_type = PixelDataType::USHORT;
		optimal_storage_size = 2;
		minimal_storage_size = 16;
		break;
	default:
		num_of_components = 0;
		is_recognized = false;
		r_size = -1;
		b_size = -1;
		g_size = -1;
		depth_size = -1;
		stencil_size = -1;
		optimal_storage_size = 0;
		is_compatible_with_buffer_textures = false;
		minimal_storage_size = -1;
	}
}

void PixelFormatTraits::ConstructCompressedFormatTraits(InternalPixelFormatCompressed internal_format_compressed)
{
	//Set default values for internal storage format parameters
	ogl_format = static_cast<GLenum>(internal_format_compressed);
	is_color = false;
	is_depth = false;
	is_stencil = false;
	is_recognized = true;	//by default we assume that the format has been successfully recognized
	is_compressed = true;
	is_signed_normalized = false;
	is_float_format = false;
	is_integer_format = false;
	is_signed = false;
	r_size = 0;
	g_size = 0;
	b_size = 0;
	a_size = 0;
	depth_size = 0;
	stencil_size = 0;
	optimal_storage_type = PixelDataType::UBYTE;
	optimal_storage_size = 4;
	is_compatible_with_buffer_textures = false;
	minimal_storage_size = -1;

	switch (internal_format_compressed)
	{
	case InternalPixelFormatCompressed::COMPRESSED_BASE_R:
		num_of_components = 1;
		is_color = true;
		is_compressed = true;
		r_size = -1;
		pixel_layout = PixelLayout::RED;
		optimal_storage_size = 1;
		break;
	case InternalPixelFormatCompressed::COMPRESSED_BASE_RG:
		num_of_components = 2;
		is_color = true;
		is_compressed = true;
		r_size = -1;
		g_size = -1;
		pixel_layout = PixelLayout::RG;
		optimal_storage_size = 2;
		break;
	case InternalPixelFormatCompressed::COMPRESSED_BASE_RGB:
		num_of_components = 3;
		is_color = true;
		is_compressed = true;
		r_size = -1;
		g_size = -1;
		b_size = -1;
		pixel_layout = PixelLayout::RGB;
		optimal_storage_size = 3;
		break;
	case InternalPixelFormatCompressed::COMPRESSED_BASE_RGBA:
		num_of_components = 4;
		is_color = true;
		is_compressed = true;
		r_size = -1;
		g_size = -1;
		b_size = -1;
		a_size = -1;
		pixel_layout = PixelLayout::RGBA;
		optimal_storage_size = 4;
		break;
	case InternalPixelFormatCompressed::COMPRESSED_BASE_SRGB:
		num_of_components = 3;
		is_color = true;
		is_compressed = true;
		r_size = -1;
		g_size = -1;
		b_size = -1;
		pixel_layout = PixelLayout::RGB;
		optimal_storage_size = 3;
		break;
	case InternalPixelFormatCompressed::COMPRESSED_BASE_SRGB_ALPHA:
		num_of_components = 4;
		is_color = true;
		is_compressed = true;
		r_size = -1;
		g_size = -1;
		b_size = -1;
		a_size = -1;
		pixel_layout = PixelLayout::RGBA;
		optimal_storage_size = 4;
		break;
	case InternalPixelFormatCompressed::COMPRESSED_R_RGTC1:
		num_of_components = 1;
		is_color = true;
		is_compressed = true;
		r_size = -1;
		pixel_layout = PixelLayout::RED;
		optimal_storage_size = 1;
		break;
	case InternalPixelFormatCompressed::COMPRESSED_SIGNED_R_RGTC1:
		num_of_components = 1;
		is_color = true;
		is_compressed = true;
		is_signed = true;
		r_size = -1;
		pixel_layout = PixelLayout::RED;
		optimal_storage_type = PixelDataType::BYTE;
		optimal_storage_size = 1;
		break;
	case InternalPixelFormatCompressed::COMPRESSED_RG_RGTC2:
		num_of_components = 2;
		is_color = true;
		is_compressed = true;
		r_size = -1;
		g_size = -1;
		pixel_layout = PixelLayout::RG;
		optimal_storage_size = 2;
		break;
	case InternalPixelFormatCompressed::COMPRESSED_SIGNED_RG_RGTC2:
		num_of_components = 2;
		is_color = true;
		is_compressed = true;
		is_signed = true;
		r_size = -1;
		g_size = -1;
		pixel_layout = PixelLayout::RG;
		optimal_storage_type = PixelDataType::BYTE;
		optimal_storage_size = 2;
		break;
	case InternalPixelFormatCompressed::COMPRESSED_RGBA_BPTC_UNORM:
		num_of_components = 4;
		is_color = true;
		is_compressed = true;
		r_size = -1;
		g_size = -1;
		b_size = -1;
		a_size = -1;
		pixel_layout = PixelLayout::RGBA;
		optimal_storage_size = 4;
		break;
	case InternalPixelFormatCompressed::COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
		num_of_components = 4;
		is_color = true;
		is_compressed = true;
		r_size = -1;
		g_size = -1;
		b_size = -1;
		a_size = -1;
		pixel_layout = PixelLayout::RGBA;
		optimal_storage_size = 4;
		break;
	case InternalPixelFormatCompressed::COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
		num_of_components = 3;
		is_color = true;
		is_compressed = true;
		is_float_format = true;
		is_signed = true;
		r_size = -1;
		g_size = -1;
		b_size = -1;
		pixel_layout = PixelLayout::RGB;
		optimal_storage_type = PixelDataType::FLOAT;
		optimal_storage_size = 12;
		break;
	case InternalPixelFormatCompressed::COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
		num_of_components = 3;
		is_color = true;
		is_compressed = true;
		is_float_format = true;
		r_size = -1;
		g_size = -1;
		b_size = -1;
		pixel_layout = PixelLayout::RGB;
		optimal_storage_type = PixelDataType::FLOAT;
		optimal_storage_size = 12;
		break;
	case InternalPixelFormatCompressed::COMPRESSED_RGB8_ETC2:
		num_of_components = 3;
		is_color = true;
		is_compressed = true;
		r_size = 8;
		g_size = 8;
		b_size = 8;
		pixel_layout = PixelLayout::RGB;
		optimal_storage_size = 3;
		break;
	case InternalPixelFormatCompressed::COMPRESSED_SRGB8_ETC2:
		num_of_components = 3;
		is_color = true;
		is_compressed = true;
		r_size = 8;
		g_size = 8;
		b_size = 8;
		pixel_layout = PixelLayout::RGB;
		optimal_storage_size = 3;
		break;
	case InternalPixelFormatCompressed::COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2:
		num_of_components = 4;
		is_color = true;
		is_compressed = true;
		r_size = 8;
		g_size = 8;
		b_size = 8;
		a_size = 1;
		pixel_layout = PixelLayout::RGBA;
		optimal_storage_size = 4;
		break;
	case InternalPixelFormatCompressed::COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2:
		num_of_components = 4;
		is_color = true;
		is_compressed = true;
		r_size = 8;
		g_size = 8;
		b_size = 8;
		a_size = 1;
		pixel_layout = PixelLayout::RGBA;
		optimal_storage_size = 4;
		break;
	case InternalPixelFormatCompressed::COMPRESSED_RGBA8_ETC2_EAC:
		num_of_components = 4;
		is_color = true;
		is_compressed = true;
		r_size = 8;
		g_size = 8;
		b_size = 8;
		a_size = 8;
		pixel_layout = PixelLayout::RGBA;
		optimal_storage_size = 4;
		break;
	case InternalPixelFormatCompressed::COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:
		num_of_components = 4;
		is_color = true;
		is_compressed = true;
		r_size = 8;
		g_size = 8;
		b_size = 8;
		a_size = 8;
		pixel_layout = PixelLayout::RGBA;
		optimal_storage_size = 4;
		break;
	case InternalPixelFormatCompressed::COMPRESSED_R11_EAC:
		num_of_components = 1;
		is_color = true;
		is_compressed = true;
		r_size = 11;
		pixel_layout = PixelLayout::RED;
		optimal_storage_type = PixelDataType::USHORT;
		optimal_storage_size = 2;
		break;
	case InternalPixelFormatCompressed::COMPRESSED_SIGNED_R11_EAC:
		num_of_components = 1;
		is_color = true;
		is_compressed = true;
		is_signed = true;
		r_size = 11;
		pixel_layout = PixelLayout::RED;
		optimal_storage_type = PixelDataType::USHORT;
		optimal_storage_size = 2;
		break;
	case InternalPixelFormatCompressed::COMPRESSED_RG11_EAC:
		num_of_components = 2;
		is_color = true;
		is_compressed = true;
		r_size = 11;
		g_size = 11;
		pixel_layout = PixelLayout::RG;
		optimal_storage_type = PixelDataType::USHORT;
		optimal_storage_size = 4;
		break;
	case InternalPixelFormatCompressed::COMPRESSED_SIGNED_RG11_EAC:
		num_of_components = 2;
		is_color = true;
		is_compressed = true;
		is_signed = true;
		r_size = 11;
		g_size = 11;
		pixel_layout = PixelLayout::RG;
		optimal_storage_type = PixelDataType::USHORT;
		optimal_storage_size = 4;
		break;
	default:
		num_of_components = 0;
		is_compressed = false;
		is_recognized = false;
		r_size = -1;
		b_size = -1;
		g_size = -1;
		depth_size = -1;
		stencil_size = -1;
		optimal_storage_size = 0;

	}
}

PixelFormatTraits::PixelFormatTraits(InternalPixelFormat internal_format)
{
	ConstructSizedFormatTraits(internal_format);
}

PixelFormatTraits::PixelFormatTraits(InternalPixelFormatCompressed internal_format_compressed)
{
	ConstructCompressedFormatTraits(internal_format_compressed);
}

PixelFormatTraits::PixelFormatTraits(BufferTextureInternalPixelFormat buffer_texture_internal_format)
{
	ConstructSizedFormatTraits(static_cast<InternalPixelFormat>(static_cast<GLenum>(buffer_texture_internal_format)));
}

PixelFormatTraits::PixelFormatTraits(GLenum ogl_internal_format)
{
	switch (ogl_internal_format)
	{
	case GL_DEPTH_COMPONENT:
	case GL_DEPTH_STENCIL:
	case GL_RED:
	case GL_RG:
	case GL_RGB:
	case GL_RGBA:
	case GL_R8:
	case GL_R8_SNORM:
	case GL_R16:
	case GL_R16_SNORM:
	case GL_RG8:
	case GL_RG8_SNORM:
	case GL_RG16:
	case GL_RG16_SNORM:
	case GL_R3_G3_B2:
	case GL_RGB4:
	case GL_RGB5:
	case GL_RGB565:
	case GL_RGB8:
	case GL_RGB8_SNORM:
	case GL_RGB10:
	case GL_RGB12:
	case GL_RGB16:
	case GL_RGB16_SNORM:
	case GL_RGBA2:
	case GL_RGBA4:
	case GL_RGB5_A1:
	case GL_RGBA8:
	case GL_RGBA8_SNORM:
	case GL_RGB10_A2:
	case GL_RGB10_A2UI:
	case GL_RGBA12:
	case GL_RGBA16:
	case GL_RGBA16_SNORM:
	case GL_SRGB8:
	case GL_SRGB8_ALPHA8:
	case GL_R16F:
	case GL_RG16F:
	case GL_RGB16F:
	case GL_RGBA16F:
	case GL_R32F:
	case GL_RG32F:
	case GL_RGB32F:
	case GL_RGBA32F:
	case GL_R11F_G11F_B10F:
	case GL_RGB9_E5:
	case GL_DEPTH_COMPONENT32F:
	case GL_DEPTH32F_STENCIL8:
	case GL_R8I:
	case GL_R8UI:
	case GL_R16I:
	case GL_R16UI:
	case GL_R32I:
	case GL_R32UI:
	case GL_RG8I:
	case GL_RG8UI:
	case GL_RG16I:
	case GL_RG16UI:
	case GL_RG32I:
	case GL_RG32UI:
	case GL_RGB8I:
	case GL_RGB8UI:
	case GL_RGB16I:
	case GL_RGB16UI:
	case GL_RGB32I:
	case GL_RGB32UI:
	case GL_RGBA8I:
	case GL_RGBA8UI:
	case GL_RGBA16I:
	case GL_RGBA16UI:
	case GL_RGBA32I:
	case GL_RGBA32UI:
	case GL_DEPTH_COMPONENT16:
	case GL_DEPTH_COMPONENT24:
	case GL_DEPTH_COMPONENT32:
	case GL_DEPTH24_STENCIL8:
	case GL_STENCIL_INDEX1:
	case GL_STENCIL_INDEX4:
	case GL_STENCIL_INDEX8:
	case GL_STENCIL_INDEX16:
		ConstructSizedFormatTraits(static_cast<InternalPixelFormat>(ogl_internal_format));
		break;


	case GL_COMPRESSED_RED:
	case GL_COMPRESSED_RG:
	case GL_COMPRESSED_RGB:
	case GL_COMPRESSED_RGBA:
	case GL_COMPRESSED_SRGB:
	case GL_COMPRESSED_SRGB_ALPHA:
	case GL_COMPRESSED_RED_RGTC1:
	case GL_COMPRESSED_SIGNED_RED_RGTC1:
	case GL_COMPRESSED_RG_RGTC2:
	case GL_COMPRESSED_SIGNED_RG_RGTC2:
	case GL_COMPRESSED_RGBA_BPTC_UNORM:
	case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
	case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
	case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
	case GL_COMPRESSED_RGB8_ETC2:
	case GL_COMPRESSED_SRGB8_ETC2:
	case GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2:
	case GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2:
	case GL_COMPRESSED_RGBA8_ETC2_EAC:
	case GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:
	case GL_COMPRESSED_R11_EAC:
	case GL_COMPRESSED_SIGNED_R11_EAC:
	case GL_COMPRESSED_RG11_EAC:
	case GL_COMPRESSED_SIGNED_RG11_EAC:
		ConstructCompressedFormatTraits(static_cast<InternalPixelFormatCompressed>(ogl_internal_format));
		break;

	default:
		num_of_components = 0;
		is_color = false;
		is_depth = false;
		is_stencil = false;
		is_recognized = false;
		is_compressed = false;
		is_signed_normalized = false;
		is_float_format = false;
		is_integer_format = false;
		is_signed = false;
		optimal_storage_type = PixelDataType::UBYTE;
		optimal_storage_size = 0;
		r_size = -1;
		g_size = -1;
		b_size = -1;
		a_size = -1;
		depth_size = -1;
		stencil_size = -1;
	}
}

unsigned short PixelFormatTraits::getNumberOfTexelComponents() const { return num_of_components; }

bool PixelFormatTraits::isColor() const { return is_color; }

bool PixelFormatTraits::isDepth() const { return is_depth; }

bool PixelFormatTraits::isStencil() const { return is_stencil; }

bool PixelFormatTraits::isRecognized() const { return is_recognized; }

bool PixelFormatTraits::isCompressed() const { return is_compressed; }

bool PixelFormatTraits::isSignedNormalized() const { return is_signed_normalized; }

bool PixelFormatTraits::isSigned() const { return is_signed; }

bool PixelFormatTraits::isFloat() const { return is_float_format; }

bool PixelFormatTraits::isInteger() const { return is_integer_format; }

bool PixelFormatTraits::isBufferTextureCompatible() const { return is_compatible_with_buffer_textures; }

short PixelFormatTraits::getRedBits() const { return r_size; }

short PixelFormatTraits::getGreenBits() const { return g_size; }

short PixelFormatTraits::getBlueBits() const { return b_size; }

short PixelFormatTraits::getAlphaBits() const { return a_size; }

short PixelFormatTraits::getDepthBits() const { return depth_size; }

short PixelFormatTraits::getStencilBits() const { return stencil_size; }

PixelLayout PixelFormatTraits::getPixelLayout() const { return pixel_layout; }

PixelDataType PixelFormatTraits::getOptimalStorageType() const { return optimal_storage_type; }

unsigned short PixelFormatTraits::getOptimalStorageSize() const { return optimal_storage_size; }

short PixelFormatTraits::getMinimalStorageSize() const { return minimal_storage_size; }

PixelStorageClass PixelFormatTraits::getStorageClass() const
{
	if (!is_color || is_compressed || !is_recognized) return PixelStorageClass::unknown;

	PixelStorageClass rv = PixelStorageClass::unknown;
	switch (getNumberOfTexelComponents())
	{
	case 4:
		if (getRedBits() == 32 && getGreenBits() == 32 && getBlueBits() == 32 && getAlphaBits() == 32)
			rv = PixelStorageClass::_4x32;

		if (getRedBits() == 16 && getGreenBits() == 16 && getBlueBits() == 16 && getAlphaBits() == 16)
			rv = PixelStorageClass::_4x16;

		if (getRedBits() == 8 && getGreenBits() == 8 && getBlueBits() == 8 && getAlphaBits() == 8)
			rv = PixelStorageClass::_4x8;
		break;

	case 3:
		if (getRedBits() == 32 && getGreenBits() == 32 && getBlueBits() == 32)
			rv = PixelStorageClass::_3x32;

		if (getRedBits() == 16 && getGreenBits() == 16 && getBlueBits() == 16)
			rv = PixelStorageClass::_3x16;

		if (getRedBits() == 8 && getGreenBits() == 8 && getBlueBits() == 8)
			rv = PixelStorageClass::_3x8;
		break;

	case 2:
		if (getRedBits() == 32 && getGreenBits() == 32)
			rv = PixelStorageClass::_2x32;

		if (getRedBits() == 16 && getGreenBits() == 16)
			rv = PixelStorageClass::_2x16;

		if (getRedBits() == 8 && getGreenBits() == 8)
			rv = PixelStorageClass::_2x8;
		break;

	case 1:
		if (getRedBits() == 32) rv = PixelStorageClass::_1x32;
		if (getRedBits() == 16) rv = PixelStorageClass::_1x16;
		if (getRedBits() == 8) rv = PixelStorageClass::_1x8;
		break;
	}

	if (ogl_format == GL_R11F_G11F_B10F) return PixelStorageClass::a;
	if (ogl_format == GL_RGB10_A2UI || ogl_format == GL_RGB10_A2) return PixelStorageClass::b;

	return rv;
}

GLenum PixelFormatTraits::getOpenGLFormatEnumerationValue() const { return ogl_format; }

bool PixelFormatTraits::operator ==(const PixelFormatTraits& other) const
{
	return num_of_components == other.num_of_components && is_color == other.is_color && is_depth == other.is_depth && is_stencil == other.is_stencil &&
		is_recognized == other.is_recognized && is_compressed == other.is_compressed && is_signed_normalized == other.is_signed_normalized &&
		is_float_format == other.is_float_format && is_integer_format == other.is_integer_format && is_signed == other.is_signed && pixel_layout == other.pixel_layout &&
		optimal_storage_type == other.optimal_storage_type && optimal_storage_size == other.optimal_storage_size && r_size == other.r_size && g_size == other.g_size && b_size == other.b_size &&
		a_size == other.a_size && depth_size == other.depth_size && stencil_size == other.stencil_size;
}

bool PixelFormatTraits::operator!=(const PixelFormatTraits& other) const { return !(*this == other); }





void PixelDataTraits::ConstructTraits()
{
	num_of_components = 0;
	is_color = false;
	is_depth = false;
	is_stencil = false;
	is_recognized = false;
	is_integer_format = false;
	has_internal_representation = false;
	supports_compression = false;
	storage_size = 0;
	r_size = -1;
	g_size = -1;
	b_size = -1;
	a_size = -1;
	depth_size = -1;
	stencil_size = -1;


	switch (static_cast<PixelDataType>(pixel_data_type))
	{
	case PixelDataType::BYTE:
		is_signed = true;

	case PixelDataType::UBYTE:
		switch (static_cast<PixelLayout>(pixel_layout))
		{
		case PixelLayout::STENCIL:
			num_of_components = 1;
			is_stencil = true;
			is_recognized = true;
			has_internal_representation = true;
			storage_size = 8;
			stencil_size = 8;
			optimal_internal_format = InternalPixelFormat::SIZED_STENCIL8;
			break;

		case PixelLayout::RED:
			num_of_components = 1;
			is_color = true;
			is_recognized = true;
			has_internal_representation = true;
			supports_compression = true;
			storage_size = 8;
			r_size = 8;
			optimal_internal_format = InternalPixelFormat::SIZED_R8;
			optimal_internal_format_compressed = is_signed ? InternalPixelFormatCompressed::COMPRESSED_SIGNED_R_RGTC1 : InternalPixelFormatCompressed::COMPRESSED_R_RGTC1;
			break;

		case PixelLayout::GREEN:
			num_of_components = 1;
			is_color = true;
			is_recognized = true;
			storage_size = 8;
			g_size = 8;
			break;

		case PixelLayout::BLUE:
			num_of_components = 1;
			is_color = true;
			is_recognized = true;
			storage_size = 8;
			b_size = 8;
			break;

		case PixelLayout::RG:
			num_of_components = 2;
			is_color = true;
			is_recognized = true;
			has_internal_representation = true;
			supports_compression = true;
			storage_size = 16;
			r_size = 8;
			g_size = 8;
			optimal_internal_format = InternalPixelFormat::SIZED_RG8;
			optimal_internal_format_compressed = is_signed ? InternalPixelFormatCompressed::COMPRESSED_SIGNED_RG_RGTC2 : InternalPixelFormatCompressed::COMPRESSED_RG_RGTC2;
			break;

		case PixelLayout::RGB:
		case PixelLayout::BGR:
			num_of_components = 3;
			is_color = true;
			is_recognized = true;
			has_internal_representation = true;
			supports_compression = true;
			storage_size = 24;
			r_size = 8;
			g_size = 8;
			b_size = 8;
			optimal_internal_format = InternalPixelFormat::SIZED_RGB8;
			optimal_internal_format_compressed = InternalPixelFormatCompressed::COMPRESSED_RGB8_ETC2;
			break;

		case PixelLayout::RGBA:
		case PixelLayout::BGRA:
			num_of_components = 4;
			is_color = true;
			is_recognized = true;
			has_internal_representation = true;
			supports_compression = true;
			storage_size = 32;
			r_size = 8;
			g_size = 8;
			b_size = 8;
			a_size = 8;
			optimal_internal_format = InternalPixelFormat::SIZED_RGBA8;
			optimal_internal_format_compressed = InternalPixelFormatCompressed::COMPRESSED_RGBA8_ETC2_EAC;
			break;

		case PixelLayout::INTEGER_RED:
			num_of_components = 1;
			is_color = true;
			is_recognized = true;
			is_integer_format = true;
			storage_size = 8;
			r_size = 8;
			break;

		case PixelLayout::INTEGER_GREEN:
			num_of_components = 1;
			is_color = true;
			is_recognized = true;
			is_integer_format = true;
			storage_size = 8;
			g_size = 8;
			break;

		case PixelLayout::INTEGER_BLUE:
			num_of_components = 1;
			is_color = true;
			is_recognized = true;
			is_integer_format = true;
			storage_size = 8;
			b_size = 8;
			break;

		case PixelLayout::INTEGER_RG:
			num_of_components = 2;
			is_color = true;
			is_recognized = true;
			is_integer_format = true;
			storage_size = 16;
			r_size = 8;
			g_size = 8;
			break;

		case PixelLayout::INTEGER_RGB:
		case PixelLayout::INTEGER_BGR:
			num_of_components = 3;
			is_color = true;
			is_recognized = true;
			is_integer_format = true;
			storage_size = 24;
			r_size = 8;
			g_size = 8;
			b_size = 8;
			break;

		case PixelLayout::INTEGER_RGBA:
		case PixelLayout::INTEGER_BGRA:
			num_of_components = 4;
			is_color = true;
			is_recognized = true;
			is_integer_format = true;
			storage_size = 32;
			r_size = 8;
			g_size = 8;
			b_size = 8;
			a_size = 8;
			break;
		}
		break;


	case PixelDataType::UBYTE_3_3_2:
	case PixelDataType::UBYTE_2_3_3_R:
		if (static_cast<PixelLayout>(pixel_layout) == PixelLayout::RGB)
		{
			num_of_components = 3;
			is_color = true;
			is_recognized = true;
			has_internal_representation = true;
			storage_size = 8;
			r_size = 3;
			g_size = 3;
			b_size = 2;
			optimal_internal_format = InternalPixelFormat::SIZED_R3_G3_B2;
		}
		break;



	case PixelDataType::SHORT:
		is_signed = true;

	case PixelDataType::USHORT:
		switch (static_cast<PixelLayout>(pixel_layout))
		{
		case PixelLayout::STENCIL:
			num_of_components = 1;
			is_stencil = true;
			is_recognized = true;
			has_internal_representation = true;
			storage_size = 16;
			stencil_size = 16;
			optimal_internal_format = InternalPixelFormat::SIZED_STENCIL16;
			break;

		case PixelLayout::DEPTH:
			num_of_components = 1;
			is_depth = true;
			is_recognized = true;
			has_internal_representation = true;
			storage_size = 16;
			depth_size = 16;
			optimal_internal_format = InternalPixelFormat::SIZED_DEPTH16;
			break;

		case PixelLayout::RED:
			num_of_components = 1;
			is_color = true;
			is_recognized = true;
			has_internal_representation = true;
			supports_compression = true;
			storage_size = 16;
			r_size = 16;
			optimal_internal_format = InternalPixelFormat::SIZED_R16;
			optimal_internal_format_compressed = is_signed ? InternalPixelFormatCompressed::COMPRESSED_SIGNED_R11_EAC : InternalPixelFormatCompressed::COMPRESSED_R11_EAC;
			break;

		case PixelLayout::GREEN:
			num_of_components = 1;
			is_color = true;
			is_recognized = true;
			storage_size = 16;
			g_size = 16;
			break;

		case PixelLayout::BLUE:
			num_of_components = 1;
			is_color = true;
			is_recognized = true;
			storage_size = 16;
			b_size = 16;
			break;

		case PixelLayout::RG:
			num_of_components = 2;
			is_color = true;
			is_recognized = true;
			has_internal_representation = true;
			supports_compression = true;
			storage_size = 32;
			r_size = 16;
			g_size = 16;
			optimal_internal_format = InternalPixelFormat::SIZED_RG16;
			optimal_internal_format_compressed = is_signed ? InternalPixelFormatCompressed::COMPRESSED_SIGNED_RG11_EAC : InternalPixelFormatCompressed::COMPRESSED_RG11_EAC;
			break;

		case PixelLayout::RGB:
		case PixelLayout::BGR:
			num_of_components = 3;
			is_color = true;
			is_recognized = true;
			has_internal_representation = true;
			supports_compression = true;
			storage_size = 48;
			r_size = 16;
			g_size = 16;
			b_size = 16;
			optimal_internal_format = InternalPixelFormat::SIZED_RGB16;
			optimal_internal_format_compressed = InternalPixelFormatCompressed::COMPRESSED_RGB8_ETC2;
			break;

		case PixelLayout::RGBA:
		case PixelLayout::BGRA:
			num_of_components = 4;
			is_color = true;
			is_recognized = true;
			has_internal_representation = true;
			supports_compression = true;
			storage_size = 64;
			r_size = 16;
			g_size = 16;
			b_size = 16;
			a_size = 16;
			optimal_internal_format = InternalPixelFormat::SIZED_RGBA16;
			optimal_internal_format_compressed = InternalPixelFormatCompressed::COMPRESSED_RGBA8_ETC2_EAC;
			break;

		case PixelLayout::INTEGER_RED:
			num_of_components = 1;
			is_color = true;
			is_recognized = true;
			is_integer_format = true;
			storage_size = 16;
			r_size = 16;
			break;

		case PixelLayout::INTEGER_GREEN:
			num_of_components = 1;
			is_color = true;
			is_recognized = true;
			is_integer_format = true;
			storage_size = 16;
			g_size = 16;
			break;

		case PixelLayout::INTEGER_BLUE:
			num_of_components = 1;
			is_color = true;
			is_recognized = true;
			is_integer_format = true;
			storage_size = 16;
			b_size = 16;
			break;

		case PixelLayout::INTEGER_RG:
			num_of_components = 2;
			is_color = true;
			is_recognized = true;
			is_integer_format = true;
			storage_size = 32;
			r_size = 16;
			g_size = 16;
			break;

		case PixelLayout::INTEGER_RGB:
		case PixelLayout::INTEGER_BGR:
			num_of_components = 3;
			is_color = true;
			is_recognized = true;
			is_integer_format = true;
			storage_size = 48;
			r_size = 16;
			g_size = 16;
			b_size = 16;
			break;

		case PixelLayout::INTEGER_RGBA:
		case PixelLayout::INTEGER_BGRA:
			num_of_components = 4;
			is_color = true;
			is_recognized = true;
			is_integer_format = true;
			storage_size = 64;
			r_size = 16;
			g_size = 16;
			b_size = 16;
			a_size = 16;
			break;
		}
		break;


	case PixelDataType::USHORT_5_6_5:
	case PixelDataType::USHORT_5_6_5_R:
		if (static_cast<PixelLayout>(pixel_layout) == PixelLayout::RGB)
		{
			num_of_components = 3;
			is_color = true;
			is_recognized = true;
			has_internal_representation = true;
			storage_size = 16;
			r_size = 5;
			g_size = 6;
			b_size = 5;
			optimal_internal_format = InternalPixelFormat::SIZED_RGB565;
		}
		break;


	case PixelDataType::USHORT_4_4_4_4:
	case PixelDataType::USHORT_4_4_4_4_R:
		if (static_cast<PixelLayout>(pixel_layout) == PixelLayout::RGBA)
		{
			num_of_components = 4;
			is_color = true;
			is_recognized = true;
			has_internal_representation = true;
			storage_size = 16;
			r_size = 4;
			g_size = 4;
			b_size = 4;
			a_size = 4;
			optimal_internal_format = InternalPixelFormat::SIZED_RGBA4;
		}
		break;


	case PixelDataType::USHORT_5_5_5_1:
	case PixelDataType::USHORT_1_5_5_5_R:
		if (static_cast<PixelLayout>(pixel_layout) == PixelLayout::RGBA)
		{
			num_of_components = 4;
			is_color = true;
			is_recognized = true;
			has_internal_representation = true;
			storage_size = 16;
			r_size = 5;
			g_size = 5;
			b_size = 5;
			a_size = 1;
			optimal_internal_format = InternalPixelFormat::SIZED_RGB5_A1;
		}
		break;


	case PixelDataType::INT:
		is_signed = true;

	case PixelDataType::UINT:
		switch (static_cast<PixelLayout>(pixel_layout))
		{
		case PixelLayout::STENCIL:
			num_of_components = 1;
			is_stencil = true;
			is_recognized = true;
			has_internal_representation = true;
			storage_size = 32;
			stencil_size = 16;
			optimal_internal_format = InternalPixelFormat::SIZED_STENCIL16;
			break;

		case PixelLayout::DEPTH:
			num_of_components = 1;
			is_depth = true;
			is_recognized = true;
			has_internal_representation = true;
			storage_size = 32;
			depth_size = 32;
			optimal_internal_format = InternalPixelFormat::SIZED_DEPTH32;
			break;

		case PixelLayout::RED:
			num_of_components = 1;
			is_color = true;
			is_recognized = true;
			has_internal_representation = true;
			supports_compression = true;
			storage_size = 32;
			r_size = 32;
			optimal_internal_format = is_signed ? InternalPixelFormat::SIZED_INT_R32 : InternalPixelFormat::SIZED_UINT_R32;
			optimal_internal_format_compressed = is_signed ? InternalPixelFormatCompressed::COMPRESSED_SIGNED_R11_EAC : InternalPixelFormatCompressed::COMPRESSED_R11_EAC;
			break;

		case PixelLayout::GREEN:
			num_of_components = 1;
			is_color = true;
			is_recognized = true;
			storage_size = 32;
			g_size = 32;
			break;

		case PixelLayout::BLUE:
			num_of_components = 1;
			is_color = true;
			is_recognized = true;
			storage_size = 32;
			b_size = 32;
			break;

		case PixelLayout::RG:
			num_of_components = 2;
			is_color = true;
			is_recognized = true;
			has_internal_representation = true;
			supports_compression = true;
			storage_size = 64;
			r_size = 32;
			g_size = 32;
			optimal_internal_format = is_signed ? InternalPixelFormat::SIZED_INT_RG32 : InternalPixelFormat::SIZED_UINT_RG32;
			optimal_internal_format_compressed = is_signed ? InternalPixelFormatCompressed::COMPRESSED_SIGNED_RG11_EAC : InternalPixelFormatCompressed::COMPRESSED_RG11_EAC;
			break;

		case PixelLayout::RGB:
		case PixelLayout::BGR:
			num_of_components = 3;
			is_color = true;
			is_recognized = true;
			has_internal_representation = true;
			supports_compression = true;
			storage_size = 96;
			r_size = 32;
			g_size = 32;
			b_size = 32;
			optimal_internal_format = is_signed ? InternalPixelFormat::SIZED_INT_RGB32 : InternalPixelFormat::SIZED_UINT_RGB32;
			optimal_internal_format_compressed = InternalPixelFormatCompressed::COMPRESSED_RGB8_ETC2;
			break;

		case PixelLayout::RGBA:
		case PixelLayout::BGRA:
			num_of_components = 4;
			is_color = true;
			is_recognized = true;
			has_internal_representation = true;
			supports_compression = true;
			storage_size = 128;
			r_size = 32;
			g_size = 32;
			b_size = 32;
			a_size = 32;
			optimal_internal_format = is_signed ? InternalPixelFormat::SIZED_INT_RGBA32 : InternalPixelFormat::SIZED_UINT_RGBA32;
			optimal_internal_format_compressed = InternalPixelFormatCompressed::COMPRESSED_RGBA8_ETC2_EAC;
			break;

		case PixelLayout::INTEGER_RED:
			num_of_components = 1;
			is_color = true;
			is_recognized = true;
			is_integer_format = true;
			storage_size = 32;
			r_size = 32;
			break;

		case PixelLayout::INTEGER_GREEN:
			num_of_components = 1;
			is_color = true;
			is_recognized = true;
			is_integer_format = true;
			storage_size = 32;
			g_size = 32;
			break;

		case PixelLayout::INTEGER_BLUE:
			num_of_components = 1;
			is_color = true;
			is_recognized = true;
			is_integer_format = true;
			storage_size = 32;
			b_size = 32;
			break;

		case PixelLayout::INTEGER_RG:
			num_of_components = 2;
			is_color = true;
			is_recognized = true;
			is_integer_format = true;
			storage_size = 64;
			r_size = 32;
			g_size = 32;
			break;

		case PixelLayout::INTEGER_RGB:
		case PixelLayout::INTEGER_BGR:
			num_of_components = 3;
			is_color = true;
			is_recognized = true;
			is_integer_format = true;
			storage_size = 96;
			r_size = 32;
			g_size = 32;
			b_size = 32;
			break;

		case PixelLayout::INTEGER_BGRA:
		case PixelLayout::INTEGER_RGBA:
			num_of_components = 4;
			is_color = true;
			is_recognized = true;
			is_integer_format = true;
			storage_size = 128;
			r_size = 32;
			g_size = 32;
			b_size = 32;
			a_size = 32;
			break;
		}
		break;


	case PixelDataType::UINT_10F_11F_11F_R:
		if (static_cast<PixelLayout>(pixel_layout) == PixelLayout::RGB)
		{
			num_of_components = 3;
			is_color = true;
			is_recognized = true;
			has_internal_representation = true;
			storage_size = 32;
			r_size = 11;
			g_size = 11;
			b_size = 10;
			optimal_internal_format = InternalPixelFormat::SIZED_FLOAT_R11_G11_B10;
		}
		break;

	case PixelDataType::UINT_10_10_10_2:
	case PixelDataType::UINT_2_10_10_10_R:
		if (static_cast<PixelLayout>(pixel_layout) == PixelLayout::RGBA)
		{
			num_of_components = 4;
			is_color = true;
			is_recognized = true;
			has_internal_representation = true;
			storage_size = 32;
			r_size = 10;
			g_size = 10;
			b_size = 10;
			a_size = 2;
			optimal_internal_format = InternalPixelFormat::SIZED_RGB10_A2;
		}
		break;


	case PixelDataType::UINT_5_9_9_9_R:
		if (static_cast<PixelLayout>(pixel_layout) == PixelLayout::RGB)
		{
			num_of_components = 3;
			is_color = true;
			is_recognized = true;
			has_internal_representation = true;
			storage_size = 32;
			optimal_internal_format = InternalPixelFormat::SIZED_FLOAT_RGB9_E5;
		}
		break;


	case PixelDataType::UINT_24_8:
		if (static_cast<PixelLayout>(pixel_layout) == PixelLayout::DEPTH_STENCIL)
		{
			num_of_components = 2;
			is_depth = true;
			is_stencil = true;
			is_recognized = true;
			has_internal_representation = true;
			storage_size = 32;
			depth_size = 24;
			stencil_size = 8;
			optimal_internal_format = InternalPixelFormat::SIZED_DEPTH24_STENCIL8;
		}
		break;



	case PixelDataType::FLOAT:
		is_signed = true;
		switch (static_cast<PixelLayout>(pixel_layout))
		{
		case PixelLayout::DEPTH:
			num_of_components = 1;
			is_depth = true;
			is_recognized = true;
			has_internal_representation = true;
			storage_size = 32;
			depth_size = 32;
			optimal_internal_format = InternalPixelFormat::SIZED_FLOAT_DEPTH32;
			break;

		case PixelLayout::RED:
			num_of_components = 1;
			is_color = true;
			is_recognized = true;
			has_internal_representation = true;
			supports_compression = true;
			storage_size = 32;
			r_size = 32;
			optimal_internal_format = InternalPixelFormat::SIZED_FLOAT_R32;
			optimal_internal_format_compressed = InternalPixelFormatCompressed::COMPRESSED_SIGNED_R11_EAC;
			break;

		case PixelLayout::GREEN:
			num_of_components = 1;
			is_color = true;
			is_recognized = true;
			storage_size = 32;
			g_size = 32;
			break;

		case PixelLayout::BLUE:
			num_of_components = 1;
			is_color = true;
			is_recognized = true;
			storage_size = 32;
			b_size = 32;
			break;

		case PixelLayout::RG:
			num_of_components = 2;
			is_color = true;
			is_recognized = true;
			has_internal_representation = true;
			supports_compression = true;
			storage_size = 64;
			r_size = 32;
			g_size = 32;
			optimal_internal_format = InternalPixelFormat::SIZED_FLOAT_RG32;
			optimal_internal_format_compressed = InternalPixelFormatCompressed::COMPRESSED_SIGNED_RG11_EAC;
			break;

		case PixelLayout::RGB:
		case PixelLayout::BGR:
			num_of_components = 3;
			is_color = true;
			is_recognized = true;
			has_internal_representation = true;
			supports_compression = true;
			storage_size = 96;
			r_size = 32;
			g_size = 32;
			b_size = 32;
			optimal_internal_format = InternalPixelFormat::SIZED_FLOAT_RGB32;
			optimal_internal_format_compressed = InternalPixelFormatCompressed::COMPRESSED_RGB_BPTC_SIGNED_FLOAT;
			break;

		case PixelLayout::RGBA:
		case PixelLayout::BGRA:
			num_of_components = 4;
			is_color = true;
			is_recognized = true;
			has_internal_representation = true;
			supports_compression = true;
			storage_size = 128;
			r_size = 32;
			g_size = 32;
			b_size = 32;
			a_size = 32;
			optimal_internal_format = InternalPixelFormat::SIZED_FLOAT_RGBA32;
			optimal_internal_format_compressed = InternalPixelFormatCompressed::COMPRESSED_RGBA8_ETC2_EAC;
			break;
		}
		break;


	case PixelDataType::FLOAT_32_UINT_24_8_R:
		if (static_cast<PixelLayout>(pixel_layout) == PixelLayout::DEPTH_STENCIL)
		{
			num_of_components = 2;
			is_depth = true;
			is_stencil = true;
			is_recognized = true;
			has_internal_representation = true;
			storage_size = 32;
			depth_size = 24;
			stencil_size = 8;
		}
		break;
	}
}

PixelDataTraits::PixelDataTraits(PixelLayout pixel_layout, PixelDataType pixel_data_type) : pixel_layout{ static_cast<GLenum>(pixel_layout) }, pixel_data_type{ static_cast<GLenum>(pixel_data_type) }
{
	ConstructTraits();
}

PixelDataTraits::PixelDataTraits(GLenum pixel_layout, GLenum pixel_data_type) : pixel_layout{ pixel_layout }, pixel_data_type{ pixel_data_type }
{
	ConstructTraits();
}

unsigned short PixelDataTraits::getNumberOfComponents() const { return num_of_components; }

bool PixelDataTraits::isColor() const { return is_color; }

bool PixelDataTraits::isDepth() const { return is_depth; }

bool PixelDataTraits::isStencil() const { return is_stencil; }

bool PixelDataTraits::isRecognized() const { return is_recognized; }

bool PixelDataTraits::isInteger() const { return is_integer_format; }

bool PixelDataTraits::isSigned() const { return is_signed; }

bool PixelDataTraits::hasInternalRepresentation() const { return has_internal_representation; }

bool PixelDataTraits::supportsCompression() const { return supports_compression; }

short PixelDataTraits::getRedBits() const { return r_size; }

short PixelDataTraits::getGreenBits() const { return g_size; }

short PixelDataTraits::getBlueBits() const { return b_size; }

short PixelDataTraits::getAlphaBits() const { return a_size; }

short PixelDataTraits::getDepthBits() const { return depth_size; }

short PixelDataTraits::getStencilBits() const { return stencil_size; }

unsigned short PixelDataTraits::getPixelStorageSize() const { return storage_size; }

std::pair<GLenum, GLenum> PixelDataTraits::getOpenGLEnumerationLayoutAndComponentType() const{ return std::make_pair(pixel_layout, pixel_data_type); }

void PixelDataTraits::getOptimalUncompressedInternalStorageFormat(InternalPixelFormat* p_optimal_internal_format) const
{
	*p_optimal_internal_format = optimal_internal_format;
}

void PixelDataTraits::getOptimalCompressedInternalStorageFormat(InternalPixelFormatCompressed* p_optimal_internal_format_compressed) const
{
	*p_optimal_internal_format_compressed = optimal_internal_format_compressed;
}

bool PixelDataTraits::operator==(const PixelDataTraits& other) const
{
	return num_of_components == other.num_of_components && is_color == other.is_color && is_depth == other.is_depth && is_stencil == other.is_stencil &&
		is_recognized == other.is_recognized && is_integer_format == other.is_integer_format && is_signed == other.is_signed && has_internal_representation == other.has_internal_representation &&
		supports_compression == other.supports_compression && optimal_internal_format == other.optimal_internal_format && optimal_internal_format_compressed == other.optimal_internal_format_compressed &&
		storage_size == other.storage_size && r_size == other.r_size && g_size == other.g_size && b_size == other.b_size && a_size == other.a_size && depth_size == other.depth_size && stencil_size == other.stencil_size;
}

bool PixelDataTraits::operator!=(const PixelDataTraits& other) const
{
	return !(*this == other);
}