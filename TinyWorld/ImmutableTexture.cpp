#include "ImmutableTexture.h"

#include <stdexcept>

using namespace tiny_world;




//***********************************Core part of  implementation of the immutable texture base infrastructure***********************************

void ImmutableTexture_Core::init_immutable_texture()
{
	GLint ppading, uppadding;
	glGetIntegerv(GL_PACK_ALIGNMENT, &ppading);
	glGetIntegerv(GL_UNPACK_ALIGNMENT, &uppadding);

	pack_padding = static_cast<TextureStorageAlignment>(ppading);
	unpack_padding = static_cast<TextureStorageAlignment>(uppadding);
}


void ImmutableTexture_Core::allocateStorage(uint32_t num_mipmap_levels, uint32_t num_array_layers, TextureSize texture_size, GLenum internal_storage_format)
{
	texture_dimension = query_dimension();
	this->num_mipmap_levels = num_mipmap_levels;
	this->num_array_layers = num_array_layers > 1 ? num_array_layers : 1;


	bool is_array = num_array_layers > 1;
	StorageSize storage_size;	//receives the storage size of the texture
	switch (texture_dimension)
	{
	case TextureDimension::_1D:
		texture_size.height = 0;
		texture_size.depth = 0;

		storage_size.width = texture_size.width;
		storage_size.height = is_array ? num_array_layers : 1;
		storage_size.depth = 1;
		storage_dimension = is_array ? TextureDimension::_2D : TextureDimension::_1D;
		break;

	case TextureDimension::_2D:
		texture_size.depth = 0;

		storage_size.width = texture_size.width;
		storage_size.height = texture_size.height;
		storage_size.depth = is_array ? num_array_layers : 1;
		storage_dimension = is_array ? TextureDimension::_3D : TextureDimension::_2D;
		break;

	case TextureDimension::_3D:
		storage_size = texture_size;
		storage_dimension = TextureDimension::_3D;
		if (is_array)
		{
			set_error_state(true);
			const char* err_msg = "Volumetric texture arrays are not supported";
			set_error_string(err_msg);
			call_error_callback(err_msg);
			return;
		}
		break;
	}

	//Perform primary initialization of the texture
	Texture::initialize(texture_size, storage_size, internal_storage_format);

	this->texture_binding = perform_allocation();	//Perform actual storage allocation

	//Number of faces equals to 6 only for cubemaps
	num_faces = texture_binding.gl_texture_target == GL_TEXTURE_CUBE_MAP || texture_binding.gl_texture_target == GL_TEXTURE_CUBE_MAP_ARRAY ? 6 : 1;
	storage_size.depth *= num_faces;

	//Perform final initialization of the texture
	Texture::initialize(texture_size, storage_size, internal_storage_format);
}


ImmutableTexture_Core::ImmutableTexture_Core(const std::string& texture_class_string_name) : Texture(texture_class_string_name)
{ 
	init_immutable_texture(); 
}


ImmutableTexture_Core::ImmutableTexture_Core(const std::string& texture_class_string_name, const std::string& texture_string_name) : 
Texture(texture_class_string_name, texture_string_name)
{
	init_immutable_texture();
}


void ImmutableTexture_Core::allocateStorage(uint32_t num_mipmap_levels, uint32_t num_array_layers,
	TextureSize texture_size, InternalPixelFormat internal_format)
{
	if (isInitialized() || getErrorState()) return;
	allocateStorage(num_mipmap_levels, num_array_layers, texture_size, static_cast<GLenum>(internal_format));
}


void ImmutableTexture_Core::allocateStorage(uint32_t num_mipmap_levels, uint32_t num_array_layers,
	TextureSize texture_size, InternalPixelFormatCompressed internal_format_compressed)
{
	if (isInitialized() || getErrorState()) return;
	allocateStorage(num_mipmap_levels, num_array_layers, texture_size, static_cast<GLenum>(internal_format_compressed));
}


void ImmutableTexture_Core::generateMipmapLevels()
{
	if (!isInitialized())
	{
		set_error_state(true);
		const char* err_msg = "Unable to generate mipmap levels: texture is not initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	if (!getErrorState())
	{
		GLuint ogl_old_texture_id = bind();		//Bind contained texture
		glGenerateMipmap(texture_binding.gl_texture_target);	//Generate mip-maps
		glBindTexture(texture_binding.gl_texture_target, ogl_old_texture_id);	//Restore old binding
	}
}


TextureStorageAlignment ImmutableTexture_Core::setPackPadding(TextureStorageAlignment new_pack_padding)
{
	TextureStorageAlignment rv = pack_padding;
	pack_padding = new_pack_padding;
	return rv;
}


TextureStorageAlignment ImmutableTexture_Core::setUnpackPadding(TextureStorageAlignment new_unpack_padding)
{
	TextureStorageAlignment rv = unpack_padding;
	unpack_padding = new_unpack_padding;
	return rv;
}


uint32_t ImmutableTexture_Core::getNumberOfArrayLayers() const { return num_array_layers; }


uint32_t ImmutableTexture_Core::getNumberOfMipmapLevels() const { return num_mipmap_levels; }


uint32_t ImmutableTexture_Core::getNumberOfFaces() const { return num_faces; }


uint32_t ImmutableTexture_Core::getNumberOfSamples() const { return 1; }


TextureDimension ImmutableTexture_Core::getDimension() const { return texture_dimension; }


bool ImmutableTexture_Core::isArrayTexture() const { return num_array_layers > 1; }


bool ImmutableTexture_Core::isCompressed() const
{
	if (!isInitialized())
	{
		set_error_state(true);
		const char* err_msg = "Unable to query compression status of the texture: texture was not initialized or initialization is not yet complete";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return false;
	}
	else
	{
		GLint compression_status;
		glGetInternalformativ(texture_binding.gl_texture_target, getOpenGLStorageInternalFormat(), GL_TEXTURE_COMPRESSED, 
			sizeof(GLint), &compression_status);

		return compression_status == GL_TRUE;
	}

}


bool ImmutableTexture_Core::isBufferTexture() const { return false; }


TextureStorageAlignment ImmutableTexture_Core::getPackPadding() const { return pack_padding; }


TextureStorageAlignment ImmutableTexture_Core::getUnpackPadding() const { return unpack_padding; }


Texture::TextureBinding ImmutableTexture_Core::getOpenGLTextureBinding() const
{
	if (!isInitialized())
	{
		set_error_state(true);
		const char* err_msg = "Unable to query texture target: texture was not initialized or initialization is not yet complete";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return TextureBinding{ 0xFFFFFFFF, 0xFFFFFFFF };
	}

	return texture_binding; 
}


GLuint ImmutableTexture_Core::bind() const
{
	GLint rv;
	glGetIntegerv(getOpenGLTextureBinding().gl_texture_binding, &rv);

	if (!isInitialized())
	{
		set_error_state(true);
		const char* err_msg = "Unable to bind texture: texture was not properly initialized or initialization is not yet complete";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return rv;
	}

	glBindTexture(getOpenGLTextureBinding().gl_texture_target, getOpenGLId());


	//Set packing and unpacking padding settings 
	glPixelStorei(GL_PACK_ALIGNMENT, static_cast<GLint>(pack_padding));
	glPixelStorei(GL_UNPACK_ALIGNMENT, static_cast<GLint>(unpack_padding));

	return static_cast<GLuint>(rv);
}


bool ImmutableTexture_Core::copyTexelData(uint32_t source_mipmap_level, uint32_t source_offset_x, uint32_t source_offset_y, uint32_t source_offset_z,
	const ImmutableTexture& destination_texture, uint32_t destination_mipmap_level, uint32_t destination_offset_x, uint32_t destination_offset_y, uint32_t destination_offset_z,
	uint32_t copy_buffer_width, uint32_t copy_buffer_height, uint32_t copy_buffer_depth) const
{
	if (getOpenGLStorageInternalFormat() != destination_texture.getOpenGLStorageInternalFormat()) return false;

	glCopyImageSubData(getOpenGLId(), texture_binding.gl_texture_target, source_mipmap_level, source_offset_x, source_offset_y, source_offset_z,
		destination_texture.getOpenGLId(), destination_texture.texture_binding.gl_texture_target, destination_mipmap_level, destination_offset_x, destination_offset_y, destination_offset_z,
		copy_buffer_width, copy_buffer_height, copy_buffer_depth);
	return true;
}



//***********************************Immutable texture interface implementation***********************************
ImmutableTexture::ImmutableTexture(const std::string& texture_class_string_name) : ImmutableTexture_Core(texture_class_string_name)
{

}


ImmutableTexture::ImmutableTexture(const std::string& texture_class_string_name, const std::string& texture_string_name) :
ImmutableTexture_Core(texture_class_string_name, texture_string_name)
{

}


GLuint ImmutableTexture::bind() const{ return ImmutableTexture_Core::bind(); }


void ImmutableTexture::attachToFBO(FramebufferAttachmentPoint attachment, uint32_t mipmap_level) const
{
	if (!isInitialized())
	{
		set_error_state(true);
		const char* err_msg = "Texture must be initialized before it gets attached to a framebuffer object";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	glFramebufferTexture(GL_DRAW_FRAMEBUFFER, static_cast<GLenum>(attachment), getOpenGLId(), mipmap_level);
}


void ImmutableTexture::attachToFBO(FramebufferAttachmentPoint attachment, uint32_t attachment_layer, uint32_t mipmap_level) const
{
	if (!isInitialized())
	{
		set_error_state(true);
		const char* err_msg = "Texture must be initialized before it gets attached to a framebuffer object";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	uint32_t num_attachment_layers =
		(getBindingSlot() == TextureSlot::TEXTURE_CUBE_MAP || getBindingSlot() == TextureSlot::TEXTURE_CUBE_MAP_ARRAY) ?
		getNumberOfArrayLayers() * 6 : getNumberOfArrayLayers();
	
	if (attachment_layer >= num_attachment_layers)
	{
		set_error_state(true);
		const char* err_msg = "Index of array layer exceeds the total number of layers contained in the texture"; 
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	glFramebufferTextureLayer(GL_DRAW_FRAMEBUFFER, static_cast<GLenum>(attachment), getOpenGLId(), mipmap_level, attachment_layer);
}


void ImmutableTexture::attachToImageUnit(uint32_t image_unit, uint32_t mipmap_level, uint32_t layer, BufferAccess access, InternalPixelFormat format) const
{
	//Check if texture is compressed
	if (isCompressed())
	{
		set_error_state(true);
		const char* err_msg = "Compressed textures can not be attached to images";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}


	//Check if specified access format is in the same class as the internal pixel storage format of the texture
	PixelFormatTraits internal_format_traits { getOpenGLStorageInternalFormat() };
	PixelFormatTraits access_format_traits{ format };

	if (!internal_format_traits.isColor())
	{
		set_error_state(true);
		const char* err_msg = "An attempt to attach non-color texture to an image unit has been made: only textures that use a non-compressed color internal storage format "
			"can be attached to image units";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	if (internal_format_traits.getStorageClass() == PixelStorageClass::unknown)
	{
		set_error_state(true);
		const char* err_msg = "Internal pixel storage format used by the texture being attached to requested image unit does not belong to any of the known storage classes";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}


	if (!access_format_traits.isColor())
	{
		set_error_state(true);
		const char* err_msg = "An attempt to use non-color access pixel format while attaching texture to requested image unit: only color pixel storage formats can be used to "
			"access texture data via image units";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	if (access_format_traits.getStorageClass() == PixelStorageClass::unknown)
	{
		set_error_state(true);
		const char* err_msg = "Unable to determine pixel storage class of the specified image access format";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}


	if (internal_format_traits.getStorageClass() == access_format_traits.getStorageClass())
	{
		set_error_state(true);
		const char* err_msg = "Internal storage format used by the texture being attached to requested image unit and specified access format of the "
			"image unit must belong to the same pixel storage class";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}


	if (getNumberOfArrayLayers() >= layer)
	{
		set_error_state(true);
		std::string err_msg = "Unable to attach texture layer " + std::to_string(layer) + "to requested image unit. Out of range error: "
			"valid layer offsets are 0-" + std::to_string(getNumberOfArrayLayers() - 1);
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	if (mipmap_level >= getNumberOfMipmapLevels())
	{
		set_error_state(true);
		std::string err_msg = "Unable to attach specified mipmap-level " + std::to_string(mipmap_level) + "to requested image unit. Out of range error: "
			"valid mipmap-level indexes are 0-" + std::to_string(getNumberOfMipmapLevels());
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}


	glBindImageTexture(image_unit, getOpenGLId(), mipmap_level, GL_FALSE, layer, static_cast<GLenum>(access), static_cast<GLenum>(format));
}


void ImmutableTexture::attachToImageUnit(uint32_t image_unit, uint32_t mipmap_level, BufferAccess access, InternalPixelFormat format) const
{
	//Check if texture is compressed
	if (isCompressed())
	{
		set_error_state(true);
		const char* err_msg = "Compressed textures can not be attached to images";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}


	//Check if specified access format is in the same class as the internal pixel storage format of the texture
	PixelFormatTraits internal_format_traits{ getOpenGLStorageInternalFormat() };
	PixelFormatTraits access_format_traits{ format };

	if (!internal_format_traits.isColor())
	{
		set_error_state(true);
		const char* err_msg = "An attempt to attach non-color texture to an image unit has been made: only textures that use a non-compressed color internal storage format "
			"can be attached to image units";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	if (internal_format_traits.getStorageClass() == PixelStorageClass::unknown)
	{
		set_error_state(true);
		const char* err_msg = "Internal pixel storage format used by the texture being attached to requested image unit does not belong to any of the known storage classes";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}


	if (!access_format_traits.isColor())
	{
		set_error_state(true);
		const char* err_msg = "An attempt to use non-color access pixel format while attaching texture to requested image unit: only color pixel storage formats can be used to "
			"access texture data via image units";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	if (access_format_traits.getStorageClass() == PixelStorageClass::unknown)
	{
		set_error_state(true);
		const char* err_msg = "Unable to determine pixel storage class of the specified image access format";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}


	if (internal_format_traits.getStorageClass() != access_format_traits.getStorageClass())
	{
		set_error_state(true);
		const char* err_msg = "Internal storage format used by the texture being attached to requested image unit and specified access format of the "
			"image unit must belong to the same pixel storage class";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	if (mipmap_level >= getNumberOfMipmapLevels())
	{
		set_error_state(true);
		std::string err_msg = "Unable to attach specified mipmap-level " + std::to_string(mipmap_level) + "to requested image unit. Out of range error: "
			"valid mipmap-level indexes are 0-" + std::to_string(getNumberOfMipmapLevels());
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}


	glBindImageTexture(image_unit, getOpenGLId(), mipmap_level, GL_TRUE, 0, static_cast<GLenum>(access), static_cast<GLenum>(format));
}