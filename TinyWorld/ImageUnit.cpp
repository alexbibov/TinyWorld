#include "ImageUnit.h"

#include <algorithm>

using namespace tiny_world;



std::vector<std::pair<bool, ImageUnit::ImageAttachmentInfoOwned>> ImageUnit::image_units{};

int ImageUnit::retrieve_unused_image_unit() const
{
	//Try to find an unused texture image unit in the cache
	image_unit_layout::const_iterator unit_offset;
	if ((unit_offset = std::find_if(image_units.begin(), image_units.end(),
		[this](const image_unit_layout::value_type element) -> bool{return !element.first; })) == image_units.end())
	{
		//If such unit can not be found, it is necessary to add a new one to the cache while
		//making sure that the maximal amount of texture image units supported by the hardware is not exceeded.
		//NOTE: the comparison is done against half the amount of actually supported texture image units. The reason for this is
		//that on some hardware when the same unit is accessed once via vertex and once via fragment shader within the same shader program, such unit usage counts as two.
		GLint max_image_units;
		glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &max_image_units);
		if (image_units.size() == static_cast<image_unit_layout::size_type>(max_image_units / 2.0)) return -1;

		image_units.push_back(std::make_pair(false, ImageAttachmentInfo{}));
		return static_cast<int>(image_units.size() - 1);
	}
	else
		return static_cast<int>(unit_offset - image_units.begin());
}

ImageUnit::ImageUnit() : Entity("ImageUnit"), ref_counter{ nullptr }
{
	if ((target_image_unit = retrieve_unused_image_unit()) == -1)
	{
		set_error_state(true);
		const char* err_msg = "Unable to initialize new image unit: all image units are in use";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	image_units[target_image_unit].first = true;
	ref_counter = new uint32_t{ 1 };
}

ImageUnit::ImageUnit(const ImageUnit& other) : Entity(other), 
ref_counter{ other.ref_counter }, target_image_unit{ other.target_image_unit }
{
	//if copy-source object has been successfully initialized, increment the reference counter
	if (ref_counter)(*ref_counter)++;
}

ImageUnit::ImageUnit(ImageUnit&& other) : Entity(std::move(other)), 
ref_counter{ other.ref_counter }, target_image_unit{ other.target_image_unit }
{
	other.ref_counter = nullptr;
	other.target_image_unit = -1;
}

ImageUnit::~ImageUnit()
{
	//Check if the object instance has been successfully initialized
	if (ref_counter && target_image_unit != -1)
	{
		//Decrement reference counter
		(*ref_counter)--;
		flush();

		//If the value of reference counter equals zero, release resources associated with the image unit
		if (!(*ref_counter))
		{
			delete ref_counter;
			image_units[target_image_unit].first = false;
			image_units[target_image_unit].second.AttachedTextureAliasPointer = nullptr;
		}
	}
}

ImageUnit& ImageUnit::operator=(const ImageUnit& other)
{
	//Account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	Entity::operator=(other);

	//If assignment source object has been correctly initialized, increment its reference counter
	if (other.ref_counter)(*other.ref_counter)++;

	//If assignment destination object has been correctly initialized, decrement its reference counter and 
	//if the counter reaches zero, release the associated resources
	if (ref_counter && target_image_unit != -1)
	{
		(*ref_counter)--;
		flush();
		
		if (!(*ref_counter))
		{
			delete ref_counter;
			image_units[target_image_unit].first = false;
			image_units[target_image_unit].second.AttachedTextureAliasPointer = nullptr;
		}
	}


	//Perform copy-assignment between the object states
	ref_counter = other.ref_counter;
	target_image_unit = other.target_image_unit;
	return *this;
}

ImageUnit& ImageUnit::operator=(ImageUnit&& other)
{
	//Account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	Entity::operator=(std::move(other));

	//If move-assignment destination object has been properly initialized, decrement its reference counter and
	//if counter's value becomes equal to zero, destroy all resources associated with the object.
	//NOTE: we do not need to do it here as the destructor of the assignment-source object will do it for us
	std::swap(ref_counter, other.ref_counter);
	std::swap(target_image_unit, other.target_image_unit);
	return *this;
}

bool ImageUnit::operator==(const ImageUnit& other) const
{
	return target_image_unit == other.target_image_unit;
}

uint32_t ImageUnit::getNumberOfAvailableImageUnits()
{
	//Retrieve the maximal number of supported texture image units
	GLint max_image_units;
	glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &max_image_units);
	
	//Retrieve the number of occupied units
	uint32_t busy_image_units = 0;
	std::for_each(image_units.begin(), image_units.end(), 
		[&busy_image_units](image_unit_layout::value_type elem) -> void{if (elem.first)++busy_image_units; });

	return (static_cast<uint32_t>(max_image_units / 2.0) - busy_image_units);
}

void ImageUnit::attachTexture(const ImmutableTexture& texture, int mipmap_level, int array_layer, BufferAccess access_level, InternalPixelFormat access_format) const
{
	if (target_image_unit == -1) return;
	texture.attachToImageUnit(target_image_unit, mipmap_level, array_layer, access_level, access_format);
	image_units[target_image_unit].second.mipmap_level = mipmap_level;
	image_units[target_image_unit].second.array_layer = array_layer;
	image_units[target_image_unit].second.access_level = access_level;
	image_units[target_image_unit].second.access_format = access_format;
	image_units[target_image_unit].second.AttachedTextureAliasPointer = std::move(std::unique_ptr < const Texture > {texture.clone()});
}

void ImageUnit::attachTexture(const ImmutableTexture& texture, int mipmap_level, BufferAccess access_level, InternalPixelFormat access_format) const
{
	if (target_image_unit == -1) return;
	texture.attachToImageUnit(target_image_unit, mipmap_level, access_level, access_format);
	image_units[target_image_unit].second.mipmap_level = mipmap_level;
	image_units[target_image_unit].second.array_layer = -1;
	image_units[target_image_unit].second.access_level = access_level;
	image_units[target_image_unit].second.access_format = access_format;
	image_units[target_image_unit].second.AttachedTextureAliasPointer = std::move(std::unique_ptr < const Texture > {texture.clone()});
}

void ImageUnit::attachTexture(const BufferTexture& texture, BufferAccess access_level, BufferTextureInternalPixelFormat access_format) const
{
	if (target_image_unit == -1) return;
	texture.attachToImageUnit(target_image_unit, access_level, access_format);
	image_units[target_image_unit].second.mipmap_level = -1;
	image_units[target_image_unit].second.array_layer = -1;
	image_units[target_image_unit].second.access_level = access_level;
	image_units[target_image_unit].second.access_format = static_cast<InternalPixelFormat>(static_cast<GLenum>(access_format));
	image_units[target_image_unit].second.AttachedTextureAliasPointer = std::move(std::unique_ptr < const Texture > {texture.clone()});
}

bool ImageUnit::hasAttachment() const
{
	return image_units[target_image_unit].second.AttachedTextureAliasPointer != nullptr;
}

ImageAttachmentInfo ImageUnit::retrieveAttachment() const
{
	if (target_image_unit != -1)
		return image_units[target_image_unit].second;

	return ImageAttachmentInfo{};
}

int ImageUnit::getBinding() const { return target_image_unit; }

void ImageUnit::flush() const 
{ 
	if (image_units[target_image_unit].second.access_level == BufferAccess::WRITE ||
		image_units[target_image_unit].second.access_level == BufferAccess::READ_WRITE)
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void ImageUnit::assign()
{
	if (target_image_unit == -1)
	{
		if ((target_image_unit = retrieve_unused_image_unit()) == -1)
		{
			set_error_state(true);
			const char* err_msg = "Unable to assign image unit: all image units are in use";
			set_error_string(err_msg);
			call_error_callback(err_msg);
			return;
		}

		image_units[target_image_unit].first = true;
		ref_counter = new uint32_t{ 1 };
	}
}

void ImageUnit::release()
{
	if (target_image_unit != -1 && !(--(*ref_counter)))
	{
		delete ref_counter;
		image_units[target_image_unit].first = false;
		image_units[target_image_unit].second.AttachedTextureAliasPointer = nullptr;
	}

	target_image_unit = -1;
}