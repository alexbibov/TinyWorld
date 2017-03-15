#include "TextureUnitBlock.h"

#include <algorithm>
#include <iterator>

using namespace tiny_world;

bool TextureUnitBlock::is_initialized = false;
TextureUnitBlock* TextureUnitBlock::myself_pointer = nullptr;

TextureUnitBlock::TextureUnitBlock() : Entity{ "TextureUnitBlock" }
{
	//Get maximal number of texture units
	GLint max_texture_units;
	glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &max_texture_units);
	this->num_of_units = max_texture_units;

	//Set 0 unit as active by default
	this->active_unit = 0;
	glActiveTexture(GL_TEXTURE0);

	//Initialize buffer to store information about texture unit block layout
	texture_unit_block_descriptor.reserve(num_of_units);

	for (uint32_t i = 0; i < num_of_units; ++i)
		texture_unit_block_descriptor.push_back(texture_unit_layout{ texture_slot_deck{}, nullptr });
}

TextureUnitBlock::~TextureUnitBlock()
{
	TextureUnitBlock::is_initialized = false;
	TextureUnitBlock::myself_pointer = nullptr;
}

TextureUnitBlock* TextureUnitBlock::initialize()
{
	if (!is_initialized)
	{
		TextureUnitBlock::myself_pointer = new TextureUnitBlock();
		TextureUnitBlock::is_initialized = true;
	}

	return TextureUnitBlock::myself_pointer;
}

void TextureUnitBlock::reset()
{
	if (is_initialized)
	{
		delete TextureUnitBlock::myself_pointer;
		TextureUnitBlock::myself_pointer = nullptr;
		TextureUnitBlock::is_initialized = false;
	}
}

uint32_t TextureUnitBlock::getNumberOfUnits() const { return num_of_units; }


uint32_t TextureUnitBlock::switchActiveTextureUnit(uint32_t new_texture_unit)
{
	if (new_texture_unit >= num_of_units)
	{
		set_error_state(true);
		const char* err_msg = "Unable to change the active texture unit: zero-based index of the new texture unit is "
			"greater or equal then the number of texture units supported by the host system";
		set_error_string(err_msg);
		call_error_callback(err_msg);
	}

	uint32_t rv = active_unit;
	this->active_unit = new_texture_unit;
	glActiveTexture(GL_TEXTURE0 + new_texture_unit);
	return rv;
}


void TextureUnitBlock::bindTexture(const Texture& texture_object)
{
	if (!texture_unit_block_descriptor[active_unit].first.
		insert(std::move(std::make_pair(texture_object.getBindingSlot(),
		std::unique_ptr < const Texture > {texture_object.clone()}))).second)
		texture_unit_block_descriptor[active_unit].first.at(texture_object.getBindingSlot()) =
		std::move(std::unique_ptr < const Texture > {texture_object.clone()});
	texture_object.bind();
}

void TextureUnitBlock::bindSampler(const TextureSampler& sampler_object)
{
	texture_unit_block_descriptor[active_unit].second = &sampler_object;
	sampler_object.bind(active_unit);
}

int32_t TextureUnitBlock::getBindingTextureUnit(const Texture& texture_object) const
{
	for (texture_unit_block_layout::const_iterator texture_unit_block_iter = texture_unit_block_descriptor.begin();
		texture_unit_block_iter != texture_unit_block_descriptor.end(); ++texture_unit_block_iter)
	{
		texture_slot_deck::const_iterator texture_slot_iter =
			texture_unit_block_iter->first.find(texture_object.getBindingSlot());
		if (texture_slot_iter != texture_unit_block_iter->first.end() &&
			*texture_slot_iter->second == texture_object)
			return static_cast<int32_t>(texture_unit_block_iter - texture_unit_block_descriptor.begin());
	}

	return -1;
}

int32_t TextureUnitBlock::getBindingTextureUnit(const TextureSampler& sampler_object) const
{
	for (texture_unit_block_layout::const_iterator texture_unit_block_iter = texture_unit_block_descriptor.begin();
		texture_unit_block_iter != texture_unit_block_descriptor.end(); ++texture_unit_block_iter)
	{
		if (texture_unit_block_iter->second->getId() == sampler_object.getId())
			return static_cast<int32_t>(texture_unit_block_iter - texture_unit_block_descriptor.begin());
	}

	return -1;
}

bool TextureUnitBlock::isBound(Texture& texture_object) const
{
	return getBindingTextureUnit(texture_object) != -1;
	return 0;
}

bool TextureUnitBlock::isBound(TextureSampler& sampler_object) const
{
	return getBindingTextureUnit(sampler_object) != -1;
	return 0;
}

const Texture *TextureUnitBlock::getBoundTexture(TextureSlot slot, uint32_t texture_unit) const
{
	texture_slot_deck::const_iterator texture_slot_iter = texture_unit_block_descriptor[texture_unit].first.find(slot);
	if (texture_slot_iter != texture_unit_block_descriptor[texture_unit].first.end())
		return texture_slot_iter->second.get();
	else
		return nullptr;
}

const TextureSampler *TextureUnitBlock::getBoundSampler(uint32_t texture_unit) const
{
	return texture_unit_block_descriptor[texture_unit].second;
}

int TextureUnitBlock::getAvailableTextureUnit() const
{
	for (int i = 0; i < static_cast<int>(num_of_units); ++i)
		if (texture_unit_block_descriptor[i].first.empty()) return i;

	return -1;
}