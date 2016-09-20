//Implements concept closely related to texture unit in OpenGL.
//This class is a container that manages a set of so-called texture units. Each unit can be thought of as a number of texture
//slots combined with a texture sampler used for data extraction from the textures. Each texture slot corresponds to an OpenGL texture type.
//When a new texture gets bound to the slot that already contains a texture, the old texture gets replaced. Each texture is
//"designed" to occupy specific slot depending on properties of the texture. The slot can be queried by calling Texture::getTextureSlot().


#ifndef TW__TEXTURE_UNIT_BLOCK__

#include <map>
#include <vector>
#include <memory>

#include "Texture.h"
#include "TextureSampler.h"

namespace tiny_world{

	class TextureUnitBlock final : Entity{
	private:
		static bool is_initialized;
		__declspec(thread) static TextureUnitBlock *myself_pointer;
		uint32_t active_unit;
		uint32_t num_of_units;

		typedef std::map<TextureSlot, std::unique_ptr<const Texture>> texture_slot_deck;
		typedef std::pair<texture_slot_deck, const TextureSampler*> texture_unit_layout;
		typedef std::vector<texture_unit_layout> texture_unit_block_layout;

		texture_unit_block_layout texture_unit_block_descriptor;

		TextureUnitBlock();		//This class is a singleton
		TextureUnitBlock(const TextureUnitBlock& other) = delete;
		TextureUnitBlock(TextureUnitBlock&& other) = delete;
		TextureUnitBlock& operator= (const TextureUnitBlock& other) = delete;
		TextureUnitBlock& operator= (TextureUnitBlock&& other) = delete;

	public:
		static TextureUnitBlock* initialize();	//Initializes singleton object
		static void reset();	//Destroys singleton making previously produced pointer incorrect. Allows to reinitialize the object in future if needed.

		uint32_t getNumberOfUnits() const;	//returns maximal amount of texture units that can be used simultaneously

		//Texture and sampler management functions

		void bindTexture(const Texture& texture_object);		//Binds texture to the currently active texture unit
		void bindSampler(const TextureSampler& sampler_object);			//Binds sampler to the currently active texture unit

		

		uint32_t switchActiveTextureUnit(uint32_t new_texture_unit);	//Switches currently active texture unit to new_texture_unit. Returns zero-based number of the previously active texture unit.

		//Information functions

		bool isBound(Texture& texture_object) const;	//Checks if texture_object is currently bound to any of the available texture units
		bool isBound(TextureSampler& sampler_object) const;		//Checks if sampler_object is currently bound to any of the available texture units

		int32_t getBindingTextureUnit(const Texture& texture_object) const;	//Returns zero-based number of the texture unit to which texture_object is currently bound. If texture_object is not bound returns -1.
		int32_t getBindingTextureUnit(const TextureSampler& sampler_object) const;	//Returns zero-based number of the texture unit to which sampler_object is currently bound. If sampler_object is not bound returns -1.

		const Texture* getBoundTexture(TextureSlot slot, uint32_t texture_unit) const;	//Returns pointer to the texture object currently bound to the texture slot "slot" in the texture unit referred by zero based index "texture_unit". If requested texture slot in the specified unit does not contain binding function returns nullptr.
		const TextureSampler* getBoundSampler(uint32_t texture_unit) const;		//Returns pointer to the texture sampler object currently bound to the texture unit referred by zero-based index "texture_unit".

		int getAvailableTextureUnit() const;	//returns identifier of a texture unit, which has no bound textures. If there is no such unit available, returns -1

		~TextureUnitBlock();	//Destructor;
	};

}





#define TW__TEXTURE_UNIT_BLOCK__
#endif