//Implements image unit object, which can be used by shaders in order to store results of their computations to texture memory

#ifndef TW__IMAGE_UNIT_H__

#include <gl/glew.h>
#include <vector>
#include <memory>

#include "ImmutableTexture.h"
#include "BufferTexture.h"

namespace tiny_world
{
	//Structure describing particulars of a texture attachment to image unit. The following templates are specialized for two versions: one is designed to be owned by 
	//internal texture image unit infrastructure and one is designed for end-user applications
	template<typename TexturePointerType>
	struct ImageAttachmentInfo_base
	{
		int mipmap_level;	//texture mipmap-level, which has been attached to image unit.
		int array_layer;	//texture array layer, which has been attached to image unit. Equals to -1 if attachment has been done for all layers at once.
		BufferAccess access_level;			//access level that affects abilities of the image unit to read and write new data to attached texture object. 
		InternalPixelFormat access_format;	//data format of attached texture object as it appears to the image unit during read and write operations.
		TexturePointerType AttachedTextureAliasPointer;	//pointer to an alias object of the texture attached to image unit. Equals to nullptr, if no proper attachment has been defined.

		//Default constructor
		ImageAttachmentInfo_base() : mipmap_level{ -1 }, array_layer{ -1 },
			access_level{ BufferAccess::READ }, access_format{ InternalPixelFormat::SIZED_RGBA8 },
			AttachedTextureAliasPointer{ nullptr }
		{

		}

		//"Aggregate type" constructor
		ImageAttachmentInfo_base(int mipmap_level, int array_layer, BufferAccess access_level, InternalPixelFormat access_format, const Texture* pTextureAlias) :
			mipmap_level{ mipmap_level }, array_layer{ array_layer }, access_level{ access_level }, access_format{ access_format }, AttachedTextureAliasPointer{ pTextureAlias }
		{

		}
	};


	template<typename TexturePointerType> struct ImageAttachmentInfo_derived : public ImageAttachmentInfo_base < TexturePointerType > {};

	template<>
	struct ImageAttachmentInfo_derived<const Texture*> : public ImageAttachmentInfo_base<const Texture*>
	{
		//Default constructor
		ImageAttachmentInfo_derived() : ImageAttachmentInfo_base{} {}

		//Initializes new instance of the object using the state of specialization ImageAttachmentInfo_base<unique_ptr<const Texture>>
		ImageAttachmentInfo_derived(const ImageAttachmentInfo_base<std::unique_ptr<const Texture>>& other) :
			ImageAttachmentInfo_base{ other.mipmap_level, other.array_layer, other.access_level, other.access_format, other.AttachedTextureAliasPointer.get() }
		{

		}

		//Assign the object with the state of specialization ImageAttachmentInfo_base<unique_ptr<const Texture>>
		ImageAttachmentInfo_derived& operator=(const ImageAttachmentInfo_base<std::unique_ptr<const Texture>>& other)
		{
			mipmap_level = other.mipmap_level;
			array_layer = other.array_layer;
			access_level = other.access_level;
			access_format = other.access_format;
			AttachedTextureAliasPointer = other.AttachedTextureAliasPointer.get();

			return *this;
		}
	};

	template<>
	struct ImageAttachmentInfo_derived<std::unique_ptr<const Texture>> : public ImageAttachmentInfo_base<std::unique_ptr<const Texture>>
	{
		//Default constructor
		ImageAttachmentInfo_derived() : ImageAttachmentInfo_base{} {}

		//Initializes new instance of the object using the state of specialization ImageAttachmentInfo_base<const ImmutableTexture*>
		ImageAttachmentInfo_derived(const ImageAttachmentInfo_base<const Texture*>& other) :
			ImageAttachmentInfo_base{ other.mipmap_level, other.array_layer, other.access_level, other.access_format, nullptr }
		{
			if (other.AttachedTextureAliasPointer)
				AttachedTextureAliasPointer = std::unique_ptr < const Texture > {other.AttachedTextureAliasPointer->clone()};
		}

		//Copy constructor
		ImageAttachmentInfo_derived(const ImageAttachmentInfo_derived& other) :
			ImageAttachmentInfo_base{ other.mipmap_level, other.array_layer, other.access_level, other.access_format, nullptr }
		{
			if (other.AttachedTextureAliasPointer)
				AttachedTextureAliasPointer = std::unique_ptr < const Texture > {other.AttachedTextureAliasPointer->clone()};
		}

		//Move constructor
		ImageAttachmentInfo_derived(ImageAttachmentInfo_derived&& other) :
			ImageAttachmentInfo_base{ other.mipmap_level, other.array_layer, other.access_level, other.access_format, nullptr }
		{
			AttachedTextureAliasPointer = std::move(other.AttachedTextureAliasPointer);
		}

		//Assigns the object with the value provided by an instance of specialization ImageAttachmentInfo_base<const ImmutableTexture*>
		ImageAttachmentInfo_derived& operator=(const ImageAttachmentInfo_base<const Texture*>& other)
		{
			mipmap_level = other.mipmap_level;
			array_layer = other.array_layer;
			access_level = other.access_level;
			access_format = other.access_format;
			AttachedTextureAliasPointer = other.AttachedTextureAliasPointer ? std::move(std::unique_ptr < const Texture > { other.AttachedTextureAliasPointer->clone() }) : nullptr;

			return *this;
		}

		//Copy assignment
		ImageAttachmentInfo_derived& operator=(const ImageAttachmentInfo_derived& other)
		{
			mipmap_level = other.mipmap_level;
			array_layer = other.array_layer;
			access_level = other.access_level;
			access_format = other.access_format;
			AttachedTextureAliasPointer = other.AttachedTextureAliasPointer ? std::move(std::unique_ptr < const Texture > {other.AttachedTextureAliasPointer->clone()}) : nullptr;

			return *this;
		}

		//Move assignment
		ImageAttachmentInfo_derived& operator=(ImageAttachmentInfo_derived&& other)
		{
			mipmap_level = other.mipmap_level;
			array_layer = other.array_layer;
			access_level = other.access_level;
			access_format = other.access_format;
			AttachedTextureAliasPointer = std::move(other.AttachedTextureAliasPointer);

			return *this;
		}
	};


	typedef ImageAttachmentInfo_derived<const Texture*> ImageAttachmentInfo;


	class ImageUnit : public Entity
	{
	private:
		typedef ImageAttachmentInfo_derived < std::unique_ptr<const Texture> > ImageAttachmentInfoOwned;

		typedef std::vector<std::pair<bool, ImageAttachmentInfoOwned>> image_unit_layout;
		
		static image_unit_layout image_units;	//vector of texture image unit attachment layouts.

		uint32_t* ref_counter;		//pointer to the reference counter of the image unit
		int target_image_unit;		//identifier of the image unit associated with image object instance

		int retrieve_unused_image_unit() const;	//retrieves an unused texture image unit, which can be safely associated with the image unit object instance. If all texture image units are in use returns -1.

	public:
		ImageUnit();	//default initialization
		ImageUnit(const ImageUnit& other);		//copy constructor
		ImageUnit(ImageUnit&& other);	//move constructor
		~ImageUnit();	//destructor

		ImageUnit& operator=(const ImageUnit& other);	//assignment operator
		ImageUnit& operator=(ImageUnit&& other);	//move-assignment operator
		bool operator==(const ImageUnit& other) const;	//returns 'true' if both image unit object instances being compared are associated with the same texture image unit. Returns 'false' otherwise


		
		//Retrieves amount of vacant texture image units supported by the hardware (the returned value is the maximal number of supported units minus the occupied units)
		static uint32_t getNumberOfAvailableImageUnits();

		//Attaches specified layer of the given mipmap-level of the given texture to the image unit
		void attachTexture(const ImmutableTexture& texture, int mipmap_level, int array_layer, BufferAccess access_level, InternalPixelFormat access_format) const;

		//Attaches given mipmap-level of the given texture to the image unit. If texture is an array texture, then all layers will be attached at once.
		void attachTexture(const ImmutableTexture& texture, int mipmap_level, BufferAccess access_level, InternalPixelFormat access_format) const;

		//Attaches buffer texture to the image unit
		void attachTexture(const BufferTexture& texture, BufferAccess access_level, BufferTextureInternalPixelFormat access_format) const;

		//Returns 'true' if image unit has a texture object attached to it. Returns 'false' otherwise
		bool hasAttachment() const;

		//Retrieves information regarding actual texture attachment on the given image unit. If no attachment has been made, the returned structure contains undefined data.
		ImageAttachmentInfo retrieveAttachment() const;

		//Returns identifier associated with the texture image unit object
		int getBinding() const;

		//Ensures that any further access to the image unit will see the effect from preceding write accesses
		void flush() const;

		//Assigns an actual image unit to the object. If the object already has an image unit assigned to it the function has no effect.
		//Note that image unit assignment is always performed during initialization of the object so the only scenario when a call to this function would
		//be needed is if the assignment has been released by calling release()
		void assign();

		//Releases association between the object and the actual image unit. Note that the image unit may remain in use if some other objects are still
		//associated with it. In order to create a new association between the object and an image unit (not necessarily the same that has been previously released)
		//one has to call assign()
		void release();
	};

}

#define TW__IMAGE_UNIT_H__
#endif