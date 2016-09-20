#ifndef TW__FRAMEBUFFER__

#include <memory>

#include "AbstractRenderingDevice.h"
#include "BufferTexture.h"


namespace tiny_world
{
	//Defines data type containing information regarding framebuffer attachment details. Such information allows 
	//to fully repeat the attachment with the same settings for another framebuffer. 
	//The data type is a structure with three components.
	template<typename TexturePointerType>
	struct tagFramebufferAttachmentDetails_base
	{
		//stores the texture mipmap level, which has been used for attachment or -1 if attachment is undefined
		int32_t mipmap_level;

		//defines layer of the texture attached to the framebuffer. If the value is negative the texture is attached as a whole
		int32_t attachment_layer;

		//pointer to attached texture (texture is referred differently for internal and external management of framebuffer attachment points)
		TexturePointerType texture;

		tagFramebufferAttachmentDetails_base() : mipmap_level{ -1 }, attachment_layer{ -1 }, texture{ nullptr } {}
		tagFramebufferAttachmentDetails_base(int32_t mipmap_level, int32_t attachment_layer, TexturePointerType texture) :
			mipmap_level{ mipmap_level }, attachment_layer{ attachment_layer }, texture{ std::move(texture) } {}

		operator bool() const { return mipmap_level != -1 && texture != nullptr; }
	};




	//The following specializations support interoperability between free and object-owned attachment descriptors
	template<typename TexturePointerType>
	struct tagFramebufferAttachmentDetails_derived : public tagFramebufferAttachmentDetails_base < TexturePointerType > {};




	template<>
	struct tagFramebufferAttachmentDetails_derived<const Texture*> :
		tagFramebufferAttachmentDetails_base < const Texture* >
	{
		tagFramebufferAttachmentDetails_derived() : tagFramebufferAttachmentDetails_base{} {}

		tagFramebufferAttachmentDetails_derived(int32_t mipmap_level, int32_t attachment_layer, const Texture* texture) :
			tagFramebufferAttachmentDetails_base{ mipmap_level, attachment_layer, texture } {}

		tagFramebufferAttachmentDetails_derived(const tagFramebufferAttachmentDetails_base<std::unique_ptr<const Texture>>& other) :
			tagFramebufferAttachmentDetails_base{ other.mipmap_level, other.attachment_layer, other.texture.get() } {}

		tagFramebufferAttachmentDetails_derived& operator=(const tagFramebufferAttachmentDetails_base<std::unique_ptr<const Texture>>& other)
		{
			mipmap_level = other.mipmap_level;
			attachment_layer = other.attachment_layer;
			texture = other.texture.get();
		}
	};




	template<>
	struct tagFramebufferAttachmentDetails_derived<std::unique_ptr<const Texture>> : 
		public tagFramebufferAttachmentDetails_base<std::unique_ptr<const Texture>>
	{
		tagFramebufferAttachmentDetails_derived() : tagFramebufferAttachmentDetails_base{} {}

		tagFramebufferAttachmentDetails_derived(int32_t mipmap_level, int32_t attachment_layer, std::unique_ptr<const Texture> texture) :
			tagFramebufferAttachmentDetails_base{ mipmap_level, attachment_layer, std::move(texture) } {}

		tagFramebufferAttachmentDetails_derived(const tagFramebufferAttachmentDetails_base<const Texture*>& other) :
			tagFramebufferAttachmentDetails_base{ other.mipmap_level, other.attachment_layer, nullptr }
		{
			if (other.texture)
				texture = std::unique_ptr < const Texture > {other.texture->clone()};
		}

		tagFramebufferAttachmentDetails_derived(const tagFramebufferAttachmentDetails_derived& other) :
			tagFramebufferAttachmentDetails_base{ other.mipmap_level, other.attachment_layer, nullptr } 
		{
			if (other.texture)
				texture = std::unique_ptr < const Texture > {other.texture->clone()};
		}

		tagFramebufferAttachmentDetails_derived(tagFramebufferAttachmentDetails_derived&& other) :
			tagFramebufferAttachmentDetails_base{ other.mipmap_level, other.attachment_layer, std::move(other.texture) }
		{}


		tagFramebufferAttachmentDetails_derived& operator=(const tagFramebufferAttachmentDetails_base<const Texture*>& other)
		{
			mipmap_level = other.mipmap_level;
			attachment_layer = other.attachment_layer;
			texture = other.texture ? std::unique_ptr < const Texture > {other.texture->clone()} : nullptr;
			return *this;
		}

		tagFramebufferAttachmentDetails_derived& operator=(const tagFramebufferAttachmentDetails_derived& other)
		{
			mipmap_level = other.mipmap_level;
			attachment_layer = other.attachment_layer;
			texture = other.texture ? std::unique_ptr < const Texture > {other.texture->clone()} : nullptr;
			return *this;
		}

		tagFramebufferAttachmentDetails_derived& operator=(tagFramebufferAttachmentDetails_derived&& other)
		{
			mipmap_level = other.mipmap_level;
			attachment_layer = other.attachment_layer;
			texture = std::move(other.texture);
			return *this;
		}
	};




	typedef tagFramebufferAttachmentDetails_derived<const Texture*> FramebufferAttachmentInfo;	//information describing details of an abstract texture attachment

	enum class FramebufferColorAttachmentPoint : GLenum
	{
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

	class Framebuffer final : public AbstractRenderingDevice
	{
	private:
		static const uint32_t max_color_attachments = 16;		//maximal number of color attachments supported
		typedef tagFramebufferAttachmentDetails_derived<std::unique_ptr<const Texture>> OwnedFramebufferAttachmentInfo;		//attachment ownership details


		GLuint ogl_framebuffer_id;		//OpenGL framebuffer identifier

		OwnedFramebufferAttachmentInfo stencil_attachment;			//stencil attachment of the framebuffer, excludes usage of stencil_depth attachment point
		OwnedFramebufferAttachmentInfo depth_attachment;			//depth attachment of the framebuffer, excludes usage of stencil_depth attachment point
		OwnedFramebufferAttachmentInfo stencil_depth_attachment;	//simultaneous attachment of the stencil and the depth attachment points; excludes usage of stencil_attachment and depth_attachment
		std::array<OwnedFramebufferAttachmentInfo, max_color_attachments> color_attachments;			//color attachments of the framebuffer


		struct ContextSnapshotFramebuffer
		{
			//The following variables store blending state of each individual color buffer enabled for the rendering device. If a blend option is defined for an individual buffer, it overrides the corresponding global option.

			std::map<uint32_t, ColorBlendFactor> rgb_sources;	//Blend factor for source RGB-channels of a color buffer attachment
			std::map<uint32_t, ColorBlendFactor> alpha_sources;	//Blend factor for source alpha-channels of a color buffer attachment
			std::map<uint32_t, ColorBlendFactor> rgb_destinations;	//Blend factor for destination RGB-channels of a color buffer attachment
			std::map<uint32_t, ColorBlendFactor> alpha_destinations;	//Blend factor for destination alpha-channels of a color buffer attachment
			std::map<uint32_t, ColorBlendEquation> rgb_blending_eqs;	//Color blend equation applied to RGB-channels for a color buffer attachment
			std::map<uint32_t, ColorBlendEquation> alpha_blending_eqs;	//Color blend equation applied to alpha-channels for a color buffer attachment

			//Color buffer masking state variables

			std::map<uint32_t, bvec4> color_buffer_masks;	//A set of color buffer masks individually applied to each enabled color buffer attachment
		}context_framebuffer;


		std::list<ContextSnapshotFramebuffer> context_framebuffer_stack;	//the stack object containing context settings of the framebuffer


		//Callback functions

		GenericFramebufferRenderer renderer;	//function, which performs rendering commands

		//Miscellaneous 

		//checks if framebuffer is complete. If is_read_complete == true, checks framebuffer for read-completeness, otherwise checks it for being draw-complete
		bool is_complete(std::string* p_string_description, bool is_read_complete) const;

	public:
		Framebuffer();		//default constructor
		explicit Framebuffer(const std::string& framebuffer_string_name);	//initializes framebuffer with the given string name
		Framebuffer(const Framebuffer& other);		//copy constructor
		Framebuffer(Framebuffer&& other);			//move constructor

		Framebuffer& operator=(const Framebuffer& other);	//copy-assignment operator
		Framebuffer& operator=(Framebuffer&& other);		//move-assignment operator

		~Framebuffer();

		//Apply OpenGL context settings to the currently active context
		void applyOpenGLContextSettings() override;

		//Pushes current state of the framebuffer down the framebuffer context state stack. Note that context state includes all the parameters related to the scissor, stencil, and depth tests as well as
		//blending settings. However, the context does not include framebuffer attachment list or renderer's linkage, therefore the state of framebuffer attachment and the renderer's linkage will not be saved on the stack. 
		//These settings can however be backed up by explicitly copying the framebuffer object
		void pushOpenGLContextSettings() override;

		//Retrieves framebuffer context settings block from the top of framebuffer context stack and applies the settings from the block to the framebuffer. Note, however, that the context settings stack does not
		//store framebuffer texture attachment list and the renderer function's linkage, hence the function does not affect texture attachments of the framebuffer and does not changes its rendering function.
		//If the stack is empty, the function will restore context settings of the framebuffer to the default values
		void popOpenGLContextSettings() override;


		//Attach a new renderer to the Framebuffer object.
		void attachRenderer(const GenericFramebufferRenderer& renderer);		


		//Returns identifier of "the highest" color attachment point supported. The minimal guaranteed return value is 7 meaning that color attachment points from 0-7 are necessarily supported by the 
		//hardware that is able to create OpenGL 4.3 rendering context
		static uint32_t getLastSupportedColorAttachmentPoint();


		//Setting and getting RGB-part of the source blend factor
		using AbstractRenderingDevice::setRGBSourceBlendFactor;
		void setRGBSourceBlendFactor(uint32_t color_buffer_index, ColorBlendFactor rgb_source_bf);	//Sets source blend factor applied to RGB-channel in color buffer attachment referred by color_buffer_index.
		ColorBlendFactor getRGBSourceBlendFactor(uint32_t color_buffer_index) const;	//Returns source blend factor currently active for RGB-channel in color buffer attachment referred by color_buffer_index.

		using AbstractRenderingDevice::setAlphaSourceBlendFactor;
		void setAlphaSourceBlendFactor(uint32_t color_buffer_index, ColorBlendFactor alpha_source_bf);	//Sets source blend factor applied to alpha-channel in color buffer attachment referred by color_buffer_index.
		ColorBlendFactor getAlphaSourceBlendFactor(uint32_t color_buffer_index) const;	//Returns source blend factor currently active for alpha-channel in color buffer attachment referred by color_buffer_index.

		using AbstractRenderingDevice::setSourceBlendFactor;
		void setSourceBlendFactor(uint32_t color_buffer_index, ColorBlendFactor source_bf);	//Sets source blend factor simultaneously applied to both RGB- and alpha- channels in color buffer attachment referred by color_buffer_index.

		using AbstractRenderingDevice::setRGBDestinationBlendFactor;
		void setRGBDestinationBlendFactor(uint32_t color_buffer_index, ColorBlendFactor rgb_destination_bf);	//Sets destination blend factor applied to RGB-channel in color buffer attachment referred by color_buffer_index.
		ColorBlendFactor getRGBDestinationBlendFactor(uint32_t color_buffer_index) const;	//Returns destination blend factor currently active for RGB-channel in color buffer attachment referred by color_buffer_index.

		using AbstractRenderingDevice::setAlphaDestinationBlendFactor;
		void setAlphaDestinationBlendFactor(uint32_t color_buffer_index, ColorBlendFactor alpha_destination_bf);	//Sets destination blend factor applied to alpha-channel in color buffer attachment referred by color_buffer_index.
		ColorBlendFactor getAlphaDestinationBlendFactor(uint32_t color_buffer_index) const;	//Returns destination blend factor currently active for alpha-channel in color buffer attachment referred by color_buffer_index.

		using AbstractRenderingDevice::setDestinationBlendFactor;
		void setDestinationBlendFactor(uint32_t color_buffer_index, ColorBlendFactor destination_bf);	//Sets destination blend factor simultaneously applied to both RGB- and alpha- channels in color buffer attachment referred by color_buffer_index.

		using AbstractRenderingDevice::setRGBBlendEquation;
		void setRGBBlendEquation(uint32_t color_buffer_index, ColorBlendEquation rgb_blend_eq);	//Sets color blend equation for RGB-channels. Selected blend equation will be individually used by single color buffer attachment located at location = color_buffer_index..
		ColorBlendEquation getRGBBlendEquation(uint32_t color_buffer_index) const;		//Retrieves currently active RGB channel blend equation used by color buffer attachment located at location = color_buffer_index.

		using AbstractRenderingDevice::setAlphaBlendEquation;
		void setAlphaBlendEquation(uint32_t color_buffer_index, ColorBlendEquation alpha_blend_eq);	//Sets color blend equation for alpha-channels. Selected blend equation will be individually used by single color buffer attachment located at location = color_buffer_index.
		ColorBlendEquation getAlphaBlendEquation(uint32_t color_buffer_index) const;		//Retrieves currently active alpha channel blend equation used by color buffer attachment located at location = color_buffer_index.

		using AbstractRenderingDevice::setBlendEquation;
		void setBlendEquation(uint32_t color_buffer_index, ColorBlendEquation blend_eq);	//Sets blend equation for both RGB- and alpha- channels. This blend equation will be individually used by color buffer attachment located at location = color_buffer_index.


		//Masking operations with color buffers
		using AbstractRenderingDevice::setColorBufferMask;
		void setColorBufferMask(uint32_t color_buffer_index, bool red, bool green, bool blue, bool alpha);	//Sets color buffer mask for an individual color buffer attachment referred by color_buffer_index
		void setColorBufferMask(uint32_t color_buffer_index, bvec4 mask);	//Sets color buffer mask for an individual color buffer attachment using 4D boolean vector
		bvec4 getColorBufferMask(uint32_t color_buffer_index) const;	//Returns color buffer mask applied to the color buffer attachment referred by color_buffer_index. The mask is returned as a 4D boolean vector with elements (x = red_mask, y = green_mask, z = blue_mask, w = alpha_mask).


		//Operations that modify the state of the framebuffer

		void makeActive() override;		//makes the Framebuffer active target for pixel reading operations and for drawing
		void makeActiveForReading() const override;	//makes the Framebuffer active target for pixel reading operations. This function does not change active rendering device of the calling thread.
		void makeActiveForDrawing() override;	//makes the Framebuffer active target for drawing operations

		void attachTexture(FramebufferAttachmentPoint attachment_point, const FramebufferAttachmentInfo& attachment_details);	//specifies texture attachment for the framebuffer
		FramebufferAttachmentInfo retrieveAttachmentDetails(FramebufferAttachmentPoint attachment_point) const;	//returns attachment details for the given attachment point of the framebuffer
		void detachTexture(FramebufferAttachmentPoint attachment_point);	//detaches texture from the given attachment point
		void setPixelReadSource(FramebufferColorAttachmentPoint read_source);	//sets source for the operations that read pixels from the framebuffer.


		//Operations that control rendering to the Framebuffer

		void update();	//fully updates settings of the framebuffer (including depth, stencil and blending options) and refreshes the rendering by calling refresh()
		void refresh();	//refreshes the contents of the framebuffer

		//Operations used to check framebuffer completeness

		//returns 'true' if framebuffer is complete for reading and 'false' otherwise. In addition, stores string description to a user-defined string object. 
		//If framebuffer is read-incomplete the description buffer will contain a possible reason for that. If the framebuffer is read-complete, the description buffer will contain string "framebuffer is complete for reading".
		//If p_description_buf = nullptr, no text description is provided on output.
		bool isReadComplete(std::string* p_description_buf = nullptr) const;	

		//returns 'true' if framebuffer is complete for drawing, returns 'false' otherwise. In addition, the function returns textual description of a possible 
		//reason for framebuffer being incomplete for drawing. The description is returned via string object provided by the caller. If framebuffer turns out to be
		//complete for drawing  the textual description will contain "framebuffer is complete for drawing". 
		//if p_description_buf = nullptr, no text description will be given on the function's output
		bool isDrawComplete(std::string* p_description_buf = nullptr) const;


		//Miscellaneous
		bool isScreenBasedDevice() const override;	//Framebuffer is a virtual rendering device and is not screen-based, hence this implementation always yields 'false'
	};



}


#define TW__FRAMEBUFFER__
#endif