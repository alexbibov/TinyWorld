#ifndef TW__ABSTRACT_RENDERING_DEVICE__

#include <GL/glew.h>
#include <cstdint>
#include <array>
#include <bitset>
#include <map>
#include <functional>
#include <list>

#include "Misc.h"
#include "ImmutableTexture1D.h"
#include "ImmutableTexture2D.h"
#include "ImmutableTexture3D.h"
#include "ImmutableTextureCubeMap.h"
#include "VectorTypes.h"
#include "ErrorBehavioral.h"

namespace tiny_world
{
	//There are three types of Rendering Device objects:
	//Screen - rendered data is forwarded to the buffers managed by window system
	//Framebuffer - stencil, depth and color data are drawn to texture attachments
	//Virtual - no actual rendering is done, framebuffer has no rendering attachments but can have unlimited "size" in memory
	//
	//All three types are inherited from class AbstractRenderingDevice. This class has the following capabilities:
	//AbstractRenderingDevice maintains internal identifier of the framebuffer
	//AbstractRenderingDevice is able to enable or disable stencil and depth tests and encapsulates enable status of these tests
	//AbstractRenderingDevice is able to enable or disable blending and multisampling and encapsulates enable status of both 
	//AbstractRenderingDevice is able to set and retrieve stencil and depth tests' parameters and encapsulates their state
	//AbstractRenderingDevice is able to set and retrieve blending and multisampling parameters and encapsulates status of both

	
	//Type wrapper over OpenGL stencil test pass criteria
	enum class StencilTestPassFunction : GLenum
	{
		NEVER = GL_NEVER,
		ALWAYS = GL_ALWAYS,
		LESS = GL_LESS,
		LESS_OR_EQUAL = GL_LEQUAL,
		EQUAL = GL_EQUAL,
		GREATER_OR_EQUAL = GL_GEQUAL,
		GREATER = GL_GREATER,
		NOT_EQUAL = GL_NOTEQUAL
	};

	//Type wrapper over OpenGL post- stencil test operation
	enum class StencilBufferUpdateOperation : GLenum
	{
		KEEP = GL_KEEP,
		ZERO = GL_ZERO,
		REPLACE = GL_REPLACE,
		INCREMENT = GL_INCR,
		DECREMENT = GL_DECR,
		INVERT = GL_INVERT,
		INCREMENT_WITH_OVERFLOW = GL_INCR_WRAP,
		DECREMENT_WITH_UNDERFLOW = GL_DECR_WRAP
	};

	//Type wrapper over OpenGL depth test pass criteria
	enum class DepthTestPassFunction : GLenum
	{
		NEVER = GL_NEVER,
		ALWAYS = GL_ALWAYS,
		LESS = GL_LESS,
		LESS_OR_EQUAL = GL_LEQUAL,
		EQUAL = GL_EQUAL,
		GREATER_OR_EQUAL = GL_GEQUAL,
		GREATER = GL_GREATER,
		NOT_EQUAL = GL_NOTEQUAL
	};


	//Blending factors
	enum class ColorBlendFactor : GLenum
	{
		ZERO = GL_ZERO,
		ONE = GL_ONE,
		SRC_COLOR = GL_SRC_COLOR,
		ONE_MINUS_SRC_COLOR = GL_ONE_MINUS_SRC_COLOR,
		DST_COLOR = GL_DST_COLOR,
		ONE_MINUS_DST_COLOR = GL_ONE_MINUS_DST_COLOR,
		SRC_ALPHA = GL_SRC_ALPHA,
		ONE_MINUS_SRC_ALPHA = GL_ONE_MINUS_SRC_ALPHA,
		DST_ALPHA = GL_DST_ALPHA,
		ONE_MINUS_DST_ALPHA = GL_ONE_MINUS_DST_ALPHA,
		CONSTANT_COLOR = GL_CONSTANT_COLOR,
		ONE_MINUS_CONSTANT_COLOR = GL_ONE_MINUS_CONSTANT_COLOR,
		CONSTANT_ALPHA = GL_CONSTANT_ALPHA,
		ONE_MINUS_CONSTANT_ALPHA = GL_ONE_MINUS_CONSTANT_ALPHA,
		SRC_ALPHA_SATURATE = GL_SRC_ALPHA_SATURATE,
		SRC1_COLOR = GL_SRC1_COLOR,
		ONE_MINUS_SRC1_COLOR = GL_ONE_MINUS_SRC1_COLOR,
		SRC1_ALPHA = GL_SRC1_ALPHA,
		ONE_MINUS_SRC1_ALPHA = GL_ONE_MINUS_SRC1_ALPHA
	};

	//Blending equations
	enum class ColorBlendEquation : GLenum
	{
		ADD = GL_FUNC_ADD,
		SUBTRACT = GL_FUNC_SUBTRACT,
		REVERSE_SUBTRACT = GL_FUNC_REVERSE_SUBTRACT,
		MIN = GL_MIN,
		MAX = GL_MAX
	};

	//Clear target bits used when resetting data in rendering buffers to certain predefined values
	enum class BufferClearTarget : GLbitfield
	{
		COLOR = GL_COLOR_BUFFER_BIT,
		DEPTH = GL_DEPTH_BUFFER_BIT,
		STENCIL = GL_STENCIL_BUFFER_BIT, 
		COLOR_DEPTH = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT,
		DEPTH_STENCIL = GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT,
		COLOR_STENCIL = GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT,
		COLOR_DEPTH_STENCIL = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT
	};

	//Color sources for pixel extraction functions readPixels(...) and readPixelsIntoTexture(...)
	enum class RenderingColorBuffer : GLenum
	{
		FRONT_LEFT = GL_FRONT_LEFT, 
		FRONT_RIGHT = GL_FRONT_RIGHT,
		BACK_LEFT = GL_BACK_LEFT,
		BACK_RIGHT = GL_BACK_RIGHT,
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
	
	typedef std::bitset<32> bitmask32;

	//StencilTestPasCriterion defines criterion that determines in what cases stencil test should pass.
	//The criterion is composed of three types: 
	//1) a boolean pass function applied to each and every bit of the stencil buffer
	//2) reference value playing role of parameter passed to the pass function
	//3) mask that is bitwise ANDed with both the reference value and the stored stencil value, with the ANDed values participating in the comparison. 
	typedef triplet<StencilTestPassFunction, uint32_t, bitmask32> StencilTestPassCriterion;


	//Type describing generic renderer object. Renderer objects are functors used to evoke drawing commands 
	class Screen;
	typedef std::function<void(Screen&)> GenericScreenRenderer;

	class Framebuffer;
	typedef std::function<void(Framebuffer&)> GenericFramebufferRenderer;


	class AbstractRenderingDevice : public Entity
	{
	private:
		__declspec(thread) static long long active_device;	//id of the rendering device active on the current calling thread


		//Viewport state variables
		std::map<uint32_t, Rectangle> viewports;		//set of viewports owned by this rendering device
		std::map <uint32_t, Rectangle> scissor_rectangles;	//set of areas determining which fragments should pass scissor test for a given viewport

		struct ContextSnapshotFront
		{
			//Cull test state variables
			Face face_to_cull;	//face (or faces) that should be dropped off by the culling test
			bool cull_test_enabled;		//determines face culling enable state

			//Clear buffer state variables
			vec4 clear_color;	//value used to clear color buffers of the rendering device
			double clear_depth;	//value used to clear depth buffer of the rendering device
			int clear_stencil;	//value used to clear stencil buffer of the rendering device

			//Scissor test state variables
			bool scissor_test_enabled;		//equals 'true' if scissor test is enabled for this rendering device

			//Stencil buffer state variables
			//For two-element arrays below the first element corresponds to front face and the second element corresponds to back face
			bool stencil_test_enabled;	//equals 'true' if stencil test is enabled for this framebuffer object. Equals 'false' otherwise.
			std::array<StencilTestPassCriterion, 2> stencil_test_pass_func;	//Defines boolean criterion determining in what cases stencil test should pass. 
			std::array<StencilBufferUpdateOperation, 2> stencil_test_op_stfail;	//Defines operation applied to the stencil buffer if stencil test fails. 
			std::array<StencilBufferUpdateOperation, 2> stencil_test_op_dtfail;	//Defines operation applied to the stencil buffer if depth test fails.
			std::array<StencilBufferUpdateOperation, 2> stencil_test_op_dtpass;	//Defines operation applied to the stencil buffer if depth test passes.
			std::array<bitmask32, 2> stencil_mask;	//Determines mask of the stencil buffer bits, which should get updated by stencil buffer operations.


			//Depth buffer state variables
			bool depth_test_enabled;	//equals 'true' if depth test is enabled for this framebuffer object. Equals 'false' otherwise.
			DepthTestPassFunction depth_test_pass_func;		//boolean criterion that determines whether depth test should pass for the currently analyzed fragment.
			bool depth_test_update_flag;		//if equals 'false' the contents of the depth buffer don't get updated after the depth test.
			bool depth_test_clamp_flag;			//determines whether the newly acquired depth values should be clamped to the range of 0 to 1. 


			//Multisampling functionality 
			bool multisampling_enabled;		//Equals 'true' if multisampling is enabled for the framebuffer owned by this rendering device


			//Color buffer blending state variables
			bool color_blend_enabled;	//equals 'true' if color blending is enabled for this rendering device
			vec4 blend_color;	//constant blend color used by some blending factors 


			//Primitive restart state variables
			bool primitive_restart_enabled;		//equals 'true' if primitive restart is enabled on primitive assembly stage for the current rendering device
			uint32_t primitive_restart_index;	//index value, which causes primitive assembly to start a new primitive on the next vertex

			//Pixel packing parameters that affect the way pixels are getting extracted into the client's memory
			bool swap_bytes;	//when equals 'true' a multi-byte component of a pixel (b0, b1, b2, b3) will be packed into memory as (b3, b2, b1, b0). Initial value is 'false'
			bool lsb_first;		//when equals 'true' the pixel extraction operations will write each byte of the data they extract beginning from the least significant bit
			TextureStorageAlignment pack_alignment;	//pack alignment used by pixel extraction operations
		}context_front;


		inline void init_device();	//helper function: performs initialization particulars upon creation of a new rendering device

	protected:
		struct ContextSnapshotBack
		{
			//The following variables store global blending state applied to all color buffers attached to the rendering device
			ColorBlendFactor g_rgb_source;	//Global blend factor for source RGB-channel applied to all color buffer attachments
			ColorBlendFactor g_alpha_source;	//Global blend factor for source alpha-channel applied to all color buffer attachments
			ColorBlendFactor g_rgb_destination;	//Global blend factor for destination RGB-channel applied to all color buffer attachments
			ColorBlendFactor g_alpha_destination;	//Global blend factor for destination alpha-channel applied to all color buffer attachments
			ColorBlendEquation g_rgb_blending_eq;	//Global color blend equation definition applied to RGB-channels of all color buffer attachments
			ColorBlendEquation g_alpha_blending_eq;	//Global color blend equation definition applied to alpha-channels of all color buffer attachments


			//Color buffer masking state variables
			bvec4 g_color_buffer_mask;		//Color buffer mask applied to all color buffer attachments of the rendering device
		}context_back;



		//Description of a stack structure used to store and retrieve snapshots of a context state
		typedef std::list<std::pair<ContextSnapshotFront, ContextSnapshotBack>> ContextSnapshotStack;
		ContextSnapshotStack context_stack;



		//AbstractRenderingDevice objects do not allow explicit instantiation

		AbstractRenderingDevice(const std::string& rendering_device_class_string_name);		//Default initialization: by default the framebuffer represents the window system front (back) buffer
		AbstractRenderingDevice(const std::string& rendering_device_class_string_name, const std::string& rendering_device_string_name);		//Default initialization: by default the framebuffer represents the window system front (back) buffer
		AbstractRenderingDevice(const AbstractRenderingDevice& other);	//Copy constructor: performs a deep copy between the rendering devices (i.e. all the contents ARE copied)
		AbstractRenderingDevice(AbstractRenderingDevice&& other);	//Move constructor
		virtual ~AbstractRenderingDevice();		//Destructor

		AbstractRenderingDevice& operator=(const AbstractRenderingDevice& other); //Assignment operator overload: during assignment a deep copy of data between the framebuffer objects is performed
		AbstractRenderingDevice& operator=(AbstractRenderingDevice&& other); //Move assignment operator

	public:
		//Miscellaneous 
		virtual void applyOpenGLContextSettings();	//applies settings encapsulated by rendering device to the currently active OpenGL state 

		//Makes rendering device current on the calling thread. That means, that all OpenGL drawing commands as well as invocations of applyOpenGLContextSettings()
		//following a call to this function will affect the rendering device and the associated OpenGL context. Therefore, raw OpenGL drawing is possible
		//even without use of update() and refresh() functions provided by the inherited classes.
		virtual void makeActive();
		virtual void makeActiveForReading() const = 0;	//makes device active for reading operations
		virtual void makeActiveForDrawing();	   //makes device active for drawing operations

		bool isActive() const;	//returns 'true' if the rendering device is active on the calling thread.
		static long long getActiveDevice();	//returns identifier of the device that is currently set active on the calling thread

		//Pushes the context settings that are active for the rendering device down the context settings stack. Note, that active viewport and scissor rectangle settings are not stored in the
		//context settings stack. However, scissor test enable state IS stored.
		virtual void pushOpenGLContextSettings();	

		//Retrieves context settings from the top of the context settings stack and re-applies them to the OpenGL context handled by the rendering device.
		//If context settings stack is empty, the function restores all settings of the context (excepting viewports and scissor boxes) to their default values.
		//The viewports and scissor boxes are left unchanged. Nevertheless, the scissor test enable state DOES get restored from the context settings stack.
		virtual void popOpenGLContextSettings();


		void clearBuffers(BufferClearTarget clear_target);	//clears one or several buffers maintained by the rendering device to the previously set values
		
		//Viewport operations
		void defineViewport(Rectangle viewport);		//defines default viewport at index = 0
		void defineViewport(float x, float y, float width, float height);	//Defines default viewport at index = 0 using explicitly provided viewport parameters.
		void defineViewport(uint32_t viewport_index, Rectangle viewport);	//defines new viewport at index referred by viewport_index.
		void defineViewport(uint32_t viewport_index, float x, float y, float width, float height);	//Defines new viewport at index = viewport_index using explicitly provided viewport parameters.
		Rectangle getViewportRectangle(uint32_t viewport_index) const; 	//Returns viewport rectangle associated with viewport index = viewport_index.

		//Scissor test functionality
		void setScissorTestEnableState(bool enabled);	//Changes enable state of scissor test
		bool getScissorTestEnableState() const;	//Retrieves currently active enable state of scissor test

		void setScissorTestPassRectangle(uint32_t viewport_index, Rectangle scissor_test_pass_area);		//Sets rectangle determining which fragments should pass scissor testing (only the fragments from within the rectangle are accepted). The scissor rectangle is applied to the viewport at index = viewport_index. 
		Rectangle getScissorTestPassRectangle(uint32_t viewport_index) const;	//Returns scissor rectangle associated with viewport at index = viewport_index.


		//Cull test functionality
		void setCullTestMode(Face face);	//sets mode used to cull faces of triangles being drawn by the rendering device
		Face getCullTestMode() const;		//returns currently active mode of face culling

		void setCullTestEnableState(bool enabled);	//allow to alter cull test enable state. If enabled equals 'true', face culling becomes active
		bool getCullTestEnableState() const;		//returns current enable state of the cull test. Returns 'true' if face culling is switched on


		//Buffer clean-up functionality
		void setClearColor(const vec4& clear_color);	//defines color to be used when clearing color buffers of the rendering device
		void setClearColor(float r, float g, float b, float a);		//defines RGBA-color to be used when clearing color buffers of the rendering device
		vec4 getClearColor() const;	//returns current setting for clear color

		void setClearDepth(double clear_depth);	//sets value to be used when clearing depth buffer of the rendering device
		double getClearDepth() const;	//returns current setting for clear depth

		void setClearStencil(int clear_stencil);	//sets value to be used when clearing stencil buffer of the rendering device
		int getClearStencil() const;	//returns currently active setting for clear stencil

		//Stencil buffer operations
		void setStencilTestEnableState(bool enabled);	//sets stencil test enable state. If enabled equals 'true', stencil test gets switched on.
		bool getStencilTestEnableState() const;		//returns currently active enable state of the stencil test associated with the framebuffer object.
		
		void setStencilTestPassFunction(Face face, StencilTestPassFunction func, uint32_t refval, bitmask32 refval_mask);		//updates pass criterion of the stencil test for the specified triangles faces.
		StencilTestPassCriterion getStencilTestPassFunction(Face face) const;		//retrieves stencil test pass criterion for the specified triangle faces.

		void setStencilTestFailStencilOperation(Face face, StencilBufferUpdateOperation op);	//updates operation applied to the stencil buffer if stencil test fails. The update is applied only for the specified triangle faces. 
		StencilBufferUpdateOperation getStencilTestFailStencilOperation(Face face) const;		//retrieves stencil test fail operation applied to the stencil buffer for the specified triangle faces.
		
		void setDepthTestFailStencilOperation(Face face, StencilBufferUpdateOperation op);	//updates operation applied to the stencil buffer if depth test fails. The update is applied only for the specified triangle faces. 
		StencilBufferUpdateOperation getDepthTestFailStencilOperation(Face face) const;		//retrieves depth test fail operation applied to the stencil buffer for the specified triangle faces.
		
		void setDepthTestPassStencilOperation(Face face, StencilBufferUpdateOperation op);	//updates operation applied to the stencil buffer if depth test passes. The update is applied only for the specified triangle faces. 
		StencilBufferUpdateOperation getDepthTestPassStencilOperation(Face face) const;		//retrieves depth test pass operation applied to the stencil buffer for the specified triangle faces.

		void setStencilMask(Face face, bitmask32 mask);	//defines bit-mask determining, which bits of the stencil buffer should be updated and which get ignored.
		bitmask32 getStencilMask(Face face) const;		//retrieves stencil buffer update mask for the specified triangle faces

		//Depth buffer operations
		void setDepthTestEnableState(bool enabled);		//sets depth test enable state. If enabled equals 'true', the depth test will be performed when drawing to encapsulated framebuffer object.
		bool getDepthTestEnableState() const;	//returns currently active enable state of the depth test.

		void setDepthTestPassFunction(DepthTestPassFunction func);		//updates pass criterion of the depth test
		DepthTestPassFunction getDepthTestPassFunction() const;		//retrieves pass function of the depth test

		void setDepthBufferUpdateFlag(bool depth_mask);	//if depth_mask equals 'false', the contents of depth buffer don't get updated regardless of the depth test result. If depth_mask equals 'true', the updates to the depth buffer are switched on and operate normally.
		bool getDepthBufferUpdateFlag() const;	//retrieves depth buffer update flag

		void setDepthBufferClampFlag(bool clamp_flag);	//sets new value for depth buffer clamp flag.
		bool getDepthBufferClampFlag() const;		//retrieves clamp flag value currently in use.

		//Color buffer blending operations
		void setColorBlendEnableState(bool enabled);	//sets enable state for color blending operations. If equals 'true', color blend is enabled for the color buffers of this rendering device. Color blend is disabled otherwise.
		bool getColorBlendEnableState() const;	//returns currently active color blend enable state

		void setBlendConstantColor(float red, float green, float blue, float alpha);	//sets constant blend color value, which is used by some blending factors.
		void setBlendConstantColor(vec4 constant_blend_color);		//sets constant blend color value, which is used by some blending factors. The input to this function defines the constant color by a 4D-vector with components (x = r, y = g, z = b, w = a).
		vec4 getBlendConstantColor() const;		//returns currently active value of the constant blend color. The returned value is a 4D vector with components (x = r, y = g, z = b, w = a).

		void setRGBSourceBlendFactor(ColorBlendFactor rgb_source_bf);	//Sets global source blend factor applied to RGB-channels in all color buffers enabled for this rendering device.
		void setAlphaSourceBlendFactor(ColorBlendFactor alpha_source_bf);	//Sets global source blend factor applied to alpha-channels in all color buffers enabled for this rendering device.
		void setSourceBlendFactor(ColorBlendFactor source_bf);	//Sets global source blend factor simultaneously applied to both RGB- and alpha- channels in all color buffer attachments enabled for this rendering device.

		void setRGBDestinationBlendFactor(ColorBlendFactor rgb_destination_bf);	//Sets global destination blend factor applied to RGB-channels in all color buffers enabled for this rendering device.
		void setAlphaDestinationBlendFactor(ColorBlendFactor alpha_destination_bf);	//Sets global destination blend factor applied to alpha-channels in all color buffers enabled for this rendering device.
		void setDestinationBlendFactor(ColorBlendFactor destination_bf);	//Sets global destination blend factor simultaneously applied to both RGB- and alpha- channels in all color buffer attachments enabled for this rendering device.

		void setRGBBlendEquation(ColorBlendEquation rgb_blend_eq);		//Sets global color blend equation for RGB-channels used by all color buffer attachments enabled for this rendering device.
		void setAlphaBlendEquation(ColorBlendEquation alpha_blend_eq);		//Sets global color blend equation for alpha-channels used by all color buffer attachments enabled for this rendering device.		
		void setBlendEquation(ColorBlendEquation blend_eq);	//Sets global blend equation simultaneously used by both RGB- and alpha- channels. Selected blend equation is global and applied to all color buffer attachments enabled for this rendering device.		


		//Logical operation with color buffer: NOT IMPLEMENTED


		//Masking operations with color buffers
		void setColorBufferMask(bool red, bool green, bool blue, bool alpha);	//Applies global color buffer mask to all enabled color buffer attachments of the rendering device
		void setColorBufferMask(bvec4 mask);	//Applies global color buffer mask to all enabled color attachments using provided 4D boolean vector		

		//Antialiasing filters: NOT IMPLEMENTED

		//Multisampling functionality
		void setMultisamplingEnableState(bool enabled);		//Changes enable state of multisampling rendering.
		bool getMultisamplingEnableState() const;	//Retrieves currently active enable state of multisampling features.

		//Primitive restart functionality
		void setPrimitiveRestartEnableState(bool enabled);	//allows to alter enable state of the primitive restart property (see OpenGL reference for further details)
		bool getPrimitiveRestartEnableState() const;	//returns current state of primitive restart property

		void setPrimitiveRestartIndexValue(uint32_t value);	//sets new index value, which causes primitive assembly to start construction of a new primitive
		uint32_t getPrimitiveRestartIndexValue() const;		//returns currently active primitive restart index value


		//Miscellaneous 
		virtual bool isScreenBasedDevice() const = 0;	//returns 'true' if the rendering device instance, on which this function is called represents a screen-based device (i.e. screen, VR-helmet etc.). Returns 'false' otherwise

		void LSBFirst(bool flag);	//if flag='true' the pixel extraction operations will use least significant bit first ordering for each byte of the data they extract
		bool isLSBFirst() const;	//Returns 'true' if pixel extraction operations write data to the client's memory starting from the least significant bit of each byte.

		void swapBytes(bool flag);	//if flag='true' then each multi-byte pixel component (b0, b1, b2, b3) of the data extracted by the pixel reading operations will be written to the client's memory as (b3, b2, b1, b0)
		bool doesSwapBytes() const;	//Returns 'true' if each multi-byte pixel component (b0, b1, b2, b3) of the data extracted by the pixel reading operations is written to the client's memory as (b3, b2, b1, b0). Returns 'false' otherwise

		void setPackPadding(TextureStorageAlignment new_pack_padding);	//sets pack padding used by the pixel extraction operations
		TextureStorageAlignment getPackPadding() const;	//returns pack padding used by the pixel extraction operations

		void readPixels(RenderingColorBuffer source_color_buffer, int x, int y, int width, int height, PixelReadLayout pixel_layout, PixelDataType pixel_type, void* data) const;	//extracts pixel data from device's memory and writes it to the supplied buffer

		//Replaces a portion of the given mipmap-level of a 1D texture (or of an array of 1D textures starting from the given layer) beginning at the given x-offset with a region of pixel data extracted from memory of the rendering device.
		//The region of pixels that replaces original part of the texture is determined by the origin (x, y) represented in viewport coordinates and by the given width.
		void readPixelsIntoTexture(RenderingColorBuffer source_color_buffer, const ImmutableTexture1D& _1d_texture, uint32_t mipmap_level, uint32_t layer, uint32_t xoffset, uint32_t x, uint32_t y, size_t width, size_t num_layers_to_modify = 1) const;

		//Replaces a portion of the given mipmap-level of a 2D texture (or of an array of 2D textures starting from the given layer) beginning at the given x- and y- offsets with a region of pixel data extracted from memory of the rendering device.
		//The region of pixels that replaces original part of the texture is determined by the origin (x, y) represented in viewport coordinates and by values of width and height.
		void readPixelsIntoTexture(RenderingColorBuffer source_color_buffer, const ImmutableTexture2D& _2d_texture, uint32_t mipmap_level, uint32_t layer, uint32_t xoffset, uint32_t yoffset, uint32_t x, uint32_t y, size_t width, size_t height) const;

		//Replaces a portion of the given mipmap-level of a 3D-texture beginning at the given x-, y-, and z- offsets with a region of pixel data extracted from memory of the rendering device. The portion of data that replaces original pixels in the texture
		//is determined by the origin (x,y) represented in viewport coordinates and by values of width and height 
		void readPixelsIntoTexture(RenderingColorBuffer source_color_buffer, const ImmutableTexture3D& _3d_texture, uint32_t mipmap_level, uint32_t xoffset, uint32_t yoffset, uint32_t zoffset, uint32_t x, uint32_t y, size_t width, size_t height) const;

		//Replaces a portion of the given mipmap-level of a cube map texture (or of an array of cube map textures at the given layer) in the given face with a region of pixel data extracted from memory of the rendering device. The region of pixel data
		//that replaces portion of the original texture is determined by the origin (x,y) represented in viewport coordinates and by value of width and height
		void readPixelsIntoTexture(RenderingColorBuffer source_color_buffer, const ImmutableTextureCubeMap& cubemap_texture, uint32_t mipmap_level, uint32_t layer, CubemapFace face, uint32_t xoffset, uint32_t yoffset, uint32_t x, uint32_t y, size_t width, size_t height) const;
	};



}


#define TW__ABSTRACT_RENDERING_DEVICE__
#endif