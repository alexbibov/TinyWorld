#ifndef TW__SCREEN__

#include <vector>
#include <list>
#include <string>
#include <functional>

#ifndef GLEW_STATIC
#define GLEW_STATIC
#endif

#include <GL/glew.h>
#include <GLFW/glfw3.h>


#include "AbstractRenderingDevice.h"
#include "Misc.h"


namespace tiny_world
{

	//Describes video mode of the screen
	struct ScreenVideoMode
	{
		uint32_t width;			//width of the screen described in pixels
		uint32_t height;		//height of the screen described in pixels
		uint32_t red_bits;		//bit depth of the red channel of the screen
		uint32_t green_bits;	//bit depth of the green channel of the screen
		uint32_t blue_bits;		//bit depth of the blue channel of the screen
		uint32_t refresh_rate;	//refresh rate of the screen, in Hz

		bool operator ==(const ScreenVideoMode& other) const;	//comparison between video modes
	};


	//Data sources available for the screen-based rendering devices
	enum class ScreenPixelSource : GLenum
	{
		FRONT = GL_FRONT,
		LEFT = GL_LEFT,
		FRONT_LEFT = GL_FRONT_LEFT,

		RIGHT = GL_RIGHT,
		FRONT_RIGHT = GL_FRONT_RIGHT,

		BACK = GL_BACK,
		BACK_LEFT = GL_BACK_LEFT,

		BACK_RIGHT = GL_BACK_RIGHT
	};


	//Multi-threading note: object instances of class Screen can only be created and destroyed by the main thread.
	//Some constructors can be called from the secondary threads, but this is not recommended since in any case the video mode
	//can only be set by the main thread. 
	//IMPORTANT: object instances of class Screen CAN NOT be destroyed by the secondary threads.
	class Screen final : public AbstractRenderingDevice
	{
	public:
		//Describes callback types used by the Screen class
		typedef std::function<void(Screen&, int, int)> OnChangeFramebufferSize;
		typedef std::function<void(Screen&, int, int)> OnChangePosition;
		typedef std::function<void(Screen&, bool)>	OnChangeFocus;
		typedef std::function<void(Screen&, bool)>	OnIconifyOrRestore;
		typedef std::function<void(Screen&)> OnClose;

	private:
		//Wrapping class around GLFW library that ensures that GLFW gets initialized when the program starts and properly terminates the library on exit
		class GLFW_wrapper final
		{
		private:
			bool initialization_status;		//status of GLFW initialization
		public:
			GLFW_wrapper();		//Constructor initializing GLFW library
			~GLFW_wrapper();	//Destructor terminating the library init state

			operator bool() const;	//returns initialization status of GLFW
			bool operator!() const;	//returns NOT applied to the initialization status
		};
		
		static GLFW_wrapper glfw_initialization_wrapper;
		static std::list<Screen*> screen_register_list;	//list of pointers to active Screen objects 
		
		bool GLEW_initialized;	//equals 'false' if GLEW has not yet been initialized


		GLFWmonitor* monitor;		//monitor object associated with the Screen rendering device
		int monitor_index;		//index of associated monitor
		std::string monitor_name;	//name of associated monitor
		std::vector<ScreenVideoMode> video_modes;	//video modes supported by the screen
		GLFWwindow* p_glfwWindow;	//pointer to the window created by GLFW. In the current implementation each screen object can own only one GLFW window. 
		GenericScreenRenderer renderer;	//renderer attached to the screen

		int vsync_interval;			//minimal number of frames to wait before swapping the front and the back buffers

		//Callback functionality
		OnChangeFramebufferSize screen_changesize_callback;		//called when framebuffer attached to the screen changes its size
		OnChangePosition screen_position_callback;				//called when position of the screen changes
		OnChangeFocus screen_focus_gain_callback;				//called when screen works in "windowed" mode and its window gets focused or looses focus
		OnIconifyOrRestore screen_iconify_callback;				//called when window associated with the screen gets iconified or restored. The second argument gets value of 'true' if the window was iconified, and the value of 'false' otherwise
		OnClose screen_close_callback;							//called when screen gets closed

		bool stereo;		//equals 'true' if the Screen object supports stereo rendering
		bool debug;			//equals 'true' if OpenGL context associated with the Screen was initialized in debug mode 
		MULTISAMPLING_MODE multisampling_mode;	//multisampling mode used by the Screen object

		inline bool initialize_glew();		//initializes GLEW 
		inline void register_callbacks();	//performs registration of GLFW callback functions

		//Callback functions
		static void window_refresh_callback(GLFWwindow* p_wnd);
		static void window_resize_callback(GLFWwindow* p_wnd, int width, int height);
		static void position_change_callback(GLFWwindow* p_wnd, int pos_x, int pos_y);
		static void changefocus_callback(GLFWwindow* p_wnd, int focused);
		static void iconify_callback(GLFWwindow* p_wnd, int iconified);
		static void close_callback(GLFWwindow* p_wnd);
	
	public:
		Screen(MULTISAMPLING_MODE ms_pixel_mode = MULTISAMPLING_MODE::MULTISAMPLING_NONE, 
			bool debug_mode = false, bool stereo_rendering = false);	//default constructor, creates object associated with the primary monitor used by the system
		
		Screen(uint32_t monitor_index, 
			MULTISAMPLING_MODE ms_pixel_mode = MULTISAMPLING_MODE::MULTISAMPLING_NONE, 
			bool debug_mode = false, bool stereo_rendering = false);	//creates Screen rendering device associated with the screen having index monitor_index
		
		Screen(std::string monitor_name, 
			MULTISAMPLING_MODE ms_pixel_mode = MULTISAMPLING_MODE::MULTISAMPLING_NONE, 
			bool debug_mode = false, bool stereo_rendering = false);	//creates Screen rendering device associated with the screen named by monitor_name
		
		Screen(uint32_t position_x, uint32_t position_y, uint32_t width, uint32_t height,
			MULTISAMPLING_MODE ms_pixel_mode = MULTISAMPLING_MODE::MULTISAMPLING_NONE, 
			bool debug_mode = false, bool stereo_rendering = false);		//initializes screen in "windowed" mode, which means that all rendering is done in a window located at screen coordinates (position_x, position_y). In this case the multi-monitor configuration is determined by currently active driver settings. This constructor can only be called from the main thread.
		
		//Copy constructor has special meaning for the Screen objects. When a newly created object A gets initialized by the state of object B, this means that the object A will share the OpenGL context with the object B.
		//Copy constructor can only be called from the main thread.
		Screen(const Screen& other);
		
		Screen(Screen&& other);
		
		~Screen();	//Screen destruction can only be performed by the main thread.



		//Copy assignment has special meaning in the context of Screen objects.
		//Basically, when Screen object A is assigned to a screen object B this means that the Screen object B will
		//share OpenGL context owned by the Screen object A. In "windowed" mode the dimensions of the GLFW window owned by B
		//will also be the same as dimensions of the GLFW window owned by A. Copy-assignment operator can only be called from the main thread.
		Screen& operator=(const Screen& other);
		Screen& operator=(Screen&& other);

		//Screen information functions
		int getScreenIndex() const;	//returns index of the monitor currently associated with the Screen rendering device. In case of "windowed" mode, the returned value is -1
		std::string getScreenName() const;	//returns string name of the monitor currently associated with the Screen rendering device. In case of "windowed" mode the returned value is the same as the window title
		std::pair<uint32_t, uint32_t> getScreenSize() const;	//returns physical size of the monitor currently associated with the Screen rendering device. The size is returned as a pair of integer values, where the first component stores width and the second component stores height of the monitor in pixels. If the Screen has been initialized in "windowed" mode, the returned value is the size of framebuffer represented in pixels.
		std::pair<uint32_t, uint32_t> getScreenPosition() const;	//returns position, in screen coordinates, of the upper-left corner of the client area of the screen. The function returns pair with the first element storing x-coordinate and the second element storing y-coordinate of the position. If the Screen has been initialized in "windowed" mode, the returned value is position of the associated window described in screen coordinates.
		ScreenVideoMode getCurrentVideoMode() const;		//returns video mode currently used by the monitor associated with the Screen rendering device. If the Screen has been initialized to work in "windowed" mode, the returned "video mode" corresponds to the size and the context settings of the window.
		std::vector<ScreenVideoMode> getVideoModes() const;	//returns list of video modes supported by the monitor associated with the Screen rendering device. If the Screen has been initialized to work in "windowed" mode, the function returns single element corresponding the current size and bit depth of the window.


		//Screen attachment to the rendering pipeline
		void setScreenVideoMode(ScreenVideoMode mode);		//sets video mode used by the Screen object. This function can only be called from the main thread.
		void attachRenderer(const GenericScreenRenderer& renderer);		//attaches renderer to the Screen object.
		void setVSyncInterval(int new_inteval);		//sets minimal number of frames to wait before swapping the front and the back buffers.
		void update();	//fully updates screen settings (including depth, stencil and blending options) and refreshes the screen by calling refresh()
		void refresh();	//refreshes the current state of the screen
		void makeActive() override;	//make the Screen current target for drawing
		void makeActiveForReading() const override;	//makes the screen active for reading operations
		void makeActiveForDrawing() override;	//makes the screen active for drawing operations



		//Screen callback functionality
		void registerOnChangeSizeCallback(const OnChangeFramebufferSize& on_change_framebuffer_size_callback);		//registers callback function, which is called when size of framebuffer changes
		void registerOnChangePositionCallback(const OnChangePosition& on_change_position_callback);	//registers callback function, which is called when position of the Screen gets changed
		void registerOnChangeFocusCallback(const OnChangeFocus& on_change_focus_callback);		//registers callback function, which is called when Screen gets or looses focus
		void registerOnIconifyOrRestoreCallback(const OnIconifyOrRestore& on_iconify_or_restore_callback);		//registers callback function, which is called when Screen's window gets iconified or restored
		void registerOnCloseCallback(const OnClose& on_close_callback);	//registers callback function, which is called when Screen's window gets closed		


		//Multi-sampling control overrides
		void setMultisamplingEnableState(bool enabled);		//allows to enable and disable multisampling usage by the rendering context


		//Miscellaneous functions
		int getVSyncInterval() const;	//returns currently active vertical synchronization interval
		bool shouldClose() const;		//returns 'true' when user attempts to close window associated with the Screen object.
		bool supportsMultisampling() const;		//returns 'true' if the Screen supports multisampling.
		bool supportsStereoRendering() const;	//returns 'true' if the screen supports stereo rendering.
		MULTISAMPLING_MODE getMultisamplingMode() const;	//returns multisampling mode used by the Screen object.
		void setStringName(const std::string& new_name);	//sets new string name for the Screen device and updates the title of GLFW window accordingly
		operator GLFWwindow*() const;	//allows to merge Screen objects with GLFW functionality 
		bool isScreenBasedDevice() const override;	//screen is a screen-based device, hence implementation of this function always returns 'true'
		void setPixelReadSource(ScreenPixelSource read_source);	//sets source for pixel extraction operations


		//Color buffer blending settings
		ColorBlendFactor getRGBSourceBlendFactor() const;		//returns RGB-part of the global source blend factor
		ColorBlendFactor getAlphaSourceBlendFactor() const;		//returns Alpha-part of the global source blend factor
		
		ColorBlendFactor getRGBDestinationBlendFactor() const;		//returns RGB-part of the global destination blend factor
		ColorBlendFactor getAlphaDestinationBlendFactor() const;		//returns Alpha-part of the global destination blend factor

		ColorBlendEquation getRGBBlendEquation() const;		//returns blend equation applied to the RGB-part of the source and destination color values multiplied by the blending factors
		ColorBlendEquation getAlphaBlendEquation() const;	//returns blend equation applied to the Alpha-part of the source and destination color values multiplied by the blending factors

		
		//Static functions that allow to query information regarding displays and video modes supported by the host system
		static uint32_t getMonitorCount();	//Retrieves number of screens currently attached to the system. Returns 0 in case of error.
		static std::vector<std::string> getMonitorNames();	//Retrieves string names of the monitors currently attached to the system. The index of the string name in returned vector object corresponds to monitor index identifier. If an error occurs at some point during a call to this function, the function does not fail and returns vector of the string names it has managed to retrieve.
		static std::vector<ScreenVideoMode> getVideoModes(std::string monitor_name);	//Retrieves all video modes supported by the monitor referred by monitor_name
		static std::vector<ScreenVideoMode> getVideoModes(uint32_t monitor_index);		//Retrieves all video modes supported by the monitor referred by monitor_index
	};


}

#define TW__SCREEN__
#endif