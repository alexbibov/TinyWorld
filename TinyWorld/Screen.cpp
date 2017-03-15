#include "Screen.h"

using namespace tiny_world;

Screen::GLFW_wrapper Screen::glfw_initialization_wrapper{};
std::list<Screen*> Screen::screen_register_list{};

//Very simple wrapper allowing to test whether "condition" is satisfied and setting the object to erroneous state in case of failure
#define SCREEN_ERROR_TEST(condition, errmsg) \
    if(!(condition)) \
        { \
        set_error_state(true); \
        set_error_string(errmsg); \
        call_error_callback(errmsg); \
        return; \
    } \





bool ScreenVideoMode::operator==(const ScreenVideoMode& other) const
{
    return (width == other.width && height == other.height &&
        red_bits == other.red_bits && green_bits == other.green_bits && blue_bits == other.blue_bits &&
        refresh_rate == other.refresh_rate);
}





Screen::GLFW_wrapper::GLFW_wrapper()
{
    initialization_status = glfwInit() == GL_TRUE;
}

Screen::GLFW_wrapper::~GLFW_wrapper()
{
    glfwTerminate();
}

Screen::GLFW_wrapper::operator bool() const { return initialization_status; }

bool Screen::GLFW_wrapper::operator!() const { return !initialization_status; }





void Screen::register_callbacks()
{
    glfwSetWindowRefreshCallback(p_glfwWindow, Screen::window_refresh_callback);	//register refresh callback
    glfwSetWindowSizeCallback(p_glfwWindow, Screen::window_resize_callback);	//register framebuffer refresh callback
    glfwSetWindowPosCallback(p_glfwWindow, Screen::position_change_callback);	//register position change callback
    glfwSetWindowFocusCallback(p_glfwWindow, Screen::changefocus_callback);		//register focus state change callback
    glfwSetWindowIconifyCallback(p_glfwWindow, Screen::iconify_callback);		//register iconified state change callback
    glfwSetWindowCloseCallback(p_glfwWindow, Screen::close_callback);			//register callback, which is called when the Screen's window is being closed
}


bool Screen::initialize_glew()
{
    if (!GLEW_initialized)
    {
        glewExperimental = GL_TRUE;
        GLEW_initialized = glewInit() == GLEW_OK;
    }

    return GLEW_initialized;
}

Screen::Screen(MULTISAMPLING_MODE ms_pixel_mode /* = MULTISAMPLING_MODE::MULTISAMPLING_NONE */,
    bool debug_mode /*= false */, bool stereo_rendering /* = false */) :
    AbstractRenderingDevice{ "Screen" },
    GLEW_initialized{ false },
    multisampling_mode{ ms_pixel_mode },
    debug{ debug_mode },
    stereo{ stereo_rendering },
    monitor{ NULL },
    monitor_index{ -1 },
    monitor_name(""),
    p_glfwWindow{ NULL },
    vsync_interval{ 0 },
    renderer{ [](Screen&)->void{} },
    screen_changesize_callback{ [](Screen&, int, int)->void{} },
    screen_position_callback{ [](Screen&, int, int)->void{} },
    screen_focus_gain_callback{ [](Screen&, bool)->void{} },
    screen_iconify_callback{ [](Screen&, bool)->void{} },
    screen_close_callback{ [](Screen&)->void{} }
{
    //Add *this* object to the register list
    Screen::screen_register_list.push_back(this);

    //Check if GLFW has been properly initialized
    SCREEN_ERROR_TEST(Screen::glfw_initialization_wrapper, "Unable to initialize GLFW library");

    //Get primary monitor
    SCREEN_ERROR_TEST((monitor = glfwGetPrimaryMonitor()) != NULL,
        "Unable to recognize primary monitor of the host system");

    //Get name of the primary monitor
    const char* primary_monitor_name;
    SCREEN_ERROR_TEST((primary_monitor_name = glfwGetMonitorName(monitor)) != NULL,
        "Unable to retrieve string name of the primary monitor of the host system");
    monitor_name = primary_monitor_name;

    //Retrieve list of all monitors connected to the host system and find index of the primary monitor
    GLFWmonitor** monitor_list;
    int count = 0;
    SCREEN_ERROR_TEST((monitor_list = glfwGetMonitors(&count)) != NULL && count != 0,
        "Can not recognize some of the monitors connected to the system");
    for (int i = 0; i < count; ++i)
    {
        const char* name;
        SCREEN_ERROR_TEST((name = glfwGetMonitorName(monitor_list[i])),
            "Unable to retrieve string name of the monitor with index " + std::to_string(i));
        if (std::string(name).compare(monitor_name) == 0)
        {
            monitor_index = i;
            break;
        }
    }

    //Retrieve list of video modes supported by the primary monitor
    const GLFWvidmode* video_mode_list;
    count = 0;
    SCREEN_ERROR_TEST((video_mode_list = glfwGetVideoModes(monitor, &count)) != NULL && count != 0,
        "Unable to retrieve the list of video modes supported by monitor \"" + monitor_name + "\"");
    for (int i = 0; i < count; ++i)
        video_modes.push_back(ScreenVideoMode{
            static_cast<uint32_t>(video_mode_list[i].width),
            static_cast<uint32_t>(video_mode_list[i].height),
            static_cast<uint32_t>(video_mode_list[i].redBits),
            static_cast<uint32_t>(video_mode_list[i].greenBits),
            static_cast<uint32_t>(video_mode_list[i].blueBits),
            static_cast<uint32_t>(video_mode_list[i].refreshRate)
    });
}


Screen::Screen(uint32_t monitor_index,
    MULTISAMPLING_MODE ms_pixel_mode /* = MULTISAMPLING_MODE::MULTISAMPLING_NONE */,
    bool debug_mode /*= false */, bool stereo_rendering /* = false */) :
    AbstractRenderingDevice{ "Screen" },
    GLEW_initialized{ false },
    monitor_index{ static_cast<int>(monitor_index) },
    multisampling_mode{ ms_pixel_mode },
    debug{ debug_mode },
    stereo{ stereo_rendering },
    monitor{ nullptr },
    monitor_name(""),
    p_glfwWindow{ nullptr },
    vsync_interval{ 0 }
    /*renderer{ [](Screen&)->void{} },
    screen_changesize_callback{ [](Screen&, int, int)->void{} },
    screen_position_callback{ [](Screen&, int, int)->void{} },
    screen_focus_gain_callback{ [](Screen&, bool)->void{} },
    screen_iconify_callback{ [](Screen&, bool)->void{} },
    screen_close_callback{ [](Screen&)->void{} }*/
{
    //Add *this* object to the register list
    Screen::screen_register_list.push_back(this);

    //Check if GLFW has been properly initialized
    SCREEN_ERROR_TEST(Screen::glfw_initialization_wrapper, "Unable to initialize GLFW library");

    //Retrieve list of all monitors connected to the host system and find index of the primary monitor
    GLFWmonitor** monitor_list;
    int count = 0;
    SCREEN_ERROR_TEST((monitor_list = glfwGetMonitors(&count)) != NULL && count != 0,
        "Can not recognize some of the monitors connected to the system");

    //Assign the monitor corresponding to monitor_index to the Screen object
    SCREEN_ERROR_TEST(monitor_index < static_cast<uint32_t>(count), "Unable to retrive monitor with monitor index " +
        std::to_string(monitor_index) + "on the host system");
    monitor = monitor_list[monitor_index];

    //Retrieve string name of the monitor
    const char* current_monitor_name;
    SCREEN_ERROR_TEST((current_monitor_name = glfwGetMonitorName(monitor)) != NULL,
        "Unable to retrieve string name of the monitor with monitor index " + std::to_string(monitor_index));
    monitor_name = current_monitor_name;

    //Retrieve list of video modes supported by selected monitor
    const GLFWvidmode* video_mode_list;
    count = 0;
    SCREEN_ERROR_TEST((video_mode_list = glfwGetVideoModes(monitor, &count)) != NULL && count != 0,
        "Unable to retrieve the list of video modes supported by monitor \"" + monitor_name + "\"");
    for (int i = 0; i < count; ++i)
        video_modes.push_back(ScreenVideoMode{
        static_cast<uint32_t>(video_mode_list[i].width),
        static_cast<uint32_t>(video_mode_list[i].height),
        static_cast<uint32_t>(video_mode_list[i].redBits),
        static_cast<uint32_t>(video_mode_list[i].greenBits),
        static_cast<uint32_t>(video_mode_list[i].blueBits),
        static_cast<uint32_t>(video_mode_list[i].refreshRate)
    });
}


Screen::Screen(std::string monitor_name,
    MULTISAMPLING_MODE ms_pixel_mode /* = MULTISAMPLING_MODE::MULTISAMPLING_NONE */,
    bool debug_mode /* = false */, bool stereo_rendering /* = false */) :
    AbstractRenderingDevice{ "Screen" }, GLEW_initialized{ false },
    monitor_name(monitor_name), multisampling_mode{ ms_pixel_mode }, debug{ debug_mode },
    stereo{ stereo_rendering }, monitor{ NULL }, monitor_index{ -1 },
    p_glfwWindow{ NULL },
    vsync_interval{ 0 },
    renderer{ [](Screen&)->void{} },
    screen_changesize_callback{ [](Screen&, int, int)->void{} },
    screen_position_callback{ [](Screen&, int, int)->void{} },
    screen_focus_gain_callback{ [](Screen&, bool)->void{} },
    screen_iconify_callback{ [](Screen&, bool)->void{} },
    screen_close_callback{ [](Screen&)->void{} }
{
    //Add *this* object to the register list
    Screen::screen_register_list.push_back(this);

    //Check if GLFW has been properly initialized
    SCREEN_ERROR_TEST(Screen::glfw_initialization_wrapper, "Unable to initialize GLFW library");

    //Retrieve list of all monitors connected to the host system and find index of the monitor with requested string name
    GLFWmonitor** monitor_list;
    int count = 0;
    SCREEN_ERROR_TEST((monitor_list = glfwGetMonitors(&count)) != NULL && count != 0,
        "Can not recognize some of the monitors connected to the system");
    for (int i = 0; i < count; ++i)
    {
        const char* name;
        SCREEN_ERROR_TEST((name = glfwGetMonitorName(monitor_list[i])),
            "Unable to retrieve string name of the monitor with index " + std::to_string(i));
        if (std::string(name).compare(monitor_name) == 0)
        {
            monitor = monitor_list[i];
            monitor_index = i;
            break;
        }
    }

    //Retrieve list of video modes supported by selected monitor
    const GLFWvidmode* video_mode_list;
    count = 0;
    SCREEN_ERROR_TEST((video_mode_list = glfwGetVideoModes(monitor, &count)) != NULL && count != 0,
        "Unable to retrieve the list of video modes supported by monitor \"" + monitor_name + "\"");
    for (int i = 0; i < count; ++i)
        video_modes.push_back(ScreenVideoMode{
        static_cast<uint32_t>(video_mode_list[i].width),
        static_cast<uint32_t>(video_mode_list[i].height),
        static_cast<uint32_t>(video_mode_list[i].redBits),
        static_cast<uint32_t>(video_mode_list[i].greenBits),
        static_cast<uint32_t>(video_mode_list[i].blueBits),
        static_cast<uint32_t>(video_mode_list[i].refreshRate)
    });
}


Screen::Screen(uint32_t position_x, uint32_t position_y, uint32_t width, uint32_t height,
    MULTISAMPLING_MODE ms_pixel_mode /* = MULTISAMPLING_MODE::MULTISAMPLING_NONE */,
    bool debug_mode /* = false */, bool stereo_rendering /* = false */) :
    AbstractRenderingDevice{ "Screen" }, GLEW_initialized{ false },
    multisampling_mode{ ms_pixel_mode }, debug{ debug_mode }, stereo{ stereo_rendering },
    monitor{ NULL }, monitor_index{ -1 },
    p_glfwWindow{ NULL }, vsync_interval{ 0 },
    renderer{ [](Screen&)->void{} },
    screen_changesize_callback{ [](Screen&, int, int)->void{} },
    screen_position_callback{ [](Screen&, int, int)->void{} },
    screen_focus_gain_callback{ [](Screen&, bool)->void{} },
    screen_iconify_callback{ [](Screen&, bool)->void{} },
    screen_close_callback{ [](Screen&)->void{} }
{
    //Add *this* object to the register list
    Screen::screen_register_list.push_back(this);

    //Check if GLFW has been properly initialized
    SCREEN_ERROR_TEST(Screen::glfw_initialization_wrapper, "Unable to initialize GLFW library");

    //Set monitor name to be equal to the name of abstract rendering device
    monitor_name = getStringName();

    //Set dummy video mode based on parameters of the windowed regime
    const GLFWvidmode *current_video_mode;
    SCREEN_ERROR_TEST((current_video_mode = glfwGetVideoMode(glfwGetPrimaryMonitor())) != NULL,
        "Unable to retrieve current video mode of the primary monitor");

    ScreenVideoMode windowed_regime_video_mode = {
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        static_cast<uint32_t>(current_video_mode->redBits),
        static_cast<uint32_t>(current_video_mode->greenBits),
        static_cast<uint32_t>(current_video_mode->blueBits),
        static_cast<uint32_t>(current_video_mode->refreshRate)
    };

    video_modes.push_back(windowed_regime_video_mode);

    //Create GLFW window
    glfwWindowHint(GLFW_VISIBLE, GL_FALSE);		//initially, the newly created window is not visible
    glfwWindowHint(GLFW_RED_BITS, windowed_regime_video_mode.red_bits);	//color channel depth values are inherited from the current video mode of the primary display
    glfwWindowHint(GLFW_GREEN_BITS, windowed_regime_video_mode.green_bits);
    glfwWindowHint(GLFW_BLUE_BITS, windowed_regime_video_mode.blue_bits);
    glfwWindowHint(GLFW_SAMPLES, static_cast<int>(ms_pixel_mode));	//define multisampling mode
    setMultisamplingEnableState(true);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, debug ? GL_TRUE : GL_FALSE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    SCREEN_ERROR_TEST((p_glfwWindow = glfwCreateWindow(windowed_regime_video_mode.width, windowed_regime_video_mode.height, getStringName().c_str(), NULL, NULL)) != NULL,
        "Unable to create GLFW window");
    glfwSetWindowPos(p_glfwWindow, position_x, position_y);

    //Register callbacks
    register_callbacks();
}


Screen::Screen(const Screen& other) : AbstractRenderingDevice{ other }, GLEW_initialized{ other.GLEW_initialized },
monitor{ other.monitor }, monitor_index{ other.monitor_index }, monitor_name(other.monitor_name),
video_modes(other.video_modes), stereo{ other.stereo }, debug{ other.debug }, multisampling_mode{ other.multisampling_mode },
p_glfwWindow(NULL), vsync_interval{ other.vsync_interval },
renderer{ other.renderer },
screen_changesize_callback{ other.screen_changesize_callback },
screen_position_callback{ other.screen_position_callback },
screen_focus_gain_callback{ other.screen_focus_gain_callback },
screen_iconify_callback{ other.screen_iconify_callback },
screen_close_callback{ other.screen_close_callback }
{
    //Add *this* object to the register list
    Screen::screen_register_list.push_back(this);

    SCREEN_ERROR_TEST(!getErrorState(), "Screen object can not be copy-initialized by an object being in an erroneous state");

    if (monitor == NULL)	//In case, where the copy source object was initialized in "windowed" mode, create GLFW window
    {
        glfwWindowHint(GLFW_VISIBLE, GL_FALSE);		//initially, GLFW windows are invisible
        glfwWindowHint(GLFW_RED_BITS, video_modes[0].red_bits);		//color channel depths are defined using the only video mode available for the "windowed" rendering regime
        glfwWindowHint(GLFW_GREEN_BITS, video_modes[0].green_bits);
        glfwWindowHint(GLFW_BLUE_BITS, video_modes[0].blue_bits);
        glfwWindowHint(GLFW_SAMPLES, static_cast<int>(multisampling_mode));		//set multisampling mode
        setMultisamplingEnableState(true);
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, debug ? GL_TRUE : GL_FALSE);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        SCREEN_ERROR_TEST((p_glfwWindow = glfwCreateWindow(video_modes[0].width, video_modes[0].height, getStringName().c_str(), NULL, other.p_glfwWindow)) != NULL,
            "Unable to create GLFW window with shared context");

        //Set position of the newly created window to be equal to position of the window owned by the copy source
        int xpos, ypos;
        glfwGetWindowPos(other.p_glfwWindow, &xpos, &ypos);
        glfwSetWindowPos(p_glfwWindow, xpos, ypos);
    }
    else	//otherwise, assign p_glfwWindow to point to the GLFW window owned by the copy source object.
            //This will allow to share OpenGL context with the assignment source object, when the destination object is attached to rendering pipeline
    {
        p_glfwWindow = other.p_glfwWindow;
    }
}


Screen::Screen(Screen&& other) : AbstractRenderingDevice{ std::move(other) }, GLEW_initialized{ other.GLEW_initialized },
monitor{ other.monitor }, monitor_index{ other.monitor_index }, monitor_name(std::move(other.monitor_name)),
video_modes(std::move(other.video_modes)), stereo{ other.stereo }, debug{ other.debug }, multisampling_mode{ other.multisampling_mode },
vsync_interval{ other.vsync_interval },
renderer{ std::move(other.renderer) },
screen_changesize_callback{ std::move(other.screen_changesize_callback) },
screen_position_callback{ std::move(other.screen_position_callback) },
screen_focus_gain_callback{ std::move(other.screen_focus_gain_callback) },
screen_iconify_callback{ std::move(other.screen_iconify_callback) },
screen_close_callback{ std::move(other.screen_close_callback) }
{
    //Add *this* object to the register list
    Screen::screen_register_list.push_back(this);

    //Capture ownership of GLFW window from the move source
    p_glfwWindow = other.p_glfwWindow;
    other.p_glfwWindow = NULL;
}


Screen& Screen::operator=(const Screen& other)
{
    //Handle the special case of "self assignment"
    if (this == &other)
        return *this;

    if (other.getErrorState())
    {
        set_error_state(true);
        const char* err_msg = "Screen object can not be copy-assigned to an object being in an erroneous state";
        set_error_string(err_msg);
        call_error_callback(err_msg);
        return *this;
    }

    //Copy the base part of the Screen objects
    AbstractRenderingDevice::operator=(other);

    //Copy OpenGL context settings
    vsync_interval = other.vsync_interval;
    renderer = other.renderer;
    screen_changesize_callback = other.screen_changesize_callback;
    screen_position_callback = other.screen_position_callback;
    screen_focus_gain_callback = other.screen_focus_gain_callback;
    screen_iconify_callback = other.screen_iconify_callback;
    screen_close_callback = other.screen_close_callback;
    GLEW_initialized = other.GLEW_initialized;
    stereo = other.stereo;
    debug = other.debug;
    multisampling_mode = other.multisampling_mode;

    //If assignment destination object contains initialized GLFW window, terminate this window
    if (p_glfwWindow != NULL)
    {
        glfwMakeContextCurrent(NULL);	//detach current OpenGL context from the main thread
        glfwDestroyWindow(p_glfwWindow);	//destroy GLFW window owned by the Screen
    }


    //If object uses "windowed" rendering mode, create new GLFW window that will share OpenGL context with the assignment source object
    if (monitor == NULL)
    {
        video_modes = other.video_modes;
        glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
        glfwWindowHint(GLFW_RED_BITS, video_modes[0].red_bits);
        glfwWindowHint(GLFW_GREEN_BITS, video_modes[0].green_bits);
        glfwWindowHint(GLFW_BLUE_BITS, video_modes[0].blue_bits);
        glfwWindowHint(GLFW_SAMPLES, static_cast<int>(multisampling_mode));
        setMultisamplingEnableState(true);
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, debug ? GL_TRUE : GL_FALSE);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        if ((p_glfwWindow = glfwCreateWindow(video_modes[0].width, video_modes[0].height, getStringName().c_str(), NULL, other.p_glfwWindow)) != NULL)
        {
            set_error_state(true);
            const char* err_msg = "Unable to create GLFW window with shared context";
            set_error_string(err_msg);
            call_error_callback(err_msg);
            return *this;
        }

        //Set position of the newly created window to be equal to position of the window owned by the copy source
        int xpos, ypos;
        glfwGetWindowPos(other.p_glfwWindow, &xpos, &ypos);
        glfwSetWindowPos(p_glfwWindow, xpos, ypos);

        //Register callbacks
        register_callbacks();
    }
    else	//otherwise, assign p_glfwWindow to point to the GLFW window owned by the assignment source object.
            //This will allow to share OpenGL context with the assignment source object, when the destination object is attached to rendering pipeline
    {
        p_glfwWindow = other.p_glfwWindow;
    }

    return *this;
}


Screen& Screen::operator=(Screen&& other)
{
    //Handle the special case, where the object gets assigned to itself
    if (this == &other)
        return *this;

    AbstractRenderingDevice::operator=(std::move(other));

    vsync_interval = other.vsync_interval;
    renderer = std::move(other.renderer);
    screen_changesize_callback = std::move(other.screen_changesize_callback);
    screen_position_callback = std::move(other.screen_position_callback);
    screen_focus_gain_callback = std::move(other.screen_focus_gain_callback);
    screen_iconify_callback = std::move(other.screen_iconify_callback);
    screen_close_callback = std::move(other.screen_close_callback);
    GLEW_initialized = other.GLEW_initialized;
    monitor = other.monitor;
    monitor_index = other.monitor_index;
    monitor_name = std::move(other.monitor_name);
    video_modes = std::move(other.video_modes);
    p_glfwWindow = other.p_glfwWindow; other.p_glfwWindow = NULL;
    stereo = other.stereo;
    debug = other.debug;
    multisampling_mode = other.multisampling_mode;

    return *this;
}


Screen::~Screen()
{
    if (p_glfwWindow)
    {
        glfwMakeContextCurrent(NULL);
        glfwDestroyWindow(p_glfwWindow);
    }

    //Remove *this* object from the register list
    Screen::screen_register_list.remove(this);
}


int Screen::getScreenIndex() const { return monitor_index; }


std::string Screen::getScreenName() const { return monitor_name; }


std::pair<uint32_t, uint32_t> Screen::getScreenSize() const
{
    if (Screen::glfw_initialization_wrapper)
    {
        int width, height;
        if (monitor != NULL)
            glfwGetMonitorPhysicalSize(monitor, &width, &height);
        else
            glfwGetWindowSize(p_glfwWindow, &width, &height);

        return std::make_pair(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
    }

    return std::make_pair(0U, 0U);
}


std::pair<uint32_t, uint32_t> Screen::getScreenPosition() const
{
    if (Screen::glfw_initialization_wrapper)
    {
        int position_x, position_y;
        if (monitor != NULL)
            glfwGetMonitorPos(monitor, &position_x, &position_y);
        else
            glfwGetWindowPos(p_glfwWindow, &position_x, &position_y);

        return std::make_pair(static_cast<uint32_t>(position_x), static_cast<uint32_t>(position_y));
    }

    return std::make_pair(0U, 0U);
}


ScreenVideoMode Screen::getCurrentVideoMode() const
{
    if (Screen::glfw_initialization_wrapper)
    {
        if (monitor != NULL)
        {
            const GLFWvidmode* current_video_mode;
            if ((current_video_mode = glfwGetVideoMode(monitor)) == NULL)
            {
                set_error_state(true);
                std::string err_msg = "Unable to retrieve current video mode of monitor \"" + monitor_name + "\"";
                set_error_string(err_msg);
                call_error_callback(err_msg);
                return ScreenVideoMode{ 0, 0, 0, 0, 0, 0 };
            }

            ScreenVideoMode screen_video_mode = {
                static_cast<uint32_t>(current_video_mode->width),
                static_cast<uint32_t>(current_video_mode->height),
                static_cast<uint32_t>(current_video_mode->redBits),
                static_cast<uint32_t>(current_video_mode->greenBits),
                static_cast<uint32_t>(current_video_mode->blueBits),
                static_cast<uint32_t>(current_video_mode->refreshRate)
            };
            return screen_video_mode;
        }
        else
            return video_modes[0];
    }
    else
        return ScreenVideoMode{ 0, 0, 0, 0, 0, 0 };
}


std::vector<ScreenVideoMode> Screen::getVideoModes() const
{
    if (Screen::glfw_initialization_wrapper)
        return video_modes;
    else
        return std::vector < ScreenVideoMode > {};
}


void Screen::setScreenVideoMode(ScreenVideoMode mode)
{
    if (!Screen::glfw_initialization_wrapper)
        return;

    SCREEN_ERROR_TEST(std::find(video_modes.begin(), video_modes.end(), mode) != video_modes.end(),
        "Requested video mode is not supported by Screen \"" + getStringName() + "\"");

    if (monitor == NULL)
        glfwShowWindow(p_glfwWindow);
    else
    {
        if (p_glfwWindow == NULL)
        {
            glfwWindowHint(GLFW_RED_BITS, mode.red_bits);
            glfwWindowHint(GLFW_GREEN_BITS, mode.green_bits);
            glfwWindowHint(GLFW_BLUE_BITS, mode.blue_bits);
            glfwWindowHint(GLFW_REFRESH_RATE, mode.refresh_rate);
            glfwWindowHint(GLFW_SAMPLES, static_cast<int>(multisampling_mode));
            glfwWindowHint(GLFW_STEREO, static_cast<int>(stereo));
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
            glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, static_cast<int>(debug));
            SCREEN_ERROR_TEST((p_glfwWindow = glfwCreateWindow(mode.width, mode.height, getStringName().c_str(), monitor, NULL)) != NULL,
                "Unable to create GLFW window");
        }
        else
        {
            glfwWindowHint(GLFW_RED_BITS, mode.red_bits);
            glfwWindowHint(GLFW_GREEN_BITS, mode.green_bits);
            glfwWindowHint(GLFW_BLUE_BITS, mode.blue_bits);
            glfwWindowHint(GLFW_REFRESH_RATE, mode.refresh_rate);
            glfwWindowHint(GLFW_SAMPLES, static_cast<int>(multisampling_mode));
            glfwWindowHint(GLFW_STEREO, static_cast<int>(stereo));
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
            glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, static_cast<int>(debug));
            SCREEN_ERROR_TEST((p_glfwWindow = glfwCreateWindow(mode.width, mode.height, getStringName().c_str(), monitor, p_glfwWindow)) != NULL,
                "Unable to create GLFW window");
        }

        //Register callbacks
        register_callbacks();
    }
}


void Screen::attachRenderer(const GenericScreenRenderer& renderer)
{
    //Make OpenGL context associated with the Screen object current
    glfwMakeContextCurrent(p_glfwWindow);

    //Initialize GLEW
    SCREEN_ERROR_TEST((GLEW_initialized = initialize_glew()), "Unable to initialize GLEW");

    //Check if the context supports OpenGL 4.3
    SCREEN_ERROR_TEST(glewIsSupported("GL_VERSION_4_3"),
        "OpenGL 4.3 is not supported by the context. Check if the latest video driver is installed");

    //Apply OpenGL context settings
    applyOpenGLContextSettings();

    //Attach renderer to the screen
    this->renderer = renderer;
}


void Screen::setVSyncInterval(int new_interval)
{
    //Check if negative swap intervals can be accepted
    SCREEN_ERROR_TEST(new_interval >= 0 ||
        glfwExtensionSupported("WGL_EXT_swap_control_tear") == GL_TRUE ||
        glfwExtensionSupported("GLX_EXT_swap_control_tear") == GL_TRUE,
        "Negative buffer swap intervals are not supported by the context");

    vsync_interval = new_interval;
    glfwSwapInterval(new_interval);
}


void Screen::refresh()
{
    if (!GLEW_initialized || !p_glfwWindow) return;
    renderer(*this);
    glfwSwapBuffers(p_glfwWindow);
    glfwPollEvents();
}

void Screen::makeActive()
{
    AbstractRenderingDevice::makeActive();
    glfwMakeContextCurrent(p_glfwWindow);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);	//make the default framebuffer current target for rendering
    applyOpenGLContextSettings();
}


void Screen::makeActiveForReading() const
{
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}


void Screen::makeActiveForDrawing()
{
    AbstractRenderingDevice::makeActiveForDrawing();
    glfwMakeContextCurrent(p_glfwWindow);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    applyOpenGLContextSettings();
}


void Screen::update()
{
    if (!GLEW_initialized || !p_glfwWindow) return;
    renderer(*this);
    glfwSwapBuffers(p_glfwWindow);
    glfwPollEvents();
    applyOpenGLContextSettings();
}


void Screen::registerOnChangeSizeCallback(const OnChangeFramebufferSize& on_change_framebuffer_size_callback)
{
    screen_changesize_callback = on_change_framebuffer_size_callback;
}


void Screen::registerOnChangePositionCallback(const OnChangePosition& on_change_position_callback)
{
    screen_position_callback = on_change_position_callback;
}


void Screen::registerOnChangeFocusCallback(const OnChangeFocus& on_change_focus_callback)
{
    screen_focus_gain_callback = on_change_focus_callback;
}


void Screen::registerOnIconifyOrRestoreCallback(const OnIconifyOrRestore& on_iconify_or_restore_callback)
{
    screen_iconify_callback = on_iconify_or_restore_callback;
}


void Screen::registerOnCloseCallback(const OnClose& on_close_callback)
{
    screen_close_callback = on_close_callback;
}


void Screen::setMultisamplingEnableState(bool enabled)
{
    if (multisampling_mode != MULTISAMPLING_MODE::MULTISAMPLING_NONE)
        AbstractRenderingDevice::setMultisamplingEnableState(enabled);
    else
        AbstractRenderingDevice::setMultisamplingEnableState(false);
}


int Screen::getVSyncInterval() const { return vsync_interval; }


bool Screen::shouldClose() const { return glfwWindowShouldClose(p_glfwWindow) == GL_TRUE; }


bool Screen::supportsMultisampling() const { return multisampling_mode != MULTISAMPLING_MODE::MULTISAMPLING_NONE; }


bool Screen::supportsStereoRendering() const { return stereo; }


MULTISAMPLING_MODE Screen::getMultisamplingMode() const { return multisampling_mode; }


void Screen::setStringName(const std::string& new_name)
{
    AbstractRenderingDevice::setStringName(new_name);
    if (Screen::glfw_initialization_wrapper && p_glfwWindow != NULL)
        glfwSetWindowTitle(p_glfwWindow, new_name.c_str());
}


Screen::operator GLFWwindow*() const { return p_glfwWindow; }


bool Screen::isScreenBasedDevice() const { return true; }


void Screen::setPixelReadSource(ScreenPixelSource read_source)
{
    GLint current_framebuffer;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &current_framebuffer);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glReadBuffer(static_cast<GLenum>(read_source));

    glBindFramebuffer(GL_READ_FRAMEBUFFER, current_framebuffer);
}


void Screen::window_refresh_callback(GLFWwindow* p_wnd)
{
    std::list<Screen*>::iterator pp_screen_object;
    if ((pp_screen_object =
        std::find_if(Screen::screen_register_list.begin(), Screen::screen_register_list.end(),
        [p_wnd](Screen* p_screen_object) -> bool { return p_screen_object->p_glfwWindow == p_wnd; })) !=
        Screen::screen_register_list.end())
    {
        (*pp_screen_object)->renderer(*(*pp_screen_object));
    }

}


void Screen::window_resize_callback(GLFWwindow* p_wnd, int width, int height)
{
    std::list<Screen*>::iterator pp_screen_object;
    if ((pp_screen_object =
        std::find_if(Screen::screen_register_list.begin(), Screen::screen_register_list.end(),
        [p_wnd](Screen* p_screen_object) -> bool { return p_screen_object->p_glfwWindow == p_wnd; })) !=
        Screen::screen_register_list.end())
    {
        (*pp_screen_object)->screen_changesize_callback(*(*pp_screen_object), width, height);
    }
}


void Screen::position_change_callback(GLFWwindow* p_wnd, int pos_x, int pos_y)
{
    std::list<Screen*>::iterator pp_screen_object;
    if ((pp_screen_object =
        std::find_if(Screen::screen_register_list.begin(), Screen::screen_register_list.end(),
        [p_wnd](Screen* p_screen_object) -> bool { return p_screen_object->p_glfwWindow == p_wnd; })) !=
        Screen::screen_register_list.end())
    {
        (*pp_screen_object)->screen_position_callback(*(*pp_screen_object), pos_x, pos_y);
    }
}


void Screen::changefocus_callback(GLFWwindow* p_wnd, int focused)
{
    std::list<Screen*>::iterator pp_screen_object;
    if ((pp_screen_object =
        std::find_if(Screen::screen_register_list.begin(), Screen::screen_register_list.end(),
        [p_wnd](Screen* p_screen_object) -> bool { return p_screen_object->p_glfwWindow == p_wnd; })) !=
        Screen::screen_register_list.end())
    {
        (*pp_screen_object)->screen_focus_gain_callback(*(*pp_screen_object), focused == GL_TRUE);
    }
}

void Screen::iconify_callback(GLFWwindow* p_wnd, int iconified)
{
    std::list<Screen*>::iterator pp_screen_object;
    if ((pp_screen_object =
        std::find_if(Screen::screen_register_list.begin(), Screen::screen_register_list.end(),
        [p_wnd](Screen* p_screen_object) -> bool { return p_screen_object->p_glfwWindow == p_wnd; })) !=
        Screen::screen_register_list.end())
    {
        (*pp_screen_object)->screen_iconify_callback(*(*pp_screen_object), iconified == GL_TRUE);
    }
}

void Screen::close_callback(GLFWwindow* p_wnd)
{
    std::list<Screen*>::iterator pp_screen_object;
    if ((pp_screen_object =
        std::find_if(Screen::screen_register_list.begin(), Screen::screen_register_list.end(),
        [p_wnd](Screen* p_screen_object) -> bool { return p_screen_object->p_glfwWindow == p_wnd; })) !=
        Screen::screen_register_list.end())
    {
        (*pp_screen_object)->screen_close_callback(*(*pp_screen_object));
    }
}





ColorBlendFactor Screen::getRGBSourceBlendFactor() const { return context_back.g_rgb_source; }

ColorBlendFactor Screen::getAlphaSourceBlendFactor() const { return context_back.g_alpha_source; }

ColorBlendFactor Screen::getRGBDestinationBlendFactor() const { return context_back.g_rgb_destination; }

ColorBlendFactor Screen::getAlphaDestinationBlendFactor() const { return context_back.g_alpha_destination; }

ColorBlendEquation Screen::getRGBBlendEquation() const { return context_back.g_rgb_blending_eq; }

ColorBlendEquation Screen::getAlphaBlendEquation() const { return context_back.g_alpha_blending_eq; }





uint32_t Screen::getMonitorCount()
{
    int count = 0;
    if (Screen::glfw_initialization_wrapper) glfwGetMonitors(&count);
    return count;
}


std::vector<std::string> Screen::getMonitorNames()
{
    std::vector<std::string> rv{};
    GLFWmonitor** monitors = NULL;
    int count = 0;
    if (Screen::glfw_initialization_wrapper && (monitors = glfwGetMonitors(&count)) != NULL)
        for (int i = 0; i < count; ++i)
        {
        const char* name = NULL;
        if ((name = glfwGetMonitorName(monitors[i])) != NULL)
            rv.push_back(name);
        }

    return rv;
}


std::vector<ScreenVideoMode> Screen::getVideoModes(std::string monitor_name)
{
    std::vector<ScreenVideoMode> rv{};

    const GLFWvidmode* monitor_video_modes = NULL;
    GLFWmonitor** monitors = NULL;
    int count = 0;
    if (Screen::glfw_initialization_wrapper && (monitors = glfwGetMonitors(&count)) != NULL)
    {
        for (int i = 0; i < count; ++i)
        {
            const char* name = NULL;
            if ((name = glfwGetMonitorName(monitors[i])) != NULL && monitor_name.compare(name) == 0)
            {
                int video_modes_count = 0;
                if ((monitor_video_modes = glfwGetVideoModes(monitors[i], &video_modes_count)) != NULL)
                    for (int j = 0; j < video_modes_count; ++j)
                        rv.push_back(ScreenVideoMode{
                        static_cast<uint32_t>(monitor_video_modes[j].width),
                        static_cast<uint32_t>(monitor_video_modes[j].height),
                        static_cast<uint32_t>(monitor_video_modes[j].redBits),
                        static_cast<uint32_t>(monitor_video_modes[j].greenBits),
                        static_cast<uint32_t>(monitor_video_modes[j].blueBits),
                        static_cast<uint32_t>(monitor_video_modes[j].refreshRate)
                    });
                break;
            }
        }
    }

    return rv;
}


std::vector<ScreenVideoMode> Screen::getVideoModes(uint32_t monitor_index)
{
    std::vector<ScreenVideoMode> rv{};

    const GLFWvidmode* monitor_video_modes = NULL;
    GLFWmonitor** monitors = NULL;
    int count = 0;
    if (Screen::glfw_initialization_wrapper && (monitors = glfwGetMonitors(&count)) != NULL)
    {
        int video_modes_count = 0;
        if ((monitor_video_modes = glfwGetVideoModes(monitors[monitor_index], &video_modes_count)) != NULL)
            for (int i = 0; i < video_modes_count; ++i)
                rv.push_back(ScreenVideoMode{
                static_cast<uint32_t>(monitor_video_modes[i].width),
                static_cast<uint32_t>(monitor_video_modes[i].height),
                static_cast<uint32_t>(monitor_video_modes[i].redBits),
                static_cast<uint32_t>(monitor_video_modes[i].greenBits),
                static_cast<uint32_t>(monitor_video_modes[i].blueBits),
                static_cast<uint32_t>(monitor_video_modes[i].refreshRate)
            });
    }

    return rv;
}