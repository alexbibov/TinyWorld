//This header contains miscelleneous functionality employed by various parts of the TinyWorld engine

#ifndef TW__TINY_WORLD_MISC__

#include <GL/glew.h>
#include <stdint.h>

namespace tiny_world
{
    //The following template helps to map standard C++ types to OpenGL implementation-dependent types
    template<typename T> struct ogl_type_mapper;
    template<> struct ogl_type_mapper < bool > { typedef GLboolean ogl_type; };
    template<> struct ogl_type_mapper < int > { typedef GLint ogl_type; };
    template<> struct ogl_type_mapper < unsigned int > { typedef GLuint ogl_type; };
    template<> struct ogl_type_mapper < short > { typedef GLshort ogl_type; };
    template<> struct ogl_type_mapper < unsigned short > { typedef GLushort ogl_type; };
    template<> struct ogl_type_mapper < float > { typedef GLfloat ogl_type; };
    template<> struct ogl_type_mapper < double > { typedef GLdouble ogl_type; };
    template<> struct ogl_type_mapper < char > { typedef GLbyte ogl_type; };
    template<> struct ogl_type_mapper < unsigned char > { typedef GLubyte ogl_type; };

    //The following template is a wrapper used to convert OpenGL data types into corresponding type traits structure
    template<typename ogl_data_type, bool special = 0> struct ogl_type_traits;
    template<> struct ogl_type_traits < GLbyte > {
        typedef GLbyte ogl_type;
        typedef char iso_c_type;
        static const GLenum ogl_data_type_enum = GL_BYTE;
    };

    template<> struct ogl_type_traits < GLshort > {
        typedef GLshort ogl_type;
        typedef short iso_c_type;
        static const GLenum ogl_data_type_enum = GL_SHORT;
    };

    template<> struct ogl_type_traits < GLint > {
        typedef GLint ogl_type;
        typedef int iso_c_type;
        static const GLenum ogl_data_type_enum = GL_INT;
    };

    template<> struct ogl_type_traits < GLfloat > {
        typedef GLfloat ogl_type;
        typedef float iso_c_type;
        static const GLenum ogl_data_type_enum = GL_FLOAT;
    };

    template<> struct ogl_type_traits < GLdouble > {
        typedef GLdouble ogl_type;
        typedef double iso_c_type;
        static const GLenum ogl_data_type_enum = GL_DOUBLE;
    };

    template<> struct ogl_type_traits < GLubyte > {
        typedef GLubyte ogl_type;
        typedef unsigned char iso_c_type;
        static const GLenum ogl_data_type_enum = GL_UNSIGNED_BYTE;
    };

    template<> struct ogl_type_traits < GLushort > {
        typedef GLushort ogl_type;
        typedef unsigned short iso_c_type;
        static const GLenum ogl_data_type_enum = GL_UNSIGNED_SHORT;
    };

    template<> struct ogl_type_traits < GLuint > {
        typedef GLuint ogl_type;
        typedef unsigned int iso_c_type;
        static const GLenum ogl_data_type_enum = GL_UNSIGNED_INT;
    };

    template<> struct ogl_type_traits < GLhalf, true > {
        typedef GLhalf ogl_type;
        typedef unsigned short iso_c_type;
        static const GLenum ogl_data_type_enum = GL_HALF_FLOAT;
    };

    template<> struct ogl_type_traits < GLfixed, true > {
        typedef GLfixed ogl_type;
        typedef int iso_c_type;
        static const GLenum ogl_data_type_enum = GL_FIXED;
    };

    //Implements triplet as generalization of std::pair yet more convenient then std::tuple
    template<typename T1, typename T2, typename T3> struct triplet
    {
        typedef T1 value_type_1;
        typedef T2 value_type_2;
        typedef T3 value_type_3;

        T1 first;
        T2 second;
        T3 third;
    };

    //Viewport descriptor
    struct Rectangle{
        float x, y, w, h;
    };

    //Type wrapper over OpenGL front and back face definition constants
    enum class Face : GLenum
    {
        FRONT = GL_FRONT,
        BACK = GL_BACK,
        FRONT_AND_BACK = GL_FRONT_AND_BACK
    };


    //Enumerates possible multi-sample pixel formats supported by the TinyWorld engine
    //Not all of these formats may be supported by OpenGL implementation provided by the video driver
    enum class MULTISAMPLING_MODE
    {
        MULTISAMPLING_NONE = 0,		//no multisampling is used (1 sample per pixel)
        MULTISAMPLING_2X = 2,		//2 samples per pixel
        MULTISAMPLING_4X = 4,		//4 samples per pixel
        MULTISAMPLING_8X = 8,		//8 samples per pixel
        MULTISAMPLING_16X = 16		//16 samples per pixel
    };

    //Enumerates frame references used to define rotation transforms
    enum class RotationFrame{ LOCAL, GLOBAL };


    //Enumerates standard object rendering modes
    #define TW_RENDERING_MODE_DEFAULT            0		//draws object using default rendering mode. Most objects are required to support this mode.
    #define TW_RENDERING_MODE_WIREFRAME          1		//draws object using "wire-frame" mode. Most objects are required to support this mode.
    #define TW_RENDERING_MODE_WIREFRAME_COMBINED 2		//combines TW_RENDERING_MODE_DEFAULT and TW_RENDERING_MODE_WIREFRAME_COMBINED into single rendering request.
    #define TW_RENDERING_MODE_SILHOUETTE		 3		//draws dark silhouette of the object. This mode is mainly used to render crepuscular  rays in post-processing pass.
    #define TW_RENDERING_MODE_RESERVED          10      //number of reserved rendering mode constants. Any constant greater than this value can be used by extensions




    //The following namespace contains constant used by atmospheric scattering computations
    namespace atmospheric_scattering_constants
    {
        const float planet_radius = 40.0f / 9.0f;	//radius of the planet
        const float sky_sphere_radius = 41.0f / 9.0f;	//radius of the sky sphere
        const float horizon_angle = std::asin(40.0f / 41.0f);	//angle from equator of the planet to the edge of horizon
        const float length_scale = 9.0f;	//length scale assumed by the ray tracers in scattering computations
        const float fH0 = 0.25f;	//non-dimensional height at which atmosphere has its average density
    }


    //Enumerates some standard return codes
    #define TW_INVALID_RETURN_VALUE 0xFFFFFFFF	//constant encoding an invalid value returned by some functions on failure
}



#define TW__TINY_WORLD_MISC__
#endif