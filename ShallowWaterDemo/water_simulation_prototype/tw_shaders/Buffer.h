//Implements a thin wrapper over OpenGL memory buffer

#ifndef TW__BUFFER__

#include <cstdint>
#include <GL/glew.h>

namespace tiny_world
{
	//Buffer access frequency patterns
	enum class BUFFER_ACCESS_FREQUENCY
	{
		STREAM, STATIC, DYNAMIC
	};

	//Buffer access type patterns
	enum class BUFFER_ACCESS_TYPE
	{
		DRAW, READ, COPY
	};

	//This is the base class for all kinds of buffers
	class Buffer
	{
	private:
		static uint32_t id;				//internal identifier of the buffer
		GLint ogl_buffer_identifier;	//internal OpenGL identifier of the buffer
		size_t buffer_size;				//size of the buffer represented in bytes
		BUFFER_ACCESS_FREQUENCY access_frequency;	//access frequency pattern used by buffer
		BUFFER_ACCESS_TYPE access_type;	//access type defined for the buffer
		bool is_initialized;			//equals 'true' if buffer has been initialized, equals 'false' otherwise

	public:
		Buffer();	//default initializer

	};
}

#define TW__BUFFER__
#endif