//Implements a thin wrapper over OpenGL memory buffer
//Note: future versions of TinyWorld with support of OpenGL 4.5 should also 
//implement immutable buffers, which may have performance impacts

#ifndef TW__BUFFER__

#include <cstdint>
#include <string>
#include <bitset>
#include <functional>
#include <GL/glew.h>

#include "ImmutableTexture.h"
#include "ErrorBehavioral.h"

namespace tiny_world
{
	//Buffer access frequency patterns
	enum class BufferUsageFrequency
	{
		STREAM, STATIC, DYNAMIC
	};

	//Buffer access type patterns
	enum class BufferUsage
	{
		DRAW, READ, COPY
	};

	//Access bits for mapped buffer ranges
	typedef std::bitset<sizeof(GLenum)*8> BufferRangeAccess;
	namespace BufferRangeAccessBits
	{
		const BufferRangeAccess READ{ GL_MAP_READ_BIT };
		const BufferRangeAccess WRITE{ GL_MAP_WRITE_BIT };
		const BufferRangeAccess PERSISTENT{ GL_MAP_PERSISTENT_BIT };
		const BufferRangeAccess COHERENT{ GL_MAP_COHERENT_BIT };
		const BufferRangeAccess INVALIDATE_RANGE{ GL_MAP_INVALIDATE_RANGE_BIT };
		const BufferRangeAccess INVALIDATE_BUFFER{ GL_MAP_INVALIDATE_BUFFER_BIT };
		const BufferRangeAccess FLUSH_EXPLICIT{ GL_MAP_FLUSH_EXPLICIT_BIT };
		const BufferRangeAccess UNSYNCHRONIZED{ GL_MAP_UNSYNCHRONIZED_BIT };
	}


	//Binding targets, to which a buffer object can be bound
	enum class BufferBindingTarget : GLenum
	{
		ARRAY = GL_ARRAY_BUFFER,
		COPY_READ = GL_COPY_READ_BUFFER,
		COPY_WRITE = GL_COPY_WRITE_BUFFER,
		DISPATCH_INDIRECT = GL_DISPATCH_INDIRECT_BUFFER,
		DRAW_INDIRECT = GL_DRAW_INDIRECT_BUFFER,
		ELEMENT_ARRAY = GL_ELEMENT_ARRAY_BUFFER,
		PIXEL_PACK = GL_PIXEL_PACK_BUFFER,
		PIXEL_UNPACK = GL_PIXEL_UNPACK_BUFFER,
		QUERY = GL_QUERY_BUFFER,
		TEXTURE = GL_TEXTURE_BUFFER,

		//The following are generic binding points for indexed binding targets
		GENERIC_TRANSFORM_FEEDBACK = GL_TRANSFORM_FEEDBACK_BUFFER,
		GENERIC_SHADER_STORAGE = GL_SHADER_STORAGE_BUFFER,
		GENERIC_UNIFORM = GL_UNIFORM_BUFFER
	};

	//Indexed binding targets, to which a buffer object can be bound
	enum class BufferIndexedBindingTarget : GLenum
	{
		TRANSFORM_FEEDBACK = GL_TRANSFORM_FEEDBACK_BUFFER,
		SHADER_STORAGE = GL_SHADER_STORAGE_BUFFER,
		UNIFORM = GL_UNIFORM_BUFFER
	};


	//This is the base class for all kinds of buffers
	class Buffer : public Entity
	{
		friend class BufferTexture_Core;
	private:
		GLuint ogl_buffer_id;	//internal OpenGL identifier of the buffer
		size_t buffer_size;				//size of the buffer represented in bytes
		BufferBindingTarget	target;		//binding target of the buffer	
		BufferUsageFrequency usage_frequency;	//usage frequency pattern used by buffer
		BufferUsage usage_type;			//usage type defined for the buffer
		bool is_initialized;			//equals 'true' if buffer has been initialized, equals 'false' otherwise

		//Helps to retrieve buffer usage flag compatible with OpenGL routines
		static GLenum retrieveUsageFlag(BufferUsageFrequency usage_frequency_pattern, BufferUsage usage_pattern);

		//Helper: retrieves GL_*_BINDING constant based on provided buffer binding target
		static GLenum retrieveBindingInfo(BufferBindingTarget target);

		//Helper: checks if provided binding target is a generic target for one of the indexed buffer binding targets
		bool isIndexedTarget(BufferBindingTarget target) const;

	public:
		//Default initializer
		Buffer(BufferBindingTarget target = BufferBindingTarget::ARRAY, 
			BufferUsage usage_pattern = BufferUsage::DRAW, 
			BufferUsageFrequency usage_frequency_pattern = BufferUsageFrequency::STATIC);

		//Creates new buffer allocates storage for it and optionally populates it with data.
		//If data = nullptr, the function just creates buffer and allocates storage for it, but
		//the contents of the storage are left undefined
		Buffer(size_t size, 
			BufferBindingTarget target = BufferBindingTarget::ARRAY, 
			BufferUsage usage_pattern = BufferUsage::DRAW,
			BufferUsageFrequency usage_frequency_pattern = BufferUsageFrequency::STATIC,
			const void* data = nullptr);

		//Copy constructor
		Buffer(const Buffer& other);

		//Move constructor
		Buffer(Buffer&& other);

		//Destructor
		virtual ~Buffer();

		//Copy-assignment operator (leads to reallocation of buffer storage)
		Buffer& operator=(const Buffer& other);

		//Move-assignment operator
		Buffer& operator=(Buffer&& other);

		//Returns 'true' if the objects being compared represent the same buffer
		bool operator==(const Buffer& other) const;

		//Yields 'true' if the objects being compared refer to different OpenGL buffers
		bool operator!=(const Buffer& other) const;


		//Returns internal OpenGL identifier of the buffer
		operator GLint() const;

		//Allocates new storage for the buffer and optionally populates it with data. If data = nullptr, the function just
		//allocates space for the storage but does not fill it with any data. If a storage has already been allocated, the function
		//erases the old storage and allocates new buffer taking into account provided access frequency and type patterns
		void allocate(size_t size, const void* data = nullptr);

		//Copies data chunk into the buffer. Note that storage must be allocated before using this function
		void setSubData(ptrdiff_t data_chunk_offset, size_t data_chunk_size, const void* data);

		//Maps the buffer onto client address space using specified access policy and returns a pointer to the start address of
		//the mapping. Note, that actual usage of the pointer by the client application may possibly break the specified access 
		//policy, but this in general could lead to significant performance degradation
		void* map(BufferAccess access_policy) const;

		//Maps PART of the buffer onto client address space using specified access policy and returns a pointer to the start 
		//address of the mapped range .Note, that actual usage of the pointer by client application may possibly break the 
		//specified access policy, but this in general could lead to significant performance penalties
		void* map(ptrdiff_t range_offset, size_t range_size, BufferRangeAccess access_bits) const;

		//Unmaps the buffer (or previously mapped buffer range) from the client space
		void unmap() const;

		//Binds buffer object to its default target identified on buffer creation
		void bind() const;

		//Binds the buffer object to the given target
		void bind(BufferBindingTarget target) const;

		//Binds buffer object to the given indexed target using the target identified on buffer creation as the base target
		void bind(uint32_t binding_point) const;

		//Binds the buffer object to the given indexed target
		void bind(BufferIndexedBindingTarget target, uint32_t binding_point) const;

		//Returns internal identifier of the buffer
		uint32_t getId() const;

		//Returns the size of the buffer
		size_t getSize() const;

		//Returns default binding target of the buffer
		BufferBindingTarget getDefaultBindingTarget() const;

		//Returns usage pattern of the buffer
		BufferUsage getUsagePattern() const;

		//Returns usage frequency pattern of the buffer
		BufferUsageFrequency getUsageFrequencyPattern() const;

		//Return 'true' if the buffer has been initialized. Returns 'false' otherwise
		bool isInitialized() const;
	};
}

#define TW__BUFFER__
#endif