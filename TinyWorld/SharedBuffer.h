#ifndef TW__SHARED_BUFFER__

#include "Buffer.h"

namespace tiny_world
{
	//Implements a wrapper over the Buffer object which provides shared ownership
	class SharedBuffer
	{
	private:
		Buffer* p_buffer;		//pointer to the buffer object owned by the shared buffer
		uint32_t* ref_counter;	//reference counter

	public:
		//Initializes the shared buffer and the buffer object owned by it using provided target and the usage hints
		SharedBuffer(BufferBindingTarget target = BufferBindingTarget::ARRAY, 
			BufferUsage usage_pattern = BufferUsage::DRAW, 
			BufferUsageFrequency usage_frequency_pattern = BufferUsageFrequency::STATIC);

		//Initializes the shared buffer and the buffer object which it owns using provided binding target and the usage hints. The buffer object owned by the shared 
		//buffer gets preallocated storage of the given size and populates it with data from the source supplied by the caller
		SharedBuffer(size_t size, BufferBindingTarget target = BufferBindingTarget::ARRAY,
			BufferUsage usage_pattern = BufferUsage::DRAW,
			BufferUsageFrequency usage_frequency_pattern = BufferUsageFrequency::STATIC, const void* data = nullptr);

		//Initializes the shared buffer using provided pointer to a Buffer object. Note that the Buffer object must reside on heap, not on stack and that
		//the ownership of the pointer must be fully delegated to the shared buffer being initialized by this pointer (i.e. the caller must keep no copies of this pointer to himself)
		SharedBuffer(Buffer* p_buffer);

		//Allows to initialize shared buffer using nullptr, which means that the owned buffer object will not be created
		SharedBuffer(decltype(nullptr) value);


		SharedBuffer(const SharedBuffer& other);	//copy constructor
		SharedBuffer(SharedBuffer&& other);		//move constructor
		~SharedBuffer();	//destructor

		SharedBuffer& operator=(const SharedBuffer& other);	//copy-assignment operator
		SharedBuffer& operator=(SharedBuffer&& other);	//move-assignment operator
		SharedBuffer& operator=(Buffer* p_buffer);	//allows to assign new Buffer object ownership to the shared buffer object. Note that the Buffer object pointer for which the ownership is transfered to the shared buffer must refer to the heap memory
		SharedBuffer& operator=(decltype(nullptr) value);	//allows to assign nullptr to the object. Releases ownership of the underlying Buffer object

		Buffer* operator->() const;		//provides access to the owned buffer object
		Buffer& operator*() const;	//provides "dereference" access to the owned buffer object
	};
}

#define TW__SHARED_BUFFER__
#endif