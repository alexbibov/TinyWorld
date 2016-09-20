#include "SharedBuffer.h"

using namespace tiny_world;


SharedBuffer::SharedBuffer(BufferBindingTarget target /* = BufferBindingTarget::ARRAY */,
	BufferUsage usage_pattern /* = BufferUsage::DRAW */,
	BufferUsageFrequency usage_frequency_pattern /* = BufferUsageFrequency::STATIC */) :
	p_buffer{ new Buffer{ target, usage_pattern, usage_frequency_pattern } }, ref_counter{ new uint32_t{ 1 } }
{

}


SharedBuffer::SharedBuffer(size_t size, BufferBindingTarget target /* = BufferBindingTarget::ARRAY */,
	BufferUsage usage_pattern /* = BufferUsage::DRAW */,
	BufferUsageFrequency usage_frequency_pattern /* = BufferUsageFrequency::STATIC */, const void* data /* = nullptr */) :
	p_buffer{ new Buffer{ size, target, usage_pattern, usage_frequency_pattern, data } }, ref_counter{ new uint32_t{ 1 } }
{

}


SharedBuffer::SharedBuffer(Buffer* p_buffer) : p_buffer{ p_buffer }, ref_counter{ new uint32_t{ 1 } }
{

}


SharedBuffer::SharedBuffer(decltype(nullptr) value) : p_buffer{ nullptr }, ref_counter{ nullptr }
{

}


SharedBuffer::SharedBuffer(const SharedBuffer& other)
{
	p_buffer = other.p_buffer;
	if (p_buffer)
	{
		ref_counter = other.ref_counter;
		++(*ref_counter);
	}
	else
	{
		ref_counter = nullptr;
	}
}


SharedBuffer::SharedBuffer(SharedBuffer&& other)
{
	p_buffer = other.p_buffer;
	ref_counter = other.ref_counter;
	if(ref_counter) ++(*ref_counter);
}


SharedBuffer::~SharedBuffer()
{
	if (ref_counter && !(--(*ref_counter)))
	{
		delete p_buffer;
		delete ref_counter;
	}
}


SharedBuffer& SharedBuffer::operator =(const SharedBuffer& other)
{
	if (this == &other)
		return *this;

	if (ref_counter && !(--(*ref_counter)))
	{
		delete p_buffer;
		delete ref_counter;
	}

	p_buffer = other.p_buffer;
	ref_counter = other.ref_counter;
	if (ref_counter) ++(*ref_counter);

	return *this;
}


SharedBuffer& SharedBuffer::operator=(SharedBuffer&& other)
{
	if (this == &other)
		return *this;

	std::swap(p_buffer, other.p_buffer);
	std::swap(ref_counter, other.ref_counter);

	return *this;
}


SharedBuffer& SharedBuffer::operator=(Buffer* p_buffer)
{
	if (ref_counter && !(--(*ref_counter))) delete p_buffer;

	if (!(*ref_counter)) *ref_counter = 1;
	else ref_counter = new uint32_t{ 1 };

	this->p_buffer = p_buffer;

	return *this;
}


SharedBuffer& SharedBuffer::operator=(decltype(nullptr) value)
{
	if (ref_counter && !(--(*ref_counter)))
	{
		delete p_buffer;
		delete ref_counter;
	}

	p_buffer = nullptr;
	ref_counter = nullptr;

	return *this;
}


Buffer* SharedBuffer::operator ->() const{ return p_buffer; }

Buffer& SharedBuffer::operator*() const { return *p_buffer; }