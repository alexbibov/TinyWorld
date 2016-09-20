#include "std140UniformBuffer.h"
#include <cinttypes>

using namespace tiny_world;



std140UniformBuffer::std140UniformBuffer() : Buffer{ BufferBindingTarget::GENERIC_UNIFORM, BufferUsage::DRAW, BufferUsageFrequency::DYNAMIC },
offset{ 0 }, binding_point{ 0 }
{

}


std140UniformBuffer::std140UniformBuffer(uint32_t binding_point) : 
Buffer{ BufferBindingTarget::GENERIC_UNIFORM, BufferUsage::DRAW, BufferUsageFrequency::DYNAMIC },
offset{ 0 }, binding_point{ binding_point }
{

}


std140UniformBuffer::std140UniformBuffer(size_t buffer_size, uint32_t binding_point) : 
Buffer{ buffer_size, BufferBindingTarget::GENERIC_UNIFORM, BufferUsage::DRAW, BufferUsageFrequency::DYNAMIC, nullptr }, 
offset{ 0 }, binding_point{ binding_point }
{

}


void std140UniformBuffer::pushScalar(bool boolean_scalar)
{
	int converted_boolean_scalar = static_cast<int>(boolean_scalar);
	pushScalar(converted_boolean_scalar);
}

void std140UniformBuffer::pushScalar(const std::vector<bool>& boolean_scalars)
{
	const ptrdiff_t aligned_offset = ((offset >> 4) << 4) + 16;
	const size_t array_size = 16 * boolean_scalars.size();

	void* std140_boolean_array = malloc(array_size);
	for (unsigned int i = 0; i < boolean_scalars.size(); ++i)
	{
		*(static_cast<GLint*>(std140_boolean_array)+4 * i) =
			static_cast<GLint>(boolean_scalars[i]);
	}


	setSubData(aligned_offset, array_size, std140_boolean_array);
	free(std140_boolean_array);
	offset = aligned_offset + array_size;
}


void std140UniformBuffer::pushVector(const bvec4& vector)
{
	pushVector(static_cast<ivec4>(vector));
}

void std140UniformBuffer::pushVector(const bvec3& vector)
{
	pushVector(static_cast<ivec3>(vector));
}

void std140UniformBuffer::pushVector(const bvec2& vector)
{
	pushVector(static_cast<ivec2>(vector));
}

void std140UniformBuffer::pushVector(const std::vector<bvec4>& vectors)
{
	std::vector<ivec4> converted_vectors;
	std::for_each(vectors.begin(), vectors.end(),
		[&converted_vectors](const bvec4& elem) -> void
	{
		converted_vectors.push_back(static_cast<ivec4>(elem));
	});

	pushVector(converted_vectors);
}

void std140UniformBuffer::pushVector(const std::vector<bvec3>& vectors)
{
	std::vector<ivec3> converted_vectors;
	std::for_each(vectors.begin(), vectors.end(),
		[&converted_vectors](const bvec3& elem) -> void
	{
		converted_vectors.push_back(static_cast<ivec3>(elem));
	});

	pushVector(converted_vectors);
}

void std140UniformBuffer::pushVector(const std::vector<bvec2>& vectors)
{
	std::vector<ivec2> converted_vectors;
	std::for_each(vectors.begin(), vectors.end(),
		[&converted_vectors](const bvec2& elem) -> void
	{
		converted_vectors.push_back(static_cast<ivec2>(elem));
	});

	pushVector(converted_vectors);
}

void std140UniformBuffer::resetOffsetCounter() { offset = 0; }


void std140UniformBuffer::setBindingPoint(uint32_t binding_point) { this->binding_point = binding_point; };

uint32_t std140UniformBuffer::getBindingPoint() const { return binding_point; }

void std140UniformBuffer::bind() const
{
	Buffer::bind(BufferIndexedBindingTarget::UNIFORM, binding_point);
}
