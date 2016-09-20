#include "Buffer.h"

using namespace tiny_world;




//Returns short textual description of the buffer usage flag
const char* getBufferUsageString(GLenum buffer_usage_flag)
{
	switch (buffer_usage_flag)
	{
	case GL_STREAM_DRAW: return "StreamDraw";
	case GL_STREAM_READ: return "StreamRead";
	case GL_STREAM_COPY: return "StreamCopy";
	case GL_STATIC_DRAW: return "StaticDraw";
	case GL_STATIC_READ: return "StaticRead";
	case GL_STATIC_COPY: return "StaticCopy";
	case GL_DYNAMIC_DRAW: return "DynamicDraw";
	case GL_DYNAMIC_READ: return "DynamicRead";
	case GL_DYNAMIC_COPY: return "DynamicCopy";
	default: return "Unknown";
	}
}

//Returns short textual description of the buffer type depending on the given buffer binding point
const char* getBufferBindingString(BufferBindingTarget buffer_binding_target)
{
	switch (buffer_binding_target)
	{
	case BufferBindingTarget::ARRAY: return "Array";
	case BufferBindingTarget::COPY_READ: return "CopyRead";
	case BufferBindingTarget::COPY_WRITE: return "CopyWrite";
	case BufferBindingTarget::DISPATCH_INDIRECT: return "DispatchIndirect";
	case BufferBindingTarget::DRAW_INDIRECT: return "DrawIndirect";
	case BufferBindingTarget::ELEMENT_ARRAY: return "ElementArray";
	case BufferBindingTarget::PIXEL_PACK: return "PixelPack";
	case BufferBindingTarget::PIXEL_UNPACK: return "PixelUnpack";
	case BufferBindingTarget::QUERY: return "Query";
	case BufferBindingTarget::TEXTURE: return "Texture";
	case BufferBindingTarget::GENERIC_TRANSFORM_FEEDBACK: return "GenericTransformFeedback";
	case BufferBindingTarget::GENERIC_SHADER_STORAGE: return "GenericShaderStorage";
	case BufferBindingTarget::GENERIC_UNIFORM: return "GenericUniform";
	default: return "Unknown";
	}
}




GLenum Buffer::retrieveUsageFlag(BufferUsageFrequency usage_frequency_pattern, BufferUsage usage_pattern)
{
	GLenum usage;
	switch (usage_frequency_pattern)
	{
	case BufferUsageFrequency::STREAM:
		switch (usage_pattern)
		{
		case BufferUsage::DRAW:
			usage = GL_STREAM_DRAW;
			break;

		case BufferUsage::READ:
			usage = GL_STREAM_READ;
			break;

		case BufferUsage::COPY:
			usage = GL_STREAM_COPY;
			break;
		}
		break;


	case BufferUsageFrequency::STATIC:
		switch (usage_pattern)
		{
		case BufferUsage::DRAW:
			usage = GL_STATIC_DRAW;
			break;

		case BufferUsage::READ:
			usage = GL_STATIC_READ;
			break;

		case BufferUsage::COPY:
			usage = GL_STATIC_COPY;
			break;
		}
		break;


	case BufferUsageFrequency::DYNAMIC:
		switch (usage_pattern)
		{
		case BufferUsage::DRAW:
			usage = GL_DYNAMIC_DRAW;
			break;

		case BufferUsage::READ:
			usage = GL_DYNAMIC_READ;
			break;

		case BufferUsage::COPY:
			usage = GL_DYNAMIC_COPY;
			break;
		}
		break;
	}

	return usage;
}

GLenum Buffer::retrieveBindingInfo(BufferBindingTarget target)
{
	switch (target)
	{
	case BufferBindingTarget::ARRAY: return GL_ARRAY_BUFFER_BINDING;
	case BufferBindingTarget::COPY_READ: return GL_COPY_READ_BUFFER_BINDING;
	case BufferBindingTarget::COPY_WRITE: return GL_COPY_WRITE_BUFFER_BINDING;
	case BufferBindingTarget::DISPATCH_INDIRECT: return GL_DISPATCH_INDIRECT_BUFFER_BINDING;
	case BufferBindingTarget::DRAW_INDIRECT: return GL_DRAW_INDIRECT_BUFFER_BINDING;
	case BufferBindingTarget::ELEMENT_ARRAY: return GL_ELEMENT_ARRAY_BUFFER_BINDING;
	case BufferBindingTarget::PIXEL_PACK: return GL_PIXEL_PACK_BUFFER_BINDING;
	case BufferBindingTarget::PIXEL_UNPACK: return GL_PIXEL_UNPACK_BUFFER_BINDING;
	case BufferBindingTarget::QUERY: return GL_QUERY_BUFFER_BINDING;
	case BufferBindingTarget::TEXTURE: return GL_TEXTURE_BUFFER_BINDING;
	case BufferBindingTarget::GENERIC_TRANSFORM_FEEDBACK: return GL_TRANSFORM_FEEDBACK_BUFFER_BINDING;
	case BufferBindingTarget::GENERIC_SHADER_STORAGE: return GL_SHADER_STORAGE_BUFFER_BINDING;
	case BufferBindingTarget::GENERIC_UNIFORM: return GL_UNIFORM_BUFFER_BINDING;
	default: return 0xFFFFFFFF;
	}
}

bool Buffer::isIndexedTarget(BufferBindingTarget target) const
{
	switch (target)
	{
	case BufferBindingTarget::ARRAY:
	case BufferBindingTarget::COPY_READ:
	case BufferBindingTarget::COPY_WRITE:
	case BufferBindingTarget::DISPATCH_INDIRECT:
	case BufferBindingTarget::DRAW_INDIRECT:
	case BufferBindingTarget::ELEMENT_ARRAY:
	case BufferBindingTarget::PIXEL_PACK:
	case BufferBindingTarget::PIXEL_UNPACK:
	case BufferBindingTarget::QUERY:
	case BufferBindingTarget::TEXTURE:
		return false;

	case BufferBindingTarget::GENERIC_TRANSFORM_FEEDBACK:
	case BufferBindingTarget::GENERIC_SHADER_STORAGE:
	case BufferBindingTarget::GENERIC_UNIFORM:
		return true;

	default: return false;
	}
}


Buffer::Buffer(BufferBindingTarget target /* = BufferBindingTarget::ARRAY */, 
	BufferUsage usage_pattern /* = BufferUsage::DRAW */, 
	BufferUsageFrequency usage_frequency_pattern /* = BufferUsageFrequency::STATIC */) : 

	Entity(getBufferBindingString(target) + std::string("Buffer(") + getBufferUsageString(Buffer::retrieveUsageFlag(usage_frequency_pattern, usage_pattern)) + ")"),

	buffer_size{ 0 }, target{ target }, 
	usage_frequency{ usage_frequency_pattern }, usage_type{ usage_pattern },
	is_initialized{ false }
{
	glGenBuffers(1, &ogl_buffer_id);
}


Buffer::Buffer(const Buffer& other) : Entity(other),
buffer_size{ other.buffer_size },
target{ other.target }, usage_frequency{ other.usage_frequency }, usage_type{ other.usage_type },
is_initialized{ other.is_initialized }
{
	//Generate new buffer object
	glGenBuffers(1, &ogl_buffer_id);

	//If source buffer has been initialized, create storage for the newly created buffer and populate it with data from the source buffer
	if (other.is_initialized)
	{
		GLenum target_binding = retrieveBindingInfo(target);
		GLint currently_bound_buffer;
		glGetIntegerv(target_binding, &currently_bound_buffer);

		if (currently_bound_buffer != ogl_buffer_id) 
			glBindBuffer(static_cast<GLenum>(target), ogl_buffer_id);

		glBufferData(static_cast<GLenum>(target), other.buffer_size, NULL, retrieveUsageFlag(usage_frequency, usage_type));

		if (currently_bound_buffer != ogl_buffer_id) 
			glBindBuffer(static_cast<GLenum>(target), currently_bound_buffer);



		GLint currently_bound_read_buffer, currently_bound_write_buffer;
		glGetIntegerv(GL_COPY_READ_BUFFER_BINDING, &currently_bound_read_buffer);
		glGetIntegerv(GL_COPY_WRITE_BUFFER_BINDING, &currently_bound_write_buffer);

		if (currently_bound_write_buffer != ogl_buffer_id)
			glBindBuffer(GL_COPY_WRITE_BUFFER, ogl_buffer_id);
		if (currently_bound_read_buffer != other.ogl_buffer_id)
			glBindBuffer(GL_COPY_READ_BUFFER, other.ogl_buffer_id);

		glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, other.buffer_size);

		if (currently_bound_write_buffer != ogl_buffer_id)
			glBindBuffer(GL_COPY_WRITE_BUFFER, currently_bound_write_buffer);
		if (currently_bound_read_buffer != other.ogl_buffer_id)
			glBindBuffer(GL_COPY_READ_BUFFER, currently_bound_read_buffer);
	}
}


Buffer::Buffer(Buffer&& other) : Entity(std::move(other)),
ogl_buffer_id{ other.ogl_buffer_id },
buffer_size{ other.buffer_size }, target{ std::move(other.target) }, usage_frequency{ std::move(other.usage_frequency) },
usage_type{ std::move(other.usage_type) }, is_initialized{ other.is_initialized }
{
	other.ogl_buffer_id = 0;
}


Buffer::~Buffer()
{
	if (ogl_buffer_id)
		glDeleteBuffers(1, &ogl_buffer_id);
}


Buffer& Buffer::operator=(const Buffer& other)
{
	//Account for the special case, when the object gets assigned to itself
	if (this == &other)
		return *this;

	Entity::operator=(other);

	//Copy buffer state from the source object
	buffer_size = other.buffer_size;
	target = other.target;
	usage_frequency = other.usage_frequency;
	usage_type = other.usage_type;
	is_initialized = other.is_initialized;

	//If the source buffer has been initialized, copy data from the source to the destination buffer
	if (other.is_initialized)
	{
		GLenum target_binding = retrieveBindingInfo(target);
		GLint currently_bound_buffer;
		glGetIntegerv(target_binding, &currently_bound_buffer);

		if (currently_bound_buffer != ogl_buffer_id)
			glBindBuffer(static_cast<GLenum>(target), ogl_buffer_id);

		glBufferData(static_cast<GLenum>(target), other.buffer_size, NULL, retrieveUsageFlag(usage_frequency, usage_type));

		if (currently_bound_buffer != ogl_buffer_id)
			glBindBuffer(static_cast<GLenum>(target), currently_bound_buffer);



		GLint currently_bound_read_buffer, currently_bound_write_buffer;
		glGetIntegerv(GL_COPY_READ_BUFFER_BINDING, &currently_bound_read_buffer);
		glGetIntegerv(GL_COPY_WRITE_BUFFER_BINDING, &currently_bound_write_buffer);

		if (currently_bound_write_buffer != ogl_buffer_id)
			glBindBuffer(GL_COPY_WRITE_BUFFER, ogl_buffer_id);
		if (currently_bound_read_buffer != other.ogl_buffer_id)
			glBindBuffer(GL_COPY_READ_BUFFER, other.ogl_buffer_id);

		glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, other.buffer_size);

		if (currently_bound_write_buffer != ogl_buffer_id)
			glBindBuffer(GL_COPY_WRITE_BUFFER, currently_bound_write_buffer);
		if (currently_bound_read_buffer != other.ogl_buffer_id)
			glBindBuffer(GL_COPY_READ_BUFFER, currently_bound_read_buffer);
	}

	return *this;
}


Buffer& Buffer::operator=(Buffer&& other)
{
	//Account for the special case, when the object gets assigned to itself
	if (this == &other)
		return *this;

	Entity::operator=(std::move(other));

	//Move buffer state from the source object

	ogl_buffer_id = other.ogl_buffer_id;
	other.ogl_buffer_id = 0;

	buffer_size = other.buffer_size;
	target = std::move(other.target);
	usage_frequency = std::move(other.usage_frequency);
	usage_type = std::move(other.usage_type);
	is_initialized = other.is_initialized;

	return *this;
}


bool Buffer::operator==(const Buffer& other) const { return ogl_buffer_id == other.ogl_buffer_id; }

bool Buffer::operator!=(const Buffer& other) const { return ogl_buffer_id != other.ogl_buffer_id; }


Buffer::operator GLint() const { return ogl_buffer_id; }


Buffer::Buffer(size_t size, 
	BufferBindingTarget target /* = BufferBindingTarget::ARRAY */, 
	BufferUsage usage_pattern /* = BufferUsage::DRAW */, 
	BufferUsageFrequency usage_frequency_pattern /* = BufferUsageFrequency::STATIC */, 
	const void* data /* = nullptr */) : 
	Entity(getBufferBindingString(target) + std::string("Buffer(") + getBufferUsageString(Buffer::retrieveUsageFlag(usage_frequency_pattern, usage_pattern)) + ")"),
	buffer_size{ size }, target{ target }, 
	usage_type{ usage_pattern }, usage_frequency{ usage_frequency_pattern }, 
	is_initialized{ true }
{
	glGenBuffers(1, &ogl_buffer_id);

	GLenum target_binding = retrieveBindingInfo(target);
	GLint currently_bound_buffer;
	glGetIntegerv(target_binding, &currently_bound_buffer);

	if (currently_bound_buffer != ogl_buffer_id)
		glBindBuffer(static_cast<GLenum>(target), ogl_buffer_id);

	glBufferData(static_cast<GLenum>(target), buffer_size, data, retrieveUsageFlag(usage_frequency_pattern, usage_pattern));

	if (currently_bound_buffer != ogl_buffer_id)
		glBindBuffer(static_cast<GLenum>(target), currently_bound_buffer);
}


void Buffer::allocate(size_t size, const void* data /* = nullptr */)
{
	buffer_size = size;

	GLenum target_binding = retrieveBindingInfo(target);
	GLint currently_bound_buffer;
	glGetIntegerv(target_binding, &currently_bound_buffer);

	if (ogl_buffer_id != currently_bound_buffer)
		glBindBuffer(static_cast<GLenum>(target), ogl_buffer_id);

	glBufferData(static_cast<GLenum>(target), size, data, retrieveUsageFlag(usage_frequency, usage_type));

	if (ogl_buffer_id != currently_bound_buffer)
		glBindBuffer(static_cast<GLenum>(target), currently_bound_buffer);

	is_initialized = true;
}


void Buffer::setSubData(ptrdiff_t data_chunk_offset, size_t data_chunk_size, const void* data)
{
	if (!is_initialized)
	{
		set_error_state(true);
		const char* err_msg = "Unable to update buffer data: the buffer has not been  initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	GLenum target_binding = retrieveBindingInfo(target);
	GLint currently_bound_buffer;
	glGetIntegerv(target_binding, &currently_bound_buffer);

	if (currently_bound_buffer != ogl_buffer_id)
		glBindBuffer(static_cast<GLenum>(target), ogl_buffer_id);

	glBufferSubData(static_cast<GLenum>(target), data_chunk_offset, data_chunk_size, data);

	if (currently_bound_buffer != ogl_buffer_id)
		glBindBuffer(static_cast<GLenum>(target), currently_bound_buffer);
}


void* Buffer::map(BufferAccess access_policy) const
{
	void* rv = nullptr;

	GLenum target_binding = retrieveBindingInfo(target);
	GLint currently_bound_buffer;
	glGetIntegerv(target_binding, &currently_bound_buffer);

	if (currently_bound_buffer != ogl_buffer_id)
		glBindBuffer(static_cast<GLenum>(target), ogl_buffer_id);

	rv = glMapBuffer(static_cast<GLenum>(target), static_cast<GLenum>(access_policy));

	if (currently_bound_buffer != ogl_buffer_id)
		glBindBuffer(static_cast<GLenum>(target), currently_bound_buffer);

	return rv;
}


void* Buffer::map(ptrdiff_t range_offset, size_t range_size, BufferRangeAccess access_policy) const
{
	void* rv = nullptr;

	GLenum target_binding = retrieveBindingInfo(target);
	GLint currently_bound_buffer;
	glGetIntegerv(target_binding, &currently_bound_buffer);

	if (currently_bound_buffer != ogl_buffer_id)
		glBindBuffer(static_cast<GLenum>(target), ogl_buffer_id);

	rv = glMapBufferRange(static_cast<GLenum>(target), range_offset, range_size, static_cast<GLenum>(access_policy.to_ulong()));

	if (currently_bound_buffer != ogl_buffer_id)
		glBindBuffer(static_cast<GLenum>(target), currently_bound_buffer);

	return rv;
}


void Buffer::unmap() const
{
	GLenum target_binding = retrieveBindingInfo(target);
	GLint currently_bound_buffer;
	glGetIntegerv(target_binding, &currently_bound_buffer);

	if (currently_bound_buffer != ogl_buffer_id)
		glBindBuffer(static_cast<GLenum>(target), ogl_buffer_id);

	glUnmapBuffer(static_cast<GLenum>(target));

	if (currently_bound_buffer != ogl_buffer_id)
		glBindBuffer(static_cast<GLenum>(target), currently_bound_buffer);
}

void Buffer::bind() const
{
	glBindBuffer(static_cast<GLenum>(target), ogl_buffer_id);
}

void Buffer::bind(BufferBindingTarget target) const
{
	glBindBuffer(static_cast<GLenum>(target), ogl_buffer_id);
}

void Buffer::bind(uint32_t binding_point) const
{
	if (!isIndexedTarget(target))
	{
		set_error_state(true);
		const char* err_msg = "Unable to bind buffer to an indexed target: default buffer binding point is not one of the generic binding points for the indexed targets";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	glBindBufferBase(static_cast<GLenum>(target), static_cast<GLuint>(binding_point), ogl_buffer_id);
}

void Buffer::bind(BufferIndexedBindingTarget target, uint32_t binding_point) const
{
	glBindBufferBase(static_cast<GLenum>(target), static_cast<GLuint>(binding_point), ogl_buffer_id);
}


size_t Buffer::getSize() const { return buffer_size; }


BufferBindingTarget Buffer::getDefaultBindingTarget() const{ return target; }


BufferUsage Buffer::getUsagePattern() const { return usage_type; }


BufferUsageFrequency Buffer::getUsageFrequencyPattern() const { return usage_frequency; }


bool Buffer::isInitialized() const { return is_initialized; }

