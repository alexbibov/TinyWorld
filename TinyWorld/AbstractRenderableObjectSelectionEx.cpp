#include "AbstractRenderableObjectSelectionEx.h"

using namespace tiny_world;


bool AbstractRenderableObjectSelectionEx::injectExtension(const ShaderProgramReferenceCode& program_ref_code, std::initializer_list<PipelineStage> program_stages)
{
	//The extension implementing selection buffer is only available for programs comprising fragment shader stage
	if (std::find(program_stages.begin(), program_stages.end(), PipelineStage::FRAGMENT_SHADER) != program_stages.end())
	{
		ShaderProgram* p_shader_program = retrieveShaderProgram(program_ref_code);

		static const char* selection_extension_shader_source =
			"#version 430 core\n"
			"uniform uvec2 uv2GeometricalObjectId;\n"
			"layout(location = 5) out vec4 v4SelectionBufferOutputValue;\n"
			"void updateSelectionBuffer()\n"
			"{\n"
			"	v4SelectionBufferOutputValue = vec4(vec2(uv2GeometricalObjectId), 0, 0);\n"
			"}";
		Shader selection_extension_shader{ GLSLSourceCode{ selection_extension_shader_source, std::strlen(selection_extension_shader_source) }, 
			ShaderType::FRAGMENT_SHADER, "AbstractRenderableObjectSelectionEx::selection_extension_shader" };
		if (!selection_extension_shader) return false;

		p_shader_program->addShader(selection_extension_shader);
		modified_program_ref_code_list.push_back(program_ref_code);
	}

	return true;
}


void AbstractRenderableObjectSelectionEx::applyExtension()
{
	//assign values for the shader uniforms
	unsigned long long this_object_id = getId();
	uvec2 uv2ScreenSize = getScreenSize();
	std::for_each(modified_program_ref_code_list.begin(), modified_program_ref_code_list.end(), 
		[this, this_object_id, &uv2ScreenSize](const ShaderProgramReferenceCode& program_ref_code) -> void
	{
		retrieveShaderProgram(program_ref_code)->assignUniformVector("uv2GeometricalObjectId", uvec2{ static_cast<uint32_t>(this_object_id), static_cast<uint32_t>(this_object_id >> 32) });
	});
}


void AbstractRenderableObjectSelectionEx::applyViewerTransform(const AbstractProjectingDevice& projecting_device)
{

}


void AbstractRenderableObjectSelectionEx::releaseExtension()
{

}


AbstractRenderableObjectSelectionEx::AbstractRenderableObjectSelectionEx() 
{

}


AbstractRenderableObjectSelectionEx::AbstractRenderableObjectSelectionEx(const AbstractRenderableObjectSelectionEx& other) :
modified_program_ref_code_list(other.modified_program_ref_code_list)
{

}


AbstractRenderableObjectSelectionEx::AbstractRenderableObjectSelectionEx(AbstractRenderableObjectSelectionEx&& other) : 
modified_program_ref_code_list(std::move(other.modified_program_ref_code_list))
{

}


AbstractRenderableObjectSelectionEx::~AbstractRenderableObjectSelectionEx()
{

}


AbstractRenderableObjectSelectionEx& AbstractRenderableObjectSelectionEx::operator=(const AbstractRenderableObjectSelectionEx& other)
{
	if (this == &other)
		return *this;

	modified_program_ref_code_list = other.modified_program_ref_code_list;

	return *this;
}


AbstractRenderableObjectSelectionEx& AbstractRenderableObjectSelectionEx::operator=(AbstractRenderableObjectSelectionEx&& other)
{
	if (this == &other)
		return *this;

	modified_program_ref_code_list = std::move(other.modified_program_ref_code_list);

	return *this;
}


unsigned long long AbstractRenderableObjectSelectionEx::getPointSelectionObjectId(const Framebuffer& target_framebuffer, const uvec2& uv2ScreenPoint)
{
	FramebufferAttachmentInfo selection_buffer_attachment_info = target_framebuffer.retrieveAttachmentDetails(FramebufferAttachmentPoint::COLOR5);
	PixelDataType selection_buffer_data_type = selection_buffer_attachment_info.texture->getStorageFormatTraits().getOptimalStorageType();
	
	//Retrieve strong identifier of the object covering the requested point of the screen
	void* values = new char[12];
	target_framebuffer.readPixels(RenderingColorBuffer::COLOR5, uv2ScreenPoint.x, uv2ScreenPoint.y, 1, 1, PixelReadLayout::RGB, selection_buffer_data_type, values);
	
	uint32_t words[2];
	switch (selection_buffer_data_type)
	{
	case PixelDataType::FLOAT:
		words[0] = static_cast<uint32_t>(reinterpret_cast<GLfloat*>(values)[0]);
		words[1] = static_cast<uint32_t>(reinterpret_cast<GLfloat*>(values)[1]);
		break;

	case PixelDataType::INT:
		words[0] = static_cast<uint32_t>(reinterpret_cast<GLint*>(values)[0]);
		words[1] = static_cast<uint32_t>(reinterpret_cast<GLint*>(values)[1]);
		break;

	case PixelDataType::UINT:
		words[0] = static_cast<uint32_t>(reinterpret_cast<GLuint*>(values)[0]);
		words[1] = static_cast<uint32_t>(reinterpret_cast<GLuint*>(values)[1]);
		break;

	default:
		delete[] values;
		return 0;
	}

	delete[] values;
	return (static_cast<unsigned long long>(words[1]) << 32) + words[0];
}


