#include "CompleteShaderProgram.h"

using namespace tiny_world;

CompleteShaderProgram::CompleteShaderProgram() : ShaderProgram("CompleteShaderProgram")
{

}

CompleteShaderProgram::CompleteShaderProgram(const std::string& program_string_name) : ShaderProgram("CompleteShaderProgram", program_string_name)
{

}

CompleteShaderProgram::CompleteShaderProgram(const CompleteShaderProgram& other) : ShaderProgram(other) 
{

}

CompleteShaderProgram::CompleteShaderProgram(CompleteShaderProgram&& other) : ShaderProgram(std::move(other))
{

}

CompleteShaderProgram::~CompleteShaderProgram()
{

}

CompleteShaderProgram& CompleteShaderProgram::operator=(const CompleteShaderProgram& other)
{
	if (this == &other)
		return *this;

	ShaderProgram::operator=(other);
	return *this;
}

CompleteShaderProgram& CompleteShaderProgram::operator=(CompleteShaderProgram&& other)
{
	if (this == &other)
		return *this;

	ShaderProgram::operator=(std::move(other));
	return *this;
}

CompleteShaderProgram::CompleteShaderProgram(const std::string& program_string_name, std::vector<std::string> file_sources, std::vector<ShaderType> shader_type) :
ShaderProgram("CompleteShaderProgram", program_string_name, file_sources, shader_type)
{

}

CompleteShaderProgram::CompleteShaderProgram(const std::string& program_string_name, const std::vector<GLSLSourceCode>& glsl_sources, std::vector<ShaderType> shader_type) :
ShaderProgram("CompleteShaderProgram", program_string_name, glsl_sources, shader_type)
{

}

CompleteShaderProgram::CompleteShaderProgram(const std::string& program_string_name, std::string shader_binary_source) :
ShaderProgram("CompleteShaderProgram", program_string_name, shader_binary_source)
{

}

CompleteShaderProgram::CompleteShaderProgram(const std::string& program_string_name, const std::vector<Shader>& shaders) :
ShaderProgram("CompleteShaderProgram", program_string_name, shaders)
{

}


bool CompleteShaderProgram::activate() const
{
	if (needsRelink()) return false;
	glUseProgram(getOpenGLProgramId());
	made_active();
	return true;
}

bool CompleteShaderProgram::isSeparate() const { return false; }

ShaderProgram* CompleteShaderProgram::clone() const
{
	return new CompleteShaderProgram{ *this };
}