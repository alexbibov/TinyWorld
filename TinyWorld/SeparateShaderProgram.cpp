#include "SeparateShaderProgram.h"
#include <algorithm>
#include <vector>
#include <set>

using namespace tiny_world;

//*************************************Functions belonging to the core part of the separate shader program implementation***********************************************

SeparateShaderProgram_Core::SeparateShaderProgram_Core() : ShaderProgram("SeparateShaderProgram")
{
	glProgramParameteri(getOpenGLProgramId(), GL_PROGRAM_SEPARABLE, GL_TRUE);
}

SeparateShaderProgram_Core::SeparateShaderProgram_Core(const std::string& program_string_name) : ShaderProgram("SeparateShaderProgram", program_string_name)
{
	glProgramParameteri(getOpenGLProgramId(), GL_PROGRAM_SEPARABLE, GL_TRUE);
}

SeparateShaderProgram_Core::SeparateShaderProgram_Core(const std::string& program_string_name, std::vector<std::string> file_sources, std::vector<ShaderType> shader_type) :
ShaderProgram("SeparateShaderProgram", program_string_name, file_sources, shader_type)
{
	glProgramParameteri(getOpenGLProgramId(), GL_PROGRAM_SEPARABLE, GL_TRUE);
}

SeparateShaderProgram_Core::SeparateShaderProgram_Core(const std::string& program_string_name, const std::vector<GLSLSourceCode>& glsl_sources, std::vector<ShaderType> shader_type) :
ShaderProgram("SeparateShaderProgram", program_string_name, glsl_sources, shader_type)
{
	glProgramParameteri(getOpenGLProgramId(), GL_PROGRAM_SEPARABLE, GL_TRUE);
}

SeparateShaderProgram_Core::SeparateShaderProgram_Core(const std::string& program_string_name, std::string shader_binary_source) :
ShaderProgram("SeparateShaderProgram", program_string_name, shader_binary_source)
{
	glProgramParameteri(getOpenGLProgramId(), GL_PROGRAM_SEPARABLE, GL_TRUE);
}

SeparateShaderProgram_Core::SeparateShaderProgram_Core(const std::string& program_string_name, const std::vector<Shader>& shaders) :
ShaderProgram("SeparateShaderProgram", program_string_name, shaders)
{
	glProgramParameteri(getOpenGLProgramId(), GL_PROGRAM_SEPARABLE, GL_TRUE);
}

SeparateShaderProgram_Core::SeparateShaderProgram_Core(const SeparateShaderProgram_Core& other) : ShaderProgram(other)
{
	glProgramParameteri(getOpenGLProgramId(), GL_PROGRAM_SEPARABLE, GL_TRUE);
}

SeparateShaderProgram_Core::SeparateShaderProgram_Core(SeparateShaderProgram_Core&& other) : ShaderProgram(std::move(other))
{

}


bool SeparateShaderProgram_Core::isSeparate() const { return true; }


//**************************************************Functions belonging to the interface part of the separate shader program implementation**************************************************
SeparateShaderProgram::SeparateShaderProgram() :
SeparateShaderProgram_Core()
{
}

SeparateShaderProgram::SeparateShaderProgram(const std::string& program_string_name) :
SeparateShaderProgram_Core(program_string_name)
{
}

SeparateShaderProgram::SeparateShaderProgram(const SeparateShaderProgram& other) :
SeparateShaderProgram_Core(other)
{
}

SeparateShaderProgram::SeparateShaderProgram(SeparateShaderProgram&& other) : SeparateShaderProgram_Core(std::move(other))
{

}

SeparateShaderProgram::SeparateShaderProgram(const std::string& program_string_name, std::vector<std::string> file_sources, std::vector<ShaderType> shader_type) :
SeparateShaderProgram_Core(program_string_name, file_sources, shader_type)
{
}

SeparateShaderProgram::SeparateShaderProgram(const std::string& program_string_name, const std::vector<GLSLSourceCode>& glsl_sources, std::vector<ShaderType> shader_type) :
SeparateShaderProgram_Core(program_string_name, glsl_sources, shader_type)
{
}

SeparateShaderProgram::SeparateShaderProgram(const std::string& program_string_name, std::string shader_binary_source) :
SeparateShaderProgram_Core(program_string_name, shader_binary_source)
{
}

SeparateShaderProgram::SeparateShaderProgram(const std::string& program_string_name, const std::vector<Shader>& shaders) :
SeparateShaderProgram_Core(program_string_name, shaders)
{
}

bool SeparateShaderProgram::installToPipeline(GLuint ogl_pipeline_id) const
{
	glUseProgramStages(ogl_pipeline_id, getProgramStages(), getOpenGLProgramId());
	return true;
}

bool SeparateShaderProgram::activate(GLuint ogl_pipeline_id) const
{
	if (needsRelink()) return false;
	glActiveShaderProgram(ogl_pipeline_id, getOpenGLProgramId());
	made_active();
	return true;
}

ShaderProgram* SeparateShaderProgram::clone() const
{
	return new SeparateShaderProgram{ *this };
}


//*************************************************Program pipeline implementation***********************************************

ProgramPipeline::ProgramPipeline() : Entity("ProgramPipeline")
{
	glGenProgramPipelines(1, &ogl_program_pipeline);
}


ProgramPipeline::ProgramPipeline(const ProgramPipeline& other) : Entity(other), pipeline_state(other.pipeline_state)
{
	glGenProgramPipelines(1, &ogl_program_pipeline);

	std::set<long long> program_set;
	std::for_each(pipeline_state.begin(), pipeline_state.end(),
		[this, &program_set](pipeline_map::value_type elem)
	{
		if (program_set.insert(elem.second->getId()).second)	//on successful insertion install program to the pipeline
			elem.second->installToPipeline(ogl_program_pipeline);
	});
}


ProgramPipeline::~ProgramPipeline()
{
	glDeleteProgramPipelines(1, &ogl_program_pipeline);
}


ProgramPipeline& ProgramPipeline::attach(const SeparateShaderProgram& separate_shader_program)
{
	const ShaderType shader_stage[6] = { ShaderType::VERTEX_SHADER, ShaderType::TESS_CONTROL_SHADER,
		ShaderType::TESS_EVAL_SHADER, ShaderType::GEOMETRY_SHADER, ShaderType::FRAGMENT_SHADER,
		ShaderType::COMPUTE_SHADER };
	for (int i = 0; i < 6; ++i)
	{
		if (separate_shader_program.containsStage(shader_stage[i]))
			if (!pipeline_state.insert(std::make_pair(shader_stage[i], &separate_shader_program)).second)
				pipeline_state.at(shader_stage[i]) = &separate_shader_program;
	}
	separate_shader_program.installToPipeline(ogl_program_pipeline);
	return *this;
}


ProgramPipeline& ProgramPipeline::operator+=(const SeparateShaderProgram& separate_shader_program)
{
	return attach(separate_shader_program);
}


bool ProgramPipeline::activate_program(long long program_id) const
{
	pipeline_map::const_iterator requested_program =
		std::find_if(pipeline_state.begin(), pipeline_state.end(),
		[program_id](pipeline_map::value_type elem)->bool
	{
		return elem.second->getId() == program_id;
	}
	);

	if (requested_program == pipeline_state.end())
		return false;
	else
	{
		requested_program->second->activate(ogl_program_pipeline);
		return true;
	}
}


bool ProgramPipeline::activate_program(std::string program_string_name) const
{
	pipeline_map::const_iterator requested_program =
		std::find_if(pipeline_state.begin(), pipeline_state.end(),
		[program_string_name](pipeline_map::value_type elem)->bool
	{
		return elem.second->getStringName() == program_string_name;
	}
	);

	if (requested_program == pipeline_state.end())
		return false;
	else
	{
		requested_program->second->activate(ogl_program_pipeline);
		return true;
	}
}


ProgramPipeline& ProgramPipeline::operator=(const ProgramPipeline& other)
{
	if (this == &other)
		return *this;

	Entity::operator=(other);

	//Reset current stages from the pipeline
	glUseProgramStages(ogl_program_pipeline, GL_ALL_SHADER_BITS, 0);

	pipeline_state = other.pipeline_state;

	std::set<long long> program_set;
	std::for_each(pipeline_state.begin(), pipeline_state.end(),
		[this, &program_set](pipeline_map::value_type elem)
	{
		if (program_set.insert(elem.second->getId()).second)	//on successful insertion install program to the pipeline
			elem.second->installToPipeline(ogl_program_pipeline);
	});

	return *this;
}


bool ProgramPipeline::bind() const
{
	glUseProgram(0);		//Detach complete shader programs from the context, if any of them were previously attached
	glBindProgramPipeline(ogl_program_pipeline);
	return true;
}


bool ProgramPipeline::isEnclosed() const
{
	if (pipeline_state.find(ShaderType::VERTEX_SHADER) == pipeline_state.end() ||
		pipeline_state.find(ShaderType::FRAGMENT_SHADER) == pipeline_state.end())
		return false;
	return true;
}

void ProgramPipeline::reset()
{
	//Reset all program stages
	glUseProgramStages(ogl_program_pipeline, GL_ALL_SHADER_BITS, 0);
	pipeline_state.clear();
}