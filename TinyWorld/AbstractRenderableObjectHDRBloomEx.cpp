#include "AbstractRenderableObjectHDRBloomEx.h"

using namespace tiny_world;


AbstractRenderableObjectHDRBloomEx::AbstractRenderableObjectHDRBloomEx() :
bloom_min_threshold{ 0.8f }, bloom_max_threshold{ 1.2f }, bloom_intensity{ 4.0f }, bloom_enabled{ 1 }
{

}

AbstractRenderableObjectHDRBloomEx::AbstractRenderableObjectHDRBloomEx(const AbstractRenderableObjectHDRBloomEx& other) :
bloom_min_threshold{ other.bloom_min_threshold }, bloom_max_threshold{ other.bloom_max_threshold },
bloom_intensity{ other.bloom_intensity }, bloom_enabled{ other.bloom_enabled }
{

}

AbstractRenderableObjectHDRBloomEx& AbstractRenderableObjectHDRBloomEx::operator=(const AbstractRenderableObjectHDRBloomEx& other)
{
	//Account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	bloom_min_threshold = other.bloom_min_threshold;
	bloom_max_threshold = other.bloom_max_threshold;
	bloom_intensity = other.bloom_intensity;
	bloom_enabled = other.bloom_enabled;

	return *this;
}

AbstractRenderableObjectHDRBloomEx::~AbstractRenderableObjectHDRBloomEx()
{

}

bool AbstractRenderableObjectHDRBloomEx::injectExtension(const ShaderProgramReferenceCode& program_ref_code, std::initializer_list<PipelineStage> program_stages)
{
	//Check if the program being created contains compute shader stage. If so, no extension should be inserted.
	if (std::find(program_stages.begin(), program_stages.end(), PipelineStage::COMPUTE_SHADER) != program_stages.end())
		return true;

	//Check if supplied program contains fragment processing stage
	if (std::find(program_stages.begin(), program_stages.end(), PipelineStage::FRAGMENT_SHADER) != program_stages.end())
	{
		Shader BloomOutput{ ShaderProgram::getShaderBaseCatalog() + "BloomOutput.fp.glsl", ShaderType::FRAGMENT_SHADER, "bloom_output_fragment_program" };
		if (!BloomOutput) return false;
		if (!retrieveShaderProgram(program_ref_code)->addShader(BloomOutput)) return false;

		modified_program_ref_code_list.push_back(program_ref_code);

		return true;
	}
	else
		return retrieveShaderProgram(program_ref_code)->isSeparate() ? true : false;
}


void AbstractRenderableObjectHDRBloomEx::applyExtension()
{
	std::for_each(modified_program_ref_code_list.begin(), modified_program_ref_code_list.end(),
		[this](const ShaderProgramReferenceCode& shader_program_ref_code) -> void
	{
		ShaderProgram* p_shader_program = retrieveShaderProgram(shader_program_ref_code);

		p_shader_program->assignUniformScalar("bBloomEnabled", bloom_enabled);
		p_shader_program->assignUniformScalar("fBloomMinThreshold", bloom_min_threshold);
		p_shader_program->assignUniformScalar("fBloomMaxThreshold", bloom_max_threshold);
		p_shader_program->assignUniformScalar("fBloomIntensity", bloom_intensity);
	});
}


void AbstractRenderableObjectHDRBloomEx::applyViewerTransform(const AbstractProjectingDevice& projecting_device)
{

}


void AbstractRenderableObjectHDRBloomEx::releaseExtension()
{

}


void AbstractRenderableObjectHDRBloomEx::setBloomMinimalThreshold(float threshold)
{
	bloom_min_threshold = threshold;
}

void AbstractRenderableObjectHDRBloomEx::setBloomMaximalThreshold(float threshold)
{
	bloom_max_threshold = threshold;
}

void AbstractRenderableObjectHDRBloomEx::useBloom(bool bloom_enable_state)
{
	bloom_enabled = bloom_enable_state ? 1 : 0;
}

void AbstractRenderableObjectHDRBloomEx::setBloomIntensity(float bloom_intensity)
{
	this->bloom_intensity = bloom_intensity;
}

float AbstractRenderableObjectHDRBloomEx::getBloomMinimalThreshold() const
{
	return bloom_min_threshold;
}

float AbstractRenderableObjectHDRBloomEx::getBloomMaximalThreshold() const
{
	return bloom_max_threshold;
}

bool AbstractRenderableObjectHDRBloomEx::isBloomInUse() const
{
	return bloom_enabled == 1;
}

float AbstractRenderableObjectHDRBloomEx::getBloomIntensity() const
{
	return bloom_intensity;
}