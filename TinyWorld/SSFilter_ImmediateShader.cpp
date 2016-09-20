#include "SSFilter_ImmediateShader.h"

using namespace tiny_world;



bool SSFilter_ImmediateShader::inject_filter_core(SeparateShaderProgram& filter_fragment_program)
{
	Shader filter_program{ ShaderProgram::getShaderBaseCatalog() + "ImmediateShader.fp.glsl", ShaderType::FRAGMENT_SHADER, "SSFilter_ImmediateShader::filter_program" };
	if (!filter_program) return false;

	filter_fragment_program.addShader(filter_program);
	return true;
}


bool SSFilter_ImmediateShader::set_filter_parameters(const AbstractProjectingDevice& projecting_device, const AbstractRenderingDevice& render_target,
	int vacant_texture_unit_id, const TextureSampler* _2d_texture_source_sampler)
{

	return true;
}


bool SSFilter_ImmediateShader::perform_post_initialization()
{
	return true;
}


SSFilter_ImmediateShader::SSFilter_ImmediateShader() : SSFilter("SSFilter_ImmediateShader")
{

}


void SSFilter_ImmediateShader::defineColorMap(const ImmutableTexture2D& color_texture)
{
	setTextureSource(0, color_texture);
}


void SSFilter_ImmediateShader::defineADMap(const ImmutableTexture2D& ad_map)
{
	setTextureSource(1, ad_map);
}


void SSFilter_ImmediateShader::defineOcclusionMap(const ImmutableTexture2D& ssao_map)
{
	setTextureSource(2, ssao_map);
}