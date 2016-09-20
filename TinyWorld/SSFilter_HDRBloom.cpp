#include  "SSFilter_HDRBloom.h"

using namespace tiny_world;


bool SSFilter_HDRBloom::inject_filter_core(SeparateShaderProgram& filter_fragment_program)
{
	Shader filter_core{ ShaderProgram::getShaderBaseCatalog() + "HDR_Bloom.fp.glsl", ShaderType::FRAGMENT_SHADER, "SSFilter_HDRBloom::fragment_program" };
	if (!filter_core) return false;

	filter_fragment_program.addShader(filter_core);
	return true;
}

bool SSFilter_HDRBloom::set_filter_parameters(const AbstractProjectingDevice& projecting_device, const AbstractRenderingDevice& render_target, 
	int vacant_texture_unit_id, const TextureSampler* _2d_texture_source_sampler)
{
	getFilterShaderProgram()->assignUniformScalar("bloom_impact", bloom_impact);
	getFilterShaderProgram()->assignUniformScalar("contrast", contrast);

	return !getFilterShaderProgram()->getErrorState();
}

bool SSFilter_HDRBloom::perform_post_initialization() { return true; }


SSFilter_HDRBloom::SSFilter_HDRBloom() : SSFilter("SSFilter_HDRBloom"), bloom_impact{ 0.7f }, contrast{ 1.4f }
{

}

void SSFilter_HDRBloom::defineColorTexture(const ImmutableTexture2D& _2d_color_texture)
{
	setTextureSource(0, _2d_color_texture);
}

void SSFilter_HDRBloom::defineBloomTexture(const ImmutableTexture2D& _2d_bloom_texture)
{
	setTextureSource(1, _2d_bloom_texture);
}

void SSFilter_HDRBloom::setBloomImpact(float impact_factor)
{
	bloom_impact = impact_factor;
}

float SSFilter_HDRBloom::getBloomImpact() const { return bloom_impact; }

void SSFilter_HDRBloom::setContrast(float contrast_value)
{
	contrast = contrast_value;
}

float SSFilter_HDRBloom::getContrast() const { return contrast; }


