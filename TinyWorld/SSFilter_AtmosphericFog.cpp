#include "SSFilter_AtmosphericFog.h"

using namespace tiny_world;


bool SSFilter_AtmosphericFog::inject_filter_core(SeparateShaderProgram& filter_fragment_program)
{
	Shader ssfilter_atmospheric_fog_shader{ ShaderProgram::getShaderBaseCatalog() + "AtmosphericFog.fp.glsl", ShaderType::FRAGMENT_SHADER, "SSFilter_AtmosphericFog::fp_shader" };
	if (!ssfilter_atmospheric_fog_shader) return false;

	filter_fragment_program.addShader(ssfilter_atmospheric_fog_shader);

	if (!filter_fragment_program) return false;
	else
		return true;
}


bool SSFilter_AtmosphericFog::perform_post_initialization()
{
	getFilterShaderProgram()->assignUniformBlockToBuffer("FogBuffer", 0);
	getFilterShaderProgram()->assignUniformScalar("fFogDistanceCutOff", fog_distance_cut_off);
	return true;
}


bool SSFilter_AtmosphericFog::set_filter_parameters(const AbstractProjectingDevice& projecting_device, const AbstractRenderingDevice& render_target, 
	int vacant_texture_unit_id, const TextureSampler* _2d_texture_source_sampler)
{
	//Apply general filter parameters
	float near_clip_plane, far_clip_plane;
	vec4 focal_plane;
	projecting_device.getProjectionVolume(&focal_plane.x, &focal_plane.y, &focal_plane.z, &focal_plane.w, &near_clip_plane, &far_clip_plane);
	getFilterShaderProgram()->assignUniformScalar("fFocalDistance", near_clip_plane);
	getFilterShaderProgram()->assignUniformScalar("fFarPlane", far_clip_plane);
	getFilterShaderProgram()->assignUniformVector("v4FocalPlane", focal_plane);

	Rectangle viewport = render_target.getViewportRectangle(0);
	getFilterShaderProgram()->assignUniformVector("v4Viewport", vec4{ viewport.x, viewport.y, viewport.w, viewport.h });

	mat4 view_transform = projecting_device.getViewTransform();
	getFilterShaderProgram()->assignUniformMatrix("m4ViewTransform", view_transform);


	//Assign in-scattering textures and bind fog buffer
	if (p_lighting_conditions)
	{
		in_scattering_sun = p_lighting_conditions->retrieveInScatteringTextures().first;
		in_scattering_moon = p_lighting_conditions->retrieveInScatteringTextures().second;

		TextureUnitBlock* p_texture_unit_block = AbstractRenderableObjectTextured::getTextureUnitBlockPointer();
		p_texture_unit_block->switchActiveTextureUnit(vacant_texture_unit_id);
		p_texture_unit_block->bindTexture(in_scattering_sun);
		p_texture_unit_block->bindSampler(in_scattering_texture_sampler);
		getFilterShaderProgram()->assignUniformScalar("s2daInScatteringSun", vacant_texture_unit_id);

		p_texture_unit_block->switchActiveTextureUnit(vacant_texture_unit_id + 1);
		p_texture_unit_block->bindTexture(in_scattering_moon);
		p_texture_unit_block->bindSampler(in_scattering_texture_sampler);
		getFilterShaderProgram()->assignUniformScalar("s2daInScatteringMoon", vacant_texture_unit_id + 1);

		p_lighting_conditions->updateFogBuffer();
		p_lighting_conditions->getFogBufferPtr()->bind();
	}

	return true;
}



SSFilter_AtmosphericFog::SSFilter_AtmosphericFog() : SSFilter("SSFilter_AtmosphericFog"), p_lighting_conditions{ nullptr }, fog_distance_cut_off{ 500.0f }
{
	in_scattering_texture_sampler.setMinFilter(SamplerMinificationFilter::LINEAR);
	in_scattering_texture_sampler.setMagFilter(SamplerMagnificationFilter::LINEAR);
	in_scattering_texture_sampler.setWrapping(SamplerWrapping{ SamplerWrappingMode::CLAMP_TO_EDGE, SamplerWrappingMode::CLAMP_TO_EDGE, SamplerWrappingMode::CLAMP_TO_EDGE });
}

SSFilter_AtmosphericFog::SSFilter_AtmosphericFog(const LightingConditions& lighting_conditions_context) : 
SSFilter("SSFilter_AtmosphericFog"), p_lighting_conditions{ &lighting_conditions_context }, fog_distance_cut_off{ 500.0f }
{
	in_scattering_texture_sampler.setMinFilter(SamplerMinificationFilter::LINEAR);
	in_scattering_texture_sampler.setMagFilter(SamplerMagnificationFilter::LINEAR);
	in_scattering_texture_sampler.setWrapping(SamplerWrapping{ SamplerWrappingMode::REPEAT, SamplerWrappingMode::CLAMP_TO_EDGE, SamplerWrappingMode::CLAMP_TO_EDGE });
}

void SSFilter_AtmosphericFog::setLightingConditions(const LightingConditions& lighting_conditions)
{
	p_lighting_conditions = &lighting_conditions;
}

void SSFilter_AtmosphericFog::defineColorTexture(const ImmutableTexture2D& color_texture)
{
	setTextureSource(0, color_texture);
}

void SSFilter_AtmosphericFog::defineLinearDepthBuffer(const ImmutableTexture2D& linear_depth_texture)
{
	setTextureSource(1, linear_depth_texture);
}

void SSFilter_AtmosphericFog::setDistanceCutOff(float distance)
{
	fog_distance_cut_off = distance;
	if (isInitialized())
		getFilterShaderProgram()->assignUniformScalar("fFogDistanceCutOff", fog_distance_cut_off);
}

float SSFilter_AtmosphericFog::getDistanceCutOff() const { return fog_distance_cut_off; }