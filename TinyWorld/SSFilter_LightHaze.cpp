#include "SSFilter_LightHaze.h"

using namespace tiny_world;




bool SSFilter_LightHaze::inject_filter_core(SeparateShaderProgram& filter_fragment_program)
{
	Shader ssfilter_light_haze_shader{ ShaderProgram::getShaderBaseCatalog() + "LightHaze.fp.glsl", ShaderType::FRAGMENT_SHADER, "SSFilter_LightHaze::fp_shader" };
	if (!ssfilter_light_haze_shader) return false;

	filter_fragment_program.addShader(ssfilter_light_haze_shader);

	if (!filter_fragment_program) return false;
	else
		return true;
}


bool SSFilter_LightHaze::set_filter_parameters(const AbstractProjectingDevice& projecting_device, const AbstractRenderingDevice& render_target,
	int vacant_texture_unit_id, const TextureSampler* _2d_texture_source_sampler)
{
	ShaderProgram& filter_shader_program = *getFilterShaderProgram();
	float left, right, bottom, top, near, far;
	projecting_device.getProjectionVolume(&left, &right, &bottom, &top, &near, &far);
	vec4 v4FocalPlane{ left, right, bottom, top };
	Rectangle viewport_rectangle = render_target.getViewportRectangle(0);
	vec4 v4Viewport{ viewport_rectangle.x, viewport_rectangle.y, viewport_rectangle.w, viewport_rectangle.h };
	mat4 m4ViewTransform = projecting_device.getViewTransform();

	
	filter_shader_program.assignUniformScalar("fFocalDistance", near);
	filter_shader_program.assignUniformVector("v4FocalPlane", v4FocalPlane);
	filter_shader_program.assignUniformVector("v4Viewport", v4Viewport);
	filter_shader_program.assignUniformMatrix("m4ViewTransform", m4ViewTransform);
	if (p_lighting_conditions) 
	{
		p_lighting_conditions->getLightBufferPtr()->bind();
		p_lighting_conditions->getFogBufferPtr()->bind();
	}

	if (!filter_shader_program) return false;
	else
		return true;
}


bool SSFilter_LightHaze::perform_post_initialization()
{
	ShaderProgram& filter_shader_program = *getFilterShaderProgram();

	filter_shader_program.assignUniformBlockToBuffer("LightBuffer", 1);
	filter_shader_program.assignUniformBlockToBuffer("FogBuffer", 0);

	if (!filter_shader_program) return false;
	else
		return true;
}



SSFilter_LightHaze::SSFilter_LightHaze() : SSFilter("SSFilter_LightHaze"), p_lighting_conditions{ nullptr }
{

}


SSFilter_LightHaze::SSFilter_LightHaze(const LightingConditions& lighting_conditions_context) : 
SSFilter("SSFilter_LightHaze"), p_lighting_conditions{ &lighting_conditions_context }
{
	
}


void SSFilter_LightHaze::setLightingConditions(const LightingConditions& lighting_conditions)
{
	p_lighting_conditions = &lighting_conditions;
}


void SSFilter_LightHaze::defineColorTexture(const ImmutableTexture2D& color_texture)
{
	setTextureSource(0, color_texture);
}


void SSFilter_LightHaze::defineLinearDepthBuffer(const ImmutableTexture2D& linear_depth_texture)
{
	setTextureSource(1, linear_depth_texture);
}