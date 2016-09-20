//Implements light haze post-processing effect for point, spot and directional light sources. The effect is applied in screen-space
//The haze filter expects two full screen maps to be provided on its input:
//channel0	color map
//channel1	linear depth map

#ifndef TW__SSFILTER_LIGHT_HAZE__

#include "SSFilter.h"
#include "LightingConditions.h"


namespace tiny_world
{
	class SSFilter_LightHaze final : public SSFilter < 2 >
	{
	private:
		const LightingConditions* p_lighting_conditions;	//pointer to the lighting conditions descriptor

		bool inject_filter_core(SeparateShaderProgram& filter_fragment_program) override;
		bool set_filter_parameters(const AbstractProjectingDevice& projecting_device, const AbstractRenderingDevice& render_target,
			int vacant_texture_unit_id, const TextureSampler* _2d_texture_source_sampler) override;
		bool perform_post_initialization() override;

	public:
		SSFilter_LightHaze();	//default constructor of the filter
		SSFilter_LightHaze(const LightingConditions& lighting_conditions_context);	//constructs filter and provides it with a lighting conditions context

		//Installs new lighting conditions context to the filter. Note that lighting conditions context object is not owned by the filter. Instead, the filter 
		//only owns a reference to it, which means that any changes made to the lighting conditions are immediately reflected by the filter
		void setLightingConditions(const LightingConditions& lighting_conditions);


		void defineColorTexture(const ImmutableTexture2D& color_texture);	//defines screen space color map
		void defineLinearDepthBuffer(const ImmutableTexture2D& linear_depth_texture);	//defines screen space linear depth map
	};
}

#define TW__SSFILTER_LIGHT_HAZE__
#endif

