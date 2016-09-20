//Implements "immediate shader" screen-space filter. The main purpose of this filter is injection of SSAO-effect
//Input channel assignments:
//channel0		color map
//channel1		AD-map (texture that contains diffusion colors multiplied by ambient modulation)
//channel2		occlusion map (preferably blurred)


#ifndef TW__SSFILTER_IMMEDIATE_SHADER__

#include "SSFilter.h"

namespace tiny_world
{
	class SSFilter_ImmediateShader final : public SSFilter < 3 >
	{
	private:
		bool inject_filter_core(SeparateShaderProgram& filter_fragment_program) override;
		bool set_filter_parameters(const AbstractProjectingDevice& projecting_device, const AbstractRenderingDevice& render_target,
			int vacant_texture_unit_id, const TextureSampler* _2d_texture_source_sampler) override;
		bool perform_post_initialization() override;

	public:
		//Default initialization
		SSFilter_ImmediateShader();

		//Defines color map to be used by immediate shader
		void defineColorMap(const ImmutableTexture2D& color_texture);

		//Defines AD-map to be used by immediate shader
		void defineADMap(const ImmutableTexture2D& ad_map);

		//Defines occlusion map to be used by immediate shader
		void defineOcclusionMap(const ImmutableTexture2D& ssao_map);
	};
}

#define TW__SSFILTER_IMMEDIATE_SHADER__
#endif
