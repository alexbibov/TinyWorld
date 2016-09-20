//Implements depth based atmospheric fog applied in screen space

#ifndef TW__SSFILTER_ATMOSPHERIC_FOG__


#include "SSFilter.h"
#include "LightingConditions.h"

namespace tiny_world
{
	class SSFilter_AtmosphericFog final : public SSFilter < 2 >
	{
	private:
		const LightingConditions* p_lighting_conditions;	//pointer to the lighting conditions context containing parameters of the fog
		ImmutableTexture2D in_scattering_sun;	//in-scattering texture contributed by sun
		ImmutableTexture2D in_scattering_moon;	//in-scattering texture contributed by moon
		TextureSampler in_scattering_texture_sampler;	//sampler object used by th in-scattering textures
		float fog_distance_cut_off;	//maximal distance assumed by the fog calculations

		bool inject_filter_core(SeparateShaderProgram& filter_fragment_program) override;
		bool set_filter_parameters(const AbstractProjectingDevice& projecting_device, const AbstractRenderingDevice& render_target,
			int vacant_texture_unit_id, const TextureSampler* _2d_texture_source_sampler) override;
		bool perform_post_initialization() override;

	public:
		SSFilter_AtmosphericFog();		//Default initializer. Creates a dummy fog filter. No actual fog is produced before a lighting conditions context is registered to the filter
		explicit SSFilter_AtmosphericFog(const LightingConditions& lighting_conditions_context);	//initializes atmospheric fog with a certain lighting conditions context

		//Installs new lighting conditions context. Note that lighting conditions context is not owned by the atmospheric fog filter meaning that any changes applied to the lighting conditions
		//are immediately reflected by behavior of the fog filter
		void setLightingConditions(const LightingConditions& lighting_conditions);

		void defineColorTexture(const ImmutableTexture2D& color_texture);	//defines color screen space texture
		void defineLinearDepthBuffer(const ImmutableTexture2D& linear_depth_texture);	//defines linear depth screen space texture

		void setDistanceCutOff(float distance);		//sets maximal allowed distance assumed when calculating amount of fog among a ray sample
		float getDistanceCutOff() const;	//returns maximal allowed distance assumed when calculating amount of fog among a ray sample
	};
}

#define TW__SSFILTER_ATMOSPHERIC_FOG__
#endif