//Implements HDR-Bloom screen-space filter
//
//Input channel assignments: 
//channel0	color texture
//channel1	bloom texture

#ifndef TW__SSFILTER_HDRBLOOM__


#include "SSFilter.h"

namespace tiny_world
{

	class SSFilter_HDRBloom final : public SSFilter<2>
	{
	private:
		float bloom_impact;		//impact of the bloom effect
		float contrast;				//contrast value; must be between 1 and 2.

		bool inject_filter_core(SeparateShaderProgram& filter_fragment_program) override;
		bool set_filter_parameters(const AbstractProjectingDevice& projecting_device, const AbstractRenderingDevice& render_target,
			int vacant_texture_unit_id, const TextureSampler* _2d_texture_source_sampler) override;
		bool perform_post_initialization() override;

	public:
		SSFilter_HDRBloom();	//default initialization of the filter

		void defineColorTexture(const ImmutableTexture2D& _2d_color_texture);	//defines color texture of the filter
		void defineBloomTexture(const ImmutableTexture2D& _2d_bloom_texture);	//defines bloom texture of the filter

		void setBloomImpact(float impact_factor);	//sets value for bloom impact factor
		float getBloomImpact() const; //returns current value of the bloom impact factor
		
		void setContrast(float contrast_value);	//sets the contrast value used by the HDR filter. Default value is 1.4
		float getContrast() const;	//returns the contrast value used by the HDR filter
	};

}


#define TW__SSFILTER_HDRBLOOM__
#endif