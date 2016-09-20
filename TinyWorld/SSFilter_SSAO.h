//Implements Screen-Space ambient occlusion technique
//Input channel assignments:
//channel0	normal map
//channel1	linear depth map

#ifndef TW__SSFILTER_SSAO__

#include <vector>
#include <cstdint>
#include <random>

#include "SSFilter.h"
#include "std140UniformBuffer.h"

namespace tiny_world
{
	class SSFilter_SSAO final : public SSFilter<2>
	{
	private:
		static const uint32_t max_sample_points = 256;	//maximal number of sample points that can be used in SSAO kernel

		std::mt19937_64 random_engine;	//random engine used to generate sample points
		std140UniformBuffer ssao_parameters_block;	//uniform buffer object containing parameters of SSAO
		ImmutableTexture2D noise_texture;	//noise texture determining random rotation of SSAO kernel
		TextureSampler noise_texture_sampler;	//sampler object used by the noise texture

		unsigned int num_samples;	//actual number of sample points used by the filter
		float kernel_radius;		//radius of the sampling kernel
		float occlusion_range;		//range of occlusion
		unsigned int noise_pattern_size;	//size of the noise pattern
		
		bool inject_filter_core(SeparateShaderProgram& filter_fragment_program) override;
		bool set_filter_parameters(const AbstractProjectingDevice& projecting_device, const AbstractRenderingDevice& render_target,
			int vacant_texture_unit_id, const TextureSampler* _2d_texture_source_sampler) override;
		bool perform_post_initialization() override;

	public:
		SSFilter_SSAO(unsigned int num_samples = 16, float kernel_radius = 1.0f, unsigned int noise_tiling = 4);	//Default initialization

		//Sets radius of SSAO kernel. The minimal value for this parameter is 0.1 and this value is used if kernel_radius does not belong to
		//the range of allowed values (i.e. iff kernel_radius < 0.1)
		void setKernelRadius(float kernel_radius);

		//Returns radius of SSAO kernel
		float getKernelRadius() const;

		//Sets SSAO range, where the range is maximal depth difference between a given fragment and occluding geometry.
		//The value of the parameter can not be smaller than the current radius of the kernel. If supplied value breaks this condition, it is
		//automatically clamped to the current value of kernel radius .
		void setOcclusionRange(float range);

		//Returns range of SSAO filter, where the range is maximal depth difference between a given fragment and occluding geometry.
		float getOcclusionRange() const;

		//Sets size of kernel random rotation pattern.
		void setNoiseSize(unsigned int noise_tiling);

		//Returns current size of kernel random rotation pattern
		unsigned int getNoiseSize() const;

		//Sets number of samples to be used by SSAO filtering kernel. The currently supported maximal number of samples is 128. Minimal number of samples is 2.
		//Supplied value is automatically clamped to the allowed range.
		void setNumberOfSamples(unsigned int num_samples);

		//Returns number of samples used by SSAO kernels
		unsigned int getNumberOfSamples() const;

		//Generates new sample points for SSAO kernel
		void updateSamplePoints();

		//Defines screen-space normal map to be used in occlusion sampling
		void defineScreenSpaceNormalMap(const ImmutableTexture2D& ss_normal_map);

		//Defines the depth map
		void defineLinearDepthBuffer(const ImmutableTexture2D& linear_depth_buffer);
	};
}


#define TW__SSFILTER_SSAO__
#endif