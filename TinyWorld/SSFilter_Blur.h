//Implements blur filter
//Input channel assignments:
//channel0	input texture to be blurred

#ifndef TW__SSFILTER_BLUR__

#include <cstdint>

#include "SSFilter.h"
#include "std140UniformBuffer.h"


namespace tiny_world
{
	class SSFilter_Blur final : public SSFilter<1>
	{
	private:
		static const uint32_t max_kernel_size = 512;	//maximal size of the kernel of the filter

		bool horizontal_blur;	//equals 'true' if blurring is performed horizontally. Equals 'false' if blur is applied vertically.
		uint32_t kernel_size;	//actual size of the kernel of the filter
		uint32_t kernel_mipmap;	//mipmap level of the input texture, which should be used for blurring
		float kernel_scale;	//scale factor of the kernel
		std::vector<float> filter_kernel;	//contains kernel of the filter
		std140UniformBuffer filter_kernel_params;	//buffer containing kernel data of the filter and related parameters

		void generate_filter_kernel();	//generates filter kernel and packs it into the corresponding uniform buffer

		bool inject_filter_core(SeparateShaderProgram& filter_fragment_program) override;
		bool set_filter_parameters(const AbstractProjectingDevice& projecting_device, const AbstractRenderingDevice& render_target,
			int vacant_texture_unit_id, const TextureSampler* _2d_texture_source_sampler) override;
		bool perform_post_initialization() override;

	public:
		enum class BlurDirection { HORIZONTAL, VERTICAL };

		//Default initialization with input parameter "size" specifying size of the filter kernel
		SSFilter_Blur(uint32_t size = 4, float kernel_scale = 3.0f);

		//Defines dimension, along which the blur filter should be applied
		void setDirection(BlurDirection direction);

		//Returns currently active blur direction
		BlurDirection getDirection() const;

		//Sets new size of the filter kernel. The minimal size is 2, the maximal size is 32.
		//If provided value of the size lies outside of the range of permissible values, it is automatically clamped to this range
		void setKernelSize(uint32_t size);

		//Returns current size of the filter kernel
		uint32_t getKernelSize() const;

		//Defines mipmap-level of the input texture that should be used to produce the blurred output. Supplied value is automatically
		//clamped to the allowed range. This value can be efficiently utilized for severe blurring
		void setKernelMipmap(uint32_t mipmap_level);

		//Returns mipmap level currently used to produce the blurred output
		uint32_t getKernelMipmap() const;

		//Assigns new value for the kernel scale
		void setKernelScale(float scale);

		//Returns current scale of the filter kernel
		float getKernelScale() const;

		//Defines input texture to be blurred
		void defineInputTexture(const ImmutableTexture2D& input_texture);
	};
}

#define TW__SSFILTER_BLUR__
#endif