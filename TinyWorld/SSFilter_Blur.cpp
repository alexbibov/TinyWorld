#include <vector>

#include "SSFilter_Blur.h"


using namespace tiny_world;



void SSFilter_Blur::generate_filter_kernel()
{
	//Generate filter kernel
	float kernel_normalization_factor = 0.0f;
	for (unsigned i = 0; i < kernel_size; ++i)
	{
		float aux_x_val = (i / (kernel_size - 1.0f) - 0.5f)*kernel_scale;
		float kernel_element = std::exp(-aux_x_val*aux_x_val / 2);
		filter_kernel[i] = kernel_element;
		kernel_normalization_factor += kernel_element;
	}

	//Normalize kernel
	std::for_each(filter_kernel.begin(), filter_kernel.end(),
		[kernel_normalization_factor](float& element)->void{element /= kernel_normalization_factor; });

	filter_kernel_params.resetOffsetCounter();
	filter_kernel_params.skipScalar<bool>(1, false);
	filter_kernel_params.skipScalar<unsigned int>(2, false);
	filter_kernel_params.pushScalar(filter_kernel);
}


bool SSFilter_Blur::inject_filter_core(SeparateShaderProgram& filter_fragment_program)
{
	Shader filter_shader{ ShaderProgram::getShaderBaseCatalog() + "Blur.fp.glsl", ShaderType::FRAGMENT_SHADER, "SSFilter_Blur::fragment_program" };
	if (!filter_shader) return false;

	filter_fragment_program.addShader(filter_shader);
	return true;
}


bool SSFilter_Blur::set_filter_parameters(const AbstractProjectingDevice& projecting_device, const AbstractRenderingDevice& render_target,
	int vacant_texture_unit_id, const TextureSampler* _2d_texture_source_sampler)
{
	filter_kernel_params.bind();
	return !filter_kernel_params.getErrorState();
}


bool SSFilter_Blur::perform_post_initialization()
{
	//Define custom sampler object (needed for the blur filter to operate properly)
	TextureSampler blur_filter_sampler{ "SSFilter_Blur::sampler" };
	blur_filter_sampler.setMinFilter(SamplerMinificationFilter::LINEAR_MIPMAP_NEAREST);
	blur_filter_sampler.setMagFilter(SamplerMagnificationFilter::LINEAR);
	blur_filter_sampler.setWrapping({ SamplerWrappingMode::CLAMP_TO_EDGE,
		SamplerWrappingMode::CLAMP_TO_EDGE,
		SamplerWrappingMode::CLAMP_TO_BORDER });
	this->setCanvasSampler(blur_filter_sampler);


	//Determine size of the uniform block containing parameters of the filter
	size_t uniform_block_size = getFilterShaderProgram()->getUniformBlockDataSize("FilterKernelParams");
	if (getFilterShaderProgram()->getErrorState()) return false;

	filter_kernel_params.allocate(uniform_block_size);	//allocate space for the uniform block

	//Generate filter kernel
	generate_filter_kernel();
	getFilterShaderProgram()->assignUniformBlockToBuffer("FilterKernelParams", filter_kernel_params);

	//Set default values for parameters of the filter
	filter_kernel_params.resetOffsetCounter();
	filter_kernel_params.pushScalar(horizontal_blur);
	filter_kernel_params.pushScalar(kernel_size);
	filter_kernel_params.pushScalar(kernel_mipmap);

	return true;
}



SSFilter_Blur::SSFilter_Blur(uint32_t size /* = 4 */, float kernel_scale /* = 3.0f */) : 
SSFilter("SSFilter_Blur"),
horizontal_blur{ true }, kernel_size{ size }, kernel_mipmap{ 0 }, kernel_scale{ kernel_scale },
filter_kernel(static_cast<std::vector<float>::size_type>(max_kernel_size)),
filter_kernel_params{ 0 }
{

}


void SSFilter_Blur::setDirection(BlurDirection direction)
{
	switch (direction)
	{
	case BlurDirection::HORIZONTAL:
		horizontal_blur = true;
		break;

	case BlurDirection::VERTICAL:
		horizontal_blur = false;
		break;
	}

	if (isInitialized())
	{
		filter_kernel_params.resetOffsetCounter();
		filter_kernel_params.pushScalar(horizontal_blur);
	}
}


SSFilter_Blur::BlurDirection SSFilter_Blur::getDirection() const
{
	if (horizontal_blur)
		return BlurDirection::HORIZONTAL;
	else
		return BlurDirection::VERTICAL;
}


void SSFilter_Blur::setKernelSize(uint32_t size)
{
	kernel_size = std::min(std::max(2U, size), max_kernel_size);
	if (isInitialized())
	{
		filter_kernel_params.resetOffsetCounter();
		filter_kernel_params.skipScalar<bool>(1, false);
		filter_kernel_params.pushScalar(kernel_size);
		generate_filter_kernel();	//update filter kernel
	}
}


uint32_t SSFilter_Blur::getKernelSize() const { return kernel_size; }


void SSFilter_Blur::setKernelMipmap(uint32_t mipmap_level)
{
	kernel_mipmap = std::min(mipmap_level, getTextureSource(0).getNumberOfMipmapLevels());
	if (isInitialized())
	{
		filter_kernel_params.resetOffsetCounter();
		filter_kernel_params.skipScalar<bool>(1, false);
		filter_kernel_params.skipScalar<unsigned int>(1, false);
		filter_kernel_params.pushScalar(kernel_mipmap);
	}
}


uint32_t SSFilter_Blur::getKernelMipmap() const { return kernel_mipmap; }


void SSFilter_Blur::setKernelScale(float scale)
{
	kernel_scale = scale;
	if (isInitialized())
		generate_filter_kernel();	//update filter kernel
}


float SSFilter_Blur::getKernelScale() const { return kernel_scale; }


void SSFilter_Blur::defineInputTexture(const ImmutableTexture2D& input_texture)
{
	setTextureSource(0, input_texture);
}