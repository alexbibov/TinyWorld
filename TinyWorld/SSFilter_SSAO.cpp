#include "SSFilter_SSAO.h"

#include <chrono>

using namespace tiny_world;


#define pi 3.1415926535897932384626433832795f


bool SSFilter_SSAO::inject_filter_core(SeparateShaderProgram& filter_fragment_program)
{
	Shader filter_core{ ShaderProgram::getShaderBaseCatalog() + "SSAO.fp.glsl", ShaderType::FRAGMENT_SHADER, "SSFilter_SSAO::fragment_program" };
	if (!filter_core) return false;

	filter_fragment_program.addShader(filter_core);
	return true;
}


bool SSFilter_SSAO::set_filter_parameters(const AbstractProjectingDevice& projecting_device, const AbstractRenderingDevice& render_target, 
	int vacant_texture_unit_id, const TextureSampler* _2d_texture_source_sampler)
{
	float left, right, bottom, top, near, far;
	projecting_device.getProjectionVolume(&left, &right, &bottom, &top, &near, &far);

	getFilterShaderProgram()->assignUniformScalar("fFocalDistance", near);
	getFilterShaderProgram()->assignUniformVector("v4FocalPlane", vec4{ left, right, bottom, top });

	Rectangle viewport = render_target.getViewportRectangle(0);
	getFilterShaderProgram()->assignUniformVector("v4Viewport", vec4{ viewport.x, viewport.y, viewport.w, viewport.h });

	getFilterShaderProgram()->assignUniformMatrix("m4ProjMat", projecting_device.getProjectionTransform());

	ssao_parameters_block.bind();

	TextureUnitBlock* p_texture_unit_block = AbstractRenderableObjectTextured::getTextureUnitBlockPointer();
	p_texture_unit_block->switchActiveTextureUnit(vacant_texture_unit_id);
	p_texture_unit_block->bindTexture(noise_texture);
	p_texture_unit_block->bindSampler(noise_texture_sampler);
	getFilterShaderProgram()->assignUniformScalar("s2dNoiseTexture", vacant_texture_unit_id);
	return true;
}


bool SSFilter_SSAO::perform_post_initialization()
{
	//Define uniform buffer object used by the filter
	ssao_parameters_block.allocate(getFilterShaderProgram()->getUniformBlockDataSize("SSAO_extra_params"));
	updateSamplePoints();
	getFilterShaderProgram()->assignUniformBlockToBuffer("SSAO_extra_params", ssao_parameters_block);

	//Assign default values for SSAO filter parameters
	getFilterShaderProgram()->assignUniformScalar("fKernelRadius", kernel_radius);
	getFilterShaderProgram()->assignUniformScalar("fOcclusionRange", occlusion_range);
	getFilterShaderProgram()->assignUniformScalar("numSamples", num_samples);

	return true;
}


SSFilter_SSAO::SSFilter_SSAO(unsigned int num_samples /* = 16 */, float kernel_radius /* = 1.0f */, unsigned int noise_tiling /* = 4 */) : 
SSFilter("SSFilter_SSAO"),
num_samples{ num_samples }, kernel_radius{ kernel_radius }, ssao_parameters_block{ 0 },
noise_texture{ "SSFilter_SSAO::noise_texture" }
{
	//Define initial value for occlusion range
	occlusion_range = kernel_radius;

	//Initialize noise pattern and sampler
	setNoiseSize(noise_tiling);
	noise_texture_sampler.setWrapping(SamplerWrapping{ SamplerWrappingMode::REPEAT, SamplerWrappingMode::REPEAT, SamplerWrappingMode::CLAMP_TO_EDGE });
	noise_texture_sampler.setMinFilter(SamplerMinificationFilter::LINEAR);
	noise_texture_sampler.setMagFilter(SamplerMagnificationFilter::LINEAR);

	//Initialize the random engine
	long long seed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	random_engine = std::mt19937_64{ static_cast<std::mt19937_64::result_type>(seed) };
}


void SSFilter_SSAO::setKernelRadius(float kernel_radius)
{
	this->kernel_radius = std::max(kernel_radius, 0.1f);
	occlusion_range = std::max(kernel_radius, occlusion_range);
	if(isInitialized()) 
	{
		getFilterShaderProgram()->assignUniformScalar("fKernelRadius", kernel_radius);
		getFilterShaderProgram()->assignUniformScalar("fOcclusionRange", occlusion_range);
	}
}


float SSFilter_SSAO::getKernelRadius() const { return kernel_radius; }


void SSFilter_SSAO::setOcclusionRange(float range)
{
	occlusion_range = std::max(range, kernel_radius);
	if (isInitialized()) getFilterShaderProgram()->assignUniformScalar("fOcclusionRange", occlusion_range);
}


float SSFilter_SSAO::getOcclusionRange() const { return occlusion_range; }


void SSFilter_SSAO::setNoiseSize(unsigned int noise_tiling)
{
	noise_pattern_size = noise_tiling;
	if (noise_texture.isInitialized())noise_texture = ImmutableTexture2D{ "SSFilter_SSAO::noise_texture" };
	noise_texture.allocateStorage(1, 1, TextureSize{ noise_pattern_size, noise_pattern_size, 1 }, InternalPixelFormat::SIZED_FLOAT_RG32);

	std::uniform_real<float> rand{ -1.0f, 1.0f };
	float* random_rotations = new float[noise_pattern_size*noise_pattern_size * 2];
	for (unsigned int i = 0; i < noise_pattern_size*noise_pattern_size; ++i)
	{
		random_rotations[2*i] = rand(random_engine);
		random_rotations[2 * i + 1] = rand(random_engine);
	}
	noise_texture.setMipmapLevelData(0, PixelLayout::RG, PixelDataType::FLOAT, random_rotations);

	delete[] random_rotations;
}


unsigned int SSFilter_SSAO::getNoiseSize() const{ return noise_pattern_size; }


void SSFilter_SSAO::setNumberOfSamples(unsigned int num_samples)
{
	this->num_samples = std::min(std::max(2U, num_samples), max_sample_points);
	if(isInitialized()) getFilterShaderProgram()->assignUniformScalar("numSamples", num_samples);
}


unsigned int SSFilter_SSAO::getNumberOfSamples() const { return num_samples; }


void SSFilter_SSAO::updateSamplePoints()
{
	std::vector<vec3> SSAO_kernel(static_cast<std::vector<vec3>::size_type>(max_sample_points));	//this vector will receive new samples
	
	//The samples are having standard normal distribution, so that the kernel has more samples closer to the fragment being tested for occlusion
	std::uniform_real<float> rand{-1.0f, 1.0f};

	for (unsigned int i = 0; i < max_sample_points; ++i)
		SSAO_kernel[i] = vec3{ rand(random_engine), rand(random_engine), (rand(random_engine) + 1.0f)*0.5f }.get_normalized();

	ssao_parameters_block.resetOffsetCounter();	//reset write offset of the uniform buffer object
	ssao_parameters_block.pushVector(SSAO_kernel);	//update sample information
}



void SSFilter_SSAO::defineScreenSpaceNormalMap(const ImmutableTexture2D& ss_normal_map)
{
	setTextureSource(0, ss_normal_map);
}

void SSFilter_SSAO::defineLinearDepthBuffer(const ImmutableTexture2D& linear_depth_buffer)
{
	setTextureSource(1, linear_depth_buffer);
}
