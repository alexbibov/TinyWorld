#include <immintrin.h>
#include <cstdlib>
#include <ctime>

#include "FractalNoise.h"
#include "CompilationDirectives.h"
#include "AbstractRenderableObject.h"


using namespace tiny_world;


#define pi 3.1415926535897932384626433832795f


//Calculates number of significant bits in 32-bit unsigned integer x
uint32_t get_number_of_significant_bits(uint32_t x)
{
	unsigned long num_bits;
	
#ifdef TARGET_OS_WINDOWS
	unsigned char dst = _BitScanReverse(&num_bits, x);
	num_bits = (num_bits + 1)*dst;
#endif

#ifdef TARGET_OS_LINUX
	num_bits = 32 - __builtin_clz(x);
#endif

	return static_cast<uint32_t>(num_bits);
}


void FractalNoise::init_shaders()
{
	std::string shader_program_suffix = std::to_string(static_cast<char>(noise_dimension)) + "D";


	perlin_noise_create_grid_compute_program.setStringName("FractalNoise" + shader_program_suffix + "::perlin_noise_create_grid_compute_program");
	perlin_noise_create_values_compute_program.setStringName("FractalNoise" + shader_program_suffix + "::perlin_noise_create_values_compute_program");



	Shader perlin_noise_create_grid_shader{ ShaderProgram::getShaderBaseCatalog() +
		"PerlinNoise" + shader_program_suffix + "CreateGrid.cp.glsl", ShaderType::COMPUTE_SHADER,
		"FractalNoise" + shader_program_suffix + "::perlin_noise_create_grid_compute_program::compute_shader" };
	if (!perlin_noise_create_grid_shader)
	{
		set_error_state(true);

		std::string err_msg =
			std::string{ "Unable to create " + shader_program_suffix + " fractal noise: " } +
			perlin_noise_create_grid_shader.getErrorString();
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}
	perlin_noise_create_grid_compute_program.addShader(perlin_noise_create_grid_shader);
	perlin_noise_create_grid_compute_program.link();
	if (!perlin_noise_create_grid_compute_program)
	{
		set_error_state(true);

		std::string err_msg =
			std::string{ "Unable to create " + shader_program_suffix + " fractal noise: " } +
			perlin_noise_create_grid_compute_program.getErrorString();
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}


	Shader perlin_noise_create_values_shader{ ShaderProgram::getShaderBaseCatalog() + 
		"PerlinNoise" + shader_program_suffix + "CreateValues.cp.glsl", ShaderType::COMPUTE_SHADER,
		"FractalNoise" + shader_program_suffix + "::perlin_noise_create_values_compute_program::compute_shader" };
	if (!perlin_noise_create_values_shader)
	{
		set_error_state(true);

		std::string err_msg =
			std::string{ "Unable to create " + shader_program_suffix + " fractal noise: " } +
			perlin_noise_create_values_shader.getErrorString();
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}
	perlin_noise_create_values_compute_program.addShader(perlin_noise_create_values_shader);
	perlin_noise_create_values_compute_program.link();
	if (!perlin_noise_create_values_compute_program)
	{
		std::string err_msg = std::string{ "Unable to create " + shader_program_suffix + " fractal noise: " } +
			perlin_noise_create_values_compute_program.getErrorString();
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}
}


FractalNoise::FractalNoise(uint32_t noise_resolution_x, uint32_t noise_resolution_y, 
	uint32_t gradient_field_resolution_x, uint32_t gradient_field_resolution_y, uint32_t num_scale_levels) :
	Entity{ "FractalNoise2D" }, noise_dimension{ NoiseDimension::_2D }, 
	grid_base_resolution_x{ std::max(2U, gradient_field_resolution_x) }, grid_base_resolution_y{ std::max(2U, gradient_field_resolution_y) }, grid_base_resolution_z{ 1 },
	out_noise_map_resolution_x{ noise_resolution_x }, out_noise_map_resolution_y{ noise_resolution_y }, out_noise_map_resolution_z{ 1 },
	first_call{ true }, update_gradients{ false }, is_periodic{ false }, perlin_noise_grid{ "FractalNoise2D::perlin_noise_grid" }, gradient_update_rate{ 0.5f },
	is_noise_map_valid{ false }
{
	if (noise_resolution_x < 2 * gradient_field_resolution_x ||
		noise_resolution_y < 2 * gradient_field_resolution_y)
	{
		set_error_state(true);
		const char* err_msg = "Unable to create 2D fractal noise: horizontal and vertical resolutions of the output noise map must be at least two times as large as "
			"correspondingly the horizontal and vertical resolutions of the gradient grid";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	//Initialize compute shaders
	init_shaders();

	//Adjust the number of scale levels so it corresponds to resolution of the gradient field
	num_scales = std::max(std::min(std::min(get_number_of_significant_bits(gradient_field_resolution_x) - 1,
		get_number_of_significant_bits(gradient_field_resolution_y) - 1), num_scale_levels), 1U);

	//Allocate storage for the buffer texture, which will contain the gradient data
	size_t storage_size = grid_base_resolution_x*grid_base_resolution_y*sizeof(GLfloat) * 2;
	for (uint32_t i = 0; i < num_scales; ++i) storage_size += (grid_base_resolution_x >> i)*(grid_base_resolution_y >> i)*sizeof(GLfloat) * 2;
	perlin_noise_grid.allocateStorage(storage_size, BufferTextureInternalPixelFormat::SIZED_FLOAT_RG32);

	//Initialize random number generator 
	srand(static_cast<unsigned int>(time(nullptr)));
}


FractalNoise::FractalNoise(uint32_t noise_resolution_x, uint32_t noise_resolution_y, uint32_t noise_resolution_z, 
	uint32_t gradient_field_resolution_x, uint32_t gradient_field_resolution_y, uint32_t gradient_field_resolution_z, uint32_t num_scale_levels) : 
	Entity{ "FractalNoise3D" }, noise_dimension{ NoiseDimension::_3D },
	grid_base_resolution_x{ gradient_field_resolution_x }, grid_base_resolution_y{ gradient_field_resolution_y }, grid_base_resolution_z{ gradient_field_resolution_z },
	out_noise_map_resolution_x{ noise_resolution_x }, out_noise_map_resolution_y{ noise_resolution_y }, out_noise_map_resolution_z{ noise_resolution_z },
	first_call{ true }, update_gradients{ false }, is_periodic{ false }, perlin_noise_grid{ "FractalNoise3D::perlin_noise_grid" }, gradient_update_rate{ 0.5f },
	is_noise_map_valid{ false }
{
	if (noise_resolution_x < 2 * gradient_field_resolution_x ||
		noise_resolution_y < 2 * gradient_field_resolution_y ||
		noise_resolution_z < 2 * gradient_field_resolution_z)
	{
		set_error_state(true);
		const char* err_msg = "Unable to create 3D fractal noise: resolution along each of the noise map dimensions must be at least two times as large as "
			"the resolution of the gradient grid along the corresponding dimension";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	//Initialize compute shaders
	init_shaders();

	//Adjust the number of scale levels to correspond to the dimensions of the gradient field
	num_scales = 
		std::max(std::min(std::min(std::min(get_number_of_significant_bits(gradient_field_resolution_x) - 1,
		get_number_of_significant_bits(gradient_field_resolution_y) - 1),
		get_number_of_significant_bits(gradient_field_resolution_z) - 1), num_scale_levels), 1U);

	//Allocate storage for the buffer texture, which will contain the gradient data
	size_t storage_size = grid_base_resolution_x*grid_base_resolution_y*grid_base_resolution_z*sizeof(GLfloat) * 4;
	for (uint32_t i = 0; i < num_scales; ++i) storage_size += (grid_base_resolution_x >> i)*(grid_base_resolution_y >> i)*(grid_base_resolution_z >> i)*sizeof(GLfloat) * 4;
	perlin_noise_grid.allocateStorage(storage_size, BufferTextureInternalPixelFormat::SIZED_FLOAT_RGBA32);

	//Initialize random number generator 
	srand(static_cast<unsigned int>(time(nullptr)));
}


FractalNoise::FractalNoise(const FractalNoise& other) : Entity{ other },
noise_dimension{ other.noise_dimension }, grid_base_resolution_x{ other.grid_base_resolution_x },
grid_base_resolution_y{ other.grid_base_resolution_y }, grid_base_resolution_z{ other.grid_base_resolution_z },
out_noise_map_resolution_x{ other.out_noise_map_resolution_x }, out_noise_map_resolution_y{ other.out_noise_map_resolution_y }, out_noise_map_resolution_z{ other.out_noise_map_resolution_z },
num_scales{ other.num_scales }, first_call{ other.first_call }, update_gradients{ other.update_gradients }, is_periodic{ other.is_periodic }, gradient_update_rate{ other.gradient_update_rate },
perlin_noise_grid{ "FractalNoise3D::perlin_noise_grid" }, perlin_noise_create_grid_compute_program{ other.perlin_noise_create_grid_compute_program }, 
perlin_noise_create_values_compute_program{ other.perlin_noise_create_values_compute_program }, is_noise_map_valid{ other.is_noise_map_valid }
{
	//Allocate storage for the buffer texture, which will contain the gradient data
	size_t storage_size = grid_base_resolution_x*grid_base_resolution_y*grid_base_resolution_z*sizeof(GLfloat) * 4;
	for (uint32_t i = 0; i < num_scales; ++i) storage_size += (grid_base_resolution_x >> i)*(grid_base_resolution_y >> i)*(grid_base_resolution_z >> i)*sizeof(GLfloat) * 4;
	perlin_noise_grid.allocateStorage(storage_size, static_cast<BufferTextureInternalPixelFormat>(other.perlin_noise_grid.getStorageFormatTraits().getOpenGLFormatEnumerationValue()));
	if (other.is_noise_map_valid) other.perlin_noise_grid.copyTexelData(perlin_noise_grid);
}


FractalNoise::FractalNoise(FractalNoise&& other) : Entity{ std::move(other) },
noise_dimension{ std::move(other.noise_dimension) }, grid_base_resolution_x{ other.grid_base_resolution_x },
grid_base_resolution_y{ other.grid_base_resolution_y }, grid_base_resolution_z{ other.grid_base_resolution_z },
out_noise_map_resolution_x{ other.out_noise_map_resolution_x }, out_noise_map_resolution_y{ other.out_noise_map_resolution_y },
out_noise_map_resolution_z{ other.out_noise_map_resolution_z }, num_scales{ other.num_scales }, first_call{ other.first_call },
update_gradients{ other.update_gradients }, is_periodic{ other.is_periodic }, gradient_update_rate{ other.gradient_update_rate },
perlin_noise_grid{ std::move(other.perlin_noise_grid) }, perlin_noise_create_grid_compute_program{ std::move(other.perlin_noise_create_grid_compute_program) },
perlin_noise_create_values_compute_program{ std::move(perlin_noise_create_values_compute_program) }, is_noise_map_valid{ other.is_noise_map_valid }
{

}


FractalNoise::~FractalNoise()
{

}


FractalNoise& FractalNoise::operator=(const FractalNoise& other)
{
	if (this == &other)
		return *this;

	noise_dimension = other.noise_dimension;
	grid_base_resolution_x = other.grid_base_resolution_x;
	grid_base_resolution_y = other.grid_base_resolution_y;
	grid_base_resolution_z = other.grid_base_resolution_z;
	out_noise_map_resolution_x = other.out_noise_map_resolution_x;
	out_noise_map_resolution_y = other.out_noise_map_resolution_y;
	out_noise_map_resolution_z = other.out_noise_map_resolution_z;
	num_scales = other.num_scales;
	first_call = other.first_call;
	update_gradients = other.update_gradients;
	is_periodic = other.is_periodic;
	gradient_update_rate = other.gradient_update_rate;
	perlin_noise_grid.allocateStorage(other.perlin_noise_grid.getTextureBufferSize(),
		static_cast<BufferTextureInternalPixelFormat>(other.perlin_noise_grid.getStorageFormatTraits().getOpenGLFormatEnumerationValue()));
	if (other.is_noise_map_valid) other.perlin_noise_grid.copyTexelData(perlin_noise_grid);
	perlin_noise_create_grid_compute_program = other.perlin_noise_create_grid_compute_program;
	perlin_noise_create_values_compute_program = other.perlin_noise_create_values_compute_program;
	is_noise_map_valid = other.is_noise_map_valid;

	return *this;
}


FractalNoise& FractalNoise::operator=(FractalNoise&& other)
{
	if (this == &other)
		return *this;

	noise_dimension = other.noise_dimension;
	grid_base_resolution_x = other.grid_base_resolution_x;
	grid_base_resolution_y = other.grid_base_resolution_y;
	grid_base_resolution_z = other.grid_base_resolution_z;
	out_noise_map_resolution_x = other.out_noise_map_resolution_x;
	out_noise_map_resolution_y = other.out_noise_map_resolution_y;
	out_noise_map_resolution_z = other.out_noise_map_resolution_z;
	num_scales = other.num_scales;
	first_call = other.first_call;
	update_gradients = other.update_gradients;
	is_periodic = other.is_periodic;
	gradient_update_rate = other.gradient_update_rate;
	perlin_noise_grid = std::move(other.perlin_noise_grid);
	perlin_noise_create_grid_compute_program = std::move(other.perlin_noise_create_grid_compute_program);
	perlin_noise_create_values_compute_program = std::move(other.perlin_noise_create_values_compute_program);
	is_noise_map_valid = other.is_noise_map_valid;
	
	return *this;
}


void FractalNoise::generateNoiseMap()
{
	uint32_t offset = 0;	//offset from the beginning of the buffer containing the gradient field

	for (uint32_t current_scale_level = 0; current_scale_level < num_scales; ++current_scale_level)
	{
		uvec3 uv3GradientGridDimension{ grid_base_resolution_x >> current_scale_level, grid_base_resolution_y >> current_scale_level, grid_base_resolution_z >> current_scale_level };

		//First step: compute the gradient field data
		uvec4 uv4Seed{ static_cast<uint32_t>(rand()), static_cast<uint32_t>(rand()), static_cast<uint32_t>(rand()), static_cast<uint32_t>(rand()) };
		perlin_noise_create_grid_compute_program.assignUniformVector("uv4Seed", uv4Seed);

		perlin_noise_create_grid_compute_program.assignUniformScalar("bFirstCall", static_cast<int>(first_call));
		perlin_noise_create_grid_compute_program.assignUniformScalar("bUpdateGradients", static_cast<int>(update_gradients));
		perlin_noise_create_grid_compute_program.assignUniformScalar("uOffset", offset);
		perlin_noise_create_grid_compute_program.assignUniformScalar("fGradientUpdateRate", gradient_update_rate);

		perlin_noise_create_grid_compute_program.activate();
		invokeNoiseMapCalculation(NoiseCalculationStage::GRADIENT_GRID_GENERATION, uv3GradientGridDimension);


		//Second step: generate noise map at the current scale level
		perlin_noise_create_values_compute_program.assignUniformScalar("uOffset", offset);
		perlin_noise_create_values_compute_program.assignUniformScalar("bFirstCall", static_cast<int>(current_scale_level == 0));
		perlin_noise_create_values_compute_program.assignUniformScalar("uNumScaleLevels", num_scales);
		perlin_noise_create_values_compute_program.assignUniformScalar("isPeriodic", static_cast<int>(is_periodic));
		perlin_noise_create_values_compute_program.activate();
		invokeNoiseMapCalculation(NoiseCalculationStage::PERLIN_NOISE_CALCULATION, uv3GradientGridDimension);


		offset += (grid_base_resolution_x >> current_scale_level)*(grid_base_resolution_y >> current_scale_level)*std::max(grid_base_resolution_z >> current_scale_level, 1U);
	}

	first_call = false;
	is_noise_map_valid = true;
}


char FractalNoise::getDimension() const { return static_cast<char>(noise_dimension); }

void FractalNoise::setContinuity(bool continuity_flag) { update_gradients = continuity_flag; }

bool FractalNoise::getContinuity() const { return update_gradients; }

void FractalNoise::setPeriodicity(bool periodicity_flag) { is_periodic = periodicity_flag; }

bool FractalNoise::getPeriodicity() const { return is_periodic; }

void FractalNoise::setEvolutionRate(float rate) { gradient_update_rate = rate; }

float FractalNoise::getEvolutionRate() const { return gradient_update_rate; }




const uvec2 FractalNoise2D::_2DWorkGroupDimension = uvec2{ 32U, 32U };


void FractalNoise2D::invokeNoiseMapCalculation(NoiseCalculationStage stage, const uvec3& uv3GradientGridResolution)
{
	ImageUnit gradientFieldStorageBuffer;
	gradientFieldStorageBuffer.setStringName("FractalNoise2D::gradientFieldStorageBuffer");
	gradientFieldStorageBuffer.attachTexture(perlin_noise_grid, BufferAccess::READ_WRITE, BufferTextureInternalPixelFormat::SIZED_FLOAT_RG32);

	switch (stage)
	{
	case NoiseCalculationStage::GRADIENT_GRID_GENERATION:
	{
		perlin_noise_create_grid_compute_program.assignUniformScalar("ibGradientGrid", gradientFieldStorageBuffer.getBinding());

		perlin_noise_create_grid_compute_program.assignUniformVector("uv2GradientGridSize", uvec2{ uv3GradientGridResolution.x, uv3GradientGridResolution.y });

		glDispatchCompute(uv3GradientGridResolution.x / _2DWorkGroupDimension.x + static_cast<GLuint>(uv3GradientGridResolution.x%_2DWorkGroupDimension.x != 0),
			uv3GradientGridResolution.y / _2DWorkGroupDimension.y + static_cast<GLuint>(uv3GradientGridResolution.y%_2DWorkGroupDimension.y != 0), 1);
		gradientFieldStorageBuffer.flush();
		break;
	}

	case NoiseCalculationStage::PERLIN_NOISE_CALCULATION:
	{
		perlin_noise_create_values_compute_program.assignUniformScalar("ibGradientGrid", gradientFieldStorageBuffer.getBinding());

		perlin_noise_create_values_compute_program.assignUniformVector("uv2GradientGridSize", uvec2{ uv3GradientGridResolution.x, uv3GradientGridResolution.y });

		ImageUnit fractalNoiseStorageBuffer;
		fractalNoiseStorageBuffer.setStringName("FractalNoise2D::fractalNoiseStorageBuffer");
		fractalNoiseStorageBuffer.attachTexture(fractal_noise, 0, BufferAccess::READ_WRITE, InternalPixelFormat::SIZED_FLOAT_R32);
		perlin_noise_create_values_compute_program.assignUniformScalar("i2dOutNoise", fractalNoiseStorageBuffer.getBinding());

		glDispatchCompute(uv2PerlinNoiseComputeGridDimension.x, uv2PerlinNoiseComputeGridDimension.y, 1);
		fractalNoiseStorageBuffer.flush();
		fractal_noise.generateMipmapLevels();
		break;
	}
	}
}


FractalNoise2D::FractalNoise2D(uint32_t noise_resolution_x /* = 512U */, uint32_t noise_resolution_y /* = 512U */,
	uint32_t gradient_field_resolution_x /* = 64U */, uint32_t gradient_field_resolution_y /* = 64U */,
	uint32_t num_scale_levels /* = 5U */) : 

	FractalNoise(noise_resolution_x, noise_resolution_y, gradient_field_resolution_x, gradient_field_resolution_y, num_scale_levels),
	uv2PerlinNoiseComputeGridDimension{ noise_resolution_x / _2DWorkGroupDimension.x + static_cast<GLuint>(noise_resolution_x%_2DWorkGroupDimension.x != 0),
	noise_resolution_y / _2DWorkGroupDimension.y + static_cast<GLuint>(noise_resolution_y%_2DWorkGroupDimension.y != 0) }

{
	uint32_t num_mipmap_levels = std::min(get_number_of_significant_bits(noise_resolution_x), 
		get_number_of_significant_bits(noise_resolution_y));

	fractal_noise.setStringName("FractalNoise2D::noise_map");
	fractal_noise.allocateStorage(num_mipmap_levels, 1, TextureSize{ noise_resolution_x, noise_resolution_y, 1 }, InternalPixelFormat::SIZED_FLOAT_R32);
}


FractalNoise2D::FractalNoise2D(const FractalNoise2D& other) : FractalNoise(other),
uv2PerlinNoiseComputeGridDimension(other.uv2PerlinNoiseComputeGridDimension)
{
	fractal_noise.allocateStorage(other.fractal_noise.getNumberOfMipmapLevels(), 1, other.fractal_noise.getTextureSize(), InternalPixelFormat::SIZED_FLOAT_R32);
	if (other.is_noise_map_valid)
	{
		TextureSize tex_size = other.fractal_noise.getTextureSize();
		for (uint32_t mipmap_level = 0; mipmap_level < other.fractal_noise.getNumberOfMipmapLevels(); ++mipmap_level)
		{
			uint32_t tex_width = tex_size.width >> mipmap_level;
			uint32_t tex_height = tex_size.height >> mipmap_level;
			other.fractal_noise.copyTexelData(mipmap_level, 0, 0, 0, fractal_noise, mipmap_level, 0, 0, 0, tex_width, tex_height, 1);
		}
	}
}


FractalNoise2D::FractalNoise2D(FractalNoise2D&& other) : FractalNoise(std::move(other)),
fractal_noise(std::move(other.fractal_noise)), uv2PerlinNoiseComputeGridDimension(std::move(other.uv2PerlinNoiseComputeGridDimension))
{

}


FractalNoise2D::~FractalNoise2D()
{

}


FractalNoise2D& FractalNoise2D::operator=(const FractalNoise2D& other)
{
	if (this == &other)
		return *this;

	FractalNoise::operator=(other);
	uv2PerlinNoiseComputeGridDimension = other.uv2PerlinNoiseComputeGridDimension;

	if (other.is_noise_map_valid)
	{
		fractal_noise = ImmutableTexture2D{ "FractalNoise2D::noise_map" };
		fractal_noise.allocateStorage(other.fractal_noise.getNumberOfMipmapLevels(), 1, other.fractal_noise.getTextureSize(), InternalPixelFormat::SIZED_FLOAT_R32);
		TextureSize tex_size = other.fractal_noise.getTextureSize();
		for (uint32_t mipmap_level = 0; mipmap_level < other.fractal_noise.getNumberOfMipmapLevels(); ++mipmap_level)
		{
			uint32_t tex_width = tex_size.width >> mipmap_level;
			uint32_t tex_height = tex_size.height >> mipmap_level;
			other.fractal_noise.copyTexelData(mipmap_level, 0, 0, 0, fractal_noise, mipmap_level, 0, 0, 0, tex_width, tex_height, 1);
		}
	}

	return *this;
}


FractalNoise2D& FractalNoise2D::operator=(FractalNoise2D&& other)
{
	if (this == &other)
		return *this;

	FractalNoise::operator=(std::move(other));

	fractal_noise = std::move(other.fractal_noise);
	uv2PerlinNoiseComputeGridDimension = std::move(other.uv2PerlinNoiseComputeGridDimension);

	return *this;
}


ImmutableTexture2D FractalNoise2D::retrieveNoiseMap() const{ return fractal_noise; }


FractalNoise2D::operator tiny_world::ImmutableTexture2D() const{ return fractal_noise; }




const uvec3 FractalNoise3D::_3DWorkGroupDimension = uvec3{ 8U, 8U, 16U };


void FractalNoise3D::invokeNoiseMapCalculation(NoiseCalculationStage stage, const uvec3& uv3GradientGridResolution)
{
	ImageUnit gradientFieldStorageBuffer;
	gradientFieldStorageBuffer.setStringName("FractalNoise3D::gradientFieldStorageBuffer");
	gradientFieldStorageBuffer.attachTexture(perlin_noise_grid, BufferAccess::READ_WRITE, BufferTextureInternalPixelFormat::SIZED_FLOAT_RGBA32);

	switch (stage)
	{
	case NoiseCalculationStage::GRADIENT_GRID_GENERATION:
	{
		perlin_noise_create_grid_compute_program.assignUniformScalar("ibGradientGrid", gradientFieldStorageBuffer.getBinding());

		perlin_noise_create_grid_compute_program.assignUniformVector("uv3GradientGridSize", uv3GradientGridResolution);

		glDispatchCompute(uv3GradientGridResolution.x / _3DWorkGroupDimension.x + static_cast<GLuint>(uv3GradientGridResolution.x % _3DWorkGroupDimension.x != 0),
			uv3GradientGridResolution.y / _3DWorkGroupDimension.y + static_cast<GLuint>(uv3GradientGridResolution.y % _3DWorkGroupDimension.y != 0),
			uv3GradientGridResolution.z / _3DWorkGroupDimension.z + static_cast<GLuint>(uv3GradientGridResolution.z % _3DWorkGroupDimension.z != 0));
		gradientFieldStorageBuffer.flush();
		break;
	}

	case NoiseCalculationStage::PERLIN_NOISE_CALCULATION:
	{
		perlin_noise_create_values_compute_program.assignUniformScalar("ibGradientGrid", gradientFieldStorageBuffer.getBinding());

		perlin_noise_create_values_compute_program.assignUniformVector("uv3GradientGridSize", uv3GradientGridResolution);

		ImageUnit fractalNoiseStorageBuffer;
		fractalNoiseStorageBuffer.setStringName("FractalNoise3D::fractalNoiseStorageBuffer");
		fractalNoiseStorageBuffer.attachTexture(fractal_noise, 0, BufferAccess::READ_WRITE, InternalPixelFormat::SIZED_FLOAT_R32);
		perlin_noise_create_values_compute_program.assignUniformScalar("i3dOutNoise", fractalNoiseStorageBuffer.getBinding());

		glDispatchCompute(uv3PerlinNoiseComputeGridDimension.x, uv3PerlinNoiseComputeGridDimension.y, uv3PerlinNoiseComputeGridDimension.z);
		fractalNoiseStorageBuffer.flush();
		fractal_noise.generateMipmapLevels();
		break;
	}
	}
}


FractalNoise3D::FractalNoise3D(uint32_t noise_resolution_x /* = 128U */, uint32_t noise_resolution_y /* = 128U */, uint32_t noise_resolution_z /* = 64U */,
	uint32_t gradient_field_resolution_x /* = 32U */, uint32_t gradient_field_resolution_y /* = 32U */, uint32_t gradient_field_resolution_z /* = 16U */, uint32_t num_scale_levels /* = 5 */) :

	FractalNoise(noise_resolution_x, noise_resolution_y, noise_resolution_z, gradient_field_resolution_x, gradient_field_resolution_y, gradient_field_resolution_z, num_scale_levels),
	uv3PerlinNoiseComputeGridDimension{ noise_resolution_x / _3DWorkGroupDimension.x + static_cast<GLuint>(noise_resolution_x % _3DWorkGroupDimension.x != 0),
	noise_resolution_y / _3DWorkGroupDimension.y + static_cast<GLuint>(noise_resolution_y % _3DWorkGroupDimension.y != 0),
	noise_resolution_z / _3DWorkGroupDimension.z + static_cast<GLuint>(noise_resolution_z % _3DWorkGroupDimension.z != 0) }

{
	uint32_t num_mipmap_levels = 
		std::min(std::min(get_number_of_significant_bits(noise_resolution_x), get_number_of_significant_bits(noise_resolution_y)),
		get_number_of_significant_bits(noise_resolution_z));

	fractal_noise.setStringName("FractalNoise3D::noise_map");
	fractal_noise.allocateStorage(num_mipmap_levels, 1, TextureSize{ noise_resolution_x, noise_resolution_y, noise_resolution_z }, InternalPixelFormat::SIZED_FLOAT_R32);
}


FractalNoise3D::FractalNoise3D(const FractalNoise3D& other) : FractalNoise(other),
uv3PerlinNoiseComputeGridDimension(other.uv3PerlinNoiseComputeGridDimension)
{
	fractal_noise.allocateStorage(other.fractal_noise.getNumberOfMipmapLevels(), 1, other.fractal_noise.getTextureSize(), InternalPixelFormat::SIZED_FLOAT_R32);
	if (other.is_noise_map_valid)
	{
		TextureSize tex_size = other.fractal_noise.getTextureSize();
		for (uint32_t mipmap_level = 0; mipmap_level < other.fractal_noise.getNumberOfMipmapLevels(); ++mipmap_level)
		{
			uint32_t tex_width = tex_size.width >> mipmap_level;
			uint32_t tex_height = tex_size.height >> mipmap_level;
			uint32_t tex_depth = tex_size.depth >> mipmap_level;
			other.fractal_noise.copyTexelData(mipmap_level, 0, 0, 0, fractal_noise, mipmap_level, 0, 0, 0, tex_width, tex_height, tex_depth);
		}
	}
}


FractalNoise3D::FractalNoise3D(FractalNoise3D&& other) :FractalNoise(std::move(other)),
fractal_noise(std::move(other.fractal_noise)), uv3PerlinNoiseComputeGridDimension(std::move(other.uv3PerlinNoiseComputeGridDimension))
{

}


FractalNoise3D::~FractalNoise3D()
{

}


FractalNoise3D& FractalNoise3D::operator=(const FractalNoise3D& other)
{
	if (this == &other)
		return *this;

	FractalNoise::operator=(other);

	uv3PerlinNoiseComputeGridDimension = other.uv3PerlinNoiseComputeGridDimension;
	if (other.is_noise_map_valid)
	{
		fractal_noise = ImmutableTexture3D{ "FractalNoise3D::noisemap" };
		fractal_noise.allocateStorage(other.fractal_noise.getNumberOfMipmapLevels(), 1, other.fractal_noise.getTextureSize(), InternalPixelFormat::SIZED_FLOAT_R32);
		TextureSize tex_size = other.fractal_noise.getTextureSize();
		for (uint32_t mipmap_level = 0; mipmap_level < other.fractal_noise.getNumberOfMipmapLevels(); ++mipmap_level)
		{
			uint32_t tex_width = tex_size.width >> mipmap_level;
			uint32_t tex_height = tex_size.height >> mipmap_level;
			uint32_t tex_depth = tex_size.depth >> mipmap_level;
			other.fractal_noise.copyTexelData(mipmap_level, 0, 0, 0, fractal_noise, mipmap_level, 0, 0, 0, tex_width, tex_height, tex_depth);
		}
	}

	return *this;
}


FractalNoise3D& FractalNoise3D::operator=(FractalNoise3D&& other)
{
	if (this == &other)
		return *this;

	FractalNoise::operator=(std::move(other));

	fractal_noise = std::move(other.fractal_noise);
	uv3PerlinNoiseComputeGridDimension = std::move(other.uv3PerlinNoiseComputeGridDimension);

	return *this;
}


ImmutableTexture3D FractalNoise3D::retrieveNoiseMap() const { return fractal_noise; }


FractalNoise3D::operator tiny_world::ImmutableTexture3D() const { return fractal_noise; }

