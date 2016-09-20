#ifndef TW__FRACTAL_NOISE__

#include <vector>
#include <cstdint>

#include "BufferTexture.h"
#include "ImmutableTexture2D.h"
#include "ImmutableTexture3D.h"
#include "ImageUnit.h"
#include "CompleteShaderProgram.h"



namespace tiny_world
{
	//Implements Perlin noise scaled multiple times and added to itself to create fractal noise.
	//The computed noise patterns are guaranteed to tile without stitches
	class FractalNoise : public Entity
	{
	private:
		enum class NoiseDimension : char { _2D = 2, _3D = 3 };


		NoiseDimension noise_dimension;		//dimension of the noise
		uint32_t grid_base_resolution_x;	//horizontal resolution of the gradient grid at the base level of the noise
		uint32_t grid_base_resolution_y;	//vertical resolution of the gradient grid at the base level of the noise
		uint32_t grid_base_resolution_z;	//depth resolution of the gradient grid at the base level of the noise
		uint32_t out_noise_map_resolution_x;	//horizontal resolution of the output noise map
		uint32_t out_noise_map_resolution_y;	//vertical resolution of the output noise map
		uint32_t out_noise_map_resolution_z;	//depth resolution of the output noise map
		uint32_t num_scales;		//number of scales used to create fractal noise

		bool first_call;	//equals 'true' until generateNoiseMap() is called for the first time
		bool update_gradients;	//equals 'true' if the gradient grid should be updated based on the values, which were generated upon the last invocation of generateNoiseMap()
		bool is_periodic;	//equals 'true' if the noise map being generated should be periodic, equals 'false' otherwise
		float gradient_update_rate;	//has effect only if update_gradients=true. This parameter describes how rapid should be the update applied to the gradients determining appearance of the noise map


		void init_shaders();	//helper function, which performs initialization of the compute shaders


	protected:
		enum class NoiseCalculationStage { GRADIENT_GRID_GENERATION, PERLIN_NOISE_CALCULATION };


		BufferTexture perlin_noise_grid;	//texture receiving random gradient field used to generate the Perlin noise
		CompleteShaderProgram perlin_noise_create_grid_compute_program;		//compute shader program, which generates random 2D vector field used to create the Perlin noise at given scale level of the Fractal noise
		CompleteShaderProgram perlin_noise_create_values_compute_program;	//compute shader program, which generates values of the Perlin noise at given scale level of the Fractal noise
		bool is_noise_map_valid;	//equals 'true' if the noise map contains valid data; equals 'false' otherwise


		//Initializes compute shader infrastructure for 2D fractal noise with the given resolutions of the output noise map and of the gradient field, and with the given number of scale levels
		FractalNoise(uint32_t noise_resolution_x, uint32_t noise_resolution_y, 
			uint32_t gradient_field_resolution_x, uint32_t gradient_field_resolution_y, uint32_t num_scale_levels);

		//Initializes compute shader infrastructure for 3D fractal noise with the given resolutions of the output noise map and of the gradient field, and with the given number of scale levels
		FractalNoise(uint32_t noise_resolution_x, uint32_t noise_resolution_y, uint32_t noise_resolution_z, 
			uint32_t gradient_field_resolution_x, uint32_t gradient_field_resolution_y, uint32_t gradient_field_resolution_z, uint32_t num_scale_levels);

		//Copy constructor
		FractalNoise(const FractalNoise& other);

		//Move constructor
		FractalNoise(FractalNoise&& other);

		//Copy assignment operator
		FractalNoise& operator=(const FractalNoise& other);

		//Move assignment operator
		FractalNoise& operator=(FractalNoise&& other);

		//Destructor
		virtual ~FractalNoise();

		//Invokes currently active compute shader. Which exactly compute shader has been activated is determined via the value of 'stage'.
		//Size of the gradient grid at the currently processed scale level is passed via uv3GradientGridResolution
		virtual void invokeNoiseMapCalculation(NoiseCalculationStage stage, const uvec3& uv3GradientGridResolution) = 0;


	public:
		//Generates new noise map in accordance with the parameters provided to the constructor
		void generateNoiseMap();

		//Returns dimension of the fractal noise
		char getDimension() const;


		//Sets whether the noise should behave as a continuous random process when generated repeatedly
		void setContinuity(bool continuity_flag);

		//Returns 'true' if the noise continuity flag is set; returns 'false' otherwise
		bool getContinuity() const;


		//Sets whether the noise map being generated should be periodic
		void setPeriodicity(bool periodicity_flag);

		//Returns 'true' if the noise periodicity flag is set; returns 'false' otherwise
		bool getPeriodicity() const;


		//Sets perceived rate of the update applied to the gradient grid of the noise map. The value passed to this
		//function will only have an effect if the continuity flag has been set to 'true' (i.e. if setContinuity(true) has been invoked prior to calling this function)
		void setEvolutionRate(float rate);

		//Returns perceived rate of the update applied to the gradient grid of the noise map
		float getEvolutionRate() const;
	};



	//Class implementing two-dimensional fractal noise
	class FractalNoise2D final : public FractalNoise
	{
	private:
		static const uvec2 _2DWorkGroupDimension;	//size of the work group employed by compute shaders when calculating a 2D fractal noise

		//Textures employed by the 2D noise generator
		ImmutableTexture2D fractal_noise;	//texture receiving resulting fractal noise

		//Miscellaneous state variables
		uvec2 uv2PerlinNoiseComputeGridDimension;	//dimension of the compute grid employed when calculating each consequent layer of the Perlin noise


		//Invokes currently active compute shader. Which exactly compute shader has been activated is determined via the value of 'stage'.
		//Size of the gradient grid at the currently processed scale level is passed via uv3GradientGridResolution
		void invokeNoiseMapCalculation(NoiseCalculationStage stage, const uvec3& uv3GradientGridResolution) override;


	public:
		//Initializes and generates fractal noise with given resolution and number of scale levels. The gradient field used to produce the noise map is based on random draws obtained from uniform distribution. 
		//Note that dimensions of the gradient field must be less than the corresponding dimensions of the noise map.
		FractalNoise2D(uint32_t noise_resolution_x = 512U, uint32_t noise_resolution_y = 512U, 
			uint32_t gradient_field_resolution_x = 64U, uint32_t gradient_field_resolution_y = 64U, uint32_t num_scale_levels = 5U);

		//Copy constructor
		FractalNoise2D(const FractalNoise2D& other);

		//Move constructor
		FractalNoise2D(FractalNoise2D&& other);

		//Destructor
		~FractalNoise2D();

		//Copy-assignment operator
		FractalNoise2D& operator=(const FractalNoise2D& other);

		//Move-assignment operator
		FractalNoise2D& operator=(FractalNoise2D&& other);

		//Returns texture alias of the last generated noise map
		ImmutableTexture2D retrieveNoiseMap() const;

		//Allows to retrieve generated noise map using C++ explicit conversion syntax.
		explicit operator ImmutableTexture2D() const;
	};



	//Class implementing three-dimensional fractal noise 
	class FractalNoise3D final : public FractalNoise
	{
	private:
		static const uvec3 _3DWorkGroupDimension;	//size of the work group employed by compute shaders when calculating a 3D fractal noise

		//Textures employed by the 3D noise generator
		ImmutableTexture3D fractal_noise;	//texture receiving resulting fractal noise

		//Miscellaneous state variables
		uvec3 uv3PerlinNoiseComputeGridDimension;	//dimension of the compute grid employed when calculating each consequent layer of the Perlin noise


		//Invokes currently active compute shader. Which exactly compute shader has been activated is determined via the value of 'stage'.
		//Size of the gradient grid at the currently processed scale level is passed via uv3GradientGridResolution
		void invokeNoiseMapCalculation(NoiseCalculationStage stage, const uvec3& uv3GradientGridResolution) override;


	public:
		//Initializes and generates fractal noise with given resolution and number of scale levels. The gradient field used to produce the noise map is based on random draws obtained from uniform distribution. 
		//Note that dimensions of the gradient field must be less than the corresponding dimensions of the noise map.
		FractalNoise3D(uint32_t noise_resolution_x = 128U, uint32_t noise_resolution_y = 128U, uint32_t noise_resolution_z = 64U,
			uint32_t gradient_field_resolution_x = 32U, uint32_t gradient_field_resolution_y = 32U, uint32_t gradient_field_resolution_z = 16U,
			uint32_t num_scale_levels = 5);

		//Copy constructor
		FractalNoise3D(const FractalNoise3D& other);

		//Move constructor
		FractalNoise3D(FractalNoise3D&& other);

		//Destructor
		~FractalNoise3D();

		//Copy-assignment operator
		FractalNoise3D& operator=(const FractalNoise3D& other);

		//Move-assignment operator
		FractalNoise3D& operator=(FractalNoise3D&& other);

		//Returns texture alias of the last generated noise map
		ImmutableTexture3D retrieveNoiseMap() const;

		//Allows to retrieve generated noise map using C++ explicit conversion syntax
		explicit operator ImmutableTexture3D() const;
	};
}

#define TW__FRACTAL_NOISE__
#endif