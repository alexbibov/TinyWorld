 //Implements shallow water based on Kurganov-Petrova scheme


#ifndef TW__KPWATER__

#include <functional>
#include <random>

#include "AbstractRenderableObject.h"
#include "AbstractRenderableObjectLightEx.h"
#include "AbstractRenderableObjectHDRBloomEx.h"
#include "AbstractRenderableObjectSelectionEx.h"
#include "KPWater_CUDA/SaintVenantSystemCore.cuh"
#include "ImageUnit.h"
#include "FullscreenRectangle.h"
#include "Framebuffer.h"
#include "FractalNoise.h"

#include "../tw_shaders/KPWaterCommonDefinitions.inc"

namespace tiny_world
{
	class KPWater final: virtual public AbstractRenderableObjectTextured,
		public AbstractRenderableObjectExtensionAggregator < AbstractRenderableObjectLightEx, AbstractRenderableObjectHDRBloomEx, AbstractRenderableObjectSelectionEx >
	{
	public:
		//Enumerates ODE-solvers employed by Kurganov-Petrova scheme
		enum class ODESolver
		{
			RungeKutta22,
			RungeKutta33,
			Euler
		};

		//Describes boundary condition as expected by Kurganov-Petrova scheme
		class BoundaryCondition
		{
		private:
			uint32_t length;	//number of elements in the boundary condition
			size_t capacity;	//number of boundary elements this object could store

			SaintVenantSystem::Numeric* w_center;	//water levels at the center of the bounding volumes
			SaintVenantSystem::Numeric* w_edge;	//water levels at the "bounding edge" of the bounding volumes

			SaintVenantSystem::Numeric* hu_center;	//horizontal flux components defined at the center points of the bounding volumes
			SaintVenantSystem::Numeric* hu_edge;	//horizontal flux components defined at the middle of the "bounding edges" of the bounding volumes

			SaintVenantSystem::Numeric* hv_center;	//vertical flux components defined at the center points of the bounding volumes
			SaintVenantSystem::Numeric* hv_edge;	//vertical flux components defined at the middle of the "bounding edges" of the bounding volumes

		public:
			BoundaryCondition();	//initializes an empty boundary condition
			BoundaryCondition(uint32_t length);	//initializes boundary condition of the specified length
			BoundaryCondition(const BoundaryCondition& other);	//copy constructor
			BoundaryCondition(BoundaryCondition&& other);	//move constructor
			~BoundaryCondition();	//destructor

			BoundaryCondition& operator=(const BoundaryCondition& other);	//copy assignment operator
			BoundaryCondition& operator=(BoundaryCondition&& other);	//move assignment operator

			//Assigns value to a boundary condition component
			void setElement(uint32_t index, SaintVenantSystem::Numeric w_center_value, SaintVenantSystem::Numeric w_edge_value,
				SaintVenantSystem::Numeric hu_center_value, SaintVenantSystem::Numeric hu_edge_value,
				SaintVenantSystem::Numeric hv_center_value, SaintVenantSystem::Numeric hv_edge_value);

			//Retrieves an element from the boundary condition
			void getElement(uint32_t index, SaintVenantSystem::Numeric& w_center_value, SaintVenantSystem::Numeric& w_edge_value,
				SaintVenantSystem::Numeric& hu_center_value, SaintVenantSystem::Numeric& hu_edge_value,
				SaintVenantSystem::Numeric& hv_center_value, SaintVenantSystem::Numeric& hv_edge_value) const;

			//Returns number of elements in the boundary condition
			uint32_t getLength() const;

			//Provides pointers to the raw data representing the boundary condition
			void getRawData(const SaintVenantSystem::Numeric** p_w_center_values, const SaintVenantSystem::Numeric** p_w_edge_values,
				const SaintVenantSystem::Numeric** p_hu_center_values, const SaintVenantSystem::Numeric** p_hu_edge_values,
				const SaintVenantSystem::Numeric** p_hv_center_values, const SaintVenantSystem::Numeric** p_hv_edge_values) const;

			//Provides pointers to the raw data representing the boundary condition
			void getRawData(SaintVenantSystem::Numeric** p_w_center_values, SaintVenantSystem::Numeric** p_w_edge_values,
				SaintVenantSystem::Numeric** p_hu_center_values, SaintVenantSystem::Numeric** p_hu_edge_values,
				SaintVenantSystem::Numeric** p_hv_center_values, SaintVenantSystem::Numeric** p_hv_edge_values);
		};

		//Type describing compute procedure used to deduce the boundary conditions
		using BoundaryConditionsComputeProcedure = std::function < void(const SaintVenantSystem::SVSCore*, const SaintVenantSystem::Numeric* p_interpolated_topography,
			BoundaryCondition&, BoundaryCondition&, BoundaryCondition&, BoundaryCondition&) >;

	private:
		SaintVenantSystem::SVSCore kpwater_cuda;	//shallow-water simulator based on Kurganov-Petrova numerical scheme. The host system is required to support CUDA
		SaintVenantSystem::SVSCore::DomainSettings domain_settings;	//settings of the computational domain
		bool is_initialized;	//equals 'true' if Kurganov-Petrova numerics has been initialized
		ODESolver ode_solver;	//ODE solver used to perform time-stepping in Kurganov-Petrova scheme	
		BoundaryConditionsComputeProcedure bc_callback;	//callback function used to compute the boundary conditions
		float g;	//gravity constant
		float theta;	//numerical dissipation constant. Must be in range [1, 2]. Larger values correspond to less dissipative but more oscillatory solutions
		float eps;	//de-singularization constant. Recommended value is max(dx^4, dy^4)

		float* p_water_heightmap_data;	//auxiliary buffer containing water height map data represented in single-precision floating point format
		float* p_topography_heightmap_data;	//auxiliary buffer containing "integer" nodes of the interpolated topography used to construct topography height map texture
		SaintVenantSystem::Numeric* p_interpolated_topography_heightmap_data;	//bilinear interpolation of the original topography height map

		float lod_factor;	//level-of-detail factor controlling tessellation of the water surface
		float max_light_penetration_depth;	//maximal water level, through which light is still able to penetrate. All the points that are located deeper under water receive "no light energy"

		Texture2DResource water_heightmap_tex_res;	//water height map texture resource
		Texture2DResource topography_heightmap_tex_res;	//texture resource containing topography height map

		Texture2DResource refraction_texture_with_caustics_tex_res;	//texture resource wrapping refraction texture modulated by caustics
		ImmutableTexture2D refraction_texture_depth_map;		//depth map corresponding to the refraction texture
		TextureSamplerReferenceCode refraction_texture_sampler_ref_code;	//reference code of the sampler object used by refraction texture
		TextureSamplerReferenceCode ripple_texture_sampler_ref_code;		//reference code of the sampler object used by the ripple and displacement maps generated by the Tessendorf algorithm
		TextureSamplerReferenceCode normal_texture_sampler_ref_code;		//reference code of the sampler object used by the normal map generated by the Tessendorf algorithm

		bool force_water_heightmap_update;	//equals 'true' when the object has been created by copying an initialized KPWater object. This is needed to avoid cloning of the height map textures between the objects

		ShaderProgramReferenceCode water_rendering_program_ref_code;	//rendering program used to shade the water surface
		ShaderProgramReferenceCode fft_compute_program_ref_code;	//OpenGL compute program responsible for computing the FFTs
		static const std::string water_rendering_program0_name;	//string name of the default water rendering program
		static const std::string fft_compute_program_name;	//string name of the compute program responsible for calculation of the FFTs

		uint32_t tess_billet_horizontal_resolution, tess_billet_vertical_resolution;	//scale factors applied to the high frequency wave maps
		float max_deep_water_wave_amplitude;	//maximal height of the deep water waves (represented using dimensional units!)
		float max_capillary_wave_amplitude;	//maximal amplitude of the capillary waves (represented using dimensional units!)
		vec2 v2RippleSimulationTime;	//current time instant of the ripple dynamics simulation for the global and the capillary scales packed in this order into a 2D vector
		vec2 v2RippleSimulationDt;	//time difference between two consequent frames of ripple simulation. This parameter includes values for two scales: global wave scale and capillary ripple scale packed into a 2D-vector in this order
		//float fresnel_power;	//exponent term applied when calculating the Fresnel factor
		vec3 v3ColorExtinctionFactors;	//specifies factors affecting how fast each of the red, green, and blue color channels gets attenuated by water. Default value is (1535, 92, 23)
		vec2 v2WindVelocity;	//direction of the wind speed affecting spatial propagation of the capillary ripples. This is not a complete 3D speed vector, but its projection onto the XZ-plane

		std::default_random_engine random_number_generator;	//random number generator used by certain initialization routines
		std::normal_distribution<float> standard_gaussian_distribution;	//represents Gaussian distribution with 0 mean and unit variance

		static const uint32_t fft_size = TW__KPWATER_FFT_SIZE__;	//Default size of the FFT
		Texture2DResource phillips_spectrum_tex_res;	//texture resource wrapping the Phillips spectrum used to compute the FFT-based waves
		Texture2DResource fft_ripples_tex_res;		//Texture resource wrapping the ripples generated by FFT
		Texture2DResource fft_displacement_map_tex_res;	//Texture resource wrapping displacement map, which is used to create choppy waves
		Texture2DResource fft_ripples_normal_map_global_scale_tex_res;	//Texture resource wrapping the normal map of the ripples generated by FFT for the global scale waves
		Texture2DResource fft_ripples_normal_map_capillary_scale_tex_res;	//Texture resource wrapping the normal map of the ripples generated by FFT for the capillary scale waves
		float choppiness;	//affects choppiness of the deep water waves. Default value is 0.1f;
		uvec2 uv2DeepWaterRippleMapTilingFactor;	//describes tiling of the ripple map generated by FFTs
		float max_wave_height_as_elevation_fraction;	//determines ratio between maximal allowed height of the deep water waves and the deepness of the water


		FractalNoise2D fractal_noise;	//fractal noise employed to eliminate repeated patterns on the water surface
		Texture2DResource fractal_noise_map_tex_res;	//texture resource associated with the fractal noise map
		uint32_t fractal_noise_update_counter;	//counter used to update the fractal noise map
		static const uint32_t fractal_noise_update_period;	//hard-coded update period of the noise map


		Framebuffer caustics_framebuffer;	//framebuffer used to receive refraction texture with caustics applied to it
		FullscreenRectangle caustics_canvas;	//full screen rectangle used to apply caustics to the refraction map in the second rendering pass
		SeparateShaderProgram caustics_rendering_program;	//caustics rendering program, implements caustics as a post-process filter applied to the refraction map
		float caustics_power;	//power, to which caustics modulation coefficient is raised
		float caustics_amplification;	//amplification factor applied to caustics modulation coefficient
		float caustics_sample_area;	//length of the side of the square domain residing in the non-dimensional model space that is used to collect the light sample contributing to caustics illumination
		uint32_t current_rendering_pass;		//active stage of object's rendering
		AbstractRenderingDevice* p_render_target;	//render target, to which the object should be eventually rendered



		//*******************************************************Raw OpenGL resources****************************************************
		GLuint ogl_buffers[2];	//OpenGL vertex and index buffers
		GLuint ogl_vertex_array_object;	//OpenGL vertex array object
		//***********************************************************************************************************************************

		void applyScreenSize(const uvec2& screen_size) override;
		bool configureRendering(AbstractRenderingDevice& render_target, uint32_t rendering_pass) override;
		void configureViewProjectionTransform(const AbstractProjectingDevice& projecting_device) override;
		bool configureRenderingFinalization() override;

		void setup_deep_water_waves();	//initializes deep water waves
		inline void setup_buffers();	//configures GPU memory buffers employed to render the object
		void setup_object();	//implements initial configuration performed upon construction of the object

		//helper function that configures parameters of underwater caustics
		bool setup_caustics_parameters(const AbstractProjectingDevice& projecting_device, const AbstractRenderingDevice& render_target, int first_texture_unit_available);


	public:
		//Describes spatial domain of the simulation. Similar to SaintVenantSystem::SVSCore::DomainSettings but is accesible to the clients
		//wishing to retrieve the domain information received by the object upon its initialization
		struct DomainSettings
		{
			float width, height;
			uint32_t resolution_width, resolution_height;
		};


		//Default initializer. Does not perform actual initialization of the numerical scheme, but is needed to allow default construction of the object.
		//Note that resolution of the tessellation billet is always rounded to the nearest multiple of 2
		KPWater(uint32_t tess_billet_horizontal_resolution = 128, uint32_t tess_billet_vertical_resolution = 128);	

		//Initializes object and numerical scheme with given parameters. Note that resolution of the tessellation billet is always rounded to the nearest multiple of 2
		KPWater(const SaintVenantSystem::Numeric* init_water_levels, const SaintVenantSystem::Numeric* init_horizontal_speed_flux, const SaintVenantSystem::Numeric* init_vertical_speed_flux,
			uint32_t domain_width, uint32_t domain_height, SaintVenantSystem::Numeric dx, SaintVenantSystem::Numeric dy, SaintVenantSystem::Numeric eps, 
			const SaintVenantSystem::Numeric* p_interpolated_topography, SaintVenantSystem::Numeric g = 9.8, SaintVenantSystem::Numeric theta = 1.1, ODESolver solver = ODESolver::RungeKutta33,
			uint32_t tess_billet_horizontal_resolution = 128, uint32_t tess_billet_vertical_resolution = 128);

		//Copy constructor
		KPWater(const KPWater& other);

		//Move constructor
		KPWater(KPWater&& other);

		//Destructor
		~KPWater();

		//Copy assignment operator
		KPWater& operator=(const KPWater& other);

		//Move assignment operator
		KPWater& operator=(KPWater&& other);
		
		//Initializes Kurganov-Petrova numerics using provided parameters
		void initialize(const SaintVenantSystem::Numeric* init_water_levels, const SaintVenantSystem::Numeric* init_horizontal_speed_flux, const SaintVenantSystem::Numeric* init_vertical_speed_flux,
			uint32_t domain_width, uint32_t domain_height, SaintVenantSystem::Numeric dx, SaintVenantSystem::Numeric dy, SaintVenantSystem::Numeric eps, 
			const SaintVenantSystem::Numeric* p_interpolated_topography, SaintVenantSystem::Numeric g = 9.8, SaintVenantSystem::Numeric theta = 1.1, ODESolver solver = ODESolver::RungeKutta33);

		//Resets Kurganov-Petrova numerics to its initial uninitialized state
		void reset();


		//Defines ODE solver to be used by the time-stepping procedure of the Kurganov-Petrova scheme
		void setODESolver(ODESolver ode_solver);

		//Returns ODE solver currently used by the time-stepping procedure of the Kurganov-Petrova scheme
		ODESolver getODESolver() const;


		//Sets boundary conditions to be used by the model integration scheme
		void registerBoundaryConditionsComputeProcedure(const BoundaryConditionsComputeProcedure& callback);

		//Integrates the model for the given time step
		void step(SaintVenantSystem::Numeric dt);

		//Updates current state (set of water levels and velocity field) of the system using provided arrays.
		//The values in the arrays are assumed to be arranged from west to east and from south to north
		//(i.e. the arrays use the row-major ordering). The function has no effect if the system has not been initialized.
		void updateSystemState(const std::vector<float>& water_levels, const std::vector<float>& horizontal_velocities, const std::vector<float>& vertical_velocities);

		//Updates water levels in the current state of the system (i.e. the velocity field is left intact).
		//The water levels are updated using the values from provided array. The values in the array are arranged from 
		//west to east and from south to north (i.e. the array uses the row-major ordering). The function has no effect if the system has not been initialized.
		void updateSystemState(const std::vector<float>& water_levels);

		//Retrieves the current state of the shallow water system. Note that repeated invocation of this function may be slow as it requires to fully copy the state.
		//Note that in order to avoid additional value-copy operations the function returns the current state via the references provided by the caller. The function has now effect if
		//the system has not been initialized
		void retrieveCurrentSystemState(std::vector<float>& water_levels, std::vector<float>& horizontal_velocities, std::vector<float>& vertical_velocities) const;

		//Updates topography used by the water dynamics solver. The system state is otherwise left intact.
		//The new values for topography are extracted from provided array with arrangement from west to east and
		//from south to north (i.e. the array uses the row-major ordering). The function has no effect if the system has not been initialized.
		//Note that the function receives the interpolated topography surface, i.e. the original surface after it has been "filtered" by computeTopographyBilinearInterpolation(...).
		//Hence, the supplied array must contain (2*W+1)*(2*H+1) elements.
		void updateTopography(const std::vector<float>& interpolated_topography_values);


		//Sets level-of-detail factor controlling tessellation density of the water surface
		void setLODFactor(float factor);

		//Returns level-of-detail factor controlling tessellation density of the water surface
		float getLODFactor() const;


		//Sets water height, at which all light energy gets effectively attenuated
		void setColorExtinctionBoundary(float water_level);

		//Returns water height, at which all light energy gets attenuated
		float getColorExtinctionBoundary() const;


		//Sets maximal amplitude of the deep water waves
		void setMaximalDeepWaveAmplitude(float amplitude);

		//Returns maximal amplitude of the deep water waves
		float getMaximalDeepWaveAmplitude() const;


		//Sets maximal amplitude of the capillary ripples
		void setMaximalRippleWaveAmplitude(float amplitude);

		//Returns maximal amplitude of the capillary ripples
		float getMaximalRippleWaveAmplitude() const;


		//Sets time difference between two consequent frames of the ripple simulation. The simulation is run in two scales: global and capillary.
		//Hence, the time step is defined for both via vec2 input parameter with the first component containing the time step for the global scale
		//simulation and the second component containing the time step for the capillary scale simulation. The default value is vec2(1e-5f, 1-e4f).
		void setRippleSimulationTimeStep(const vec2& dt);

		//Returns time step of the ripple simulation at the global and the capillary scales packed in this order into a vec2
		vec2 getRippleSimulationTimeStep() const;


		//Sets exponent term, which is applied when computing the Fresnel factor. Higher values result in refraction effects dominating over reflection
		//void setFresnelPower(float value);

		//Returns Fresnel power term
		//float getFresnelPower() const;


		//Sets color extinction factors affecting how fast each of the red, green, and blue color channels get attenuated by water. Default value is 1535 for red, 92 for green, and 23 for blue
		void setColorExtinctionFactors(float red_extinction_factor, float green_extinction_factor, float blue_extinction_factor);

		//Sets color extinction factors affecting how fast each of the red, green, and blue color channels get attenuated by water. Default value is 1535 for red, 92 for green, and 23 for blue
		void setColorExtinctionFactors(const vec3& color_extinction_factors);

		//Returns color extinction factors for red, green and blue color channels packed into a 3D vector in this order
		vec3 getColorExtinctionFactors() const;


		//Sets projection of the wind velocity vector onto the water surface plane. This parameter affects direction of capillary ripples spatial propagation
		void setWindVelocity(const vec2& wind_speed);

		//Returns projection of the wind velocity vector onto the water surface plane.
		vec2 getWindVelocity() const;


		//Sets parameter that affects choppiness of the deep water waves. Default value of this parameter is 0.1f
		void setDeepWaterWavesChoppiness(float choppiness);

		//Returns choppiness parameter of the deep water waves
		float getDeepWaterWavesChoppiness() const;


		//Sets tiling factor applied to the deep water ripple map generated by FFTs. The first parameter determines tiling of the geometrical waves, and the second parameter is used for bump mapping
		void setDeepWaterRippleMapTilingFactor(uint32_t geometry_waves_tiling_factor, uint32_t bump_map_tiling_factor);

		//Returns tiling factors applied to the deep water ripple map generated by FFTs. The tiling factors in the returned value are packed into a 2D vector of unsigned integers with the first component
		//containing the tiling factor applied when shaping geometrical waves and the second component containing the tiling factor used for generation of the bump map
		uvec2 getDeepWaterRippleMapTilingFactor() const;


		//Sets ratio between maximal allowed amplitude of the deep water waves and the deepness of the water at a given location
		void setMaximalWaveHeightAsElevationFraction(float ratio);

		//Returns ratio between maximal allowed amplitude of the deep water waves and the deepness of the water at a given location
		float getMaximalWaveHeightAsElevationFraction() const;


		//Sets exponent parameter of caustics
		void setCausticsPower(float power);

		//Returns exponent parameter of caustics
		float getCausticsPower() const;


		//Sets amplification factor of caustics
		void setCausticsAmplificationFactor(float factor);

		//Returns amplification factor of caustics
		float getCausticsAmplificationFactor() const;


		//Sets size of the non-dimensional area, which is used to collect light samples producing caustics
		void setCausticsSampleArea(float area);

		//Returns size of the non-dimensional area, which is used to collect light samples producing caustics
		float getCausticsSampleArea() const;


		//Returns structure containing settings of the spatial domain used by the simulation
		DomainSettings getDomainDetails() const;



		//Updates refraction texture. Note that refraction texture must be updated on each re-draw
		void updateRefractionTexture(const ImmutableTexture2D& refraction_texture, const ImmutableTexture2D& refraction_texture_depth_map);



		//Scales the "water object" in the object space. Note that repetitive scale transforms are multiplied
		void scale(float x_scale_factor, float y_scale_factor, float z_scale_factor);

		//Applies scaling transform defined by the given vector in object space. Note that repetitive scale transforms are multiplied
		void scale(const vec3& new_scale_factors);		



		//The following functions implement basic infrastructure of AbstractRenderableObject

		bool supportsRenderingMode(uint32_t rendering_mode) const override;
		uint32_t getNumberOfRenderingPasses(uint32_t rendering_mode) const override;
		bool render() override;



		//KPWater can only simulate water dynamics on top of the surfaces that are bilinear forms. Hence, sometimes it is necessary to perform bilinear interpolation of a given topography height map that is not known to be bilinear.
		//This function takes pointer to a buffer containing original height map, width and height of the height map and writes result of bilinear interpolation into "interpolated_topography". The buffer, to which the result of interpolation is returned
		//must be preallocated by the caller and must have enough storage to accommodate at least (2*width+1)*(2*height+1) single-precision floating point elements
		static void computeTopographyBilinearInterpolation(const float* topography, uint32_t topography_width, uint32_t topography_height, float* interpolated_topography);


		//Extracts the "integer" nodes from the given interpolated topography surface. "Integer" nodes are central cell points in the staggered grid defined in Kurganov-Petrova scheme. These central points are the points, for which actual water elevation
		//values are computed by the scheme. For correct rendering topography should be drawn using the height map returned by this function. Note that "topography_width" and "topography_height" are width and height of the ORIGINAL (not interpolated!)
		//topography map, hence "inerpolated_topography" must contain  (2*topography_width+1)*(2*topography_height+1) values and the function will write topography_width*topography_height values into  "topography_principal_component".
		static void extractTopographyPrincipalComponent(const float* interpolated_topography, uint32_t topography_width, uint32_t topography_height, float* topography_principal_component);
	};
}

#define TW__KPWATER__
#endif