//Implements atmospheric clouds with physical scattering properties

#ifndef TW__CLOUDS__

#include "AbstractRenderableObject.h"
#include "AbstractRenderableObjectHDRBloomEx.h"
#include "AbstractRenderableObjectSelectionEx.h"
#include "LightingConditions.h"
#include "std140UniformBuffer.h"

#include <random>

namespace tiny_world
{
	class Clouds final : virtual public AbstractRenderableObjectTextured, 
		public AbstractRenderableObjectExtensionAggregator<AbstractRenderableObjectHDRBloomEx, AbstractRenderableObjectSelectionEx>
	{
	private:
		//Texture resource associated with probability density functions that control the rates of evaporation (red-channel), 
		//humidification (green-channel), and phase transition (blue-channel) of the clouds. Each layer of the simulation grid
		//is represented by a single layer in 2D texture array
		TextureBufferResource cloud_dynamics_pdfs_tex_res;

		//Texture resource associated with the texture containing the current state of the cellular automaton, which is used
		//to simulate dynamics of the cloud formation process
		TextureBufferResource cloud_dynamics_cell_automaton_state;

		//Texture resource associated with the texture containing the current state of the cloud simulation with the mean smoothing filter
		//applied. The smoothing is needed as the cellular automaton that simulates the cloud formation is only capable of generating 
		//binary distributions.
		TextureBufferResource cloud_dynamics_cell_automaton_smoothed_state;

		//Compute program implementing single step of the cloud formation simulator
		CompleteShaderProgram cloud_formation_compute_program;

		static const std::string cloud_formation_compute_program_name;	//string name of the cloud formation compute program

		std140UniformBuffer cloud_formation_parameters;		//std140 uniform buffer receiving parameters of the cloud formation compute program

		//Resolution of the cloud simulation grid
		uvec3 uv3SimulationGridResolution;

		//Spatial size of the cloud simulation domain
		vec3 v3SimulationDomainSize;

		//Wind force at each vertical layer of the simulation domain
		std::vector<uint32_t> wind_force;

		//Wind direction, must have unit length
		vec2 v2WindDirection;


		//Lighting conditions affecting color of the clouds
		const LightingConditions* p_lighting_conditions;

		//Rendering program implementing shading of the clouds
		CompleteShaderProgram cloud_shading_program;

		static const std::string cloud_shading_program_name;	//string name of the cloud formation compute program


		//Random engine needed to generate initial distributions used by cloud formation processes
		std::default_random_engine random_engine;

		std::uniform_real_distribution<float> uniform_distribution;	//uniform distribution wrapper over the random engine
		std::normal_distribution<float>	normal_distribution;	//normal distribution wrapper over the random engine


		//Modulator used to affect probability of air humidification. Higher values correspond to higher probabilities of humidification
		float humidification_modulator;	

		//Modulator used to affect probability of the state transition. Higher values correspond to higher probability of humidity being turned into clouds
		float phase_transition_modulator;

		//Modulator used to affect probability of cloud extinction. Higher values correspond to higher probability of extinction
		float cloud_extinction_modulator;


		void applyScreenSize(const uvec2& screen_size) override;
		bool configureRendering(AbstractRenderingDevice& render_target, uint32_t rendering_pass) override;
		void configureViewProjectionTransform(const AbstractProjectingDevice& projecting_device) override;
		bool configureRenderingFinalization() override;


		inline void setup_object();	//runs initial procedures required on object's initialization
		inline void generate_initial_PDFs();	//generates probability density functions of humidification, phase transition, and cloud extinction


	public:
		//Default initialization
		Clouds();	
		
		//Initializes clouds simulation given the spatial size and the resolution of the simulation domain
		Clouds(float cloud_domain_size_x, float cloud_domain_size_y, float cloud_domain_size_z, uint32_t cloud_domain_resolution_x = 512, uint32_t cloud_domain_resolution_y = 512,
			uint32_t cloud_domain_resolution_z = 64);

		//Copy initialization
		Clouds(const Clouds& other);

		//Move initialization
		Clouds(Clouds&& other);

		//Copy assignment
		Clouds& operator=(const Clouds& other);

		//Move assignment
		Clouds& operator=(Clouds&& other);

		//Destructor
		~Clouds();


		void setDomainDimensions(float cloud_domain_size_x, float cloud_domain_size_y, float cloud_domain_size_z);	//sets spatial dimensions of the cloud domain
		void setDomainDimensions(const vec3& cloud_domain_size);	//sets spatial dimensions of the cloud domain
		vec3 getDomainDimensions() const;	//retrieves spatial dimensions of the cloud domain


		void step();	//runs single simulation step


		//Standard infrastructure of a drawable object

		bool supportsRenderingMode(uint32_t rendering_mode) const override;
		uint32_t getNumberOfRenderingPasses(uint32_t rendering_mode) const override;
		bool render() override;
	};
}

#define TW__CLOUDS__
#endif
