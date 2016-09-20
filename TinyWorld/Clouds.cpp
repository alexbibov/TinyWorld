#include "Clouds.h"

#include <chrono>

using namespace tiny_world;



#define pi 3.1415926535897932384626433832795f

const std::string Clouds::cloud_formation_compute_program_name = "Clouds::cloud_formation_compute_program";
const std::string Clouds::cloud_shading_program_name = "Clouds::cloud_shading_program";

typedef AbstractRenderableObjectExtensionAggregator<AbstractRenderableObjectHDRBloomEx, AbstractRenderableObjectSelectionEx> ExtensionAggregator;



void Clouds::setup_object()
{
	//Allocate storage for the textures
	cloud_dynamics_pdfs_tex_res.first.allocateStorage(uv3SimulationGridResolution.x*uv3SimulationGridResolution.y*uv3SimulationGridResolution.z * 3 * sizeof(float),
		BufferTextureInternalPixelFormat::SIZED_FLOAT_RGB32);
	cloud_dynamics_cell_automaton_state.first.allocateStorage(uv3SimulationGridResolution.x*uv3SimulationGridResolution.y*uv3SimulationGridResolution.z,
		BufferTextureInternalPixelFormat::SIZED_UINT_R8);
	cloud_dynamics_cell_automaton_smoothed_state.first.allocateStorage(uv3SimulationGridResolution.x*uv3SimulationGridResolution.y*uv3SimulationGridResolution.z,
		BufferTextureInternalPixelFormat::SIZED_UINT_R16);


	//Configure compute shader program
	Shader cloud_formation_simulator_shader{ ShaderProgram::getShaderBaseCatalog() + "CloudsSimulator.cp.glsl", ShaderType::COMPUTE_SHADER, "Clouds::cloud_formation_simulator_shader" };
	cloud_formation_compute_program.addShader(cloud_formation_simulator_shader);
	cloud_formation_compute_program.link();


	//Configure std140 uniform buffer receiving simulation parameters for the compute program
	//cloud_formation_parameters.
}


void Clouds::generate_initial_PDFs()
{
	//Setup initial state for the random engine
	random_engine.seed(static_cast<unsigned long>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock{}.now().time_since_epoch()).count()));

	//Preallocate memory for auxiliary buffers that will receive the random fields implementing the PDFs driving the process and the initial state for the cloud formation simulator
	float* p_gaussian_sum_field = new float[uv3SimulationGridResolution.x*uv3SimulationGridResolution.y * 3];
	char* p_init_state = new char[uv3SimulationGridResolution.x*uv3SimulationGridResolution.y];


	for (uint32_t i = 0; i < uv3SimulationGridResolution.z * 3; ++i)
	{
		//Get the modulating factor corresponding to the field currently being processed. If the currently processed random field is humidification PDF
		//than the auxiliary buffer should be filled with zeros
		float modulation_factor;
		char bit_shift_factor;
		switch (i % 3)
		{
		case 0:
			std::fill(p_gaussian_sum_field, p_gaussian_sum_field + uv3SimulationGridResolution.x*uv3SimulationGridResolution.y * 3, 0.0f);
			std::fill(p_init_state, p_init_state + uv3SimulationGridResolution.x*uv3SimulationGridResolution.y, 0);
			modulation_factor = humidification_modulator;
			bit_shift_factor = 1;
			break;

		case 1:
			modulation_factor = phase_transition_modulator;
			bit_shift_factor = 2;
			break;

		case 2:
			modulation_factor = cloud_extinction_modulator;
			bit_shift_factor = 4;
			break;
		}


		//Obtain number of Gaussian hills that will participate in the sum
		uint32_t num_gaussian_hills = static_cast<uint32_t>(std::min(std::round(uniform_distribution(random_engine)*10.0 + 0.5), 10.0));

		//Fill the random field
		for (uint32_t j = 0; j < num_gaussian_hills; ++j)
		{
			//Generate variance for the current Gaussian hill
			vec2 v2Sigma = vec2{ uniform_distribution(random_engine), uniform_distribution(random_engine) };

			//Generate mean value for the current Gaussian hill
			vec2 v2Mean = vec2{ uniform_distribution(random_engine) - 0.5f, uniform_distribution(random_engine) - 0.5f };

			//Fill the map with values 
			for (uint32_t k = 0; k < uv3SimulationGridResolution.x * uv3SimulationGridResolution.y; ++k)
			{
				uint32_t y = k / uv3SimulationGridResolution.x;
				uint32_t x = k % uv3SimulationGridResolution.x;

				vec2 v2Aux = vec2(static_cast<float>(x), static_cast<float>(y)) - v2Mean;
				float aux = std::exp(-(v2Aux.x*v2Aux.x / (v2Sigma.x*v2Sigma.x) + v2Aux.y*v2Aux.y / (v2Sigma.y*v2Sigma.y)) / 2) / (2 * pi*v2Sigma.x*v2Sigma.y) * modulation_factor;
				p_gaussian_sum_field[3 * k + i % 3] += aux;
				p_init_state[k] |= (aux >= 0.5)*bit_shift_factor;
			}
		}


		//Check if we have finished filling the current vertical layer. If the layer is filled we need to transfer the contents from the auxiliary buffers to the corresponding buffer textures
		if (i % 3 == 2)
		{
			void* p_pdf_dest_buffer = cloud_dynamics_pdfs_tex_res.first.map(uv3SimulationGridResolution.x*uv3SimulationGridResolution.y*(i - 2)*sizeof(float),
				uv3SimulationGridResolution.x*uv3SimulationGridResolution.y * 3 * sizeof(float), BufferRangeAccessBits::WRITE);
			memcpy(p_pdf_dest_buffer, p_gaussian_sum_field, uv3SimulationGridResolution.x*uv3SimulationGridResolution.y * 3 * sizeof(float));
			cloud_dynamics_pdfs_tex_res.first.unmap();

			void* p_state_dest_buffer = cloud_dynamics_cell_automaton_state.first.map(uv3SimulationGridResolution.x*uv3SimulationGridResolution.y*(i - 2) / 3,
				uv3SimulationGridResolution.x*uv3SimulationGridResolution.y, BufferRangeAccessBits::WRITE);
			memcpy(p_state_dest_buffer, p_init_state, uv3SimulationGridResolution.x*uv3SimulationGridResolution.y);
			cloud_dynamics_cell_automaton_state.first.unmap();
		}
	}
}


Clouds::Clouds() : 
AbstractRenderableObject{ "Clouds" },
cloud_dynamics_pdfs_tex_res{ BufferTexture{ "Clouds::dynamics_PDFs_texture_array" }, TextureReferenceCode{} },
cloud_dynamics_cell_automaton_state{ BufferTexture{ "Clouds::dynamics_cell_automaton" }, TextureReferenceCode{} },
cloud_dynamics_cell_automaton_smoothed_state{ BufferTexture{ "Clouds::dynamics_cell_automaton_smoothed_state" }, TextureReferenceCode{} },
cloud_formation_compute_program{ cloud_formation_compute_program_name }, uv3SimulationGridResolution{ 512, 512, 64 },
v3SimulationDomainSize{ 1.0f, 1.0f, 1.0f }, wind_force(64, 1U), v2WindDirection{ 1.0f, 0.0f }, p_lighting_conditions{ nullptr },
cloud_shading_program{ cloud_shading_program_name },
humidification_modulator{ 1.0f }, phase_transition_modulator{ 1.0f }, cloud_extinction_modulator{ 1.0f }
{

}


Clouds::Clouds(float cloud_domain_size_x, float cloud_domain_size_y, float cloud_domain_size_z,
	uint32_t cloud_domain_resolution_x /* = 512 */, uint32_t cloud_domain_resolution_y /* = 512 */, uint32_t cloud_domain_resolution_z /* = 64 */) :
	AbstractRenderableObject{ "Clouds" },
	cloud_dynamics_pdfs_tex_res{ BufferTexture{ "Clouds::dynamics_PDFs_texture_array" }, TextureReferenceCode{} },
	cloud_dynamics_cell_automaton_state{ BufferTexture{ "Clouds::dynamics_cell_automaton" }, TextureReferenceCode{} },
	cloud_dynamics_cell_automaton_smoothed_state{ BufferTexture{ "Clouds::dynamics_cell_automaton_smoothed_state" }, TextureReferenceCode{} },
	cloud_formation_compute_program{ cloud_formation_compute_program_name },
	uv3SimulationGridResolution{ cloud_domain_resolution_x, cloud_domain_resolution_y, cloud_domain_resolution_z },
	v3SimulationDomainSize{ cloud_domain_size_x, cloud_domain_size_y, cloud_domain_size_z },
	wind_force(64, 1U), v2WindDirection{ 1.0f, 0.0f }, p_lighting_conditions{ nullptr },
	cloud_shading_program{ cloud_shading_program_name }
{

}


