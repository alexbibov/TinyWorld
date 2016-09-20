#ifndef TW_TESSELLATED_TERRAIN_H__

#include "AbstractRenderableObject.h"
#include "AbstractRenderableObjectLightEx.h"
#include "AbstractRenderableObjectHDRBloomEx.h"
#include "AbstractRenderableObjectSelectionEx.h"
#include "CompleteShaderProgram.h"
#include "ImmutableTexture2D.h"
#include "FractalNoise.h"

namespace tiny_world
{
	class TessellatedTerrain : virtual public AbstractRenderableObjectTextured, 
		public AbstractRenderableObjectExtensionAggregator<AbstractRenderableObjectLightEx, AbstractRenderableObjectHDRBloomEx, AbstractRenderableObjectSelectionEx>
	{
	private:
		static const std::string tess_terrain_rendering_program0_name;

		uint32_t num_u_base_nodes;		//base nodes used for tessellation along u-axis
		uint32_t num_v_base_nodes;		//base nodes used for tessellation along v-axis

		float lod;				//Level-Of-Detail factor

		//********************************************************Raw OpenGL resources**************************************************************
		GLuint ogl_vertex_attribute_object;		//OpenGL identifier of vertex attribute object
		GLuint ogl_vertex_buffer_object;		//OpenGL identifier of the buffer storing u- and v- base nodes
		GLuint ogl_index_buffer_object;			//OpenGL identifier of the index buffer that specifies the tessellation billet
		//**********************************************************************************************************************************************

		TextureSamplerReferenceCode height_map_sampler_ref_code;		//Texture sampler used to sample data from the height map
		TextureReferenceCode height_map_ref_code;		//Reference code of the height map texture
		TextureSamplerReferenceCode terrain_texture_sampler_ref_code;	//Texture sampler used to sample data from terrain texture
		TextureReferenceCode terrain_texture_ref_code;	//Reference code of the terrain texture
		FractalNoise2D fractal_noise;	//two-dimensional fractal noise
		Texture2DResource fractal_noise_map_tex_res;	//Reference code associated with the fractal noise map
		static const uint32_t fractal_noise_resolution;		//hard-coded resolution of the fractal noise
		static const float fractal_noise_scaling_weight;	//hard-coded scaling weight of the fractal noise, which has been hand-picked to deliver visually attractive results
		uint32_t height_map_texture_u_res;		//horizontal resolution of the height map texture
		uint32_t height_map_texture_v_res;		//vertical resolution of the height map texture
		std::vector<float> height_map_numeric_data;		//full height map represented by an array of floats
		bool is_heightmap_normalized;	//equals 'true' if the terrain height map has been normalized

		//The following texture scale factors determine physical size of the texture applied to the terrain surface scaled to the unit square region.
		//For example if u-scale and v-scale factors are both equal to 0.5, the terrain surface will be wrapped by four full-size texture terrain tiles
		float texture_u_scale;		//texture scale in horizontal direction
		float texture_v_scale;		//texture scale in vertical direction

		float height_map_normalization_constant;		//normalizing constant applied to the height map (the maximal height level in the given height map)
		float height_map_level_offset;			//the absolute minimum of the height map. While loaded, the texture map is normalized and shifted so that its lowest level is aligned with 0.

		ShaderProgramReferenceCode tess_terrain_rendering_program0_ref_code;		//Rendering programs of the object
		
		uint32_t *ref_counter;					//Reference counter used to monitor lifetime of OpenGL objects owned by this type


		void applyScreenSize(const uvec2& screen_size) override;
		bool configureRendering(AbstractRenderingDevice& render_target, uint32_t rendering_pass) override;
		void configureViewProjectionTransform(const AbstractProjectingDevice& projecting_device) override;
		bool configureRenderingFinalization() override;

		void init_tess_terrain();		//implements spin-up initialization routines for tessellated terrain object 
		void free_ogl_resources();		//performs clean-up of OpenGL resources owned by the object

	public:
		//Default initialization with optional possibility to set custom LOD and tessellation billet
		TessellatedTerrain(float lod = 20.0f, uint32_t num_u_base_nodes = 64, uint32_t num_v_base_nodes = 64, float texture_u_scale = 0.1f, float texture_v_scale = 0.1f);

		//Default initialization with user-defined string name and (optionally) custom LOD factor and tessellation billet
		TessellatedTerrain(const std::string& tess_terrain_string_name, float lod = 20.0f, 
			uint32_t num_u_base_nodes = 64, uint32_t num_v_base_nodes = 64, 
			float texture_u_scale = 0.1f, float texture_v_scale = 0.1f);
		
		//Loads height map from an ASCII text file containing numerical matrix with rows separated by end-line characters and columns separated by comma or spaces.
		//This constructor also allows to define custom LOD value, resolution of the tessellation billet, and texture scale factors
		TessellatedTerrain(const std::string& tess_terrain_string_name, const std::string& file_height_map, bool normalize_height_map = true,
			float lod = 20.0f, uint32_t num_u_base_nodes = 64, uint32_t num_v_base_nodes = 64,
			float texture_u_scale = 0.1f, float texture_v_scale = 0.1f);
		
		//Loads height map of given resolution from the given address in memory. This function also allows to define desired LOD value, resolution of
		//the tessellation billet, and texture scale factors.
		TessellatedTerrain(const std::string& tess_terrain_string_name, const float* height_map, uint32_t u_resolution, uint32_t v_resolution, bool normalize_height_map = true,
			float lod = 1.0f, uint32_t num_u_base_nodes = 64, uint32_t num_v_base_nodes = 64,
			float texture_u_scale = 0.1f, float texture_v_scale = 0.1f);
		
		TessellatedTerrain(const TessellatedTerrain& other);	//copy constructor
		TessellatedTerrain(TessellatedTerrain&& other);		//move constructor

		~TessellatedTerrain();	//destructor

		TessellatedTerrain& operator=(const TessellatedTerrain& other);		//copy-assignment operator
		TessellatedTerrain& operator=(TessellatedTerrain&& other);		//move-assignment operator


		void defineHeightMap(const std::string& file_height_map, bool normalize_height_map = true);	//reads new height map from an ASCII-file containing matrix of comma separated height values
		void defineHeightMap(const float* height_map, uint32_t u_resolution, uint32_t v_resolution, bool normalize_height_map = true);	//feeds tessellation engine with a new height map contained in application's memory
		void installTexture(const ImmutableTexture2D& terrain_texture, float texture_u_scale = 0.1f, float texture_v_scale = 0.1f);		//applies new texture to the terrain object and (optionally) sets desired scale factors for the texture

		float setLODFactor(float new_lod_factor);	//sets new value for the LOD factor and returns previously used value

		//Returns absolute minimal height level of the height map. The reason to return this value is to compensate for normalization, which occurs while the height map is 
		//being loaded and maps the physical heights to the range [0,1]
		float getHeightLevelOffset() const;

		float getHeightNormalizationConstant() const;		//returns constant used as divisor when normalizing original height map (the highest point in the original map)
		bool isHeightMapNormalized() const;	//returns 'true' if the height map has been normalized. Returns 'false' otherwise
		void retrieveHeightMap(const float** height_map, uint32_t& width, uint32_t& height) const;	//retrieves a pointer to the height map currently used by the tessellated terrain. If no height map has been defined the returned pointer is nullptr.
		float getLODFactor() const;		//returns currently used LOD factor


		//Scales the terrain object in object space. Repetitive scale transforms are multiplied
		void scale(float x_scale_factor, float y_scale_factor, float z_scale_factor);

		//Uses components of the given vector to define scaling transform. All previous scaling transforms are merged with the current transform by means of transform superposition
		void scale(const vec3& new_scale_factors);



		bool supportsRenderingMode(uint32_t rendering_mode) const override;
		uint32_t getNumberOfRenderingPasses(uint32_t rendering_mode) const override;
		bool render() override;

		//Helper function: allows to parse a height map stored in ASCII format as a comma (or space) separated rectangular table. Function returns 'true' on success or 'false' otherwise.
		//Parsed data is returned via reference parameter "loaded_height_map". The dimensions of the map are written to the references "width" and "height". On successful invocation, and if 
		//"normalize_height_map" equals 'true' the function shifts the output data so that the minimal height value is zero, and then normalizes the data by dividing each value by the original data 
		//range (i.e. the maximal height minus the minimal height). The shift offset (which is effectively the minimal height in the original data) as well as the maximal height (shifted by the offset!) 
		//are written to the output reference parameters "offset" and "range". If the function fails, the parsed data is undefined, and the "parse_error" is written to the output reference 
		//argument "parse_error". If the function executes without errors the "parse_error" string is undefined. Finally, if "normalize_height_map" equals 'false', the function simply parses the data in the file
		//and writes the raw values into "loaded_height_map". In this case "offset" will be set to 0 and "range" will be set to 1. The behavior in case of error when "normalize_height_map" equals 'false' is 
		//identical to that of the case when "normalize_height_map" equals 'true'
		static bool parseHeightMapFromFile(const std::string& file_height_map, std::vector<float>& loaded_height_map, uint32_t& width, uint32_t& height, float& offset, float& range, std::string& parse_error, bool normalize_height_map = true);
	};


}

#define TW_TESSELLATED_TERRAIN_H__
#endif