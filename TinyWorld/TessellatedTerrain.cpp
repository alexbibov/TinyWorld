#include "TessellatedTerrain.h"

#include <istream>
#include <fstream>
#include <vector>
#include <cctype>
#include <type_traits>


using namespace tiny_world;

typedef AbstractRenderableObjectExtensionAggregator<AbstractRenderableObjectLightEx, AbstractRenderableObjectHDRBloomEx, AbstractRenderableObjectSelectionEx> ExtensionAggregator;

const std::string TessellatedTerrain::tess_terrain_rendering_program0_name = "TessellatedTerrain::rendering_program0";
const uint32_t TessellatedTerrain::fractal_noise_resolution = 4096U;
const float TessellatedTerrain::fractal_noise_scaling_weight = 1.365333333333f;


void TessellatedTerrain::applyScreenSize(const uvec2& screen_size)
{

}


void TessellatedTerrain::init_tess_terrain()
{
	//Allocate reference counter for the current context
	ref_counter = new uint32_t{ 1 };


	//Create new vertex attribute object associated with the tessellated terrain
	glGenVertexArrays(1, &ogl_vertex_attribute_object);
	glBindVertexArray(ogl_vertex_attribute_object);
	glEnableVertexAttribArray(vertex_attribute_position::getId());
	vertex_attribute_position::setVertexAttributeBufferLayout(0, 0);
	glEnableVertexAttribArray(vertex_attribute_texcoord::getId());
	vertex_attribute_texcoord::setVertexAttributeBufferLayout(vertex_attribute_position::getCapacity(), 0);


	//Create new vertex buffer object and populate it with tessellation billet
	glGenBuffers(1, &ogl_vertex_buffer_object);
	glBindBuffer(GL_ARRAY_BUFFER, ogl_vertex_buffer_object);

	void* vertex_buf = 
		malloc((vertex_attribute_position::getCapacity() + vertex_attribute_texcoord::getCapacity()) * num_u_base_nodes * num_v_base_nodes);
	float u_step = 1.0f / (num_u_base_nodes - 1.0f), v_step = 1.0f / (num_v_base_nodes - 1.0f);

	for (int i = 0; i < static_cast<int>(num_v_base_nodes); ++i)
		for (int j = 0; j < static_cast<int>(num_u_base_nodes); ++j)
		{
		//NOTE: tessellated terrain is generated in X-Z plane

		//Pointer to memory segment containing position of the current vertex
		vertex_attribute_position::value_type* p_position =
			reinterpret_cast<vertex_attribute_position::value_type*>(static_cast<char*>(vertex_buf)+
			(i * num_u_base_nodes + j)*(vertex_attribute_position::getCapacity() + vertex_attribute_texcoord::getCapacity()));


		//Pointer to memory segment containing texture coordinates of the current vertex
		vertex_attribute_texcoord::value_type* p_texcoord =
			reinterpret_cast<vertex_attribute_texcoord::value_type*>(reinterpret_cast<char*>(p_position)+vertex_attribute_position::getCapacity());

		//x-coordinate
		p_position[0] = j*u_step - 0.5f;

		//y-coordinate
		p_position[1] = 0.0f;

		//z-coordinate
		p_position[2] = -i*v_step + 0.5f;

		//w-coordinate
		p_position[3] = 1.0f;

		//Texture u-coordinate
		p_texcoord[0] = j*u_step / texture_u_scale;

		//Texture v-coordinate
		p_texcoord[1] = i*v_step / texture_v_scale;
		}
	glBufferData(GL_ARRAY_BUFFER, 
		num_u_base_nodes*num_v_base_nodes*(vertex_attribute_position::getCapacity()+vertex_attribute_texcoord::getCapacity()), 
		vertex_buf, GL_STATIC_DRAW);
	delete[] vertex_buf;

	glBindVertexBuffer(0, ogl_vertex_buffer_object, 0, vertex_attribute_position::getCapacity() + vertex_attribute_texcoord::getCapacity());

	//Create index buffer object and populate it with indexes
	glGenBuffers(1, &ogl_index_buffer_object);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ogl_index_buffer_object);

	GLuint* index_buf = new GLuint[(num_u_base_nodes - 1)*(num_v_base_nodes - 1) * 4];

	for (int i = 0; i < static_cast<int>(num_v_base_nodes - 1); ++i)
		for (int j = 0; j < static_cast<int>(num_u_base_nodes - 1); ++j)
		{
			index_buf[(i*(num_u_base_nodes - 1) + j) * 4 + 0] = i*num_u_base_nodes + j;			//lower-left corner of the patch
			index_buf[(i*(num_u_base_nodes - 1) + j) * 4 + 1] = i*num_u_base_nodes + j+1;		//lower-right corner of the patch
			index_buf[(i*(num_u_base_nodes - 1) + j) * 4 + 2] = (i + 1)*num_u_base_nodes + j + 1;	//upper-right corner of the patch
			index_buf[(i*(num_u_base_nodes - 1) + j) * 4 + 3] = (i + 1)*num_u_base_nodes + j;		//upper-left corner of the patch	
		}
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, (num_u_base_nodes - 1)*(num_v_base_nodes - 1) * 4 * sizeof(GLuint), index_buf, GL_STATIC_DRAW);
	delete[] index_buf;



	//Initialize default rendering program
	if (!tess_terrain_rendering_program0_ref_code)
	{
		tess_terrain_rendering_program0_ref_code =
			createCompleteShaderProgram(tess_terrain_rendering_program0_name,
			{ PipelineStage::VERTEX_SHADER, PipelineStage::TESS_CONTROL_SHADER, PipelineStage::TESS_EVAL_SHADER,
			PipelineStage::GEOMETRY_SHADER, PipelineStage::FRAGMENT_SHADER });

		Shader vertex_shader{ ShaderProgram::getShaderBaseCatalog() + "TessellatedTerrain_default.vp.glsl", ShaderType::VERTEX_SHADER,
			"TessellatedTerrain_VertexPreprocessor_Default" };
		Shader tess_control_shader{ ShaderProgram::getShaderBaseCatalog() + "TessellatedTerrain_default.tcp.glsl", ShaderType::TESS_CONTROL_SHADER,
			"TessellatedTerrain_TessellationControlProgram_Default" };
		Shader tess_eval_shader{ ShaderProgram::getShaderBaseCatalog() + "TessellatedTerrain_default.tep.glsl", ShaderType::TESS_EVAL_SHADER,
			"TessellatedTerrain_TessellationEvaluationProgram_Default" };
		Shader geometry_shader{ ShaderProgram::getShaderBaseCatalog() + "TessellatedTerrain_default.gp.glsl", ShaderType::GEOMETRY_SHADER,
			"TessellatedTerrain_GeometryInterpolator_Default" };
		Shader fragment_shader{ ShaderProgram::getShaderBaseCatalog() + "TessellatedTerrain_default.fp.glsl", ShaderType::FRAGMENT_SHADER,
			"TessellatedTerrain_FragmentRasterizer_Default" };

		retrieveShaderProgram(tess_terrain_rendering_program0_ref_code)->addShader(vertex_shader);
		retrieveShaderProgram(tess_terrain_rendering_program0_ref_code)->addShader(tess_control_shader);
		retrieveShaderProgram(tess_terrain_rendering_program0_ref_code)->addShader(tess_eval_shader);
		retrieveShaderProgram(tess_terrain_rendering_program0_ref_code)->addShader(geometry_shader);
		retrieveShaderProgram(tess_terrain_rendering_program0_ref_code)->addShader(fragment_shader);

		retrieveShaderProgram(tess_terrain_rendering_program0_ref_code)->bindVertexAttributeId("tess_billet_vertex_position", vertex_attribute_position::getId());
		retrieveShaderProgram(tess_terrain_rendering_program0_ref_code)->bindVertexAttributeId("tess_billet_texcoord", vertex_attribute_texcoord::getId());

		retrieveShaderProgram(tess_terrain_rendering_program0_ref_code)->link();

		//Set uniform values that remain constant during existence of this object
		//retrieveShaderProgram(tess_terrain_rendering_program0_ref_code)->assignUniformScalar("num_u_base_nodes", num_u_base_nodes);
		//retrieveShaderProgram(tess_terrain_rendering_program0_ref_code)->assignUniformScalar("num_v_base_nodes", num_v_base_nodes);
	}


	//Initialize wire frame mode rendering program

	//Setup texture samplers
	height_map_sampler_ref_code = createTextureSampler("TessellatedTerrain::height_map_sampler", SamplerMagnificationFilter::LINEAR, SamplerMinificationFilter::LINEAR_MIPMAP_NEAREST,
		SamplerWrapping{ SamplerWrappingMode::CLAMP_TO_EDGE, SamplerWrappingMode::CLAMP_TO_EDGE, SamplerWrappingMode::CLAMP_TO_EDGE });

	terrain_texture_sampler_ref_code = createTextureSampler("TessellatedTerrain::terrain_texture_sampler", SamplerMagnificationFilter::LINEAR, SamplerMinificationFilter::LINEAR_MIPMAP_NEAREST,
		SamplerWrapping{ SamplerWrappingMode::REPEAT, SamplerWrappingMode::REPEAT, SamplerWrappingMode::CLAMP_TO_EDGE });


	//Setup custom normal, specular and emission texture samplers
	defineProceduralNormalMap("TessTerrainNormalMapFunc");
	defineProceduralSpecularMap("TessTerrainSpecularMapFunc");
	defineProceduralEmissionMap("TessTerrainEmissionMapFunc");

	//Configure the fractal noise
	fractal_noise.setPeriodicity(true);
	fractal_noise_map_tex_res.second = registerTexture(fractal_noise_map_tex_res.first, terrain_texture_sampler_ref_code);
	vec3 v3ScalingFactors = getObjectScale();
	retrieveShaderProgram(tess_terrain_rendering_program0_ref_code)->assignUniformVector("v2FractalNoiseScaling", 
		vec2{ v3ScalingFactors.x / fractal_noise_resolution, v3ScalingFactors.z / fractal_noise_resolution }*fractal_noise_scaling_weight);
}


void TessellatedTerrain::free_ogl_resources()
{
	if (ogl_vertex_attribute_object)	//if vertex attribute object exists, destroy it
		glDeleteVertexArrays(1, &ogl_vertex_attribute_object);

	if (ogl_vertex_buffer_object)		//if vertex buffer object exists, destroy it
		glDeleteBuffers(1, &ogl_vertex_buffer_object);

	if (ogl_index_buffer_object)		//if element array object exists, destroy it
		glDeleteBuffers(1, &ogl_index_buffer_object);
}



TessellatedTerrain::TessellatedTerrain(float lod /* = 20.0f */, 
	uint32_t num_u_base_nodes /* = 64 */, uint32_t num_v_base_nodes /* = 64 */, float texture_u_scale /* = 0.1f */, float texture_v_scale /* = 0.1f */) :

	AbstractRenderableObject("TessellatedTerrain"),

	ExtensionAggregator(std::initializer_list<std::pair<PipelineStage, std::string>>({ std::make_pair(PipelineStage::FRAGMENT_SHADER, ShaderProgram::getShaderBaseCatalog() + "TessellatedTerrain_default.fp.ext.glsl") })),

	num_u_base_nodes{ num_u_base_nodes }, num_v_base_nodes{ num_v_base_nodes }, lod{ lod }, 
	fractal_noise{ fractal_noise_resolution, fractal_noise_resolution, 1024, 1024, 8U }, fractal_noise_map_tex_res{ fractal_noise.retrieveNoiseMap(), TextureReferenceCode{} },
	height_map_texture_u_res{ 0 }, height_map_texture_v_res{ 0 }, is_heightmap_normalized{ false },
	texture_u_scale{ texture_u_scale }, texture_v_scale{ texture_v_scale }
{
	init_tess_terrain();
}

TessellatedTerrain::TessellatedTerrain(const std::string& tess_terrain_string_name, float lod /* = 20.0f */, 
	uint32_t num_u_base_nodes /* = 64 */, uint32_t num_v_base_nodes /* = 64 */, float texture_u_scale /* = 0.1f */, float texture_v_scale /* = 0.1f */) :

	AbstractRenderableObject("TessellatedTerrain", tess_terrain_string_name),
	ExtensionAggregator(std::initializer_list<std::pair<PipelineStage, std::string>>({ std::make_pair(PipelineStage::FRAGMENT_SHADER, ShaderProgram::getShaderBaseCatalog() + "TessellatedTerrain_default.fp.ext.glsl") })),

	num_u_base_nodes{ num_u_base_nodes }, num_v_base_nodes{ num_v_base_nodes }, lod{ lod },
	fractal_noise{ fractal_noise_resolution, fractal_noise_resolution, 1024, 1024, 8U }, fractal_noise_map_tex_res{ fractal_noise.retrieveNoiseMap(), TextureReferenceCode{} },
	height_map_texture_u_res{ 0 }, height_map_texture_v_res{ 0 }, is_heightmap_normalized{ false },
	texture_u_scale{ texture_u_scale }, texture_v_scale{ texture_v_scale }
{
	init_tess_terrain();
}


TessellatedTerrain::TessellatedTerrain(const std::string& tess_terrain_string_name, const std::string& file_height_map, bool normalize_height_map /* = true */,
	float lod /* = 20.0f */, uint32_t num_u_base_nodes /* = 64 */, uint32_t num_v_base_nodes /* = 64 */,
	float texture_u_scale /* = 0.1f */, float texture_v_scale /* = 0.1f */) :

	AbstractRenderableObject("TessellatedTerrain", tess_terrain_string_name),
	ExtensionAggregator(std::initializer_list<std::pair<PipelineStage, std::string>>({ std::make_pair(PipelineStage::FRAGMENT_SHADER, ShaderProgram::getShaderBaseCatalog() + "TessellatedTerrain_default.fp.ext.glsl") })),
	
	lod{ lod }, fractal_noise{ fractal_noise_resolution, fractal_noise_resolution, 1024, 1024, 8U }, fractal_noise_map_tex_res{ fractal_noise.retrieveNoiseMap(), TextureReferenceCode{} },
	height_map_texture_u_res{ 0 }, height_map_texture_v_res{ 0 },
	num_u_base_nodes{ num_u_base_nodes }, num_v_base_nodes{ num_v_base_nodes },
	texture_u_scale{ texture_u_scale }, texture_v_scale{ texture_v_scale }
{
	init_tess_terrain();
	defineHeightMap(file_height_map, normalize_height_map);
}


TessellatedTerrain::TessellatedTerrain(const std::string& tess_terrain_string_name, const float* height_map, 
	uint32_t u_resolution, uint32_t v_resolution, bool normalize_height_map /* = true */,  float lod /* = 1.0f */,
	uint32_t num_u_base_nodes /* = 64 */, uint32_t num_v_base_nodes /* = 64 */,
	float texture_u_scale /* = 0.1f */, float texture_v_scale /* = 0.1f */) :

	AbstractRenderableObject("TessellatedTerrain", tess_terrain_string_name),
	ExtensionAggregator(std::initializer_list<std::pair<PipelineStage, std::string>>({ std::make_pair(PipelineStage::FRAGMENT_SHADER, ShaderProgram::getShaderBaseCatalog() + "TessellatedTerrain_default.fp.ext.glsl") })),

	num_u_base_nodes{ num_u_base_nodes }, num_v_base_nodes{ num_v_base_nodes }, lod{ lod }, 
	fractal_noise{ fractal_noise_resolution, fractal_noise_resolution, 1024, 1024, 8U }, fractal_noise_map_tex_res{ fractal_noise.retrieveNoiseMap(), TextureReferenceCode{} },
	height_map_texture_u_res{ 0 }, height_map_texture_v_res{ 0 },
	texture_u_scale{ texture_u_scale }, texture_v_scale{ texture_v_scale }
{
	init_tess_terrain();
	defineHeightMap(height_map, u_resolution, v_resolution, normalize_height_map);
}


TessellatedTerrain::TessellatedTerrain(const TessellatedTerrain& other) : 

AbstractRenderableObject(other), AbstractRenderableObjectTextured(other), ExtensionAggregator(other), 
num_u_base_nodes{ other.num_u_base_nodes }, num_v_base_nodes{ other.num_v_base_nodes }, lod{ other.lod },
ogl_vertex_attribute_object{ other.ogl_vertex_attribute_object }, ogl_vertex_buffer_object{ other.ogl_vertex_buffer_object },
ogl_index_buffer_object{ other.ogl_index_buffer_object }, height_map_sampler_ref_code{ other.height_map_sampler_ref_code }, 
height_map_ref_code{ other.height_map_ref_code },
height_map_texture_u_res{ other.height_map_texture_u_res }, 
height_map_texture_v_res{ other.height_map_texture_v_res },
height_map_numeric_data(other.height_map_numeric_data), is_heightmap_normalized{ other.is_heightmap_normalized },
texture_u_scale{ other.texture_u_scale },
texture_v_scale{ other.texture_v_scale },
height_map_level_offset{ other.height_map_level_offset },
height_map_normalization_constant{ other.height_map_normalization_constant },
terrain_texture_ref_code{ other.terrain_texture_ref_code }, terrain_texture_sampler_ref_code{ other.terrain_texture_sampler_ref_code },
fractal_noise{ other.fractal_noise },
fractal_noise_map_tex_res{ fractal_noise.retrieveNoiseMap(), other.fractal_noise_map_tex_res.second },
tess_terrain_rendering_program0_ref_code(other.tess_terrain_rendering_program0_ref_code),
ref_counter{ other.ref_counter }

{
	(*ref_counter)++;	//increment reference counter related to the OpenGL objects owned by TessellatedTerrain 

	updateTexture(fractal_noise_map_tex_res.second, fractal_noise_map_tex_res.first, terrain_texture_sampler_ref_code);
}

TessellatedTerrain::TessellatedTerrain(TessellatedTerrain&& other) : 

AbstractRenderableObject(std::move(other)), AbstractRenderableObjectTextured(std::move(other)), ExtensionAggregator(std::move(other)), 
num_u_base_nodes{ other.num_u_base_nodes }, num_v_base_nodes{ other.num_v_base_nodes }, lod{ other.lod },
ogl_vertex_attribute_object{ other.ogl_vertex_attribute_object }, ogl_vertex_buffer_object{ other.ogl_vertex_buffer_object },
ogl_index_buffer_object{ other.ogl_index_buffer_object }, height_map_sampler_ref_code{ std::move(other.height_map_sampler_ref_code) },
height_map_ref_code{ other.height_map_ref_code }, 
fractal_noise{ std::move(other.fractal_noise) }, fractal_noise_map_tex_res{ std::move(other.fractal_noise_map_tex_res) },
height_map_texture_u_res{ other.height_map_texture_u_res },
height_map_texture_v_res{ other.height_map_texture_v_res },
height_map_numeric_data(std::move(other.height_map_numeric_data)), is_heightmap_normalized{ other.is_heightmap_normalized },
texture_u_scale{ other.texture_u_scale },
texture_v_scale{ other.texture_v_scale },
height_map_level_offset{ other.height_map_level_offset },
height_map_normalization_constant{ other.height_map_normalization_constant },
terrain_texture_ref_code{ other.terrain_texture_ref_code }, terrain_texture_sampler_ref_code{ std::move(other.terrain_texture_sampler_ref_code) },
tess_terrain_rendering_program0_ref_code(other.tess_terrain_rendering_program0_ref_code),
ref_counter{ other.ref_counter }

{
	(*ref_counter)++;	//increment reference counter related to the owned OpenGL objects

	//Invalidate OpenGL objects owned by the move source
	other.ogl_vertex_attribute_object = 0;
	other.ogl_vertex_buffer_object = 0;
	other.ogl_index_buffer_object = 0;
}


TessellatedTerrain::~TessellatedTerrain()
{
	(*ref_counter)--;	//decrement reference counter related to the OpenGL objects owned by TessellatedTerrain

	//if reference counter equals 0, destroy OpenGL objects owned by the current context
	if (!(*ref_counter))
	{
		free_ogl_resources();
		delete ref_counter;
	}
}


TessellatedTerrain& TessellatedTerrain::operator=(const TessellatedTerrain& other)
{
	//Account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	//Increment reference counter of the source of assignment
	(*other.ref_counter)++;

	//Decrement reference counter of the current context
	(*ref_counter)--;

	//If current context is no any longer referred by any other object, destroy OpenGL resources
	if (!(*ref_counter))
		free_ogl_resources();

	//Perform object assignment

	//Begin by assigning the base classes
	AbstractRenderableObject::operator=(other);
	AbstractRenderableObjectTextured::operator=(other);
	ExtensionAggregator::operator=(other);

	//Assign the rest of the object
	num_u_base_nodes = other.num_u_base_nodes;
	num_v_base_nodes = other.num_v_base_nodes;
	lod = other.lod;
	texture_u_scale = other.texture_u_scale;
	texture_v_scale = other.texture_v_scale;

	ogl_vertex_attribute_object = other.ogl_vertex_attribute_object;
	ogl_vertex_buffer_object = other.ogl_vertex_buffer_object;
	ogl_index_buffer_object = other.ogl_index_buffer_object;

	height_map_ref_code = other.height_map_ref_code;
	height_map_sampler_ref_code = other.height_map_sampler_ref_code;

	fractal_noise = other.fractal_noise;
	fractal_noise_map_tex_res.first = fractal_noise.retrieveNoiseMap();
	fractal_noise_map_tex_res.second = other.fractal_noise_map_tex_res.second;
	updateTexture(fractal_noise_map_tex_res.second, fractal_noise_map_tex_res.first, terrain_texture_sampler_ref_code);

	height_map_texture_u_res = other.height_map_texture_u_res;
	height_map_texture_v_res = other.height_map_texture_v_res;
	height_map_numeric_data = other.height_map_numeric_data;
	is_heightmap_normalized = other.is_heightmap_normalized;
	height_map_level_offset = other.height_map_level_offset;
	height_map_normalization_constant = other.height_map_normalization_constant;

	terrain_texture_ref_code = other.terrain_texture_ref_code;
	terrain_texture_sampler_ref_code = other.terrain_texture_sampler_ref_code;

	tess_terrain_rendering_program0_ref_code = other.tess_terrain_rendering_program0_ref_code;
	ref_counter = other.ref_counter;

	return *this;
}

TessellatedTerrain& TessellatedTerrain::operator=(TessellatedTerrain&& other)
{
	//Account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	//Move-assign the base objects
	AbstractRenderableObject::operator=(std::move(other));
	AbstractRenderableObjectTextured::operator=(std::move(other));
	ExtensionAggregator::operator=(std::move(other));

	//Increment reference counter of the source of assignment
	(*other.ref_counter)++;

	//Swap reference counters between assignment source and assignment destination
	std::swap(ref_counter, other.ref_counter);

	//Swap OpenGL resources between assignment source and assignment destination
	std::swap(ogl_vertex_attribute_object, other.ogl_vertex_attribute_object);
	std::swap(ogl_vertex_buffer_object, other.ogl_vertex_buffer_object);
	std::swap(ogl_index_buffer_object, other.ogl_index_buffer_object);

	//Implement move-assignment for the rest of the current context's state
	num_u_base_nodes = other.num_u_base_nodes;
	num_v_base_nodes = other.num_v_base_nodes;
	texture_u_scale = other.texture_u_scale;
	texture_v_scale = other.texture_v_scale;
	lod = other.lod;

	height_map_ref_code = other.height_map_ref_code;
	height_map_sampler_ref_code = std::move(other.height_map_sampler_ref_code);

	fractal_noise = std::move(other.fractal_noise);
	fractal_noise_map_tex_res = std::move(other.fractal_noise_map_tex_res);

	height_map_texture_u_res = other.height_map_texture_u_res;
	height_map_texture_v_res = other.height_map_texture_v_res;
	height_map_numeric_data = std::move(other.height_map_numeric_data);
	is_heightmap_normalized = other.is_heightmap_normalized;
	height_map_level_offset = other.height_map_level_offset;
	height_map_normalization_constant = other.height_map_normalization_constant;

	terrain_texture_ref_code = other.terrain_texture_ref_code;
	terrain_texture_sampler_ref_code = std::move(other.terrain_texture_sampler_ref_code);

	tess_terrain_rendering_program0_ref_code = other.tess_terrain_rendering_program0_ref_code;

	return *this;
}


void TessellatedTerrain::defineHeightMap(const std::string& file_height_map, bool normalize_height_map /* = true */)
{
	std::string parse_error;
	if (!parseHeightMapFromFile(file_height_map, height_map_numeric_data, height_map_texture_u_res, height_map_texture_v_res, 
		height_map_level_offset, height_map_normalization_constant, parse_error, normalize_height_map))
	{
		set_error_state(true);
		set_error_string(parse_error);
		call_error_callback(parse_error);
		return;
	}
	is_heightmap_normalized = normalize_height_map;

	//Initialize height map 2D-texture
	TextureSize height_map_texture_size;
	height_map_texture_size.width = height_map_texture_u_res;
	height_map_texture_size.height = height_map_texture_v_res;
	height_map_texture_size.depth = 0;

	ImmutableTexture2D height_map_texture;
	height_map_texture.allocateStorage(1, 1, height_map_texture_size, InternalPixelFormat::SIZED_FLOAT_R32);
	height_map_texture.setMipmapLevelLayerData(0, 0, PixelLayout::RED, PixelDataType::FLOAT, height_map_numeric_data.data());
	
	if (!height_map_ref_code)
		height_map_ref_code = registerTexture(height_map_texture, height_map_sampler_ref_code);
	else
		updateTexture(height_map_ref_code, height_map_texture);
}

void TessellatedTerrain::defineHeightMap(const float* height_map, uint32_t u_resolution, uint32_t v_resolution, bool normalize_height_map /* = true */)
{
	//Update height map texture resolution settings
	height_map_texture_u_res = u_resolution;
	height_map_texture_v_res = v_resolution;

	//Normalize height map data
	float* height_map_copy = new float[u_resolution*v_resolution];
	memcpy(height_map_copy, height_map, sizeof(float)*u_resolution*v_resolution);

	is_heightmap_normalized = normalize_height_map;
	if (normalize_height_map)
	{
		height_map_normalization_constant = std::numeric_limits<float>::min();
		height_map_level_offset = std::numeric_limits<float>::max();
		for (int i = 0; i < static_cast<signed>(u_resolution*v_resolution); ++i)
		{
			if (height_map_copy[i] < height_map_level_offset) height_map_level_offset = height_map_copy[i];
			if (height_map_copy[i] > height_map_normalization_constant) height_map_normalization_constant = height_map_copy[i];
		}
		height_map_normalization_constant -= height_map_level_offset;
		for (int i = 0; i < static_cast<signed>(u_resolution*v_resolution); ++i)
		{
			height_map_copy[i] -= height_map_level_offset;
			height_map_copy[i] /= height_map_normalization_constant;
		}
	}
	else
	{
		height_map_normalization_constant = 1;
		height_map_level_offset = 0;
	}
	height_map_numeric_data.clear();
	height_map_numeric_data.insert(height_map_numeric_data.end(), height_map_copy, height_map_copy + u_resolution*v_resolution);

	TextureSize height_map_texture_size;
	height_map_texture_size.width = height_map_texture_u_res;
	height_map_texture_size.height = height_map_texture_v_res;
	height_map_texture_size.depth = 0;

	ImmutableTexture2D height_map_texture;
	height_map_texture.allocateStorage(1, 1, height_map_texture_size, InternalPixelFormat::SIZED_FLOAT_R32);
	height_map_texture.setMipmapLevelLayerData(0, 0, PixelLayout::RED, PixelDataType::FLOAT, height_map_copy);
	delete[] height_map_copy;

	if (!height_map_ref_code)
		height_map_ref_code = registerTexture(height_map_texture, height_map_sampler_ref_code);
	else
		updateTexture(height_map_ref_code, height_map_texture);
}

void TessellatedTerrain::installTexture(const ImmutableTexture2D& terrain_texture, 
	float texture_u_scale /* = 0.1f */, float texture_v_scale /* = 0.1f */)
{
	if (!terrain_texture_ref_code)
		terrain_texture_ref_code = registerTexture(terrain_texture, terrain_texture_sampler_ref_code);
	else
		updateTexture(terrain_texture_ref_code, terrain_texture);

	this->texture_u_scale = texture_u_scale;
	this->texture_v_scale = texture_v_scale;

	fractal_noise.generateNoiseMap();
}

float TessellatedTerrain::setLODFactor(float new_lod_factor)
{
	float rv = lod;
	lod = new_lod_factor;
	return rv;
}

void TessellatedTerrain::scale(float x_scale_factor, float y_scale_factor, float z_scale_factor)
{
	retrieveShaderProgram(tess_terrain_rendering_program0_ref_code)->assignUniformVector("v2FractalNoiseScaling", vec2{ x_scale_factor / fractal_noise_resolution, z_scale_factor / fractal_noise_resolution }*fractal_noise_scaling_weight);

	AbstractRenderableObject::scale(x_scale_factor, y_scale_factor, z_scale_factor);
}

void TessellatedTerrain::scale(const vec3& new_scale_factors)
{
	scale(new_scale_factors.x, new_scale_factors.y, new_scale_factors.z);
}

float TessellatedTerrain::getHeightLevelOffset() const { return height_map_level_offset; }

float TessellatedTerrain::getHeightNormalizationConstant() const { return height_map_normalization_constant; }

bool TessellatedTerrain::isHeightMapNormalized() const { return is_heightmap_normalized; }

void TessellatedTerrain::retrieveHeightMap(const float** height_map, uint32_t& width, uint32_t& height) const
{
	*height_map = height_map_numeric_data.size() ? height_map_numeric_data.data() : nullptr;
	width = height_map_texture_u_res;
	height = height_map_texture_v_res;
}

float TessellatedTerrain::getLODFactor() const { return lod; }

uint32_t TessellatedTerrain::getNumberOfRenderingPasses(uint32_t rendering_mode) const 
{ 
	switch (rendering_mode)
	{
	case 0:
	case 1:
		return 1;
	default: 
		return 0;
	}
}

bool TessellatedTerrain::supportsRenderingMode(uint32_t rendering_mode) const
{
	switch (rendering_mode)
	{
	case TW_RENDERING_MODE_DEFAULT:
	case TW_RENDERING_MODE_SILHOUETTE:
		return true;
	default: 
		return false;
	}
}

void TessellatedTerrain::configureViewProjectionTransform(const AbstractProjectingDevice& projecting_device)
{
	switch (getActiveRenderingMode())
	{

	case TW_RENDERING_MODE_DEFAULT:
	{
		mat4 ModelViewTransform = projecting_device.getViewTransform()*getObjectTransform()*getObjectScaleTransform();
		retrieveShaderProgram(tess_terrain_rendering_program0_ref_code)->assignUniformVector("Scale", getObjectScale());
		retrieveShaderProgram(tess_terrain_rendering_program0_ref_code)->assignUniformMatrix("ModelViewTransform", ModelViewTransform);
		retrieveShaderProgram(tess_terrain_rendering_program0_ref_code)->assignUniformMatrix("ProjectionTransform", projecting_device.getProjectionTransform());
		break;
	}

	}
	
}

bool TessellatedTerrain::configureRendering(AbstractRenderingDevice& render_target, uint32_t rendering_pass)
{
	if (rendering_pass > 0) return false;

	//If requested render target is not yet active, activate it
	if (!render_target.isActive())
		render_target.makeActive();

	switch (getActiveRenderingMode())
	{
	case TW_RENDERING_MODE_DEFAULT:
		//Bind object's data buffer
		glBindVertexArray(ogl_vertex_attribute_object);

		if (height_map_ref_code)
			bindTexture(height_map_ref_code);

		retrieveShaderProgram(tess_terrain_rendering_program0_ref_code)->assignUniformScalar("height_map_sampler", getBindingUnit(height_map_ref_code));
		retrieveShaderProgram(tess_terrain_rendering_program0_ref_code)->assignUniformScalar("terrain_tex_sampler", getBindingUnit(terrain_texture_ref_code));
		retrieveShaderProgram(tess_terrain_rendering_program0_ref_code)->assignUniformScalar("lod", lod);
		retrieveShaderProgram(tess_terrain_rendering_program0_ref_code)->assignUniformVector("screen_size", getScreenSize());
		retrieveShaderProgram(tess_terrain_rendering_program0_ref_code)->assignUniformScalar("s2dFractalNoise", getBindingUnit(fractal_noise_map_tex_res.second));
		
		if (terrain_texture_ref_code)
			bindTexture(terrain_texture_ref_code);

		if (fractal_noise_map_tex_res.second)
			bindTexture(fractal_noise_map_tex_res.second);


		COMPLETE_SHADER_PROGRAM_CAST(retrieveShaderProgram(tess_terrain_rendering_program0_ref_code)).activate();

		glPatchParameteri(GL_PATCH_VERTICES, 4);

		return true;

	case TW_RENDERING_MODE_SILHOUETTE:
		return true;

	default:
		return false;
	}
	
	
	return true;
}
bool TessellatedTerrain::render()
{
	glDrawElements(GL_PATCHES, (num_u_base_nodes - 1)*(num_v_base_nodes - 1) * 4, GL_UNSIGNED_INT, NULL);
	return true;
}

bool TessellatedTerrain::configureRenderingFinalization()
{
	return true;
}



bool TessellatedTerrain::parseHeightMapFromFile(const std::string& file_height_map, std::vector<float>& loaded_height_map, uint32_t& width, uint32_t& height, 
	float& offset, float& range, std::string& parse_error, bool normalize_height_map /* = true */)
{
	std::ifstream file_height_map_input_stream{ file_height_map };		//input stream of the height map file source

	//if file has not been found, throw an exception
	if (!file_height_map_input_stream.good())
		throw(std::runtime_error("Unable to open requested height map source file"));

	int num_of_entries{ -1 };		//number of entries in each row of the height map
	int row_counter{ 0 };	//index of the row currently being parsed

	do
	{
		++row_counter;	//increment current row number

		//Determine length of the ensuing row
		std::streampos current_position = file_height_map_input_stream.tellg();
		file_height_map_input_stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		if (file_height_map_input_stream.eof())		//if end-of-file has been reached clear input stream state bits, so that the stream can still be used
			file_height_map_input_stream.clear();
		unsigned long long line_length = file_height_map_input_stream.tellg() - current_position + 1;
		file_height_map_input_stream.seekg(current_position);	//caret return

		//Allocate buffer and read new line from file into the buffer
		char* height_map_raw_data = new char[line_length];
		file_height_map_input_stream.getline(height_map_raw_data, line_length);
		line_length = file_height_map_input_stream.gcount() + 1;
		height_map_raw_data[line_length - 1] = '\n';

		//Parse the new line
		std::string current_line{ height_map_raw_data, static_cast<std::size_t>(line_length) };	//wrap the line into a C++ string object
		delete[] height_map_raw_data;	//release raw data buffer

		//Parse the line and identify numeric tokens
		uint32_t current_line_num_of_entries{ 0 };	//number of elements in the line currently being parsed
		std::string::size_type token_start_pos{ 0 };	//start position of string token
		std::string::size_type token_end_pos{ std::numeric_limits<std::string::size_type>::max() };	//end position of string token
		std::string token;		//string token

		do
		{
			token_start_pos = current_line.find_first_not_of(" \t,\n", token_end_pos + 1);
			if (token_start_pos == std::string::npos) break;	//no more tokens left in the current line: cancel parsing

			token_end_pos = current_line.find_first_of(" \t,\n", token_start_pos);
			if (token_end_pos == std::string::npos) token_end_pos = current_line.size();	//no more token separators detected: succeeding token ends at the end-of-file


			token = current_line.substr(token_start_pos, token_end_pos - token_start_pos);
			//remove prefixing and trailing spaces
			std::string::size_type _s = token.find_first_not_of(" \t");		//find first non-space character in the token
			std::string::size_type _e = token.find_last_not_of(" \t");		//find last non-space character in the token
			token = token.substr(_s, _e - _s + 1);
			bool token_empty = true;	//if token has only non-printable characters, then it's considered empty and discarded
			for (char c : token)
			{
				if (std::isprint(c))
				{
					token_empty = false;
					break;
				}
			}
			if (token_empty) continue; //empty tokens do not count

			try{
				loaded_height_map.push_back(std::stof(token));
			}
			catch (std::invalid_argument inv_arg_exception)
			{
				parse_error = "Unable to parse ASCII height map table in file " + file_height_map + ": token " + token +
					" has no valid floating point representation";
				return false;
			}
			catch (std::out_of_range out_of_range_exception)
			{
				parse_error = "Unable to parse ASCII height map table in file " + file_height_map + ": token " + token +
					" can not be represented by a floating point number of required precision";
				return false;
			}


			++current_line_num_of_entries;
		} while (token_end_pos < current_line.size() - 1);

		if (!current_line_num_of_entries)	//an empty line was parsed: this is not an error, just skip it
		{
			--row_counter;
			continue;
		}

		if (num_of_entries == -1)num_of_entries = current_line_num_of_entries;

		if (current_line_num_of_entries != num_of_entries)
		{
			parse_error = "Unable to parse ASCII height map table in file " + file_height_map + ": the number of entries in row " + std::to_string(row_counter) + " differs from the number of entries in the previous rows";
			return false;
		}
			

	} while (!file_height_map_input_stream.eof());

	//Normalize height map data if necessary
	if (normalize_height_map)
	{
		range = static_cast<float>(*std::max_element(loaded_height_map.begin(), loaded_height_map.end()));
		offset = static_cast<float>(*std::min_element(loaded_height_map.begin(), loaded_height_map.end()));

		range -= offset;
		for (float& height : loaded_height_map)
		{
			height -= offset;
			height /= range;
		}
	}
	else
	{
		range = 1;
		offset = 0;
	}

	//Swap row ordering, so that it corresponds to how OpenGL represents texture data
	std::vector<float> row;
	row.resize(num_of_entries);
	for (int i = 0; i < row_counter / 2; ++i)
	{
		std::copy(loaded_height_map.begin() + i * num_of_entries,
			loaded_height_map.begin() + (i + 1) * num_of_entries, row.begin());

		std::copy(loaded_height_map.end() - (i + 1) * num_of_entries,
			loaded_height_map.end() - i * num_of_entries,
			loaded_height_map.begin() + i*num_of_entries);

		std::copy(row.begin(), row.end(), loaded_height_map.end() - (i + 1)*num_of_entries);
	}

	width = static_cast<uint32_t>(row_counter);
	height = static_cast<uint32_t>(num_of_entries);

	return true;
}