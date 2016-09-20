#include "AbstractRenderableObjectLightEx.h"

using namespace tiny_world;


void AbstractRenderableObjectLightEx::setup_extension()
{
	//Initialize default sampler used by the lighting extension
	light_texture_sampler_ref_code = createTextureSampler("AbstractRenderableObjectLightEx::light_textire_sampler", SamplerMagnificationFilter::LINEAR, SamplerMinificationFilter::LINEAR_MIPMAP_NEAREST,
		SamplerWrapping{ SamplerWrappingMode::REPEAT, SamplerWrappingMode::REPEAT, SamplerWrappingMode::CLAMP_TO_EDGE });



	//Register dummy textures to the renderable object in order to guarantee proper bindings of the samplers
	char aux_value[48]{0};

	ImmutableTexture2D dummy_normal_map{ "AbstractRenderableObjectLightEx::dummy_normal_map" };
	dummy_normal_map.allocateStorage(1, 1, TextureSize{ 1, 1, 0 }, InternalPixelFormat::SIZED_R8);
	dummy_normal_map.setMipmapLevelData(0, PixelLayout::RED, PixelDataType::BYTE, &aux_value[0]);
	normal_map_ref_code = registerTexture(dummy_normal_map, light_texture_sampler_ref_code);

	ImmutableTexture2D dummy_array_normal_map{ "AbstractRenderableObjectLightEx::dummy_array_normal_map" };
	dummy_array_normal_map.allocateStorage(1, 2, TextureSize{ 1, 1, 0 }, InternalPixelFormat::SIZED_R8);
	dummy_array_normal_map.setMipmapLevelLayerData(0, 0, PixelLayout::RED, PixelDataType::BYTE, &aux_value[0]);
	dummy_array_normal_map.setMipmapLevelLayerData(0, 1, PixelLayout::RED, PixelDataType::BYTE, &aux_value[1]);
	array_normal_map_ref_code = registerTexture(dummy_array_normal_map, light_texture_sampler_ref_code);


	ImmutableTexture2D dummy_specular_map{ "AbstractRenderableObjectLightEx::dummy_specular_map" };
	dummy_specular_map.allocateStorage(1, 1, TextureSize{ 1, 1, 0 }, InternalPixelFormat::SIZED_R8);
	dummy_specular_map.setMipmapLevelData(0, PixelLayout::RED, PixelDataType::BYTE, &aux_value[0]);
	specular_map_ref_code = registerTexture(dummy_specular_map, light_texture_sampler_ref_code);

	ImmutableTexture2D dummy_array_specular_map{ "AbstractRenderableObjectLightEx::dummy_array_specular_map" };
	dummy_array_specular_map.allocateStorage(1, 2, TextureSize{ 1, 1, 0 }, InternalPixelFormat::SIZED_R8);
	dummy_array_specular_map.setMipmapLevelLayerData(0, 0, PixelLayout::RED, PixelDataType::BYTE, &aux_value[0]);
	dummy_array_specular_map.setMipmapLevelLayerData(0, 1, PixelLayout::RED, PixelDataType::BYTE, &aux_value[1]);
	array_specular_map_ref_code = registerTexture(dummy_array_specular_map, light_texture_sampler_ref_code);


	ImmutableTexture2D dummy_emission_map{ "AbstractRenderableObjectLightEx::dummy_emission_map" };
	dummy_emission_map.allocateStorage(1, 1, TextureSize{ 1, 1, 0 }, InternalPixelFormat::SIZED_R8);
	dummy_emission_map.setMipmapLevelData(0, PixelLayout::RED, PixelDataType::BYTE, &aux_value[0]);
	emission_map_ref_code = registerTexture(dummy_emission_map, light_texture_sampler_ref_code);

	ImmutableTexture2D dummy_array_emission_map{ "AbstractRenderableObjectLightEx::dummy_array_emission_map" };
	dummy_array_emission_map.allocateStorage(1, 2, TextureSize{ 1, 1, 0 }, InternalPixelFormat::SIZED_R8);
	dummy_array_emission_map.setMipmapLevelLayerData(0, 0, PixelLayout::RED, PixelDataType::BYTE, &aux_value[0]);
	dummy_array_emission_map.setMipmapLevelLayerData(0, 1, PixelLayout::RED, PixelDataType::BYTE, &aux_value[1]);
	array_emission_map_ref_code = registerTexture(dummy_array_emission_map, light_texture_sampler_ref_code);


	ImmutableTexture2D dummy_environment_map{ "AbstractRenderableObjectLightEx::dummy_environment_map" };
	dummy_environment_map.allocateStorage(1, 1, TextureSize{ 1, 1, 0 }, InternalPixelFormat::SIZED_R8);
	dummy_environment_map.setMipmapLevelData(0, PixelLayout::RED, PixelDataType::BYTE, &aux_value[0]);
	environment_map_ref_code = registerTexture(dummy_environment_map, light_texture_sampler_ref_code);

	ImmutableTexture2D dummy_array_environment_map{ "AbstractRenderableObjectLightEx::dummy_array_environment_map" };
	dummy_array_environment_map.allocateStorage(1, 2, TextureSize{ 1, 1, 0 }, InternalPixelFormat::SIZED_R8);
	dummy_array_environment_map.setMipmapLevelLayerData(0, 0, PixelLayout::RED, PixelDataType::BYTE, &aux_value[0]);
	dummy_array_environment_map.setMipmapLevelLayerData(0, 1, PixelLayout::RED, PixelDataType::BYTE, &aux_value[1]);
	array_environment_map_ref_code = registerTexture(dummy_array_environment_map, light_texture_sampler_ref_code);

	ImmutableTextureCubeMap dummy_cube_environment_map{ "AbstractRenderableObjectLightEx::dummy_cube_environment_map" };
	dummy_cube_environment_map.allocateStorage(1, 1, TextureSize{ 1, 1, 0 }, InternalPixelFormat::SIZED_R8);
	dummy_cube_environment_map.setMipmapLevelData(0, PixelLayout::RED, PixelDataType::BYTE, &aux_value[0], &aux_value[1], &aux_value[2], &aux_value[3], &aux_value[4], &aux_value[5]);
	cube_environment_map_ref_code = registerTexture(dummy_cube_environment_map, light_texture_sampler_ref_code);

	ImmutableTextureCubeMap dummy_cube_array_environment_map{ "AbstractRenderableObjectLightEx::dummy_cube_array_environment_map" };
	dummy_cube_array_environment_map.allocateStorage(1, 2, TextureSize{ 1, 1, 0 }, InternalPixelFormat::SIZED_R8);
	dummy_cube_array_environment_map.setMipmapLevelLayerData(0, 0, PixelLayout::RED, PixelDataType::BYTE, &aux_value[0], &aux_value[1], &aux_value[2], &aux_value[3], &aux_value[4], &aux_value[5]);
	dummy_cube_array_environment_map.setMipmapLevelLayerData(0, 1, PixelLayout::RED, PixelDataType::BYTE, &aux_value[6], &aux_value[7], &aux_value[8], &aux_value[9], &aux_value[10], &aux_value[11]);
	cube_array_environment_map_ref_code = registerTexture(dummy_cube_array_environment_map, light_texture_sampler_ref_code);
}


AbstractRenderableObjectLightEx::AbstractRenderableObjectLightEx() : 
p_lighting_conditions{ nullptr },
has_normal_map{ false }, supports_array_normal_maps{ false }, enable_normal_map{ true }, 
normal_map_sample_retriever_variation("DefaultNormalMapSampleRetriever"), 
has_specular_map{ false }, supports_array_specular_maps{ false }, enable_specular_map{ true }, 
specular_map_sample_retriever_variation("DefaultSpecularMapSampleRetriever"),
has_emission_map{ false }, supports_array_emission_maps{ false }, enable_emission_map{ true }, 
emission_map_sample_retriever_variation("DefaultEmissionMapSampleRetriever"),
has_environment_map{ false }, supports_array_environment_maps{ false }, enable_environment_map{ true }, 
uses_custom_environment_map_sample_retriever{ false }, environment_map_sample_retriever_variation("SphericalEnvironmentMapSampleRetriever"),
uses_custom_array_environment_map_sample_retriever{ false }, array_environment_map_sample_retriever_variation("SphericalArrayEnvironmentMapSampleRetriever"),
uses_custom_cube_environment_map_sample_retriever{ false }, cube_environment_map_sample_retriever_variation("CubicEnvironmentMapSampleRetriever"),
uses_custom_cube_array_environment_map_sample_retriever{ false }, cube_array_environment_map_sample_retriever_variation("CubicArrayEnvironmentMapSampleRetriever"),
reflection_mapper_variation("DefaultReflectionMapper"),
viewer_location{ 0.0f, 0.0f, 0.0f },
viewer_rotation{ 1.0f },
default_diffuse_color{ 1.0f, 1.0f, 1.0f, 1.0f },
default_specular_color{ 1.0f, 1.0f, 1.0f },
default_specular_exponent{ 1.0f },
default_emission_color{ 0.0f, 0.0f, 0.0f }
{
	setup_extension();
}




AbstractRenderableObjectLightEx::AbstractRenderableObjectLightEx(std::initializer_list<std::pair<PipelineStage, std::string>> auxiliary_shader_sources) : 
p_lighting_conditions{ nullptr },
has_normal_map{ false }, supports_array_normal_maps{ false }, enable_normal_map{ true },
normal_map_sample_retriever_variation("DefaultNormalMapSampleRetriever"),
has_specular_map{ false }, supports_array_specular_maps{ false }, enable_specular_map{ true },
specular_map_sample_retriever_variation("DefaultSpecularMapSampleRetriever"),
has_emission_map{ false }, supports_array_emission_maps{ false }, enable_emission_map{ true },
emission_map_sample_retriever_variation("DefaultEmissionMapSampleRetriever"),
has_environment_map{ false }, supports_array_environment_maps{ false }, enable_environment_map{ true },
uses_custom_environment_map_sample_retriever{ false }, environment_map_sample_retriever_variation("SphericalEnvironmentMapSampleRetriever"),
uses_custom_array_environment_map_sample_retriever{ false }, array_environment_map_sample_retriever_variation("SphericalArrayEnvironmentMapSampleRetriever"),
uses_custom_cube_environment_map_sample_retriever{ false }, cube_environment_map_sample_retriever_variation("CubicEnvironmentMapSampleRetriever"),
uses_custom_cube_array_environment_map_sample_retriever{ false }, cube_array_environment_map_sample_retriever_variation("CubicArrayEnvironmentMapSampleRetriever"),
reflection_mapper_variation("DefaultReflectionMapper"),
viewer_location{ 0.0f, 0.0f, 0.0f },
viewer_rotation{ 1.0f },
default_diffuse_color{ 1.0f, 1.0f, 1.0f, 1.0f },
default_specular_color{ 1.0f, 1.0f, 1.0f },
default_specular_exponent{ 1.0f },
default_emission_color{ 0.0f, 0.0f, 0.0f }
{
	setup_extension();

	auxiliary_glsl_sources = 
		std::map<PipelineStage, std::string>{ std::make_pair(PipelineStage::VERTEX_SHADER, std::string("")), std::make_pair(PipelineStage::TESS_CONTROL_SHADER, std::string("")),
		std::make_pair(PipelineStage::TESS_EVAL_SHADER, std::string("")), std::make_pair(PipelineStage::GEOMETRY_SHADER, std::string("")),
		std::make_pair(PipelineStage::FRAGMENT_SHADER, std::string("")), std::make_pair(PipelineStage::COMPUTE_SHADER, std::string("")) };

	for (std::pair<PipelineStage, const std::string&> source : auxiliary_shader_sources)
	{
		std::string error_message;
		std::pair<bool, std::string> parse_result = Shader::parseShaderSource(source.second, &error_message);
		if (!parse_result.first)
		{
			set_error_state(true);
			set_error_string(error_message);
			call_error_callback(error_message);
			return;
		}

		auxiliary_glsl_sources[source.first] += parse_result.second + "\n";
	}
}




AbstractRenderableObjectLightEx::AbstractRenderableObjectLightEx(const AbstractRenderableObjectLightEx& other) :
auxiliary_glsl_sources(other.auxiliary_glsl_sources),

p_lighting_conditions{ other.p_lighting_conditions },

has_normal_map{ other.has_normal_map }, supports_array_normal_maps{ other.supports_array_normal_maps }, enable_normal_map{ other.enable_normal_map }, 
normal_map_sample_retriever_variation(other.normal_map_sample_retriever_variation), 
normal_map_ref_code{ other.normal_map_ref_code }, array_normal_map_ref_code{ other.array_normal_map_ref_code },

has_specular_map{ other.has_specular_map }, supports_array_specular_maps{ other.supports_array_specular_maps }, enable_specular_map{ other.enable_specular_map }, 
specular_map_sample_retriever_variation(other.specular_map_sample_retriever_variation),
specular_map_ref_code{ other.specular_map_ref_code }, array_specular_map_ref_code{ other.array_specular_map_ref_code },

has_emission_map{ other.has_emission_map }, supports_array_emission_maps{ other.supports_array_emission_maps }, enable_emission_map{ other.enable_emission_map }, 
emission_map_sample_retriever_variation(other.emission_map_sample_retriever_variation),
emission_map_ref_code{ other.emission_map_ref_code }, array_emission_map_ref_code{ other.array_emission_map_ref_code },

has_environment_map{ other.has_environment_map }, supports_array_environment_maps{ other.supports_array_environment_maps }, enable_environment_map{ other.enable_environment_map },
environment_map_type{ other.environment_map_type }, environment_map_ref_code{ other.environment_map_ref_code }, array_environment_map_ref_code{ other.array_environment_map_ref_code },
cube_environment_map_ref_code{ other.cube_environment_map_ref_code }, cube_array_environment_map_ref_code{ other.cube_array_environment_map_ref_code },

uses_custom_environment_map_sample_retriever{ other.uses_custom_environment_map_sample_retriever }, environment_map_sample_retriever_variation(other.environment_map_sample_retriever_variation), 
uses_custom_array_environment_map_sample_retriever{ other.uses_custom_array_environment_map_sample_retriever }, array_environment_map_sample_retriever_variation(other.array_environment_map_sample_retriever_variation),
uses_custom_cube_environment_map_sample_retriever{ other.uses_custom_cube_environment_map_sample_retriever }, cube_environment_map_sample_retriever_variation(other.cube_environment_map_sample_retriever_variation),
uses_custom_cube_array_environment_map_sample_retriever{ other.uses_custom_cube_array_environment_map_sample_retriever }, cube_array_environment_map_sample_retriever_variation(other.cube_array_environment_map_sample_retriever_variation),
reflection_mapper_variation(other.reflection_mapper_variation),

viewer_location{ other.viewer_location },
viewer_rotation{ other.viewer_rotation },
default_diffuse_color{ other.default_diffuse_color },
default_specular_color{ other.default_specular_color },
default_specular_exponent{ other.default_specular_exponent },
default_emission_color{ other.default_emission_color },
light_texture_sampler_ref_code{ other.light_texture_sampler_ref_code }
{
	
}

AbstractRenderableObjectLightEx::AbstractRenderableObjectLightEx(AbstractRenderableObjectLightEx&& other) :
auxiliary_glsl_sources(std::move(other.auxiliary_glsl_sources)),

p_lighting_conditions{ other.p_lighting_conditions },

has_normal_map{ other.has_normal_map }, supports_array_normal_maps{ other.supports_array_normal_maps }, enable_normal_map{ other.enable_normal_map }, 
normal_map_sample_retriever_variation(std::move(other.normal_map_sample_retriever_variation)),
normal_map_ref_code{ std::move(other.normal_map_ref_code) }, array_normal_map_ref_code{ std::move(other.array_normal_map_ref_code) },

has_specular_map{ other.has_specular_map }, supports_array_specular_maps{ other.supports_array_specular_maps }, enable_specular_map{ other.enable_specular_map }, 
specular_map_sample_retriever_variation(std::move(other.specular_map_sample_retriever_variation)),
specular_map_ref_code{ std::move(other.specular_map_ref_code) }, array_specular_map_ref_code{ std::move(other.array_specular_map_ref_code) },

has_emission_map{ other.has_emission_map }, supports_array_emission_maps{ other.supports_array_emission_maps }, enable_emission_map{ other.enable_emission_map }, 
emission_map_sample_retriever_variation(std::move(other.emission_map_sample_retriever_variation)),
emission_map_ref_code{ std::move(other.emission_map_ref_code) }, array_emission_map_ref_code{ std::move(other.array_emission_map_ref_code) }, 

has_environment_map{ other.has_environment_map }, supports_array_environment_maps{ other.supports_array_environment_maps }, enable_environment_map{ other.enable_environment_map }, 
environment_map_type{ std::move(other.environment_map_type) }, environment_map_ref_code{ std::move(other.environment_map_ref_code) },
array_environment_map_ref_code{ std::move(other.array_environment_map_ref_code) }, cube_environment_map_ref_code{ std::move(other.cube_environment_map_ref_code) },
cube_array_environment_map_ref_code{ std::move(other.cube_array_environment_map_ref_code) },

uses_custom_environment_map_sample_retriever{ other.uses_custom_environment_map_sample_retriever }, environment_map_sample_retriever_variation(std::move(other.environment_map_sample_retriever_variation)),
uses_custom_array_environment_map_sample_retriever{ other.uses_custom_array_environment_map_sample_retriever }, array_environment_map_sample_retriever_variation(std::move(other.array_environment_map_sample_retriever_variation)),
uses_custom_cube_environment_map_sample_retriever{ other.uses_custom_cube_environment_map_sample_retriever }, cube_environment_map_sample_retriever_variation(std::move(other.cube_environment_map_sample_retriever_variation)),
uses_custom_cube_array_environment_map_sample_retriever{ other.uses_custom_cube_array_environment_map_sample_retriever }, cube_array_environment_map_sample_retriever_variation(std::move(other.cube_array_environment_map_sample_retriever_variation)),
reflection_mapper_variation(std::move(other.reflection_mapper_variation)),

viewer_location{ std::move(other.viewer_location) },
viewer_rotation{ std::move(other.viewer_rotation) },
default_diffuse_color{ std::move(other.default_diffuse_color) },
default_specular_color{ std::move(other.default_specular_color) },
default_specular_exponent{ other.default_specular_exponent },
default_emission_color{ std::move(other.default_emission_color) },
light_texture_sampler_ref_code{ std::move(other.light_texture_sampler_ref_code) }
{
	
}

AbstractRenderableObjectLightEx& AbstractRenderableObjectLightEx::operator=(const AbstractRenderableObjectLightEx& other)
{
	//Account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	//Copy the rest of the object state from the source of the assignment
	auxiliary_glsl_sources = other.auxiliary_glsl_sources;

	p_lighting_conditions = other.p_lighting_conditions;

	has_normal_map = other.has_normal_map;
	supports_array_normal_maps = other.supports_array_normal_maps;
	enable_normal_map = other.enable_normal_map;
	normal_map_sample_retriever_variation = other.normal_map_sample_retriever_variation;
	normal_map_ref_code = other.normal_map_ref_code;
	array_normal_map_ref_code = other.array_normal_map_ref_code;


	has_specular_map = other.has_specular_map;
	supports_array_specular_maps = other.supports_array_specular_maps;
	enable_specular_map = other.enable_specular_map;
	specular_map_sample_retriever_variation = other.specular_map_sample_retriever_variation;
	specular_map_ref_code = other.specular_map_ref_code;
	array_specular_map_ref_code = other.array_specular_map_ref_code;


	has_emission_map = other.has_emission_map;
	supports_array_emission_maps = other.supports_array_emission_maps;
	enable_emission_map = other.enable_emission_map;
	emission_map_sample_retriever_variation = other.emission_map_sample_retriever_variation;
	emission_map_ref_code = other.emission_map_ref_code;
	array_emission_map_ref_code = other.array_emission_map_ref_code;


	has_environment_map = other.has_environment_map;
	supports_array_environment_maps = other.supports_array_environment_maps;
	enable_environment_map = other.enable_environment_map;
	environment_map_type = other.environment_map_type;
	environment_map_ref_code = other.environment_map_ref_code;
	array_environment_map_ref_code = other.array_environment_map_ref_code;
	cube_environment_map_ref_code = other.cube_environment_map_ref_code;
	cube_array_environment_map_ref_code = other.cube_array_environment_map_ref_code;


	uses_custom_environment_map_sample_retriever = other.uses_custom_environment_map_sample_retriever;
	environment_map_sample_retriever_variation = other.environment_map_sample_retriever_variation;
	
	uses_custom_array_environment_map_sample_retriever = other.uses_custom_array_environment_map_sample_retriever;
	array_environment_map_sample_retriever_variation = other.array_environment_map_sample_retriever_variation;

	uses_custom_cube_environment_map_sample_retriever = other.uses_custom_cube_environment_map_sample_retriever;
	cube_environment_map_sample_retriever_variation = other.cube_environment_map_sample_retriever_variation;

	uses_custom_cube_array_environment_map_sample_retriever = other.uses_custom_cube_array_environment_map_sample_retriever;
	cube_array_environment_map_sample_retriever_variation = other.cube_array_environment_map_sample_retriever_variation;

	reflection_mapper_variation = other.reflection_mapper_variation;


	viewer_location = other.viewer_location;
	viewer_rotation = other.viewer_rotation;
	default_diffuse_color = other.default_diffuse_color;
	default_specular_color = other.default_specular_color;
	default_specular_exponent = other.default_specular_exponent;
	default_emission_color = other.default_emission_color;

	light_texture_sampler_ref_code = other.light_texture_sampler_ref_code;


	return *this;
}

AbstractRenderableObjectLightEx& AbstractRenderableObjectLightEx::operator=(AbstractRenderableObjectLightEx&& other)
{
	//Account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	//Move state from the source to the destination object
	auxiliary_glsl_sources = std::move(other.auxiliary_glsl_sources);

	p_lighting_conditions = other.p_lighting_conditions;

	has_normal_map = other.has_normal_map;
	supports_array_normal_maps = other.supports_array_normal_maps;
	enable_normal_map = other.enable_normal_map;
	normal_map_sample_retriever_variation = std::move(other.normal_map_sample_retriever_variation);
	normal_map_ref_code = std::move(other.normal_map_ref_code);
	array_normal_map_ref_code = std::move(other.array_normal_map_ref_code);


	has_specular_map = other.has_specular_map;
	supports_array_specular_maps = other.supports_array_specular_maps;
	enable_specular_map = other.enable_specular_map;
	specular_map_sample_retriever_variation = std::move(other.specular_map_sample_retriever_variation);
	specular_map_ref_code = std::move(other.specular_map_ref_code);
	array_specular_map_ref_code = std::move(other.array_specular_map_ref_code);


	has_emission_map = other.has_emission_map;
	supports_array_emission_maps = other.supports_array_emission_maps;
	enable_emission_map = other.enable_emission_map;
	emission_map_sample_retriever_variation = std::move(other.emission_map_sample_retriever_variation);
	emission_map_ref_code = std::move(other.emission_map_ref_code);
	array_emission_map_ref_code = std::move(other.array_emission_map_ref_code);


	has_environment_map = other.has_environment_map;
	supports_array_environment_maps = other.supports_array_environment_maps;
	enable_environment_map = other.enable_environment_map;
	environment_map_type = std::move(other.environment_map_type);
	environment_map_ref_code = std::move(other.environment_map_ref_code);
	array_environment_map_ref_code = std::move(other.array_environment_map_ref_code);
	cube_environment_map_ref_code = std::move(other.cube_environment_map_ref_code);
	cube_array_environment_map_ref_code = std::move(other.cube_array_environment_map_ref_code);


	uses_custom_environment_map_sample_retriever = other.uses_custom_environment_map_sample_retriever;
	environment_map_sample_retriever_variation = std::move(other.environment_map_sample_retriever_variation);

	uses_custom_array_environment_map_sample_retriever = other.uses_custom_array_environment_map_sample_retriever;
	array_environment_map_sample_retriever_variation = std::move(other.array_environment_map_sample_retriever_variation);

	uses_custom_cube_environment_map_sample_retriever = other.uses_custom_cube_environment_map_sample_retriever;
	cube_environment_map_sample_retriever_variation = std::move(other.cube_environment_map_sample_retriever_variation);

	uses_custom_cube_array_environment_map_sample_retriever = other.uses_custom_cube_array_environment_map_sample_retriever;
	cube_array_environment_map_sample_retriever_variation = std::move(other.cube_array_environment_map_sample_retriever_variation);

	reflection_mapper_variation = std::move(other.reflection_mapper_variation);

	viewer_location = std::move(other.viewer_location);
	viewer_rotation = std::move(other.viewer_rotation);
	default_diffuse_color = std::move(other.default_diffuse_color);
	default_specular_color = std::move(other.default_specular_color);
	default_specular_exponent = other.default_specular_exponent;
	default_emission_color = std::move(other.default_emission_color);

	light_texture_sampler_ref_code = std::move(other.light_texture_sampler_ref_code);


	return *this;
}

AbstractRenderableObjectLightEx::~AbstractRenderableObjectLightEx()
{

}

void AbstractRenderableObjectLightEx::applyViewerTransform(const AbstractProjectingDevice& projecting_device)
{
	mat4 viewer_transform = projecting_device.getViewTransform();

	mat3 vt_rotation_part{ viewer_transform[0][0], viewer_transform[1][0], viewer_transform[2][0],
		viewer_transform[0][1], viewer_transform[1][1], viewer_transform[2][1],
		viewer_transform[0][2], viewer_transform[1][2], viewer_transform[2][2] };
	viewer_location = projecting_device.getLocation();

	mat4 ot = getObjectTransform();
	mat3 ot_rotation_part{ ot[0][0], ot[1][0], ot[2][0],
		ot[0][1], ot[1][1], ot[2][1],
		ot[0][2], ot[1][2], ot[2][2] };
	viewer_rotation = vt_rotation_part*ot_rotation_part;

	std::for_each(ShaderProgramListBegin(), ShaderProgramListEnd(),
		[this](ShaderProgram& shader_program) -> void
	{
		if (!shader_program.containsStage(ShaderType::COMPUTE_SHADER))
		{
			shader_program.assignUniformVector("v3ViewerLocation", viewer_location);
			shader_program.assignUniformMatrix("m3ViewerRotation", viewer_rotation);
		}
	});
}

bool AbstractRenderableObjectLightEx::injectExtension(const ShaderProgramReferenceCode& program_ref_code, std::initializer_list<PipelineStage> pipeline_stages)
{
	//Check if the program being created contains compute shader stage. If so, no extension should be inserted.
	if (std::find(pipeline_stages.begin(), pipeline_stages.end(), PipelineStage::COMPUTE_SHADER) != pipeline_stages.end())
		return true;

	ShaderProgram* p_shader_program = retrieveShaderProgram(program_ref_code);

	//Determine the last vertex processing stage declared for the program
	PipelineStage last_vertex_processing_stage;

	//Auxiliary value that helps to determine which stage is the last in the vertex processing part of the pipeline.
	//The following values are possible upon completion of the tests:
	//1	— vertex shader
	//2 — tessellation control program
	//3 — tessellation evaluation program
	//4 — geometry shader
	char aux = 0;

	std::array<std::pair<PipelineStage, char>, 4> vertex_processing_stages =
	{ std::make_pair(PipelineStage::GEOMETRY_SHADER, 4), std::make_pair(PipelineStage::TESS_EVAL_SHADER, 3),
	std::make_pair(PipelineStage::TESS_CONTROL_SHADER, 2), std::make_pair(PipelineStage::VERTEX_SHADER, 1) };

	for (std::pair<PipelineStage, char> stage : vertex_processing_stages)
	{
		if (std::find(pipeline_stages.begin(), pipeline_stages.end(), stage.first) != pipeline_stages.end())
			aux = aux < stage.second ? stage.second : aux;
	}

	switch (aux)
	{
	case 1:
		last_vertex_processing_stage = PipelineStage::VERTEX_SHADER;
		break;

	case 2:
		last_vertex_processing_stage = PipelineStage::TESS_CONTROL_SHADER;
		break;

	case 3:
		last_vertex_processing_stage = PipelineStage::TESS_EVAL_SHADER;
		break;

	case 4:
		last_vertex_processing_stage = PipelineStage::GEOMETRY_SHADER;
		break;

	default:
		if (!p_shader_program->isSeparate())
			return false;
	}


	/*Shader light_model_vertex_shader{ ShaderProgram::getShaderBaseCatalog() + "LightModel.vp.glsl",
		last_vertex_processing_stage, "light_model_vertex_program" };
	Shader light_model_fragment_shader{ ShaderProgram::getShaderBaseCatalog() + "LightModel.fp.glsl",
		ShaderType::FRAGMENT_SHADER, "light_model_fragment_program" };*/

	std::map<PipelineStage, std::string> sources =
	{ std::make_pair(PipelineStage::VERTEX_SHADER, std::string("")), std::make_pair(PipelineStage::TESS_CONTROL_SHADER, std::string("")),
	std::make_pair(PipelineStage::TESS_EVAL_SHADER, std::string("")), std::make_pair(PipelineStage::GEOMETRY_SHADER, std::string("")),
	std::make_pair(PipelineStage::FRAGMENT_SHADER, std::string("")), std::make_pair(PipelineStage::COMPUTE_SHADER, std::string("")) };

	if (aux)
	{
		std::string error_message;
		std::pair<bool, std::string> vertex_program_parse_result = Shader::parseShaderSource(ShaderProgram::getShaderBaseCatalog() + "LightModel.vp.glsl", &error_message);
		if (!vertex_program_parse_result.first) 
			return false;

		sources[last_vertex_processing_stage] += vertex_program_parse_result.second + "\n";
	}

	if (std::find(pipeline_stages.begin(), pipeline_stages.end(), PipelineStage::FRAGMENT_SHADER) != pipeline_stages.end())
	{
		std::string error_message;
		std::pair<bool, std::string> fragment_program_parse_result = Shader::parseShaderSource(ShaderProgram::getShaderBaseCatalog() + "LightModel.fp.glsl", &error_message);
		if (!fragment_program_parse_result.first)
			return false;

		sources[PipelineStage::FRAGMENT_SHADER] += fragment_program_parse_result.second + "\n";
	}
	

	for (const std::pair<PipelineStage, std::string>& elem : auxiliary_glsl_sources)
		sources[elem.first] += elem.second;

	for (const std::pair<PipelineStage, std::string>& elem : sources)
	{
		if (!elem.second.length()) continue;
		Shader shader{ GLSLSourceCode{ elem.second.c_str(), elem.second.length() }, elem.first, "light_extension_shader_#" + std::to_string(static_cast<uint32_t>(elem.first)) };
		if (!shader) return false;
		if (!p_shader_program->addShader(shader)) return false;
	}

	//Add reference code of the shader program to the list of modified programs
	modified_program_ref_code_list.push_back(program_ref_code);

	return true;
}

void AbstractRenderableObjectLightEx::applyExtension()
{
	//Bind reflection textures
	bindTexture(array_normal_map_ref_code);
	bindTexture(normal_map_ref_code);
	bindTexture(array_specular_map_ref_code);
	bindTexture(specular_map_ref_code);
	bindTexture(array_emission_map_ref_code);
	bindTexture(emission_map_ref_code);
	bindTexture(array_environment_map_ref_code);
	bindTexture(environment_map_ref_code);
	bindTexture(cube_array_environment_map_ref_code);
	bindTexture(cube_environment_map_ref_code);


	std::for_each(modified_program_ref_code_list.begin(), modified_program_ref_code_list.end(),
		[this](const ShaderProgramReferenceCode& shader_program_ref_code) -> void
	{
		//Retrieve a pointer for the target shader program
		ShaderProgram* p_shader_program = retrieveShaderProgram(shader_program_ref_code);

		//Assign lighting uniforms
		mat4 LightTransform = getObjectTransform().inverse();
		p_shader_program->assignUniformScalar("bLightingEnabled", static_cast<unsigned int>(p_lighting_conditions != nullptr));
		p_shader_program->assignUniformMatrix("m4LightTransform", LightTransform);

		//Update lighting conditions
		if (p_lighting_conditions)
		{
			p_lighting_conditions->getLightBufferPtr()->bind();
			p_shader_program->assignUniformBlockToBuffer("LightBuffer", 1);
			p_lighting_conditions->updateLightBuffer();
		}

		//Apply light maps
		p_shader_program->assignUniformScalar("bHasNormalMap", static_cast<GLint>(has_normal_map));
		p_shader_program->assignUniformScalar("bNormalMapEnabled", static_cast<GLint>(enable_normal_map));
		p_shader_program->assignSubroutineUniform("funcNormalMap", PipelineStage::FRAGMENT_SHADER, normal_map_sample_retriever_variation);

		p_shader_program->assignUniformScalar("bHasSpecularMap", static_cast<GLint>(has_specular_map));
		p_shader_program->assignUniformScalar("bSpecularMapEnabled", static_cast<GLint>(enable_specular_map));
		p_shader_program->assignSubroutineUniform("funcSpecularMap", PipelineStage::FRAGMENT_SHADER, specular_map_sample_retriever_variation);

		p_shader_program->assignUniformScalar("bHasEmissionMap", static_cast<GLint>(has_emission_map));
		p_shader_program->assignUniformScalar("bEmissionMapEnabled", static_cast<GLint>(enable_emission_map));
		p_shader_program->assignSubroutineUniform("funcEmissionMap", PipelineStage::FRAGMENT_SHADER, emission_map_sample_retriever_variation);

		p_shader_program->assignUniformScalar("bHasEnvironmentMap", static_cast<GLint>(has_environment_map));
		p_shader_program->assignUniformScalar("bEnvironmentMapEnabled", static_cast<GLint>(enable_environment_map));
		p_shader_program->assignUniformScalar("uiEnvironmentMapType", static_cast<GLuint>(environment_map_type));

		p_shader_program->assignUniformScalar("bSupportsArrayNormalMaps", static_cast<GLint>(supports_array_normal_maps));
		p_shader_program->assignUniformScalar("bSupportsArraySpecularMaps", static_cast<GLint>(supports_array_specular_maps));
		p_shader_program->assignUniformScalar("bSupportsArrayEmissionMaps", static_cast<GLint>(supports_array_emission_maps));
		p_shader_program->assignUniformScalar("bSupportsArrayEnvironmentMaps", static_cast<GLint>(supports_array_environment_maps));


		p_shader_program->assignUniformVector("v4DefaultDiffuseColor", default_diffuse_color);
		p_shader_program->assignUniformVector("v3DefaultSpecularColor", default_specular_color);
		p_shader_program->assignUniformScalar("fDefaultSpecularExponent", default_specular_exponent);
		p_shader_program->assignUniformVector("v3DefaultEmissionColor", default_emission_color);

		//Assign texture unit identifiers to the sampler objects in the light shader
		p_shader_program->assignUniformScalar("s2dNormalMap", getBindingUnit(normal_map_ref_code));
		p_shader_program->assignUniformScalar("s2daNormalArrayMap", getBindingUnit(array_normal_map_ref_code));

		p_shader_program->assignUniformScalar("s2dSpecularMap", getBindingUnit(specular_map_ref_code));
		p_shader_program->assignUniformScalar("s2daSpecularArrayMap", getBindingUnit(array_specular_map_ref_code));

		p_shader_program->assignUniformScalar("s2dEmissionMap", getBindingUnit(emission_map_ref_code));
		p_shader_program->assignUniformScalar("s2daEmissionArrayMap", getBindingUnit(array_emission_map_ref_code));

		p_shader_program->assignUniformScalar("s2dEnvironmentMap", getBindingUnit(environment_map_ref_code));
		p_shader_program->assignUniformScalar("s2daEnvironmentArrayMap", getBindingUnit(array_environment_map_ref_code));
		p_shader_program->assignUniformScalar("scEnvironmentMap", getBindingUnit(cube_environment_map_ref_code));
		p_shader_program->assignUniformScalar("scaEnvironmentArrayMap", getBindingUnit(cube_array_environment_map_ref_code));
		

		p_shader_program->assignSubroutineUniform("funcEnvironmentMap", PipelineStage::FRAGMENT_SHADER, environment_map_sample_retriever_variation);
		p_shader_program->assignSubroutineUniform("funcArrayEnvironmentMap", PipelineStage::FRAGMENT_SHADER, array_environment_map_sample_retriever_variation);
		p_shader_program->assignSubroutineUniform("funcCubeEnvironmentMap", PipelineStage::FRAGMENT_SHADER, cube_environment_map_sample_retriever_variation);
		p_shader_program->assignSubroutineUniform("funcCubeArrayEnvironmentMap", PipelineStage::FRAGMENT_SHADER, cube_array_environment_map_sample_retriever_variation);
		p_shader_program->assignSubroutineUniform("funcReflection", PipelineStage::FRAGMENT_SHADER, reflection_mapper_variation);
	});
}

void AbstractRenderableObjectLightEx::releaseExtension()
{

}

bool AbstractRenderableObjectLightEx::doesHaveNormalMap() const { return has_normal_map; }

bool AbstractRenderableObjectLightEx::doesSupportArrayNormalMaps() const { return supports_array_normal_maps; }

void AbstractRenderableObjectLightEx::applyNormalMapSourceTexture(const ImmutableTexture2D& normal_map_source_texture)
{
	has_normal_map = true;
	supports_array_normal_maps = normal_map_source_texture.isArrayTexture();
	//normal_map_sample_retriever_variation = "DefaultNormalMapSampleRetriever";

	if (supports_array_normal_maps)
		updateTexture(array_normal_map_ref_code, normal_map_source_texture, light_texture_sampler_ref_code);
	else
		updateTexture(normal_map_ref_code, normal_map_source_texture, light_texture_sampler_ref_code);
}

void AbstractRenderableObjectLightEx::setNormalMapEnableState(bool enable_state)
{
	enable_normal_map = enable_state;
}

bool AbstractRenderableObjectLightEx::getNormalMapEnableState() const { return enable_normal_map; }



bool AbstractRenderableObjectLightEx::doesHaveSpecularMap() const { return has_specular_map; }

bool AbstractRenderableObjectLightEx::doesSupportArraySpecularMaps() const { return supports_array_specular_maps; }

void AbstractRenderableObjectLightEx::applySpecularMapSourceTexture(const ImmutableTexture2D& specular_map_source_texture)
{
	has_specular_map = true;
	supports_array_specular_maps = specular_map_source_texture.isArrayTexture();
	//specular_map_sample_retriever_variation = "DefaultSpecularMapSampleRetriever";

	if (supports_array_specular_maps)
		updateTexture(array_specular_map_ref_code, specular_map_source_texture, light_texture_sampler_ref_code);
	else
		updateTexture(specular_map_ref_code, specular_map_source_texture, light_texture_sampler_ref_code);
}

void AbstractRenderableObjectLightEx::setSpecularMapEnableState(bool enable_state)
{
	enable_specular_map = enable_state;
}

bool AbstractRenderableObjectLightEx::getSpecularMapEnableState() const { return enable_specular_map; }



bool AbstractRenderableObjectLightEx::doesHaveEmissionMap() const { return has_emission_map; }

bool AbstractRenderableObjectLightEx::doesSupportArrayEmissionMaps() const { return supports_array_emission_maps; }

void AbstractRenderableObjectLightEx::applyEmissionMapSourceTexture(const ImmutableTexture2D& emission_map_source_texture)
{
	has_emission_map = true;
	supports_array_emission_maps = emission_map_source_texture.isArrayTexture();
	//emission_map_sample_retriever_variation = "DefaultEmissionMapSampleRetriever";

	if (supports_array_emission_maps)
		updateTexture(array_emission_map_ref_code, emission_map_source_texture, light_texture_sampler_ref_code);
	else
		updateTexture(emission_map_ref_code, emission_map_source_texture, light_texture_sampler_ref_code);
}

void AbstractRenderableObjectLightEx::setEmissionMapEnableState(bool enable_state)
{
	enable_emission_map = true;
}

bool AbstractRenderableObjectLightEx::getEmissionMapEnableState() const { return enable_emission_map; }



bool AbstractRenderableObjectLightEx::doesHaveEnvironmentMap() const { return enable_environment_map; }

bool AbstractRenderableObjectLightEx::doesSupportArrayEnvironmentMaps() const { return supports_array_environment_maps; }

void AbstractRenderableObjectLightEx::applyEnvironmentMap(const SphericalEnvironmentMap& environment_map)
{
	ImmutableTexture2D environment_map_texture = static_cast<ImmutableTexture2D>(environment_map);
	has_environment_map = true;
	supports_array_environment_maps = environment_map_texture.isArrayTexture();
	environment_map_type = EnvironmentMapType::Spherical;

	if (supports_array_environment_maps)
		updateTexture(array_environment_map_ref_code, environment_map_texture, light_texture_sampler_ref_code);
	else
		updateTexture(environment_map_ref_code, environment_map_texture, light_texture_sampler_ref_code);


	if (supports_array_environment_maps && !uses_custom_array_environment_map_sample_retriever)
		array_environment_map_sample_retriever_variation = "SphericalArrayEnvironmentMapSampleRetriever";

	if (!supports_array_environment_maps && !uses_custom_environment_map_sample_retriever)
		environment_map_sample_retriever_variation = "SphericalEnvironmentMapSampleRetriever";
}

void AbstractRenderableObjectLightEx::applyEnvironmentMap(const EquirectangularEnvironmentMap& environment_map)
{
	ImmutableTexture2D environment_map_texture = static_cast<ImmutableTexture2D>(environment_map);
	has_environment_map = true;
	supports_array_environment_maps = environment_map_texture.isArrayTexture();
	environment_map_type = EnvironmentMapType::Equirectangular;

	if (supports_array_environment_maps)
		updateTexture(array_environment_map_ref_code, environment_map_texture, light_texture_sampler_ref_code);
	else
		updateTexture(environment_map_ref_code, environment_map_texture, light_texture_sampler_ref_code);


	if (supports_array_environment_maps && !uses_custom_array_environment_map_sample_retriever)
		array_environment_map_sample_retriever_variation = "EquirectangularArrayEnvironmentMapSampleRetriever";

	if (!supports_array_environment_maps && !uses_custom_environment_map_sample_retriever)
		environment_map_sample_retriever_variation = "EquirectangularEnvironmentMapSampleRetriever";
}

void AbstractRenderableObjectLightEx::applyEnvironmentMap(const CubeEnvironmentMap& environment_map)
{
	ImmutableTextureCubeMap environment_map_texture = static_cast<ImmutableTextureCubeMap>(environment_map);
	has_environment_map = true;
	supports_array_environment_maps = environment_map_texture.isArrayTexture();
	environment_map_type = EnvironmentMapType::Cubic;

	if (supports_array_environment_maps)
		updateTexture(cube_array_environment_map_ref_code, environment_map_texture, light_texture_sampler_ref_code);
	else
		updateTexture(cube_environment_map_ref_code, environment_map_texture, light_texture_sampler_ref_code);
	

	if (supports_array_environment_maps && !uses_custom_cube_array_environment_map_sample_retriever)
		cube_array_environment_map_sample_retriever_variation = "CubicArrayEnvironmentMapSampleRetriever";
	
	if (!supports_array_environment_maps && !uses_custom_cube_environment_map_sample_retriever)
		cube_environment_map_sample_retriever_variation = "CubicEnvironmentMapSampleRetriever";
}

void AbstractRenderableObjectLightEx::setEnvironmentMapEnableState(bool enable_state) 
{ 
	enable_environment_map = enable_state; 
}

bool AbstractRenderableObjectLightEx::getEnvironmentMapEnableState() const{ return enable_environment_map; }

EnvironmentMapType AbstractRenderableObjectLightEx::getEnvironmentMapType() const { return environment_map_type; }

void AbstractRenderableObjectLightEx::defineProceduralNormalMap(const std::string& procedure_name)
{
	has_normal_map = true;
	normal_map_sample_retriever_variation = procedure_name;
 }

void AbstractRenderableObjectLightEx::disbandProceduralNormalMap()
{
	normal_map_sample_retriever_variation = "DefaultNormalMapSampleRetriever";
}

void AbstractRenderableObjectLightEx::defineProceduralSpecularMap(const std::string& procedure_name)
{
	has_specular_map = true;
	specular_map_sample_retriever_variation = procedure_name;
}

void AbstractRenderableObjectLightEx::disbandProceduralSpecularMap()
{
	specular_map_sample_retriever_variation = "DefaultSpecularMapSampleRetriever";
}

void AbstractRenderableObjectLightEx::defineProceduralEmissionMap(const std::string& procedure_name)
{
	has_emission_map = true;
	emission_map_sample_retriever_variation = procedure_name;
}

void AbstractRenderableObjectLightEx::disbandProceduralEmissionMap()
{
	specular_map_sample_retriever_variation = "DefaultEmissionMapSampleRetriever";
}

void AbstractRenderableObjectLightEx::defineCustomEnvironmentMapSampleRetriever(const std::string& variation_name)
{
	uses_custom_environment_map_sample_retriever = true;
	environment_map_sample_retriever_variation = variation_name;
}

void AbstractRenderableObjectLightEx::defineCustomArrayEnvironmentMapSampleRetriever(const std::string& variation_name)
{
	uses_custom_array_environment_map_sample_retriever = true;
	array_environment_map_sample_retriever_variation = variation_name;
}

void AbstractRenderableObjectLightEx::defineCustomCubeEnvironmentMapSampleRetriever(const std::string& variation_name)
{
	uses_custom_cube_environment_map_sample_retriever = true;
	cube_environment_map_sample_retriever_variation = variation_name;
}

void AbstractRenderableObjectLightEx::defineCustomCubeArrayEnvironmentMapSampleRetriever(const std::string& variation_name)
{
	uses_custom_cube_array_environment_map_sample_retriever = true;
	cube_array_environment_map_sample_retriever_variation = variation_name;
}

void AbstractRenderableObjectLightEx::defineCustomReflectionMapper(const std::string& variation_name)
{
	reflection_mapper_variation = variation_name;
}

void AbstractRenderableObjectLightEx::resetEnvironmentMapCustomVariations()
{
	uses_custom_environment_map_sample_retriever = false;
	uses_custom_array_environment_map_sample_retriever = false;
	uses_custom_cube_environment_map_sample_retriever = false;
	uses_custom_cube_array_environment_map_sample_retriever = false;

	if (has_environment_map)
	{
		switch (environment_map_type)
		{
		case EnvironmentMapType::Spherical:
			if (supports_array_environment_maps) array_environment_map_sample_retriever_variation = "SphericalArrayEnvironmentMapSampleRetriever";
			else environment_map_sample_retriever_variation = "SphericalEnvironmentMapSampleRetriever";
			break;

		case EnvironmentMapType::Cubic:
			if (supports_array_environment_maps) cube_array_environment_map_sample_retriever_variation = "CubicArrayEnvironmentMapSampleRetriever";
			else cube_environment_map_sample_retriever_variation = "CubicEnvironmentMapSampleRetriever";
			break;

		case EnvironmentMapType::Equirectangular:
			if (supports_array_environment_maps) array_environment_map_sample_retriever_variation = "EquirectangularArrayEnvironmentMapSampleRetriever";
			else environment_map_sample_retriever_variation = "EquirectangularEnvironmentMapSampleRetriever";
			break;
		}
	}
}

void AbstractRenderableObjectLightEx::resetReflectionMapper()
{
	reflection_mapper_variation = "DefaultReflectionMapper";
}


vec4 AbstractRenderableObjectLightEx::getDiffuseColor() const { return default_diffuse_color; }

void AbstractRenderableObjectLightEx::setDiffuseColor(const vec4& new_diffuse_color)
{
	default_diffuse_color = new_diffuse_color;
}

vec3 AbstractRenderableObjectLightEx::getSpecularColor() const { return default_specular_color; }

void AbstractRenderableObjectLightEx::setSpecularColor(const vec3& new_specular_color)
{
	default_specular_color = new_specular_color;
}

float AbstractRenderableObjectLightEx::getSpecularExponent() const { return default_specular_exponent; }

void AbstractRenderableObjectLightEx::setSpecularExponent(float new_specular_exponent)
{
	default_specular_exponent = new_specular_exponent;
}

vec3 AbstractRenderableObjectLightEx::getEmissionColor() const { return default_emission_color; }

void AbstractRenderableObjectLightEx::setEmissionColor(const vec3& new_emission_color)
{
	default_emission_color = new_emission_color;
}

void AbstractRenderableObjectLightEx::applyLightingConditions(const LightingConditions& lighting_conditions_descriptor)
{
	p_lighting_conditions = &lighting_conditions_descriptor;
}

const LightingConditions* AbstractRenderableObjectLightEx::retrieveLightingConditionsPointer() const { return p_lighting_conditions; }