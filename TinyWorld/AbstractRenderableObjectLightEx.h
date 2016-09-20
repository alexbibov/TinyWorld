//Implements lighting extension of AbstractRenderableObject class

#ifndef TW__ABSTRACT_RENDERABLE_OBJECT_LIGHT_EX__

#include <map>

#include "AbstractRenderableObject.h"
#include "LightingConditions.h"
#include "EnvironmentMap.h"

namespace tiny_world
{
	//Describes an abstract drawable object that is lit by means of the Phong lighting model
	//The shader part of the extension uses the following color outputs:
	//COLOR2	screen-space normal map
	//COLOR3	screen-space linear depth map
	//COLOR4	screen-space diffuse color scaled by the isotropic ambient light intensity
	class AbstractRenderableObjectLightEx : virtual public AbstractRenderableObjectTextured
	{
	private:
		std::map<PipelineStage, std::string> auxiliary_glsl_sources;	//raw GLSL codes implementing auxiliary features required by the object

		const LightingConditions*	p_lighting_conditions;	//pointer to the lighting conditions configuration descriptor

		TextureSamplerReferenceCode light_texture_sampler_ref_code;	//sampler object used to sample values from normal, specular, emission and environment maps

		bool has_normal_map;		//equals 'true' if the object uses bump-mapping
		bool supports_array_normal_maps;	//equals 'true' if the object supports array bump-maps
		bool enable_normal_map;		//equals 'true' if normal mapping is enabled for the object
		std::string normal_map_sample_retriever_variation;	//name of the shader procedure variation used to retrieve values from normal maps
		TextureReferenceCode normal_map_ref_code;		//reference code of the normal map
		TextureReferenceCode array_normal_map_ref_code;	//reference code used by array normal maps

		bool has_specular_map;		//equals 'true' if the object has a specular map
		bool supports_array_specular_maps;		//equals 'true' if the object supports array specular maps
		bool enable_specular_map;	//equals 'true' if specular map is enabled for the object
		std::string specular_map_sample_retriever_variation;	//name of the shader procedure variation used to retrieve values from specular maps
		TextureReferenceCode specular_map_ref_code;		//reference code of the specular map
		TextureReferenceCode array_specular_map_ref_code;	//reference code used by array specular maps

		bool has_emission_map;		//equals 'true' if the object supports emission mapping
		bool supports_array_emission_maps;	//equals 'true' if the object supports array emission maps
		bool enable_emission_map;	//equals 'true' if emission map is enabled for the object
		std::string emission_map_sample_retriever_variation;	//name of the shader procedure variation used to retrieve values from emission maps
		TextureReferenceCode emission_map_ref_code;		//reference code of the emission map
		TextureReferenceCode array_emission_map_ref_code;	//reference code used by array emission maps

		bool has_environment_map;	//equals 'true' if the object has an environment map
		bool supports_array_environment_maps;	//equals 'true' if the object uses array environment maps
		bool enable_environment_map;	//equals 'true' if environment mapping is enabled
		EnvironmentMapType environment_map_type;	//type of the environment map, which currently is in use
		TextureReferenceCode environment_map_ref_code;	//reference code used by simple 2D environment maps
		TextureReferenceCode array_environment_map_ref_code;	//reference code used by environment maps based on 2D array textures
		TextureReferenceCode cube_environment_map_ref_code;	//reference code used by environment maps based on cubemap textures
		TextureReferenceCode cube_array_environment_map_ref_code;	//reference code used by environment maps based on cubemap array textures


		bool uses_custom_environment_map_sample_retriever;	//equals 'true' if the object defines a custom sample retriever for non-array environment maps based on 2D textures
		std::string environment_map_sample_retriever_variation;	//name of a subroutine variation, which will be used to extract samples from non-array environment maps based on 2D textures

		bool uses_custom_array_environment_map_sample_retriever;	//equals 'true' if the object defines a custom sample retriever subroutine to be used with array environment maps based on 2D array textures
		std::string array_environment_map_sample_retriever_variation;	//name of a subroutine variation, which will be used with array environment maps based on 2D array textures

		bool uses_custom_cube_environment_map_sample_retriever;	//equals 'true' if object defines a custom variation to extract samples from environment maps based on cube textures
		std::string cube_environment_map_sample_retriever_variation;	//name of a sample retriever variation, which will be used to obtain samples from environment maps based on cube textures

		bool uses_custom_cube_array_environment_map_sample_retriever;	//equals 'true' if the object defines a custom variation to sample data from array environment maps based on cubemap array textures
		std::string cube_array_environment_map_sample_retriever_variation;	//name of a sample retriever subroutine, which will be used to retrieve data from array environment maps based on cubemap array textures

		std::string reflection_mapper_variation;	//name of a subroutine variation, which will be used to compute reflection vectors


		vec3 viewer_location;	//location of the viewer in global space
		mat3 viewer_rotation;	//rotation part of the viewer transform (i.e. the transform that converts the global space into the viewer space)

		vec4 default_diffuse_color;		//default diffuse color
		vec3 default_specular_color;	//default specular color
		float default_specular_exponent;	//default specular exponent
		vec3 default_emission_color;	//default emission color

		std::list<ShaderProgramReferenceCode> modified_program_ref_code_list;		//list of reference codes of the shader programs that have been modified by injecting lighting shaders into them


		void setup_extension();	//performs required initialization particulars

	protected:
		//Links functionality provided by the light model to a shader program maintained by the drawable object.
		//Function returns 'true' if the linkage was successful and 'false' otherwise
		bool injectExtension(const ShaderProgramReferenceCode& program_ref_code, std::initializer_list<PipelineStage> program_stages) override;

		//Informs extension about the coordinate transform that converts the world space into the viewer space
		void applyViewerTransform(const AbstractProjectingDevice& projecting_device) override;

		//This function applies properties of the lighting model to the object being rendered. Hence, it binds texture objects required by lighting computations and installs
		//corresponding samplers.
		void applyExtension() override;

		//Allows extension to release the resources it has allocated before the rendering. This function is invoked automatically immediately after the drawing commands have been executed
		//This is especially useful when extension makes use of image units, since ImageUnit object should be released as soon as possible upon completion of the rendering to vacate the 
		//resource it holds
		void releaseExtension() override;

		AbstractRenderableObjectLightEx();		//Default initialization of a lit object

		//Initializes lighting extension and equips it with additional shader sources. The list must be filled by pairs of the target shader stages and the path strings describing location of the source files containing GLSL code to be parsed.
		//Note that the parsed sources are simply concatenated together and added to the source code of the extension, so avoid repetitive declarations
		explicit AbstractRenderableObjectLightEx(std::initializer_list<std::pair<PipelineStage, std::string>> auxiliary_shader_sources);	

		AbstractRenderableObjectLightEx(const AbstractRenderableObjectLightEx& other);	//Copy construction of a lit object
		AbstractRenderableObjectLightEx(AbstractRenderableObjectLightEx&& other);	//Move constructor of a lit object
		AbstractRenderableObjectLightEx& operator=(const AbstractRenderableObjectLightEx& other);	//Copy-assignment of lit objects
		AbstractRenderableObjectLightEx& operator=(AbstractRenderableObjectLightEx&& other);	//Move-assignment of lit objects

		void defineProceduralNormalMap(const std::string& procedure_name);	//defines custom procedure used to obtain per-pixel surface normals
		void disbandProceduralNormalMap();	//turns off procedural normal mapping

		void defineProceduralSpecularMap(const std::string& procedure_name);	//defines custom procedure used to obtain per-pixel surface specular modulation coefficients
		void disbandProceduralSpecularMap();	//turns off procedural specular mapping

		void defineProceduralEmissionMap(const std::string& procedure_name);	//defines custom procedure used to obtain per-pixel surface emission modulation coefficients
		void disbandProceduralEmissionMap();	//turns off procedural emission mapping

		void defineCustomEnvironmentMapSampleRetriever(const std::string& variation_name);	//defines custom sample retriever, which will be used with non-array environment maps based on 2D textures
		void defineCustomArrayEnvironmentMapSampleRetriever(const std::string& variation_name);	//defines custom sample retriever, which will be used to extract data from array environment maps based on 2D array textures
		void defineCustomCubeEnvironmentMapSampleRetriever(const std::string& variation_name);	//defines custom sample retriever, which will be used to sample data from environment maps based on cubemap textures
		void defineCustomCubeArrayEnvironmentMapSampleRetriever(const std::string& variation_name);	//defines custom sample retriever, which will be used to extract data from array environment maps based on cubemap array textures
		void defineCustomReflectionMapper(const std::string& variation_name);	//defines a custom subroutine, which will be used to compute reflection vectors given incident vectors
		void resetEnvironmentMapCustomVariations();	//forces usage of default subroutines to sample data from environment maps of all types
		void resetReflectionMapper();	//forces usage of default procedure when computing reflection vectors



	public:
		virtual ~AbstractRenderableObjectLightEx();

		bool doesHaveNormalMap() const;		//returns "true" if the object supports bump-mapping
		bool doesSupportArrayNormalMaps() const;	//returns true if the object can use array textures as the source for bump mapping
		void applyNormalMapSourceTexture(const ImmutableTexture2D& normal_map_source_texture);	//introduces a new texture to be used as a source for the normal mapping
		void setNormalMapEnableState(bool enable_state);	//sets enable state for normal mapping
		bool getNormalMapEnableState() const;	//retrieves enable state of normal mapping

		bool doesHaveSpecularMap() const;	//returns "true" if the object has a specular map
		bool doesSupportArraySpecularMaps() const;	//returns "true" if the object can use array textures as the source of its specular maps
		void applySpecularMapSourceTexture(const ImmutableTexture2D& specular_map_source_texture);	//introduces a new texture to be used as the source of object's specular maps
		void setSpecularMapEnableState(bool enable_state);	//sets enable state for specular mapping
		bool getSpecularMapEnableState() const;	//retrieves enable state of specular mapping

		bool doesHaveEmissionMap() const;	//returns "true" if the object has an emission map
		bool doesSupportArrayEmissionMaps() const;	//returns "true" if the object can use a array textures as the source of of its emission maps
		void applyEmissionMapSourceTexture(const ImmutableTexture2D& emission_map_source_texture);	//introduces new texture to be used as the source of object's emission maps
		void setEmissionMapEnableState(bool enable_state);	//sets enable state for emission mapping
		bool getEmissionMapEnableState() const;	//retrieves enable state of emission mapping

		bool doesHaveEnvironmentMap() const;	//returns "true" if the object has an environment map
		bool doesSupportArrayEnvironmentMaps() const; //returns "true" if the object uses an array texture to retrieve environmental data
		void applyEnvironmentMap(const SphericalEnvironmentMap& environment_map);	//applies hemispherical environment map to the object. This setting overrides all existing environmental data.
		void applyEnvironmentMap(const EquirectangularEnvironmentMap& environment_map);	//applies equirectangular environment map to the object. This setting overrides all existing environmental data.
		void applyEnvironmentMap(const CubeEnvironmentMap& environment_map);	//applies cube environment map to the object. This settings overrides all existing environmental data.
		void setEnvironmentMapEnableState(bool enable_state);	//sets enable state for environmental mapping
		bool getEnvironmentMapEnableState() const;	//retrieves enable state of environmental mapping
		EnvironmentMapType getEnvironmentMapType() const;	//returns type of the environmental data that is currently used by the object

		vec4 getDiffuseColor() const;	//returns diffuse color of the object
		void setDiffuseColor(const vec4& new_diffuse_color);	//sets diffuse color of the object

		vec3 getSpecularColor() const;	//returns specular color of the object
		void setSpecularColor(const vec3& new_specular_color);	//sets specular color for the object
		float getSpecularExponent() const;	//returns value of the specular exponent used in the lighting equation
		void setSpecularExponent(float new_specular_exponent);	//sets new value of the specular exponent

		vec3 getEmissionColor() const;	//returns emission color of the object
		void setEmissionColor(const vec3& new_emission_color);	//sets new emission color for the object

		//Registers lighting conditions descriptor to the drawable object. The settings provided by the descriptor
		//will be used to light the object. Note however, that drawable object does not own lighting conditions descriptor,
		//which means that if lighting settings change, it is immediately reflected in the lighting computations
		//of all objects that share the corresponding lighting conditions descriptor. If no lighting conditions
		//descriptor has been provided, the lighting computations are performed as if there was no light sources
		void applyLightingConditions(const LightingConditions& lighting_conditions_descriptor);

		//Retrieves a pointer for the lighting conditions object used by the lit entity
		const LightingConditions* retrieveLightingConditionsPointer() const;
	};
}


#define TW__ABSTRACT_RENDERABLE_OBJECT_LIGHT_EX__
#endif