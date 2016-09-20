//Implements the concept of lighting conditions. Here thew term "lighting conditions" refer to the set of properties needed to compute lighting effects,
//such as interaction between light and an object's surface or between light and atmosphere

#ifndef TW__LIGHTING_CONDITIONS__


#include <list>

#include "Light.h"
#include "std140UniformBuffer.h"
#include "Skydome.h"

namespace tiny_world
{
	//Encapsulates set of properties that are used in order to light the scene. This is the core part of the object that
	//contains implementation of the basic functionality and is not intended for explicit usage. Use LightingConditions object instead.
	class LightingConditions_Core : public Entity
	{
	private:
		//Maximal number of directional light sources
		static const unsigned int max_directional_lights = 100;

		//Maximal number of point light sources
		static const unsigned int max_point_lights = 100;

		//Maximal number of spot light sources
		static const unsigned int max_spot_lights = 100;

		//Pointer to the ambient light source registered to the context
		const AmbientLight* p_ambient_light;

		//List of directional light sources
		std::list<const DirectionalLight*> directional_lights;

		//List of point light sources
		std::list<const PointLight*> point_lights;

		//List of spot light sources
		std::list<const SpotLight*> spot_lights;


		//**************The following parameters are dedicated to atmospheric effects caused by interactions of light and atmosphere***************

		//Additional color modulation applied to the atmospheric fog. This parameters may be handy to create certain special effects, like a fog that
		//appears reddish or a white mist that covers the sky dome.
		vec3 v3AtmosphericFogColor;

		//Global density of the fog
		float atmospheric_fog_global_density;

		//Height fall-off parameter of the fog
		float atmospheric_fog_height_fall_off;

		//Parameter of the phase function used to simulate Mie scattering in the atmospheric fog.
		float atmospheric_fog_mie_phase_function_param;

		//Global light haze attenuation factor
		float light_haze_attenuation_factor;

		//************************************************************************************************************************************


	protected:
		//Uniform buffer object complying with STD140 layout rules. This buffer contains data related to the light sources
		mutable std140UniformBuffer light_buffer;

		//Uniform buffer object complying with STD140 layout rules. This buffer contains parameters that
		//describe appearance of the atmospheric fog
		mutable std140UniformBuffer fog_buffer;

		//Skydome object used to compute correct color of the atmospheric fog
		const Skydome* p_skydome;


		//Default initializer
		LightingConditions_Core();

		//Copy constructor
		LightingConditions_Core(const LightingConditions_Core& other) = default;

		//Move constructor (not defaulted for compatibility with Microsoft compilers)
		LightingConditions_Core(LightingConditions_Core&& other);

		//Copy assignment operator
		LightingConditions_Core& operator=(const LightingConditions_Core& other) = default;

		//Move assignment operator (not defaulted for compatibility with Microsoft compilers)
		LightingConditions_Core& operator=(LightingConditions_Core&& other);

		//Standard destructor (user-defined to make it virtual)
		virtual ~LightingConditions_Core();


		//Updates current configuration of the light sources so that the latest changes become visible to the shaders.
		//This function must be called on each frame update if light sources are updated dynamically
		void updateLightBuffer() const;

		//Updates current configuration of the atmospheric fog so that the last changes become visible to 
		//the corresponding shaders. This function must be called on each frame if fog parameters are dynamically updated.
		//An update in appearance of the sky dome does not induce necessity to call this function
		void updateFogBuffer() const;

		//Returns pair of ImmutableTexture2D objects containing in-scattering values used by implementation of the atmospheric fog.
		//The first element of the pair contains in-scattering contributed by sun, and the second element contains in-scattering contributed by moon
		std::pair<ImmutableTexture2D, ImmutableTexture2D> retrieveInScatteringTextures() const;

	public:
		//Registers new light source to the lighting conditions context and returns 'true' if the light has been successfully
		//added or 'false' otherwise. Note, that lighting conditions can only include single ambient light source and
		//no light source can be added to the context repeatedly. Doing so will produce no effect and the function will return
		//'false' meaning that registration of the new light source has failed
		bool addLight(const AbstractLight& light);

		//Removes light from the list of light sources considered by the lighting conditions based on provided
		//strong identifier. Returns 'true' on successful removal and 'false' otherwise
		bool removeLight(unsigned int light_id);

		//Removes light from the list of light sources considered by the lighting conditions based on provided
		//string name. If there are several light sources having the same string name, which was the subject for
		//removal, then only one light source will be actually removed. The choice of which light among the candidates 
		//should be removed from the list is based on the search order that is organized as follows (smaller entries in the table
		//below mean that the corresponding entries are examined by the search earlier):
		//1) Ambient light
		//2) Directional lights in the order of addition
		//3) Point lights in the order of addition
		//4) Spot lights in the order of addition
		//This means, that if for example one needs to remove a light source with string name "MyLight" and there are two directional
		//lights and one spot light having this name, then the directional light that has been added the first will actually be removed.
		bool removeLight(const std::string& light_string_name);



		//Returns pointer for the ambient light source registered to the lighting conditions. If the lighting conditions do not incorporate an ambient light source the function returns nullptr
		const AmbientLight* getAmbientLightSource() const;


		//Returns list of pointers for directional light sources registered to the lighting conditions. If the lighting conditions do not have directional light sources the function returns an empty list
		std::list<const DirectionalLight*> getDirectionalLightSources() const;


		//Returns list of pointers for point light sources registered to the lighting conditions. If the lighting conditions do not have point light sources the function returns an empty list
		std::list<const PointLight*> getPointLightSources() const;


		//Returns list of pointers for spot light sources registered to the lighting conditions. If the lighting conditions do not have spot light sources the function returns an empty list
		std::list<const SpotLight*> getSpotLightSources() const;
		


		//Sets additional color modulation to be applied to the atmospheric fog. This feature can be used to produce certain special effects like
		//a fog that looks reddish or a white dense mist that covers the world. The default value of this parameter is (1, 1, 1)  (i.e. the modulation has no effect)
		void setAtmosphericFogColor(const vec3& fog_color);

		//Returns additional color modulation currently applied to the scattered color of the atmospheric fog
		vec3 getAtmosphericFogColor() const;

		//Defines global density of the atmospheric fog. The default value is 1. This density factor must be non-negative. 
		//If provided value is less than zero, the function has no effect. The value of zero effectively turns the fog off.
		void setAtmosphericFogGlobalDensity(float density_factor);

		//Retrieves global density of the atmospheric fog
		float getAtmosphericFogGlobalDensity() const;

		//Defines height fall-off coefficient determining how fast the density of the atmospheric fog decreases with increase of the altitude above the "sea level".
		//The default value for this parameter is 1. The height fall-off coefficient must be positive. If supplied value is negative or zero, the function has no effect.
		void setAtmosphericFogHeightFallOff(float height_fall_off_coefficient);

		//Returns height fall-off coefficient of the atmospheric fog distribution
		float getAtmosphericFogHeightFallOff() const;

		//Defines parameter, which will be used by the Henyey-Greenstein phase function to produce Mie scattering effect in the atmospheric fog.
		//Negative values mean that more light is scattered towards the observer, whereas positive values would make more light scattered back toward
		//the light source. This parameter must belong to the interval (-1, 1). If provided value lies outside the allowed range, the function has no effect.
		//The default value for this parameter is computed based on the phase function definition extracted from the Skydome object registered to the lighting
		//conditions context. If no Skydome object has been registered, the default value is -0.9
		void setAtmosphericFogMiePhaseFunctionParameter(float mie_phase_function_parameter);

		//Returns current value of the Henyey-Greenstein phase function parameter, which is used to produce Mie scattering effect in the atmospheric fog.
		float getAtmosphericFogMiePhaseFunctionParameter() const;


		//Sets global light haze attenuation factor (attenuation due to the density of liquid particles in the atmosphere). The default value of this parameter is 1.0
		void setLightHazeAttenuationFactor(float factor);

		//Returns global light haze attenuation factor
		float getLightHazeAttenuationFactor() const;


		//Assigns in-scattering textures and positions of the sky bodies to be used for computing color of the fog. After this function is called, the fog color
		//overriding set by setAtmosphericFogColor() gets ignored and, whereas the fog color becomes computed procedurally based on data
		//contained in the supplied Skydome object
		void setSkydome(const Skydome* p_skydome_object);

		//Retrieves pointer for the skydome object employed by the lighting conditions. If the lighting conditions do not take the skydome into account the function returns nullptr
		const Skydome* getSkydome() const;
	};




	//Provides interface for LightingConditions_Core. The separation of LightingConditions_Core and LightingConditions objects is needed
	//for ability to reveal the private part of LightingConditions without fully revealing the private part of LightingConditions_Core
	class LightingConditions final : public LightingConditions_Core
	{
		friend class AbstractRenderableObjectLightEx;
		friend class SSFilter_AtmosphericFog;
		friend class SSFilter_LightHaze;

	private:
		//Updates current configuration of the lighting conditions so that the latest changes become visible to the shaders.
		//This function must be called on each frame update if lightings conditions are updated dynamically
		void updateLightBuffer() const;

		//Updates current configuration of the atmospheric fog so that the last changes become visible to 
		//the corresponding shaders. This function must be called on each frame if fog parameters are dynamically updated.
		//An update in appearance of the sky dome does not induce necessity to call this function
		void updateFogBuffer() const;

		//Returns pointer to the light buffer object encapsulated by the lighting conditions context
		const std140UniformBuffer* getLightBufferPtr() const;

		//Returns pointer to the fog buffer object encapsulated by the lighting conditions context
		const std140UniformBuffer* getFogBufferPtr() const;

		//Returns pair of ImmutableTexture2D objects containing in-scattering values used by implementation of the atmospheric fog.
		//The first element of the pair contains in-scattering contributed by sun, and the second element contains in-scattering contributed by moon
		std::pair<ImmutableTexture2D, ImmutableTexture2D> retrieveInScatteringTextures() const;
	};
}

#define TW__LIGHTING_CONDITIONS__
#endif
