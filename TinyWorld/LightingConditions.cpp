#include "LightingConditions.h"

using namespace tiny_world;


LightingConditions_Core::LightingConditions_Core() : Entity("LightingConditions"),
p_ambient_light{ nullptr }, light_buffer{ 1 }, fog_buffer{ 0 }, 
v3AtmosphericFogColor{ 1.0f }, atmospheric_fog_global_density{ 1.0f },
atmospheric_fog_height_fall_off{ 1.0f }, atmospheric_fog_mie_phase_function_param{ -0.9f },
light_haze_attenuation_factor{ 1.0f }
{
	//size_t light_buffer_allocation_size = 48 + 16 * (2 * max_directional_lights + 5 * max_point_lights + 8 * max_spot_lights);	//compute size of allocation consumed by the light buffer
	size_t light_buffer_allocation_size =
		std140UniformBuffer::getMinimalStorageCapacity <
		vec3,
		uint32_t, std::vector<vec3>, std::vector<vec3>,
		uint32_t, std::vector<vec3>, std::vector<vec3>,
		std::vector<vec3>, std::vector<float>, std::vector<float>,
		uint32_t, std::vector<vec3>, std::vector<vec3>, std::vector<vec3>,
		std::vector<float>, std::vector<vec3>,
		std::vector<float>, std::vector<float>, std::vector<float >> (max_directional_lights, max_directional_lights, 
		max_point_lights, max_point_lights, max_point_lights, max_point_lights, max_point_lights,
		max_spot_lights, max_spot_lights, max_spot_lights, max_spot_lights, max_spot_lights, max_spot_lights, max_spot_lights, max_spot_lights);
	light_buffer.allocate(light_buffer_allocation_size);	//allocate memory for the light buffer


	//size_t fog_buffer_allcation_size = 52;	//compute size of allocation consumed by the fog buffer
	size_t fog_buffer_allcation_size =
		std140UniformBuffer::getMinimalStorageCapacity<vec3, float, vec3, float, vec3, float, float>();

	fog_buffer.allocate(fog_buffer_allcation_size);	//allocate memory for the fog buffer
}

LightingConditions_Core::LightingConditions_Core(LightingConditions_Core&& other) :
Entity(std::move(other)),
p_ambient_light{ other.p_ambient_light },
directional_lights(std::move(other.directional_lights)),
point_lights(std::move(other.point_lights)),
spot_lights(std::move(other.spot_lights)),
v3AtmosphericFogColor{ std::move(other.v3AtmosphericFogColor) },
atmospheric_fog_global_density{ other.atmospheric_fog_global_density },
atmospheric_fog_height_fall_off{ other.atmospheric_fog_height_fall_off },
atmospheric_fog_mie_phase_function_param{ other.atmospheric_fog_mie_phase_function_param },
light_haze_attenuation_factor{ other.light_haze_attenuation_factor },
light_buffer{ std::move(other.light_buffer) }, fog_buffer{ std::move(other.fog_buffer) },
p_skydome{ other.p_skydome }
{

}

LightingConditions_Core& LightingConditions_Core::operator=(LightingConditions_Core&& other)
{
	//Account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	Entity::operator=(std::move(other));
	p_ambient_light = other.p_ambient_light;
	directional_lights = std::move(other.directional_lights);
	point_lights = std::move(other.point_lights);
	spot_lights = std::move(other.spot_lights);
	v3AtmosphericFogColor = std::move(other.v3AtmosphericFogColor);
	atmospheric_fog_global_density = other.atmospheric_fog_global_density;
	atmospheric_fog_height_fall_off = other.atmospheric_fog_height_fall_off;
	atmospheric_fog_mie_phase_function_param = other.atmospheric_fog_mie_phase_function_param;
	light_haze_attenuation_factor = other.light_haze_attenuation_factor;
	light_buffer = std::move(other.light_buffer);
	fog_buffer = std::move(other.fog_buffer);
	p_skydome = other.p_skydome;

	return *this;
}

LightingConditions_Core::~LightingConditions_Core()
{

}


bool LightingConditions_Core::addLight(const AbstractLight& light)
{
	switch (light.getLightType())
	{
	case LightType::AMBIENT:
		if (p_ambient_light) return false;	//only one ambient light can be included into lighting conditions
		p_ambient_light = dynamic_cast<const AmbientLight*>(&light);
		return true;

	case LightType::DIRECTIONAL:
		//Check if the light being added is already in the corresponding list
		if (std::find_if(directional_lights.begin(), directional_lights.end(),
			[&light](const DirectionalLight* elem) -> bool
		{return elem->getId() == light.getId(); }) != directional_lights.end()) return false;
		else
		{
			directional_lights.push_back(dynamic_cast<const DirectionalLight*>(&light));
			return true;
		}

	case LightType::POINT:
		//Check if the light being added is already in the corresponding list
		if (std::find_if(point_lights.begin(), point_lights.end(),
			[&light](const PointLight* elem) -> bool
		{return elem->getId() == light.getId(); }) != point_lights.end()) return false;
		else
		{
			point_lights.push_back(dynamic_cast<const PointLight*>(&light));
			return true;
		}

	case LightType::SPOT:
		//Check if the light being added is already in the corresponding list
		if (std::find_if(spot_lights.begin(), spot_lights.end(),
			[&light](const SpotLight* elem) -> bool
		{return elem->getId() == light.getId(); }) != spot_lights.end()) return false;
		else
		{
			spot_lights.push_back(dynamic_cast<const SpotLight*>(&light));
			return true;
		}
	}

	return false;
}

bool LightingConditions_Core::removeLight(unsigned int light_id)
{
	//Check if the light that has to be removed is the ambient light that has been previously included into the lighting conditions
	if (p_ambient_light && p_ambient_light->getId() == light_id)
	{
		p_ambient_light = nullptr;
		return true;
	}

	//Check if the light that has to be removed is a directional light
	std::list<const DirectionalLight*>::iterator directional_light_to_remove;
	if ((directional_light_to_remove = std::find_if(directional_lights.begin(), directional_lights.end(),
		[light_id](const DirectionalLight* elem) -> bool{return elem->getId() == light_id; })) != directional_lights.end())
	{
		directional_lights.erase(directional_light_to_remove);
		return true;
	}

	//Check if the light that has to be removed is a point light
	std::list<const PointLight*>::iterator point_light_to_remove;
	if ((point_light_to_remove = std::find_if(point_lights.begin(), point_lights.end(),
		[light_id](const PointLight* elem) -> bool{return elem->getId() == light_id; })) != point_lights.end())
	{
		point_lights.erase(point_light_to_remove);
		return true;
	}

	//Check if the light that has to be removed is a spot light
	std::list<const SpotLight*>::iterator spot_light_to_remove;
	if ((spot_light_to_remove = std::find_if(spot_lights.begin(), spot_lights.end(),
		[light_id](const SpotLight* elem) -> bool{return elem->getId() == light_id; })) != spot_lights.end())
	{
		spot_lights.erase(spot_light_to_remove);
		return true;
	}

	return false;
}

bool LightingConditions_Core::removeLight(const std::string& light_string_name)
{
	//Check if the ambient light (if it has been included into the lighting conditions) has the requested string name
	if (p_ambient_light && p_ambient_light->getStringName() == light_string_name)
	{
		p_ambient_light = nullptr;
		return true;
	}

	//Check if there is a directional light that has the requested string name
	std::list<const DirectionalLight*>::iterator directional_light_to_remove;
	if ((directional_light_to_remove = std::find_if(directional_lights.begin(), directional_lights.end(),
		[&light_string_name](const DirectionalLight* elem) -> bool{return elem->getStringName() == light_string_name; })) != directional_lights.end())
	{
		directional_lights.erase(directional_light_to_remove);
		return true;
	}

	//Check if there is a point light that has the requested string name
	std::list<const PointLight*>::iterator point_light_to_remove;
	if ((point_light_to_remove = std::find_if(point_lights.begin(), point_lights.end(),
		[&light_string_name](const PointLight* elem) -> bool{return elem->getStringName() == light_string_name; })) != point_lights.end())
	{
		point_lights.erase(point_light_to_remove);
		return true;
	}

	//Check if there is a spot light that has the requested string name
	std::list<const SpotLight*>::iterator spot_light_to_remove;
	if ((spot_light_to_remove = std::find_if(spot_lights.begin(), spot_lights.end(),
		[&light_string_name](const SpotLight* elem) -> bool{return elem->getStringName() == light_string_name; })) != spot_lights.end())
	{
		spot_lights.erase(spot_light_to_remove);
		return true;
	}

	return false;
}



const AmbientLight* LightingConditions_Core::getAmbientLightSource() const
{
	return p_ambient_light;
}

std::list<const DirectionalLight*> LightingConditions_Core::getDirectionalLightSources() const
{
	return directional_lights;
}

std::list<const PointLight*> LightingConditions_Core::getPointLightSources() const
{
	return point_lights;
}

std::list<const SpotLight*> LightingConditions_Core::getSpotLightSources() const
{
	return spot_lights;
}



void LightingConditions_Core::setAtmosphericFogColor(const vec3& fog_color){ v3AtmosphericFogColor = fog_color; }

vec3 LightingConditions_Core::getAtmosphericFogColor() const { return v3AtmosphericFogColor; }


void LightingConditions_Core::setAtmosphericFogGlobalDensity(float density_factor) { atmospheric_fog_global_density = density_factor; }

float LightingConditions_Core::getAtmosphericFogGlobalDensity() const { return atmospheric_fog_global_density; }


void LightingConditions_Core::setAtmosphericFogHeightFallOff(float height_fall_off_coefficient) { atmospheric_fog_height_fall_off = height_fall_off_coefficient; }

float LightingConditions_Core::getAtmosphericFogHeightFallOff() const { return atmospheric_fog_height_fall_off; }


void LightingConditions_Core::setAtmosphericFogMiePhaseFunctionParameter(float mie_phase_function_parameter){ atmospheric_fog_mie_phase_function_param = mie_phase_function_parameter; }

float LightingConditions_Core::getAtmosphericFogMiePhaseFunctionParameter() const { return atmospheric_fog_mie_phase_function_param; }


void LightingConditions_Core::setLightHazeAttenuationFactor(float factor)
{
	light_haze_attenuation_factor = factor;
}

float LightingConditions_Core::getLightHazeAttenuationFactor() const { return light_haze_attenuation_factor; }


void LightingConditions_Core::setSkydome(const Skydome* p_skydome_object){ p_skydome = p_skydome_object; }

const Skydome* LightingConditions_Core::getSkydome() const { return p_skydome; }


void LightingConditions_Core::updateLightBuffer() const
{
	//Ensure that the light buffer is being filled from the beginning
	light_buffer.resetOffsetCounter();

	//Apply configuration of the ambient light
	if (p_ambient_light)
		light_buffer.pushVector(p_ambient_light->getColor());
	else
		light_buffer.pushVector(vec3{ 0.0f });


	//Apply configuration of the directional lights
	light_buffer.pushScalar(static_cast<unsigned int>(directional_lights.size()));
	std::vector<vec3> directional_light_directions(static_cast<std::vector<vec3>::size_type>(max_directional_lights), vec3{ 0 });
	std::vector<vec3> directional_light_intensities(static_cast<std::vector<vec3>::size_type>(max_directional_lights), vec3{ 0 });
	{
		unsigned int i = 0;
		for (const DirectionalLight* p_directional_light : directional_lights)
		{
			directional_light_directions[i] = p_directional_light->getDirection().get_normalized();
			directional_light_intensities[i] = p_directional_light->getColor();
			++i;
		}
	}
	light_buffer.pushVector(directional_light_directions);
	light_buffer.pushVector(directional_light_intensities);


	//Apply configuration of the point lights
	light_buffer.pushScalar(static_cast<unsigned int>(point_lights.size()));
	std::vector<vec3> point_light_locations(static_cast<std::vector<vec3>::size_type>(max_point_lights), vec3{ 0 });
	std::vector<vec3> point_light_attenuation_factors(static_cast<std::vector<vec3>::size_type>(max_point_lights), vec3{ 0 });
	std::vector<vec3> point_light_intensities(static_cast<std::vector<vec3>::size_type>(max_point_lights), vec3{ 0 });
	std::vector<float> point_light_haze_intensity_factors(static_cast<std::vector<float>::size_type>(max_point_lights), float{ 1.0f });
	std::vector<float> point_light_haze_location_decay_factors(static_cast<std::vector<float>::size_type>(max_point_lights), float{ 1.0f });
	{
		unsigned int i = 0;
		for (const PointLight* p_point_light : point_lights)
		{
			point_light_locations[i] = p_point_light->getLocation();

			triplet<float, float, float> attenuation_factors = p_point_light->getAttenuation();
			point_light_attenuation_factors[i] = vec3{ attenuation_factors.first, attenuation_factors.second, attenuation_factors.third };

			point_light_intensities[i] = p_point_light->getColor();

			point_light_haze_intensity_factors[i] = p_point_light->getHazeIntensity();

			point_light_haze_location_decay_factors[i] = p_point_light->getHazeLocationDecay();

			++i;
		}
	}
	light_buffer.pushVector(point_light_locations);
	light_buffer.pushVector(point_light_attenuation_factors);
	light_buffer.pushVector(point_light_intensities);
	light_buffer.pushScalar(point_light_haze_intensity_factors);
	light_buffer.pushScalar(point_light_haze_location_decay_factors);


	//Apply configuration of the spot lights
	light_buffer.pushScalar(static_cast<unsigned int>(spot_lights.size()));
	std::vector<vec3> spot_light_locations(static_cast<std::vector<vec3>::size_type>(max_spot_lights), vec3{ 0 });
	std::vector<vec3> spot_light_directions(static_cast<std::vector<vec3>::size_type>(max_spot_lights), vec3{ 0 });
	std::vector<vec3> spot_light_attenuation_factors(static_cast<std::vector<vec3>::size_type>(max_spot_lights), vec3{ 0 });
	std::vector<float> spot_light_exponents(static_cast<std::vector<float>::size_type>(max_spot_lights), 0);
	std::vector<vec3> spot_light_intensities(static_cast<std::vector<vec3>::size_type>(max_spot_lights), vec3{ 0 });
	std::vector<float> spot_light_haze_intensity_factors(static_cast<std::vector<float>::size_type>(max_spot_lights), float{ 1.0f });
	std::vector<float> spot_light_haze_direction_decay_factors(static_cast<std::vector<float>::size_type>(max_point_lights), float{ 1.0f });
	std::vector<float> spot_light_haze_location_decay_factors(static_cast<std::vector<float>::size_type>(max_point_lights), float{ 1.0f });
	{
		unsigned int  i = 0;
		for (const SpotLight* p_spot_light : spot_lights)
		{
			spot_light_locations[i] = p_spot_light->getLocation();

			spot_light_directions[i] = p_spot_light->getDirection().get_normalized();

			triplet<float, float, float> attenuation_factors = p_spot_light->getAttenuation();
			spot_light_attenuation_factors[i] = vec3{ attenuation_factors.first, attenuation_factors.second, attenuation_factors.third };

			spot_light_exponents[i] = p_spot_light->getSpotExponent();

			spot_light_intensities[i] = p_spot_light->getColor();

			spot_light_haze_intensity_factors[i] = p_spot_light->getHazeIntensity();

			spot_light_haze_direction_decay_factors[i] = p_spot_light->getHazeDirectionDecay();

			spot_light_haze_location_decay_factors[i] = p_spot_light->getHazeLocationDecay();

			++i;
		}
	}
	light_buffer.pushVector(spot_light_locations);
	light_buffer.pushVector(spot_light_directions);
	light_buffer.pushVector(spot_light_attenuation_factors);
	light_buffer.pushScalar(spot_light_exponents);
	light_buffer.pushVector(spot_light_intensities);
	light_buffer.pushScalar(spot_light_haze_intensity_factors);
	light_buffer.pushScalar(spot_light_haze_location_decay_factors);
	light_buffer.pushScalar(spot_light_haze_direction_decay_factors);
}

void LightingConditions_Core::updateFogBuffer() const
{
	//Ensure that writing caret points at the beginning of the buffer
	fog_buffer.resetOffsetCounter();

	//Apply parameters of the atmospheric fog
	fog_buffer.pushVector(v3AtmosphericFogColor);
	fog_buffer.pushScalar(atmospheric_fog_global_density);
	if (p_skydome) fog_buffer.pushVector(p_skydome->sun_direction);
	else fog_buffer.skipVector<vec3>(1, false);
	fog_buffer.pushScalar(atmospheric_fog_height_fall_off);
	if (p_skydome) fog_buffer.pushVector(p_skydome->moon_direction);
	else fog_buffer.skipVector<vec3>(1, false);
	fog_buffer.pushScalar(atmospheric_fog_mie_phase_function_param);
	fog_buffer.pushScalar(light_haze_attenuation_factor);
}

std::pair<ImmutableTexture2D, ImmutableTexture2D> LightingConditions_Core::retrieveInScatteringTextures() const
{
	return std::make_pair(p_skydome->in_scattering_sun, p_skydome->in_scattering_moon);
}



void LightingConditions::updateLightBuffer() const { LightingConditions_Core::updateLightBuffer(); }

void LightingConditions::updateFogBuffer() const { LightingConditions_Core::updateFogBuffer(); }

const std140UniformBuffer* LightingConditions::getLightBufferPtr() const { return &light_buffer; }

const std140UniformBuffer* LightingConditions::getFogBufferPtr() const { return &fog_buffer; }

std::pair<ImmutableTexture2D, ImmutableTexture2D> LightingConditions::retrieveInScatteringTextures() const
{
	return LightingConditions_Core::retrieveInScatteringTextures();
}

