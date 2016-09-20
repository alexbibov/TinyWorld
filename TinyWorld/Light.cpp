#include "Light.h"

using namespace tiny_world;




AbstractLight::AbstractLight(const std::string& light_class_string_name) : 
Entity(light_class_string_name), color{ 1.0f, 1.0f, 1.0f }
{

}

AbstractLight::AbstractLight(const std::string& light_class_string_name, const std::string& light_string_name) : 
Entity(light_class_string_name, light_string_name), color{ 1.0f, 1.0f, 1.0f }
{

}

AbstractLight::AbstractLight(const std::string& light_class_string_name, const std::string& light_string_name, const vec3& color) : 
Entity(light_class_string_name, light_string_name), color{ color }
{

}

AbstractLight::AbstractLight(const std::string& light_class_string_name, const std::string& light_string_name, float color_r, float color_g, float color_b) : 
Entity(light_class_string_name, light_string_name), color{ color_r, color_g, color_b }
{

}

AbstractLight::AbstractLight(const AbstractLight& other) : 
Entity(other), color{ other.color }
{

}

AbstractLight::AbstractLight(AbstractLight&& other) : 
Entity(std::move(other)), color{ std::move(other.color) }
{

}

AbstractLight::~AbstractLight()
{

}

AbstractLight& AbstractLight::operator=(const AbstractLight& other)
{
	//Account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	Entity::operator=(other);

	color = other.color;
	return *this;
}

AbstractLight& AbstractLight::operator=(AbstractLight&& other)
{
	//Account for the special case of "assignment to itself"
	if (this == &other)
		return *this;
	
	Entity::operator=(std::move(other));

	color = std::move(other.color);
	return *this;
}

void AbstractLight::setColor(const vec3& new_color)
{
	color = new_color;
}

void AbstractLight::setColor(float color_r, float color_g, float color_b)
{
	color.x = color_r;
	color.y = color_g;
	color.z = color_b;
}

vec3 AbstractLight::getColor() const { return color; }




GlobalLight::GlobalLight() : AbstractLight("GlobalLight") {}

GlobalLight::GlobalLight(const GlobalLight& other) : AbstractLight(other)
{

}

GlobalLight::GlobalLight(GlobalLight&& other) : AbstractLight(std::move(other))
{

}

GlobalLight& GlobalLight::operator=(const GlobalLight& other)
{
	return *this;
}

GlobalLight& GlobalLight::operator=(GlobalLight&& other)
{ 
	return *this; 
}




LocalLight::LocalLight() : AbstractLight("LocalLight"),
location{ 0.0f, 0.0f, 0.0f }, 
attenuation_factor_constant{ 1.0f }, attenuation_factor_linear{ 0.0f }, attenuation_factor_quadratic{ 0.0f },
haze_intensity_factor{ 1.0f }, haze_location_decay{ 1.0f }
{

}

LocalLight::LocalLight(const vec3& location) : AbstractLight("LocalLight"),
location{ location },
attenuation_factor_constant{ 1.0f }, attenuation_factor_linear{ 0.0f }, attenuation_factor_quadratic{ 0.0f },
haze_intensity_factor{ 1.0f }, haze_location_decay{ 1.0f }
{

}

LocalLight::LocalLight(float location_x, float location_y, float location_z) : AbstractLight("LocalLight"), 
	location{ vec3{ location_x, location_y, location_z } },
	attenuation_factor_constant{ 1.0f }, attenuation_factor_linear{ 0.0f }, attenuation_factor_quadratic{ 0.0f }, 
	haze_intensity_factor{ 1.0f }, haze_location_decay{ 1.0f }
{

}

LocalLight::LocalLight(const LocalLight& other) : AbstractLight(other)

{

}

LocalLight::LocalLight(LocalLight&& other) : AbstractLight(std::move(other)), 
location{ std::move(other.location) }, attenuation_factor_constant{ other.attenuation_factor_constant },
attenuation_factor_linear{ other.attenuation_factor_linear }, attenuation_factor_quadratic{ other.attenuation_factor_quadratic }, 
haze_intensity_factor{ other.haze_intensity_factor }, haze_location_decay{ other.haze_location_decay }
{

}

LocalLight& LocalLight::operator=(const LocalLight& other)
{
	//Account for the case of assignment to itself
	if (this == &other)
		return *this;

	location = other.location;
	attenuation_factor_constant = other.attenuation_factor_constant;
	attenuation_factor_linear = other.attenuation_factor_linear;
	attenuation_factor_quadratic = other.attenuation_factor_quadratic;
	haze_intensity_factor = other.haze_intensity_factor;
	haze_location_decay = other.haze_location_decay;

	return *this;
}

LocalLight& LocalLight::operator=(LocalLight&& other)
{
	//Account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	location = std::move(other.location);
	attenuation_factor_constant = other.attenuation_factor_constant;
	attenuation_factor_linear = other.attenuation_factor_linear;
	attenuation_factor_quadratic = other.attenuation_factor_quadratic;
	haze_intensity_factor = other.haze_intensity_factor;
	haze_location_decay = other.haze_location_decay;

	return *this;
}

vec3 LocalLight::getLocation() const { return location; }

void LocalLight::setLocation(const vec3& new_location)
{
	location = new_location;
}

void LocalLight::setLocation(float location_x, float location_y, float location_z)
{
	location.x = location_x;
	location.y = location_y;
	location.z = location_z;
}

void LocalLight::setAttenuation(float constant_factor, float linear_factor, float quadratic_factor)
{
	attenuation_factor_constant = constant_factor > 0 ? constant_factor : 0.0f;
	attenuation_factor_linear = linear_factor > 0 ? linear_factor : 0.0f;
	attenuation_factor_quadratic = quadratic_factor > 0 ? quadratic_factor : 0.0f;

	if (attenuation_factor_constant == 0 && attenuation_factor_linear == 0 && attenuation_factor_quadratic == 0)
		attenuation_factor_constant = 1.0f;
}

triplet<float, float, float> LocalLight::getAttenuation() const
{
	return triplet < float, float, float > {attenuation_factor_constant, attenuation_factor_linear, attenuation_factor_quadratic};
}

void LocalLight::setHazeIntensity(float factor) { haze_intensity_factor = factor; }

float LocalLight::getHazeIntensity() const { return haze_intensity_factor; }

void LocalLight::setHazeLocationDecay(float decay_factor) { haze_location_decay = decay_factor; }

float LocalLight::getHazeLocationDecay() const { return haze_location_decay; }




DirectedLight::DirectedLight() : 
AbstractLight("DirectedLight"), direction{ 0.0f, -1.0f, 0.0f }
{

}

DirectedLight::DirectedLight(const vec3& direction) : 
AbstractLight("DirectedLight"), direction{ direction }
{

}

DirectedLight::DirectedLight(float direction_x, float direction_y, float direction_z) : 
AbstractLight("DirectedLight"), direction{ direction_x, direction_y, direction_z }
{

}

DirectedLight::DirectedLight(const DirectedLight& other) : 
AbstractLight(other), direction{ other.direction }
{

}

DirectedLight::DirectedLight(DirectedLight&& other) : 
AbstractLight(std::move(other)), direction{ std::move(other.direction) }
{

}

DirectedLight& DirectedLight::operator=(const DirectedLight& other)
{
	//Account for the special case of assignment to itself
	if (this == &other)
		return *this;

	direction = other.direction;

	return *this;
}

DirectedLight& DirectedLight::operator=(DirectedLight&& other)
{
	//Account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	direction = std::move(other.direction);

	return *this;
}

vec3 DirectedLight::getDirection() const { return direction; }

void DirectedLight::setDirection(const vec3& new_direction)
{
	direction = new_direction;
}




AmbientLight::AmbientLight() : AbstractLight("AmbientLight"), GlobalLight()
{

}

AmbientLight::AmbientLight(const std::string& ambient_light_string_name) : AbstractLight("AmbientLight", ambient_light_string_name), GlobalLight()
{

}

AmbientLight::AmbientLight(const std::string& ambient_light_string_name, const vec3& color) : AbstractLight("AmbientLight", ambient_light_string_name, color), GlobalLight()
{

}

AmbientLight::AmbientLight(const std::string& ambient_light_string_name, float color_r, float color_g, float color_b) :
AbstractLight("AmbientLight", ambient_light_string_name, color_r, color_g, color_b), GlobalLight()
{

}

AmbientLight::AmbientLight(const AmbientLight& other) : AbstractLight(other), GlobalLight(other)
{

}

AmbientLight::AmbientLight(AmbientLight&& other) : AbstractLight(std::move(other)), GlobalLight(std::move(other))
{

}

AmbientLight& AmbientLight::operator=(const AmbientLight& other)
{
	if (this == &other)
		return *this;

	AbstractLight::operator=(other);
	GlobalLight::operator=(other);

	return *this;
}

AmbientLight& AmbientLight::operator=(AmbientLight&& other)
{
	if (this == &other)
		return *this;

	AbstractLight::operator=(std::move(other));
	GlobalLight::operator=(std::move(other));

	return *this;
}

AbstractLight* AmbientLight::clone() const { return new AmbientLight{ *this }; }

LightType AmbientLight::getLightType() const { return LightType::AMBIENT; }




DirectionalLight::DirectionalLight() : AbstractLight("DirectionalLight"), GlobalLight(), DirectedLight()
{

}

DirectionalLight::DirectionalLight(const std::string& directional_light_string_name) : 
AbstractLight("DirectionalLight", directional_light_string_name), GlobalLight(), DirectedLight()
{

}

DirectionalLight::DirectionalLight(const std::string& directional_light_string_name, const vec3& color, const vec3& direction) :
AbstractLight("DirectionalLight", directional_light_string_name, color), GlobalLight(), DirectedLight(direction)
{

}

DirectionalLight::DirectionalLight(const std::string& directional_light_string_name,
	float color_r, float color_g, float color_b,
	float direction_x, float direction_y, float direction_z) : 
	AbstractLight("DirectionalLight", directional_light_string_name, color_r, color_g, color_b),
	GlobalLight(), 
	DirectedLight(direction_x, direction_y, direction_z)
{

}

DirectionalLight::DirectionalLight(const DirectionalLight& other) : AbstractLight(other), GlobalLight(other), DirectedLight(other)
{

}

DirectionalLight::DirectionalLight(DirectionalLight&& other) : AbstractLight(std::move(other)), GlobalLight(std::move(other)), DirectedLight(std::move(other))
{

}

DirectionalLight& DirectionalLight::operator=(const DirectionalLight& other)
{
	if (this == &other)
		return *this;

	AbstractLight::operator=(other);
	GlobalLight::operator=(other);
	DirectedLight::operator=(other);

	return *this;
}

DirectionalLight& DirectionalLight::operator=(DirectionalLight&& other)
{
	if (this == &other)
		return *this;

	AbstractLight::operator=(std::move(other));
	GlobalLight::operator=(std::move(other));
	DirectedLight::operator=(std::move(other));

	return *this;
}

AbstractLight* DirectionalLight::clone() const { return new DirectionalLight{ *this }; }

LightType DirectionalLight::getLightType() const { return LightType::DIRECTIONAL; }




PointLight::PointLight() : AbstractLight("PointLight"), LocalLight()
{

}

PointLight::PointLight(const std::string& point_light_string_name) : AbstractLight("PointLight", point_light_string_name), LocalLight()
{

}

PointLight::PointLight(const std::string& point_light_string_name, const vec3& color, const vec3& location) : 
AbstractLight("PointLight", point_light_string_name, color), LocalLight(location)
{

}

PointLight::PointLight(const std::string& point_light_string_name,
	float color_r, float color_g, float color_b,
	float location_x, float location_y, float location_z) : 
	AbstractLight("PointLight", point_light_string_name, color_r, color_g, color_b), LocalLight(location_x, location_y, location_z)
{

}

PointLight::PointLight(const PointLight& other) : AbstractLight(other), LocalLight(other)
{


}

PointLight::PointLight(PointLight&& other) : AbstractLight(std::move(other)), LocalLight(std::move(other))
{

}

PointLight& PointLight::operator=(const PointLight& other)
{
	if (this == &other)
		return *this;

	AbstractLight::operator=(other);
	LocalLight::operator=(other);

	return *this;
}

PointLight& PointLight::operator=(PointLight&& other)
{
	if (this == &other)
		return *this;

	AbstractLight::operator=(std::move(other));
	LocalLight::operator=(std::move(other));

	return *this;
}

AbstractLight* PointLight::clone() const { return new PointLight{ *this }; }

LightType PointLight::getLightType() const { return LightType::POINT; }




SpotLight::SpotLight() : AbstractLight("SpotLight"), LocalLight(), DirectedLight(), spot_exponent{ 1.0f }, haze_direction_decay{ 1.0f }
{

}

SpotLight::SpotLight(const std::string& spot_light_string_name) : 
AbstractLight("SpotLight", spot_light_string_name), LocalLight(), DirectedLight(), spot_exponent{ 1.0f }, haze_direction_decay{ 1.0f }
{

}

SpotLight::SpotLight(const std::string& spot_light_string_name, const vec3& color, const vec3& location, const vec3& direction) :
AbstractLight("SpotLight", spot_light_string_name, color), LocalLight(location), DirectedLight(direction), spot_exponent{ 1.0f }, haze_direction_decay{ 1.0f }
{

}

SpotLight::SpotLight(const std::string& spot_light_string_name,
	float color_r, float color_g, float color_b,
	float location_x, float location_y, float location_z,
	float direction_x, float direction_y, float direction_z) :
	AbstractLight("SpotLight", spot_light_string_name, color_r, color_g, color_b),
	LocalLight(location_x, location_y, location_z),
	DirectedLight(direction_x, direction_y, direction_z), spot_exponent{ 1.0f },
	haze_direction_decay{ 1.0f }
{

}

SpotLight::SpotLight(const SpotLight& other) : AbstractLight(other), LocalLight(other), DirectedLight(other),
spot_exponent{ other.spot_exponent }, haze_direction_decay{ other.haze_direction_decay }
{

}

SpotLight::SpotLight(SpotLight&& other) : AbstractLight(std::move(other)), LocalLight(std::move(other)), DirectedLight(std::move(other)),
spot_exponent{ other.spot_exponent }, haze_direction_decay{ other.haze_direction_decay }
{

}

SpotLight& SpotLight::operator=(const SpotLight& other)
{
	if (this == &other)
		return *this;

	AbstractLight::operator=(other);
	LocalLight::operator=(other);
	DirectedLight::operator=(other);

	spot_exponent = other.spot_exponent;
	haze_direction_decay = other.haze_direction_decay;

	return *this;
}

SpotLight& SpotLight::operator=(SpotLight&& other)
{
	if (this == &other)
		return *this;

	AbstractLight::operator=(std::move(other));
	LocalLight::operator=(std::move(other));
	DirectedLight::operator=(std::move(other));

	spot_exponent = other.spot_exponent;
	haze_direction_decay = other.haze_direction_decay;

	return *this;
}

AbstractLight* SpotLight::clone() const { return new SpotLight{ *this }; }

LightType SpotLight::getLightType() const { return LightType::SPOT; }

void SpotLight::setSpotExponent(float exponent)
{
	spot_exponent = exponent > 0 ? exponent : 0.0f;
}

float SpotLight::getSpotExponent() const { return spot_exponent; }

void SpotLight::setHazeDirectionDecay(float decay_factor) { haze_direction_decay = decay_factor; }

float SpotLight::getHazeDirectionDecay() const { return haze_direction_decay; }