#ifndef TW__LIGHT_H__


#include <string>
#include <cinttypes>

#include "VectorTypes.h"
#include "Misc.h"
#include "Entity.h"

namespace tiny_world
{
	//Describes all available types of light sources
	enum class LightType{ AMBIENT, DIRECTIONAL, POINT, SPOT };



	//Implements concept of generic light: not all lights have outspoken direction or position, but all of them have initial intensity (color)
	class AbstractLight : public Entity
	{
	private:
		vec3 color;		//intensity of the light source

	protected:
		//Constructor infrastructure is protected because AbstractLight can not exist on its own 

		explicit AbstractLight(const std::string& light_class_string_name);	//Creates basic white light
		AbstractLight(const AbstractLight& other);	//copy constructor
		AbstractLight(AbstractLight&& other);	//move constructor
		AbstractLight(const std::string& light_class_string_name, const std::string& light_string_name);		//Creates basic white light and associates the given string name with it
		AbstractLight(const std::string& light_class_string_name, const std::string& light_string_name, const vec3& color);		//Creates light having the given color
		AbstractLight(const std::string& light_class_string_name, const std::string& light_string_name, float color_r, float color_g, float color_b);	//Creates light with color defined by the triplet (color_r, color_g, color_b)

		AbstractLight& operator=(const AbstractLight& other);
		AbstractLight& operator=(AbstractLight&& other);

	public:
		~AbstractLight();	//destructor
		virtual AbstractLight* clone() const = 0;	//allocates copy of the light object on the heap memory
		virtual LightType getLightType() const = 0;	//returns the light type implemented by the object

		void setColor(const vec3& new_color);		//sets a new color for the light source
		void setColor(float color_r, float color_g, float color_b);	//sets a new color for the lights source determined by the triplet (color_r, color_g, color_b)

		vec3 getColor() const;	//returns the current color of the light source
	};



	//Global lights don't have position, but may have a direction. This class is needed just for inheritance consistency
	class GlobalLight : virtual public AbstractLight
	{
	protected:
		GlobalLight();
		GlobalLight(const GlobalLight& other);
		GlobalLight(GlobalLight&& other);
		
		GlobalLight& operator=(const GlobalLight& other);
		GlobalLight& operator=(GlobalLight&& other);
	};



	//Local lights have position but not necessarily have direction
	class LocalLight : virtual public AbstractLight
	{
	private:
		vec3 location;		//position of the light object in 3D-space
		float attenuation_factor_constant;	//constant attenuation factor
		float attenuation_factor_linear;	//linear attenuation factor
		float attenuation_factor_quadratic;	//quadratic attenuation factor
		float haze_intensity_factor;		//intensity factor of the atmospheric haze
		float haze_location_decay;			//location decay of the atmospheric haze

	protected:
		LocalLight();	//Creates basic local light located at the origin
		explicit LocalLight(const vec3& location);		//Creates light located at given location
		LocalLight(float location_x, float location_y, float location_z);	//Creates light located at (location_x, location_y, location_z)
		LocalLight(const LocalLight& other);
		LocalLight(LocalLight&& other);

		LocalLight& operator=(const LocalLight& other);
		LocalLight& operator=(LocalLight&& other);

	public:
		vec3 getLocation() const;	//returns current position of the local light
		void setLocation(const vec3& new_location);		//sets position of the local light
		void setLocation(float location_x, float location_y, float location_z);	//sets location of the local light to (position_x, position_y, postion_z)
		void setAttenuation(float constant_factor, float linear_factor, float quadratic_factor);	//sets attenuation factors for the local light. Negative values are clamped to 0.
		triplet<float, float, float> getAttenuation() const;	//returns constant, linear and quadratic attenuation factors packed into a triplet
		void setHazeIntensity(float factor);	//sets intensity factor applied when computing the atmospheric haze effect produced by the light source. The value of zero turns atmospheric haze off. The default value is 1.
		float getHazeIntensity() const;	//returns intensity factor of the atmospheric haze
		void setHazeLocationDecay(float decay_factor);	//sets decay factor used to compute atmospheric haze effect caused by presence of the light. This factor determines how fast the haze disappears as distance to the light source increases. The default value of this parameter is 1.0
		float getHazeLocationDecay() const;	//returns location decay factor used to compute haze effect caused by the light source. This factor determines attenuation of the haze glowing halo with respect to distance from the light source.
	};



	//Directed lights are special in the sense that they have direction
	class DirectedLight : virtual public AbstractLight
	{
	private:
		vec3 direction;		//direction of the light object in 3D-space

	protected:
		DirectedLight();	//Creates light with orientation (0, -1, 0) (directed downwards) 
		explicit DirectedLight(const vec3& direction);		//Creates light with given direction
		DirectedLight(float direction_x, float direction_y, float direction_z);	//Creates light with direction (direction_x, direction_y, direction_z)
		DirectedLight(const DirectedLight& other);
		DirectedLight(DirectedLight&& other);

		DirectedLight& operator=(const DirectedLight& other);
		DirectedLight& operator=(DirectedLight&& other);

	public:
		vec3 getDirection() const;	//returns current direction of the directed light
		void setDirection(const vec3& new_direction);	//sets new direction of the directed light
	};



	//Implements ambient light
	class AmbientLight final : public GlobalLight
	{
	public:
		AmbientLight();		//Creates basic ambient light of white color
		explicit AmbientLight(const std::string& ambient_light_string_name);	//creates white ambient light and weakly associates it with the given string name
		AmbientLight(const std::string& ambient_light_string_name, const vec3& color);	//creates an ambient light with given color and weakly identified by the given string name
		AmbientLight(const std::string& ambient_light_string_name, float color_r, float color_g, float color_b);		//creates ambient light with color (color_r, color_g, color_b) weakly identified by the given string name
		AmbientLight(const AmbientLight& other);	//copy constructor
		AmbientLight(AmbientLight&& other);		//move constructor

		AmbientLight& operator=(const AmbientLight& other);	//copy-assignment operator
		AmbientLight& operator=(AmbientLight&& other);	//move-assignment

		AbstractLight* clone() const override;
		LightType getLightType() const override;
	};



	//Implements directional light
	class DirectionalLight final : public GlobalLight, public DirectedLight
	{
	public:
		DirectionalLight();		//Creates basic white directional light with orientation (0, -1, 0)
		explicit DirectionalLight(const std::string& directional_light_string_name);		//creates basic white directional light directed downwards, which is weakly identified by the given string name
		DirectionalLight(const std::string& directional_light_string_name, const vec3& color, const vec3& direction);	//creates directional light with given color and direction weakly identified by the given string name
		DirectionalLight(const std::string& directional_light_string_name,
			float color_r, float color_g, float color_b,
			float direction_x, float direction_y, float direction_z);	//creates directional light with color (color_r, color_b, color_b) and direction (direction_x, direction_y, direction_z) weakly identified by the given string name
		DirectionalLight(const DirectionalLight& other);	//copy constructor
		DirectionalLight(DirectionalLight&& other);		//move constructor

		DirectionalLight& operator=(const DirectionalLight& other);	//copy-assignment operator
		DirectionalLight& operator=(DirectionalLight&& other);	//move-assignment operator

		AbstractLight* clone() const override;
		LightType getLightType() const override;
	};



	//Implements point light source
	class PointLight final : public LocalLight
	{
	public:
		PointLight();	//creates basic white point light located at the origin
		explicit PointLight(const std::string& point_light_string_name);	//creates basic white point light located at the origin weakly identified by the given string name
		PointLight(const std::string& point_light_string_name, const vec3& color, const vec3& location);		//creates point light with given color and position weakly identified by the given string name 
		PointLight(const std::string& point_light_string_name,
			float color_r, float color_g, float color_b,
			float location_x, float location_y, float location_z);	//creates point light with color (color_r, color_g, color_b) located at (position_x, position_y, position_z)
		PointLight(const PointLight& other);	//copy-constructor for the point light
		PointLight(PointLight&& other);		//move-constructor for the point light

		PointLight& operator=(const PointLight& other);	//copy-assignment operator
		PointLight& operator=(PointLight&& other);	//move-assignment operator

		AbstractLight* clone() const override;
		LightType getLightType() const override;
	};



	//Implements spot light source
	class SpotLight final : public LocalLight, public DirectedLight
	{
	private:
		float spot_exponent;	//spot attenuation when incident ray moves away from the spot direction
		float haze_direction_decay;	//direction decay of the atmospheric haze

	public:
		SpotLight();	//creates basic white spot light located in the origin and directed downwards (orientation is defined by vector (0, -1, 0))
		explicit SpotLight(const std::string& spot_light_string_name);	//creates white spot light weakly identified by the given string name
		SpotLight(const std::string& spot_light_string_name, const vec3& color, const vec3& location, const vec3& direction);	//creates spot light with given color, location and direction, which is weakly identified by the given string name
		SpotLight(const std::string& spot_light_string_name,
			float color_r, float color_g, float color_b,
			float location_x, float location_y, float location_z,
			float direction_x, float direction_y, float direction_z);	//creates spot light with color (color_r, color_g, color_b) weakly identified by the given string name and having position (position_x, position_y, position_z) and direction (direction_x, direction_y, direction_z)
		SpotLight(const SpotLight& other);	//copy constructor
		SpotLight(SpotLight&& other);	//move constructor

		SpotLight& operator=(const SpotLight& other);	//copy-assignment operator
		SpotLight& operator=(SpotLight&& other);	//move-assignment operator

		AbstractLight* clone() const override;
		LightType getLightType() const override;

		void setSpotExponent(float exponent);	//set new attenuation exponent for the spot light source. The value is clamped to 0 when negative.
		float getSpotExponent() const;	//returns spot light attenuation exponent

		void setHazeDirectionDecay(float decay_factor);	//sets decay factor used to compute haze effect caused by the light source. This parameter determines decay of the haze as the distance between a given point and direction axis of the light source increases. The default value of this parameter is 1.0
		float getHazeDirectionDecay() const;	//returns decay factor determining attenuation of the atmospheric haze at a given point with respect to distance between this point and direction axis of the light source.
	};
}


#define TW__LIGHT_H__
#endif