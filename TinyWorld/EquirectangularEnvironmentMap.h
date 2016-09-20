#ifndef TW__EQUIRECTANGULAR_ENVIRONMENT_MAP__

#include "ImmutableTexture2D.h"

namespace tiny_world
{
	class EquirectangularEnvironmentMap
	{
	private:
		ImmutableTexture2D environment_map;		//texture alias of the environment map


	public:
		//Default initialization constructor
		EquirectangularEnvironmentMap();

		//Initializes equirectangular environment map using provided 2D texture
		EquirectangularEnvironmentMap(const ImmutableTexture2D& environment_map_texture);


		//Installs texture to provide environmental data
		void setEnvironmentalData(const ImmutableTexture2D& environment_map_texture);

		//Returns texture providing environmental data
		ImmutableTexture2D getEnvironmentalData() const;


		
	};
}

#define TW__EQUIRECTANGULAR_ENVIRONMENT_MAP__
#endif