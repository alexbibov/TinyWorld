#ifndef TW__ENVIRONMENT_MAP__

#include "ImmutableTexture2D.h"
#include "ImmutableTextureCubeMap.h"

namespace tiny_world
{
	//Simple wrapper around ImmutableTexture2D. Needed to identify different types of environment maps
	class EquirectangularEnvironmentMap
	{
	private:
		ImmutableTexture2D environment_map;		//texture alias of the environment map

	public:
		//Initializes equirectangular environment map using provided 2D texture
		explicit EquirectangularEnvironmentMap(const ImmutableTexture2D& environment_map_texture);

		//Extracts alias of the texture containing environmental map data
		operator ImmutableTexture2D() const;
	};


	//Simple wrapper around ImmutableTexture2D. Needed to identify different types of environment maps
	class SphericalEnvironmentMap
	{
	private:
		ImmutableTexture2D environemnt_map;		//texture alias of the environment map

	public:
		//Initializes hemispherical map using provided 2D texture
		SphericalEnvironmentMap(const ImmutableTexture2D& environment_map_texture);

		//Extracts alias of the texture containing environmental map data
		operator ImmutableTexture2D() const;
	};


	//Simple wrapper around ImmutableTextureCubeMap. Needed to identify different types of environment maps
	class CubeEnvironmentMap
	{
	private:
		ImmutableTextureCubeMap environment_map;	//texture alias of the environment map

	public:
		//Initializes cube environment map using provided cubemap texture
		CubeEnvironmentMap(const ImmutableTextureCubeMap& environment_map_texture);

		//Extracts alias of the texture containing environmental map data
		operator ImmutableTextureCubeMap() const;
	};


	//Enumeration listing the supported formats of environmental data
	enum class EnvironmentMapType : uint32_t
	{
		Spherical = 0,
		Cubic = 1,
		Equirectangular = 2
	};
} 

#define TW__ENVIRONMENT_MAP__
#endif