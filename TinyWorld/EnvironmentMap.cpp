#include "EnvironmentMap.h"

using namespace tiny_world;


EquirectangularEnvironmentMap::EquirectangularEnvironmentMap(const ImmutableTexture2D& environment_map_texture) : environment_map{ environment_map_texture }
{

}

EquirectangularEnvironmentMap::operator tiny_world::ImmutableTexture2D() const
{
	return environment_map;
}


SphericalEnvironmentMap::SphericalEnvironmentMap(const ImmutableTexture2D& environment_map_texture) : environemnt_map{ environment_map_texture }
{

}

SphericalEnvironmentMap::operator tiny_world::ImmutableTexture2D() const
{
	return environemnt_map;
}


CubeEnvironmentMap::CubeEnvironmentMap(const ImmutableTextureCubeMap& environment_map_texture) : environment_map{ environment_map_texture }
{

}

CubeEnvironmentMap::operator tiny_world::ImmutableTextureCubeMap() const
{
	return environment_map;
}