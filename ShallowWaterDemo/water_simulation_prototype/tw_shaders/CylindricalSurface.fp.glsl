#version 430 core


in GS_DATA
{
	vec2 v2TextureCoordinate;
}gs_in;


uniform sampler2D surface_diffuse_texture;


out layout(location = 0) vec4 v4FragColor;
out layout(location = 1) vec4 v4BloomColor;


vec4 computeLightContribution(vec4 v4DiffuseColor, vec2 v2NormalMapTexCoords, vec2 v2SpecularMapTexCoords, vec2 v2EmissionMapTexCoords, 
		float fNormalMapLayer, float fSpecularMapLayer, float fEmissionMapLayer, float fEnvironmentMapLayer);
vec4 computeBloomFragment(vec4 v4FragmentColor);


void main()
{
	v4FragColor = computeLightContribution(texture(surface_diffuse_texture, gs_in.v2TextureCoordinate), 
		gs_in.v2TextureCoordinate, gs_in.v2TextureCoordinate, gs_in.v2TextureCoordinate, 0, 0, 0, 0);

	v4BloomColor = computeBloomFragment(v4FragColor);
}