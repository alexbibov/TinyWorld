#version 430 core

#define MAX_FLOAT_VALUE 3.402823466e38f

uniform float bloom_impact;			//determines how much bloom effect affects the final scene colour
uniform float contrast;		//contrast value used by the HDR filter

const float fGaussianBlurWeights[] = 
	float[](0.0024499299678342,
0.0043538453346397,
0.0073599963704157,
0.0118349786570722,
0.0181026699707781,
0.0263392293891488,
0.0364543006660986,
0.0479932050577658,
0.0601029809166942,
0.0715974486241365,
0.0811305381519717,
0.0874493212267511,
0.0896631113333857,
0.0874493212267511,
0.0811305381519717,
0.0715974486241365,
0.0601029809166942,
0.0479932050577658,
0.0364543006660986,
0.0263392293891488,
0.0181026699707781,
0.0118349786570722,
0.0073599963704157,
0.0043538453346397,
0.0024499299678342);
	
in vec2 tex_coord;



//Computes luminance corresponding to the given color
float computeLuminance(vec3 v3Color)
{
	return dot(v3Color, vec3(0.299f, 0.587f, 0.144f));
}

//Computes exposure that should be applied to a color with given luminance
float computeExposure(float fLuminance, float fBias)
{
	return min(sqrt(1.0f / (fLuminance + fBias)), MAX_FLOAT_VALUE);
}

//source0 — color texture defined in the screen space
//source1 — bloom texture defined in the screen space
vec4 do_filtering(in sampler2D source0, in sampler2D source1)
{
	vec4 v4MixedColor = vec4(0.0f);
	vec4 v4BloomColor = vec4(0.0f);
	vec4 v4HDRColor = vec4(0.0f);
	vec2 v2TextureScale = textureSize(source0, 0) - 1;
	float fLuminance[25];
	
	for(int i = 0; i < 25; ++i)
		v4BloomColor += fGaussianBlurWeights[i] * 
		texture(source1, tex_coord + (-vec2(0.5f) + vec2(i / 5 - 2, i % 5 - 2))/v2TextureScale);
	
	v4HDRColor = texture(source0, tex_coord - vec2(0.5f) / v2TextureScale);
	v4MixedColor.rgb =  v4HDRColor.rgb + bloom_impact * v4BloomColor.rgb;
	v4MixedColor.a = v4HDRColor.a;
	
	for(int i = 0; i < 25; ++i)
	{
		v4HDRColor = texture(source0, (gl_FragCoord.xy + vec2(i / 5 - 2, i % 5 - 2)) / v2TextureScale);
		v4HDRColor.rgb = v4HDRColor.rgb + bloom_impact * v4BloomColor.rgb;
		fLuminance[i] = computeLuminance(v4HDRColor.rgb);
	}
	
	float fFilteredLuminance = (1.0f * (fLuminance[0] + fLuminance[4] + fLuminance[20] + fLuminance[24]) + 
		4.0f * (fLuminance[1] + fLuminance[3] + fLuminance[5] + fLuminance[9] + fLuminance[15] + fLuminance[19] + fLuminance[21] + fLuminance[23]) + 
		7.0f * (fLuminance[2] + fLuminance[10] + fLuminance[14] + fLuminance[22]) + 
		16.0f * (fLuminance[6] + fLuminance[8] + fLuminance[16] + fLuminance[18]) + 
		26.0f * (fLuminance[7] + fLuminance[11] + fLuminance[13] + fLuminance[17]) + 
		41.0f * fLuminance[12]) / 273.0f;
		
	float fExposure = computeExposure(fFilteredLuminance, 0.1f);

	vec4 v4Color;
	v4Color.rgb = vec3(1.0f)-exp(-pow(min(fExposure*v4MixedColor.rgb, pow(MAX_FLOAT_VALUE, 1/contrast ) ), vec3(contrast) ));
	v4Color.a  = v4MixedColor.a;
	return v4Color;
} 

