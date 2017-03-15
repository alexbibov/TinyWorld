#version 430 core

//Channel description:
//source0	color map
//source1	AD-map
//source2	occlusion map
vec4 do_filtering(in sampler2D source0, in sampler2D source1, in sampler2D source2)
{
	ivec2 iv2FragCoord = ivec2(gl_FragCoord.xy);
	vec4 v4Color = texelFetch(source0, iv2FragCoord, 0); 
	vec3 v3ResultingColor = v4Color.rgb + texelFetch(source1, iv2FragCoord, 0).rgb*(texelFetch(source2, iv2FragCoord, 0).r - 1.0f);

	return vec4(v3ResultingColor.rgb, v4Color.a);
}