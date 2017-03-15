#version 430

const unsigned int max_kernel_size = 512;	//maximal size of the filter kernel

layout(std140) uniform FilterKernelParams
{
	int horizontal_blur;	//if equals 'true' the filter performs horizontal blurring. If equals 'false' the filter performs vertical blurring
	unsigned int size;	//size of the kernel
	unsigned int mipmap_level;	//mipmap level used for blurring
	float kernel[max_kernel_size];	//filter kernel
}params;


//source0 — input texture to be blurred
vec4 do_filtering(in sampler2D source0)
{
	vec2 v2TextureSize = textureSize(source0, 0);
	vec4 v4BlurredColor = vec4(0);

	int dimension_selector = 0;
	if(params.horizontal_blur == 0) dimension_selector = 1;

		
	for(unsigned int i = 0; i < params.size; ++i)
	{
		vec2 v2Pos = gl_FragCoord.xy - 0.5f;
		v2Pos[dimension_selector] += i - (params.size - 1.0f) / 2.0f;
		v4BlurredColor += textureLod(source0, v2Pos/(v2TextureSize - 1.0f), float(params.mipmap_level))*params.kernel[i];
	}
		

	return v4BlurredColor;
}