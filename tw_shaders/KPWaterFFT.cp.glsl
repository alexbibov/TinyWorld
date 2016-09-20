#version 430 core

#include "KPWaterCommonDefinitions.inc"

layout(local_size_x = TW__KPWATER_FFT_HALF_SIZE__, local_size_y = 1, local_size_z = 1) in;

#define nSignificantBits 9
#define pi 3.1415926535897932384626433832795f


uniform sampler2D s2dPhillipsSpectrum;	//Used to sample values from the Phillips spectrum

layout(rg32f) uniform coherent restrict image2D i2dOutput1;	//Used to write output values computed by FFT
layout(rgba32f) uniform coherent restrict image2D i2dOutput2;	//Used to write output values computed by FFT
layout(rgba32f) uniform coherent restrict image2D i2dOutput3;	//Used to write output values computed by FFT
layout(rgba32f) uniform coherent restrict image2D i2dOutput4;	//Used to write output values computed by FFT


uniform uint uiStep;	//step of the computation
//Below given brief descriptions on how computation is implemented.
//
//Step 0.
//FFT is computed for each row assuming that s2dPhillipsSpectrum contains (e1*h(K), e2*h(K), e1*h(-K), -e2*h(-K)) in its 
//red, green, blue, and alpha channels respectively. Here e1 and e2 are random draws from the standard Gaussian distribution 
//and h(k)=1/sqrt(2)*sqrt(P), where P is the Phillips distribution. The results are written to i2dOutput1, i2dOutput2, and to i2dOutput3
//and are organized as follows: red and green channels of i2dOutput1 contain row-wise transform of the basic Tessendorf spectrum; 
//red, green, blue and alpha channels of i2dOutput2 contain row-wise transforms of the x- and y- values of the spectrum used to compute spatial 
//displacement that is needed to create choppy waves; red, green, blue, and alpha channels of i2dOutput3 contain row-wise transforms of the 
//normal map spectrum. 
//All data are written to memory in transposed order. Note that each result of the raw-major transforms uses two channels. This is needed to store both 
//real and imaginary parts of the result.
//
//Step 1.		
//The data is read back from i2dOutput1, i2dOutput2, and i2dOutput3 and raw-major FFTs are applied to the results calculated on Step 1.
//The resulting computations are written into the same outputs and are organized in the following way: red channel of i2dOutput1 contains 
//height map of the waves, the green channel is not in use; red and green channels of i2dOutput2 contain x- and y- values of the displacement map, 
//the blue and the alpha channels of this output are not in use; finally red and green color channels of i2dOutput3 contain x- and y- values of
//the normal vectors of the wave height map, blue and alpha channels are not used.


uniform float fChoppiness;	//Parameter affecting choppiness of the waves
uniform float fGravityConstant;	//Acceleration due to gravity
uniform float fTimeGlobalScale;	//Current time instant used to produce result at the global scale
uniform float fTimeCapillaryScale;	//Current time instant used to produce result at the capillary scale
uniform vec2 v2DomainSize;	//Width and height of the water domain packed in this order into a 2D vector



shared vec2 fFFTData1[2*gl_WorkGroupSize.x];	//receives wave height map
shared vec4 fFFTData2[2*gl_WorkGroupSize.x];	//receives x- and y- values of the displacement map
shared vec4 fFFTData3[2*gl_WorkGroupSize.x];	//receives x- and y- values of the normal map computed at the global scale
shared vec4 fFFTData4[2*gl_WorkGroupSize.x];	//receives x- and y- values of the normal map computed at the capillary scale



//Reverses first S bits of the given unsigned integer vector and drops all other bits to 0
uvec2 reverse_bits(uvec2 x, uint S)
{
	x = (x & 0xAAAAAAAA) >> 1 | (x & 0x55555555) << 1;
	x = (x & 0xCCCCCCCC) >> 2 | (x & 0x33333333) << 2;
	x = (x & 0xF0F0F0F0) >> 4 | (x & 0x0F0F0F0F) << 4;
	x = (x & 0xFF00FF00) >> 8 | (x & 0x00FF00FF) << 8;
	x = (x & 0xFFFF0000) >> 16 | (x & 0x0000FFFF) << 16;

	return x >> (32 - S);
}


//Computes product of two complex numbers represented by 2D vectors with the x-value
//as the real part and the y-value as the imaginary part
vec2 complex_product(vec2 a, vec2 b)
{
	return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}


//Performs full pass of the Cooley-Tukey algorithm
void makeCooleyTukeyPass()
{
	for(int iCooleyTukeyStep = 2; iCooleyTukeyStep <= 2*gl_WorkGroupSize.x; iCooleyTukeyStep *= 2)
	{
		int iElement = int(gl_LocalInvocationID.x) % (iCooleyTukeyStep/2);
		int iGroup = int(gl_LocalInvocationID.x) / (iCooleyTukeyStep/2);

		vec2 v2TwiddleFactor = vec2(cos(2.0f*pi*iElement/iCooleyTukeyStep), -sin(2.0f*pi*iElement/iCooleyTukeyStep));


		vec2 v2E11 = fFFTData1[iGroup*iCooleyTukeyStep + iElement];
		vec2 v2E12 = complex_product(v2TwiddleFactor, fFFTData1[iGroup*iCooleyTukeyStep + iElement + iCooleyTukeyStep/2]);

		vec4 v4E21 = fFFTData2[iGroup*iCooleyTukeyStep + iElement];
		vec4 v4E22 = vec4(complex_product(v2TwiddleFactor, fFFTData2[iGroup*iCooleyTukeyStep + iElement + iCooleyTukeyStep/2].xy), 
			complex_product(v2TwiddleFactor, fFFTData2[iGroup*iCooleyTukeyStep + iElement + iCooleyTukeyStep/2].zw));

		vec4 v4E31 = fFFTData3[iGroup*iCooleyTukeyStep + iElement];
		vec4 v4E32 = vec4(complex_product(v2TwiddleFactor, fFFTData3[iGroup*iCooleyTukeyStep + iElement + iCooleyTukeyStep/2].xy), 
			complex_product(v2TwiddleFactor, fFFTData3[iGroup*iCooleyTukeyStep + iElement + iCooleyTukeyStep/2].zw));

		vec4 v4E41 = fFFTData4[iGroup*iCooleyTukeyStep + iElement];
		vec4 v4E42 = vec4(complex_product(v2TwiddleFactor, fFFTData4[iGroup*iCooleyTukeyStep + iElement + iCooleyTukeyStep/2].xy), 
			complex_product(v2TwiddleFactor, fFFTData4[iGroup*iCooleyTukeyStep + iElement + iCooleyTukeyStep/2].zw));


		fFFTData1[iGroup*iCooleyTukeyStep + iElement] = v2E11 + v2E12;
		fFFTData1[iGroup*iCooleyTukeyStep + iElement + iCooleyTukeyStep/2] = v2E11 - v2E12;

		fFFTData2[iGroup*iCooleyTukeyStep + iElement] = v4E21 + v4E22;
		fFFTData2[iGroup*iCooleyTukeyStep + iElement + iCooleyTukeyStep/2] = v4E21 - v4E22;

		fFFTData3[iGroup*iCooleyTukeyStep + iElement] = v4E31 + v4E32;
		fFFTData3[iGroup*iCooleyTukeyStep + iElement + iCooleyTukeyStep/2] = v4E31 - v4E32;

		fFFTData4[iGroup*iCooleyTukeyStep + iElement] = v4E41 + v4E42;
		fFFTData4[iGroup*iCooleyTukeyStep + iElement + iCooleyTukeyStep/2] = v4E41 - v4E42;

		barrier();
		memoryBarrierShared();
	}
}


//Writes data from shared memory into the image outputs. Parameter bShouldTranspose determines if the data
//should be transposed when written to the output buffers
void writeOutput(bool bShouldTranspose)
{
	ivec2 iv2Coord1, iv2Coord2;
	if(bShouldTranspose) 
	{
		iv2Coord1 = ivec2(gl_GlobalInvocationID.yx);
		iv2Coord2 = ivec2(gl_GlobalInvocationID.y, gl_GlobalInvocationID.x + gl_WorkGroupSize.x);
	}
	else
	{
		iv2Coord1 = ivec2(gl_GlobalInvocationID.xy);
		iv2Coord2 = ivec2(gl_GlobalInvocationID.x + gl_WorkGroupSize.x, gl_GlobalInvocationID.y);
	}

	imageStore(i2dOutput1, iv2Coord1, vec4(fFFTData1[gl_LocalInvocationID.x], vec2(0)));
	imageStore(i2dOutput1, iv2Coord2, 
		vec4(fFFTData1[gl_LocalInvocationID.x + gl_WorkGroupSize.x], vec2(0)));

	imageStore(i2dOutput2, iv2Coord1, fFFTData2[gl_LocalInvocationID.x]);
	imageStore(i2dOutput2, iv2Coord2, fFFTData2[gl_LocalInvocationID.x + gl_WorkGroupSize.x]);

	imageStore(i2dOutput3, iv2Coord1, fFFTData3[gl_LocalInvocationID.x]);
	imageStore(i2dOutput3, iv2Coord2, fFFTData3[gl_LocalInvocationID.x + gl_WorkGroupSize.x]);

	imageStore(i2dOutput4, iv2Coord1, fFFTData4[gl_LocalInvocationID.x]);
	imageStore(i2dOutput4, iv2Coord2, fFFTData4[gl_LocalInvocationID.x + gl_WorkGroupSize.x]);
}


//Computes wave number corresponding to the given element of the Phillips spectrum texture
vec2 getWaveNumber(uvec2 uv2TexCoords)
{
	uvec2 uv2GridSize = uvec2(2*gl_WorkGroupSize.x, gl_NumWorkGroups.y);
	ivec2 iv2WaveNumberIndex = ivec2(reverse_bits(uv2TexCoords, nSignificantBits));
	iv2WaveNumberIndex = ivec2(notEqual(iv2WaveNumberIndex, vec2(0)))*ivec2(uv2GridSize - iv2WaveNumberIndex);
	iv2WaveNumberIndex -= ivec2(greaterThanEqual(iv2WaveNumberIndex, ivec2(uv2GridSize)/2))*ivec2(uv2GridSize);
	return 2*pi*iv2WaveNumberIndex/v2DomainSize;
}


//Computes wave frequency corresponding to the given non-dimensional spatial position inside the water domain
float getFrequency(uvec2 uv2Position, float fk)
{
	uvec2 uv2GridSize = uvec2(2*gl_WorkGroupSize.x, gl_NumWorkGroups.y);
	vec2 v2NDPointPosition = uv2Position / (uv2GridSize - 1.0f);
	return sqrt(fGravityConstant*fk);
}




//Implements step 0 of the computation
void step0()
{
	//Compute wave numers corresponding to the current thread
	vec2 v2K1 = getWaveNumber(gl_GlobalInvocationID.xy);
	vec2 v2K2 = getWaveNumber(gl_GlobalInvocationID.xy + uvec2(gl_WorkGroupSize.x, 0));
	float fk1 = length(v2K1);
	float fk2 = length(v2K2);

	//Compute the wave frequencies corresponding to the current thread
	float fOmega1 = getFrequency(gl_GlobalInvocationID.xy, fk1);
	float fOmega2 = getFrequency(gl_GlobalInvocationID.xy + uvec2(gl_WorkGroupSize.x, 0), fk2);



	//Load currently processed FFT block into the shared memory
	vec4 v4PhillipsfSpectrum1 = texelFetch(s2dPhillipsSpectrum, ivec2(gl_LocalInvocationID.x,  gl_WorkGroupID.y), 0);
	vec4 v4PhillipsfSpectrum2 = texelFetch(s2dPhillipsSpectrum, ivec2(gl_LocalInvocationID.x + gl_WorkGroupSize.x,  gl_WorkGroupID.y), 0);
	float fC1_GS = cos(fOmega1*fTimeGlobalScale), fS1_GS = sin(fOmega1*fTimeGlobalScale);
	float fC2_GS = cos(fOmega2*fTimeGlobalScale), fS2_GS = sin(fOmega2*fTimeGlobalScale);
	float fC1_CS = cos(fOmega1*fTimeCapillaryScale), fS1_CS = sin(fOmega1*fTimeCapillaryScale);
	float fC2_CS = cos(fOmega2*fTimeCapillaryScale), fS2_CS = sin(fOmega2*fTimeCapillaryScale);



	vec2 v2TessendorfSpectrum1_GS = 
		complex_product(vec2(v4PhillipsfSpectrum1.rg), vec2(fC1_GS, fS1_GS)) + 
		complex_product(vec2(v4PhillipsfSpectrum1.ba), vec2(fC1_GS, -fS1_GS));
	vec2 v2TessendorfSpectrum2_GS = 
		complex_product(vec2(v4PhillipsfSpectrum2.rg), vec2(fC2_GS, fS2_GS)) + 
		complex_product(vec2(v4PhillipsfSpectrum2.ba), vec2(fC2_GS, -fS2_GS));

	vec2 v2TessendorfSpectrum1_CS = 
		complex_product(vec2(v4PhillipsfSpectrum1.rg), vec2(fC1_CS, fS1_CS)) + 
		complex_product(vec2(v4PhillipsfSpectrum1.ba), vec2(fC1_CS, -fS1_CS));
	vec2 v2TessendorfSpectrum2_CS = 
		complex_product(vec2(v4PhillipsfSpectrum2.rg), vec2(fC2_CS, fS2_CS)) + 
		complex_product(vec2(v4PhillipsfSpectrum2.ba), vec2(fC2_CS, -fS2_CS));



	vec4 v4NormalMapSpectra1_GS = 
		vec4(complex_product(v2K1.x * v2TessendorfSpectrum1_GS, vec2(0, 1)),
		complex_product(v2K1.y * v2TessendorfSpectrum1_GS, vec2(0, 1)));
	vec4 v4NormalMapSpectra2_GS = 
		vec4(complex_product(v2K2.x * v2TessendorfSpectrum2_GS, vec2(0, 1)),
		complex_product(v2K2.y * v2TessendorfSpectrum2_GS, vec2(0, 1)));

	vec4 v4NormalMapSpectra1_CS = 
		vec4(complex_product(v2K1.x * v2TessendorfSpectrum1_CS, vec2(0, 1)),
		complex_product(v2K1.y * v2TessendorfSpectrum1_CS, vec2(0, 1)));
	vec4 v4NormalMapSpectra2_CS = 
		vec4(complex_product(v2K2.x * v2TessendorfSpectrum2_CS, vec2(0, 1)),
		complex_product(v2K2.y * v2TessendorfSpectrum2_CS, vec2(0, 1)));



	vec4 v4SpatialDisplacementSpectra1 = vec4(0);
	if(fk1 != 0) v4SpatialDisplacementSpectra1 = -v4NormalMapSpectra1_GS / fk1 * fChoppiness;

	vec4 v4SpatialDisplacementSpectra2 = vec4(0);
	if(fk2 != 0) v4SpatialDisplacementSpectra2 = -v4NormalMapSpectra2_GS / fk2 * fChoppiness;




	fFFTData1[gl_LocalInvocationID.x] = v2TessendorfSpectrum1_GS;
	fFFTData1[gl_LocalInvocationID.x + gl_WorkGroupSize.x] = v2TessendorfSpectrum2_GS;

	fFFTData2[gl_LocalInvocationID.x] = v4SpatialDisplacementSpectra1;
	fFFTData2[gl_LocalInvocationID.x + gl_WorkGroupSize.x] = v4SpatialDisplacementSpectra2;

	fFFTData3[gl_LocalInvocationID.x] = v4NormalMapSpectra1_GS;
	fFFTData3[gl_LocalInvocationID.x + gl_WorkGroupSize.x] = v4NormalMapSpectra2_GS;

	fFFTData4[gl_LocalInvocationID.x] = v4NormalMapSpectra1_CS;
	fFFTData4[gl_LocalInvocationID.x + gl_WorkGroupSize.x] = v4NormalMapSpectra2_CS;

	barrier();
	memoryBarrierShared();


	//Perform Cooley-Tukey steps on the shared memory blocks
	makeCooleyTukeyPass();


	//Write intermediate data to the image outputs
	writeOutput(true);
}


//Implements step 1 of the computation
void step1()
{
	//Load data into shared memory, but now using what has been stored into the images
	fFFTData1[gl_LocalInvocationID.x] = imageLoad(i2dOutput1, ivec2(gl_GlobalInvocationID.xy)).rg;
	fFFTData1[gl_LocalInvocationID.x + gl_WorkGroupSize.x] = imageLoad(i2dOutput1, ivec2(gl_GlobalInvocationID.xy) + ivec2(gl_WorkGroupSize.x, 0)).rg;

	fFFTData2[gl_LocalInvocationID.x] = imageLoad(i2dOutput2, ivec2(gl_GlobalInvocationID.xy));
	fFFTData2[gl_LocalInvocationID.x + gl_WorkGroupSize.x] = imageLoad(i2dOutput2, ivec2(gl_GlobalInvocationID.xy) + ivec2(gl_WorkGroupSize.x, 0));

	fFFTData3[gl_LocalInvocationID.x] = imageLoad(i2dOutput3, ivec2(gl_GlobalInvocationID.xy));
	fFFTData3[gl_LocalInvocationID.x + gl_WorkGroupSize.x] = imageLoad(i2dOutput3, ivec2(gl_GlobalInvocationID.xy) + ivec2(gl_WorkGroupSize.x, 0));

	fFFTData4[gl_LocalInvocationID.x] = imageLoad(i2dOutput4, ivec2(gl_GlobalInvocationID.xy));
	fFFTData4[gl_LocalInvocationID.x + gl_WorkGroupSize.x] = imageLoad(i2dOutput4, ivec2(gl_GlobalInvocationID.xy) + ivec2(gl_WorkGroupSize.x, 0));

	barrier();
	memoryBarrierShared();


	//Perform Cooley-Tukey steps on the shared memory blocks
	makeCooleyTukeyPass();


	//Write final results to the image outputs
	writeOutput(false);
}



void main()
{
	switch(uiStep)
	{
		case 0: 
		step0();
		break;

		case 1:
		step1();
		break;
	}
}
