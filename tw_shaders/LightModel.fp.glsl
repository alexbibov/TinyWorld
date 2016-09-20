#version 430 core

//This fragment program implements per-pixel lighting models


//Environment map types
#define EnvironmentMapTypeSpehrical			0
#define EnvironmentMapTypeCubic				1
#define EnvironmentMapTypeEquirectangular	2




//*********************************************Parameters related to the lit object******************************************
uniform bool bLightingEnabled;	//equals 'true' if lighting is enabled for this object
uniform vec3 v3ViewerLocation;	//position of the viewer in global space
uniform mat3 m3ViewerRotation;	//rotation component of the transform converting scaled object space into the viewer space
uniform mat4 m4LightTransform;	//light transformation matrix (transforms lights from the global to scaled model space)
//***************************************************************************************************************************


//************************************************Object material properties*************************************************
uniform bool bHasNormalMap;			//equals 'true' if object uses bump-mapping
uniform bool bNormalMapEnabled;		//enable state of the normal maps

uniform bool bHasSpecularMap;		//equals 'true' if object has a specular map
uniform bool bSpecularMapEnabled;	//enable state of the specular maps

uniform bool bHasEmissionMap;		//equals 'true' if object has an emission map
uniform bool bEmissionMapEnabled;	//enable state of the emission maps

uniform bool bSupportsArrayNormalMaps;		//equals 'true' if bump-mapping is implemented using array textures
uniform bool bSupportsArraySpecularMaps;		//equals 'true' if specular reflection is described by array specular map
uniform bool bSupportsArrayEmissionMaps;		//equals 'true' if emission can be described by an array texture
uniform bool bSupportsArrayEnvironmentMaps;		//equals 'true' if environment mapping is implemented by an array texture

uniform bool bHasEnvironmentMap;	//equals 'true' if object has an environment map
uniform bool bEnvironmentMapEnabled;	//equals 'true' if environment map is enabled
uniform uint uiEnvironmentMapType;	//determines type of the currently used environment map

uniform sampler2D s2dNormalMap;		//normal map sampler
uniform sampler2D s2dSpecularMap;		//specular map sampler
uniform sampler2D s2dEmissionMap;		//emission map sampler
uniform sampler2D s2dEnvironmentMap;	//2D environment map sampler
uniform samplerCube scEnvironmentMap;	//cube environment map sampler

uniform sampler2DArray s2daNormalArrayMap;		//used instead of s2dNormalMap if bSupportsArrayNormalMaps = true
uniform sampler2DArray s2daSpecularArrayMap;	//used instead of s2dSpecularMap if bSupportsArraySpecularMaps = true
uniform sampler2DArray s2daEmissionArrayMap;	//used instead of s2dEmissionMap if bSupportsArrayEmissionMaps = true
uniform sampler2DArray s2daEnvironmentArrayMap;	//used instead of s2dEnvironmentMap if bSupportsArrayEnvironmentMaps = true
uniform samplerCubeArray scaEnvironmentArrayMap; //used instead of scEnvironmentMap if bSupportsArrayEnvironmentMaps = true

uniform vec4 v4DefaultDiffuseColor;			//default diffuse colour of the object
uniform float fDefaultSpecularExponent;		//default specular exponent of the object
uniform vec3 v3DefaultSpecularColor;		//default specular colour of the object
uniform vec3 v3DefaultEmissionColor;		//default emission colour
//***************************************************************************************************************************


//************************************Subroutine types used by environment mapping*******************************************
subroutine vec3 EnvironmentMapSampleRetriever(sampler2D s2dEnvironmentMap, vec3 v3ReflectionVector);
subroutine vec3 ArrayEnvironmentMapSampleRetriever(sampler2DArray s2daEnvironmentMap, vec3 v3ReflectionVector, uint layer);
subroutine vec3 CubeEnvironmentMapSampleRetriever(samplerCube scEnvironmentMap, vec3 v3ReflectionVector);
subroutine vec3 CubeArrayEnvironmentMapSampleRetriever(samplerCubeArray scaEnvironmentMap, vec3 v3ReflectionVector, uint layer);
subroutine vec3 ReflectionMapper(vec3 v3IncidentVector, vec3 v3Normal);
//***************************************************************************************************************************


//***********************************Subroutine types used to sample values from textures************************************
subroutine vec3 ProceduralNormalMapSampleRetriever(vec2 v2TexCoords, float fArrayLayer);
subroutine vec3 ProceduralSpecularMapSampleRetriever(vec2 v2TexCoords, float fArrayLayer);
subroutine vec3 ProceduralEmissionMapSampleRetriever(vec2 v2TexCoords, float fArrayLayer);  
//***************************************************************************************************************************


//***********************************Subroutine uniforms used by environment mapping*****************************************
subroutine uniform EnvironmentMapSampleRetriever funcEnvironmentMap;
subroutine uniform CubeEnvironmentMapSampleRetriever funcCubeEnvironmentMap;
subroutine uniform ArrayEnvironmentMapSampleRetriever funcArrayEnvironmentMap;
subroutine uniform CubeArrayEnvironmentMapSampleRetriever funcCubeArrayEnvironmentMap;
subroutine uniform ReflectionMapper funcReflection;
//***************************************************************************************************************************


//***************************************Subroutine uniforms used by texture samplers****************************************
subroutine uniform ProceduralNormalMapSampleRetriever funcNormalMap;
subroutine uniform ProceduralSpecularMapSampleRetriever funcSpecularMap;
subroutine uniform ProceduralEmissionMapSampleRetriever funcEmissionMap;
//***************************************************************************************************************************




#include "common/LightBuffer.include.glsl"	//Include definition of the light buffer
#include "common/SphericalEnvironmentMap.include.glsl"	//include sample retriever for spherical environment maps
#include "common/EquirectangularEnvironmentMap.include.glsl"	//include sample retriever for equirectangular environment maps
#include "common/CubicEnvironmentMap.include.glsl"	//include sample retriever for cubic environment maps


//Interface block that declares extra output data from the vertex shader
in LIGHT_MODEL_VERTEX_DATA
{
    //Normal vector represented in the viewer space
    //vec3 v3Normal;

    //Linear depth of the vertex represented in the viewer space
    float fLinearDepth;	

    //Tangent, bi-normal, and normal vectors corresponding to the current vertex
    vec3 v3T, v3B, v3N;

    //Position of a vertex being lit in scaled object space
    vec3 v3VertexLocation_SOS;

    //Relative location of the viewer represented in scaled object space
    vec3 v3ViewerRelativeLocation_SOS;
}light_model_vertex_data;


//Screen-space normal map output
layout(location = 2) out vec3 v3Normal;

//Linear depth output
layout(location = 3) out float fLinearDepth;

//"Unlit" fragment output
layout(location = 4) out vec3 v3AD;

//Global parameter containing unit vector pointing from position of the viewer towards the current fragment.
//This vector is represented in the scaled object space.
vec3 v3IncidentVector;


//Default subroutine used to compute reflection vectors
subroutine(ReflectionMapper) vec3 DefaultReflectionMapper(vec3 v3IncidentVector, vec3 v3Normal) 
{ 
    return reflect(v3IncidentVector, v3Normal);
}


//Default subroutine used to retrieve values from normal maps
subroutine(ProceduralNormalMapSampleRetriever) vec3 DefaultNormalMapSampleRetriever(vec2 v2TexCoords, float fArrayLayer)
{
    vec3 v3NormalVector;
    if(bSupportsArrayNormalMaps)
    {
        vec3 v3NormalVector1 = texture(s2daNormalArrayMap, vec3(v2TexCoords, floor(fArrayLayer))).rgb * 2.0f - 1.0f;
        vec3 v3NormalVector2 = texture(s2daNormalArrayMap, vec3(v2TexCoords, ceil(fArrayLayer))).rgb * 2.0f - 1.0f;
        v3NormalVector = mix(v3NormalVector1, v3NormalVector2, fract(fArrayLayer));
    }
    else
    {
        v3NormalVector = texture(s2dNormalMap, v2TexCoords).rgb * 2.0f - 1.0f;
    }
        
    return(normalize(v3NormalVector));
}


//Default subroutine used to retrieve values from specular maps
subroutine(ProceduralSpecularMapSampleRetriever) vec3 DefaultSpecularMapSampleRetriever(vec2 v2TexCoords, float fArrayLayer)
{
    vec3 v3SpecularModulation;

    if(bSupportsArraySpecularMaps)
    {
        vec3 v3SpecularModulation1 = texture(s2daSpecularArrayMap, vec3(v2TexCoords, floor(fArrayLayer))).rgb;
        vec3 v3SpecularModulation2 = texture(s2daSpecularArrayMap, vec3(v2TexCoords, ceil(fArrayLayer))).rgb;
        v3SpecularModulation = mix(v3SpecularModulation1, v3SpecularModulation2, fract(fArrayLayer));
    }
    else
    {
        v3SpecularModulation = texture(s2dSpecularMap, v2TexCoords).rgb;
    }

    return v3SpecularModulation;
}


//Default subroutine used to retrieve values from emission maps
subroutine(ProceduralEmissionMapSampleRetriever) vec3 DefaultEmissionMapSampleRetriever(vec2 v2TexCoords, float fArrayLayer)
{
    vec3 v3EmissionModulation;

    if(bSupportsArrayEmissionMaps)
    {
        vec3 v3EmissionModulation1 = texture(s2daEmissionArrayMap, vec3(v2TexCoords, floor(fArrayLayer))).rgb;
        vec3 v3EmissionModulation2 = texture(s2daEmissionArrayMap, vec3(v2TexCoords, floor(fArrayLayer))).rgb;
        v3EmissionModulation = mix(v3EmissionModulation1, v3EmissionModulation2, fract(fArrayLayer));
    }
    else
    {
        v3EmissionModulation = texture(s2dEmissionMap, v2TexCoords).rgb;
    }

    return v3EmissionModulation;
}







//Helper function, which returns 'true' if supplied environment map type is based on cube maps
bool isCubicEnvironmentMap(uint uiEnvironmentMapType)
{
    switch(uiEnvironmentMapType)
    {
        case EnvironmentMapTypeCubic:
            return true;

        default: return false;
    }
}


//This function computes colour component of the current fragment, which appears due to the presence of light sources
//The return value of this function should be assigned to the final colour of the fragment.
//DETAILED DESCRIPTION:
//v4DiffuseColor — main colour of the object. This might be a colour sampled from texture or some other procedurally generated value.
//v2NormalMapTexCoords — texture coordinates used by bump-mapping look-ups. This input is ignored if object does not implement bump-mapping.
//v2SpecularMapTexCoords — texture coordinates used by specular mapping look-ups. This input is ignored if object does not implement specular mapping.
//v2EmissionMapTexCoords — texture coordinates used by emission mapping look-ups. This input is ignored if object does not implement emission mapping.
//fNormalMapLayer, fSpecularMapLayer, fEmissionMapLayer, fEnvironmentMapLayer — if either normal, specular, emission or environment maps are represented by array textures, these inputs 
//declare which layer to sample from. If a map is not represented by an array texture, the corresponding f*MapLayer parameter is ignored.

mat3 m3TBN;	//transforms coordinates from scaled object space to tangent space

vec4 computeLightContribution(vec4 v4DiffuseColor, vec2 v2NormalMapTexCoords, vec2 v2SpecularMapTexCoords, vec2 v2EmissionMapTexCoords, 
        float fNormalMapLayer, float fSpecularMapLayer, float fEmissionMapLayer, float fEnvironmentMapLayer)
{
    //Store linear depth value of the fragment
    fLinearDepth = light_model_vertex_data.fLinearDepth;


    //Dissect light transform
    mat3 m3LightTransform3D = mat3(m4LightTransform[0].xyz, m4LightTransform[1].xyz, m4LightTransform[2].xyz);
    vec3 v3LightTransformShift = m4LightTransform[3].xyz;


    //Account for ambient lighting
    vec3 v3Result = v4DefaultDiffuseColor.rgb * v4DiffuseColor.rgb * light_buffer.v3AmbientLightIntensity;
    v3AD = v3Result;	//write data to the AD-map


    //Compute TBN transform
    float fT_norm;
    float fB_norm;
    float fN_norm = length(light_model_vertex_data.v3N);

    vec3 v3T = light_model_vertex_data.v3T;
    vec3 v3B = light_model_vertex_data.v3B;
    vec3 v3N = light_model_vertex_data.v3N;

    v3N = fN_norm > 0 ? v3N / fN_norm : vec3(0, 0, 1);

    v3T = v3T - dot(v3T, v3N)*v3N;
    fT_norm = length(light_model_vertex_data.v3T); 
    v3T = fT_norm > 0 ? v3T / fT_norm : vec3(1, 0, 0);

    v3B = v3B - dot(v3B, v3T)*v3T - dot(v3B, v3N)*v3N;
    fB_norm = length(light_model_vertex_data.v3B);
    v3B = fB_norm > 0 ? v3B / fB_norm : vec3(0, 1, 0);

    m3TBN = transpose(mat3(v3T, v3B, v3N));


    //Extract viewer's location
    vec3 v3ViewerRelativeLocation_T = normalize(m3TBN * light_model_vertex_data.v3ViewerRelativeLocation_SOS);

    
    //Extract normal vector
    vec3 v3NormalVector;
    if(bHasNormalMap && bNormalMapEnabled)
    {
        v3NormalVector = funcNormalMap(v2NormalMapTexCoords, fNormalMapLayer);
    }
    else
    {
        v3NormalVector = vec3(0.0f, 0.0f, 1.0f);
    }
    //Store screen-space normal map fragment
    v3Normal = m3ViewerRotation*transpose(m3TBN)*v3NormalVector;
    //v3Normal = normalize(light_model_vertex_data.v3Normal);

    
    //Extract specular modulation
    vec3 v3SpecularModulation;
    if(bHasSpecularMap && bSpecularMapEnabled)
    {
        v3SpecularModulation = funcSpecularMap(v2SpecularMapTexCoords, fSpecularMapLayer);
    }
    else
    {
        v3SpecularModulation = vec3(1.0f);	//no specular modulation
    }
        
        
    //Extract emission modulation factor
    vec3 v3EmissionModulation;
    if(bHasEmissionMap && bEmissionMapEnabled)
    {
        v3EmissionModulation = funcEmissionMap(v2EmissionMapTexCoords, fEmissionMapLayer);
    }
    else
        v3EmissionModulation = vec3(1.0f);




    //Account for directional lighting
    uint uiNumDirectionalLights = bLightingEnabled ?  light_buffer.nDirectionalLights : 0;
    for(uint i = 0; i < uiNumDirectionalLights; ++i)
    {
        //Compute directional light transformed direction
        vec3 v3LightDirection_T = m3TBN * m3LightTransform3D * light_buffer.v3DirectionalLightDirections[i];
        
        //Compute diffuse component
        float fDiffuseIntensity = max(dot(v3NormalVector, -v3LightDirection_T), 0.0f);
        v3Result += v4DefaultDiffuseColor.rgb * v4DiffuseColor.rgb * (fDiffuseIntensity * light_buffer.v3DirectionalLightIntensities[i]);
        
        //Compute specular component
        //vec3 R = -reflect(-DLTDir, normal_vector);
        //float specular_intensity = max(dot(R, VTLoc), 0.0f);
        vec3 v3H = normalize(-v3LightDirection_T + v3ViewerRelativeLocation_T);
        float fSpecularIntensity = max(dot(v3H, v3NormalVector), 0.0f);
        fSpecularIntensity = pow(fSpecularIntensity, fDefaultSpecularExponent);
        
        if(dot(-v3LightDirection_T, v3NormalVector) <= 0) fSpecularIntensity = 0;
        v3Result += v3DefaultSpecularColor * v3SpecularModulation * (fSpecularIntensity * light_buffer.v3DirectionalLightIntensities[i]);
    }
    
    
    //Account for point light sources
    uint uiNumPointLights = bLightingEnabled ? light_buffer.nPointLights : 0;
    for(uint i = 0; i < uiNumPointLights; ++i)
    {
        //Compute light source attenuation
        vec3 v3LightRelativeLocation_T = 
            m3TBN * (m3LightTransform3D * light_buffer.v3PointLightLocations[i] + v3LightTransformShift - 
            light_model_vertex_data.v3VertexLocation_SOS);
        float fD = length(v3LightRelativeLocation_T);
        float fAttenuation = 1.0f / 
        (light_buffer.v3PointLightAttenuationFactors[i].x + 
        light_buffer.v3PointLightAttenuationFactors[i].y * fD + 
        light_buffer.v3PointLightAttenuationFactors[i].z * fD * fD);
    
        
        //Extract point light transformed relative location
        vec3 v3LightRelativeDirection_T = normalize(v3LightRelativeLocation_T);
    
        //Compute diffuse component 
        float fDiffuseIntensity = max(dot(v3NormalVector, v3LightRelativeDirection_T), 0.0f);
        v3Result += v4DefaultDiffuseColor.rgb * v4DiffuseColor.rgb * (fDiffuseIntensity * (light_buffer.v3PointLightIntensities[i] * fAttenuation));
        
        //Compute specular component
        //vec3 R = -reflect(PLTLoc, normal_vector);
        //float specular_intensity = max(dot(R, VTLoc), 0.0f);
        vec3 v3H = normalize(v3LightRelativeDirection_T + v3ViewerRelativeLocation_T);
        float fSpecularIntensity = max(dot(v3H, v3NormalVector), 0.0f);
        fSpecularIntensity = pow(fSpecularIntensity, fDefaultSpecularExponent);
        
        if(dot(v3LightRelativeDirection_T, v3NormalVector) <= 0) fSpecularIntensity = 0;
        v3Result += v3DefaultSpecularColor * v3SpecularModulation * (fSpecularIntensity * (light_buffer.v3PointLightIntensities[i] * fAttenuation));
    }
    
    
    //Account for spot light sources
    uint uiNumSpotLights = bLightingEnabled ? light_buffer.nSpotLights : 0;
    for(uint i = 0; i < uiNumSpotLights; ++i)
    {
        //Extract spot light transformed direction
        vec3 v3LightDirection_T = m3TBN * m3LightTransform3D * light_buffer.v3SpotLightDirections[i];
        
        //Extract spot light transformed location
        vec3 v3LightRelativeLocation_T = 
            m3TBN * (m3LightTransform3D * light_buffer.v3SpotLightLocations[i] + v3LightTransformShift - 
            light_model_vertex_data.v3VertexLocation_SOS);
        vec3 v3LightRelativeDirection_T = normalize(v3LightRelativeLocation_T);
    
        //Compute light source attenuation
        float fD = length(v3LightRelativeLocation_T);
        float fAttenuation = max(dot(-v3LightDirection_T, v3LightRelativeDirection_T), 0.0f);
        fAttenuation = pow(fAttenuation, light_buffer.fSpotLightExponents[i]);
        fAttenuation /= 
        (light_buffer.v3SpotLightAttenuationFactors[i].x + 
        light_buffer.v3SpotLightAttenuationFactors[i].y * fD + 
        light_buffer.v3SpotLightAttenuationFactors[i].z * fD * fD);
    
        //Compute diffuse component
        float fDiffuseIntensity = max(dot(v3NormalVector, v3LightRelativeDirection_T), 0.0f);
        v3Result += v4DefaultDiffuseColor.rgb * v4DiffuseColor.rgb * (fDiffuseIntensity * (light_buffer.v3SpotLightIntensities[i] * fAttenuation));
        
        //Compute specular component
        //vec3 R = -reflect(SLTLoc, normal_vector);
        //float specular_intensity = max(dot(R, VTLoc), 0.0f);
        vec3 v3H = normalize(v3LightRelativeDirection_T + v3ViewerRelativeLocation_T);
        float fSpecularIntensity = max(dot(v3H, v3NormalVector), 0.0f);
        fSpecularIntensity = pow(fSpecularIntensity, fDefaultSpecularExponent);
        
        if(dot(v3LightRelativeDirection_T, v3NormalVector) <= 0) fSpecularIntensity = 0.0f;
        v3Result += v3DefaultSpecularColor * v3SpecularModulation * (fSpecularIntensity * (light_buffer.v3SpotLightIntensities[i] * fAttenuation));
    }
    
    v3Result += v3DefaultEmissionColor * v3EmissionModulation;


    //Incorporate reflection effects
    v3IncidentVector = -normalize(light_model_vertex_data.v3ViewerRelativeLocation_SOS);
    if(bHasEnvironmentMap && bEnvironmentMapEnabled)
    {
        vec3 v3ReflectionVector = transpose(m3LightTransform3D)*funcReflection(v3IncidentVector, transpose(m3TBN) * v3NormalVector);
        vec3 v3ReflectionSample;
        if(bSupportsArrayEnvironmentMaps)
            if(isCubicEnvironmentMap(uiEnvironmentMapType))
            {
                vec3 v3ReflectionSample1 = funcCubeArrayEnvironmentMap(scaEnvironmentArrayMap, v3ReflectionVector, uint(floor(fEnvironmentMapLayer)));
                vec3 v3ReflectionSample2 = funcCubeArrayEnvironmentMap(scaEnvironmentArrayMap, v3ReflectionVector, uint(ceil(fEnvironmentMapLayer)));
                v3ReflectionSample = mix(v3ReflectionSample1, v3ReflectionSample2, fract(fEnvironmentMapLayer));
            }
            else
            {
                vec3 v3ReflectionSample1 = funcArrayEnvironmentMap(s2daEnvironmentArrayMap, v3ReflectionVector, uint(floor(fEnvironmentMapLayer)));
                vec3 v3ReflectionSample2 = funcArrayEnvironmentMap(s2daEnvironmentArrayMap, v3ReflectionVector, uint(ceil(fEnvironmentMapLayer)));
                v3ReflectionSample = mix(v3ReflectionSample1, v3ReflectionSample2, fract(fEnvironmentMapLayer));
            }
        else
            if(isCubicEnvironmentMap(uiEnvironmentMapType))
                v3ReflectionSample = funcCubeEnvironmentMap(scEnvironmentMap, v3ReflectionVector);
            else
                v3ReflectionSample = funcEnvironmentMap(s2dEnvironmentMap, v3ReflectionVector);

        v3Result += v3ReflectionSample;
    }

    
    return vec4(v3Result, v4DefaultDiffuseColor.a * v4DiffuseColor.a);
}
