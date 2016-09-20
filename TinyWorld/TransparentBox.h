//This object implements the concept of a "transperent box filled with light-scattering points". This is an important concept, which
//allows to perform volumetric rendering, and simulate clouds of smoke, atmospheric clouds, blasts, etc. It is important to understand,
//that transparent boxes make use of a non-Phong lighting model and therefore are not inherited from AbstractRenderableObjectLit.
//Instead they implement their own light system based on a forward multiple scattering model. For optimization reasons it is a good idea
//to register just a small subset of lights used for the Phong illumination to be taken into account for shading of the transparent boxes.
//In addition, the position of light is irrelevant for the transparent boxes as it is assumed that the light travels in vacuum before it hits the medium.


#ifndef TW__TRANSPARENT_BOX__


//Transparent boxes do not support regular rendering modes. Instead they have two of their own
#define TW_RENDERING_MODE_RAY_CAST_GAS				10		//shading is implemented by a "gas-like" light scattering algorithm based on ray casting approach
#define TW_RENDERING_MODE_RAY_CAST_ABSORBENT		11		//shading is implemented using simple emission-absorption model and ray-casting
#define TW_RENDERING_MODE_PROXY_GEOMETRY_ABSORBENT	12		//shading is implemented using emission-absorption model and proxy geometry blending


#include "AbstractRenderableObject.h"
#include "AbstractRenderableObjectHDRBloomEx.h"
#include "Light.h"
#include "ImmutableTexture3D.h"
#include "CompleteShaderProgram.h"
#include "Framebuffer.h"
#include "SSFilter_HDRBloom.h"

#include <list>

namespace tiny_world
{

	class TransparentBox : public AbstractRenderableObjectTextured, public AbstractRenderableObjectExtensionAggregator<AbstractRenderableObjectHDRBloomEx>
	{
	private:
		static const std::string rendering_program0_name;		//name of the rendering program implementing volume rendering algorithms based on ray casting
		static const std::string rendering_program1_name;		//name of the shading program that implements light attenuation for TW_RENDERING_MODE_PROXY_GEOMETRY_ABSORBENT
		static const std::string rendering_program2_name;		//name of the shading program that implements eye blend-in rendering pass for TW_RENDERING_MODE_PROXY_GEOMETRY_ABSORBENT

		float width, height, depth;		//width, height and depth of the transparent box
		std::list<const DirectionalLight*> light_source_directions;	//list of directions to the light sources taken into account during shading of the transparent box
		vec3 v3AverageLightDirection;	//average direction of the light sources that have been added to the object's list. This value is used by rendering mode TW_RENDERING_MODE_PROXY_GEOMETRY_ABSORBENT
		vec3 v3CumulativeLightColor;	//cumulative color of the light sources that have been added to the list maintained by the object. This value is used by rendering mode TW_RENDERING_MODE_PROXY_GEOMETRY_ABSORBENT

		int num_primary_samples_old;	//this variable is needed to detect changes in number of primary samples used by the object. The initial value should always be equal to -1
		uint32_t num_primary_samples;	//number of samples to be taken along each ray cast toward the observer or number of cutting planes that form proxy geometry
		uint32_t num_secondary_samples;	//number of samples to be taken along each ray cast toward each of the light sources. Note that this setting has effect only for optical models based on ray casting
		float solid_angle;		//solid angle, within which optical model approximates forward scattering. This value only has effect for rendering mode TW_RENDERING_MODE_RAY_CAST_GAS
		

		//The following variables are not initialized during creation of the object. Their values are getting updated by applyViewProjectionTransform(...)

		bool is_viewer_inside;	//equals 'true' if viewer is located inside of the transparent box. Equals 'false' otherwise.
		vec3 v3ViewerLocation;	//location of the viewer represented in world space coordinates
		vec3 v3ViewerDirection;	//direction of viewer's sight represented in world space coordinates
		vec3 v3ViewerUpVector;	//"up vector" of the viewer


		//The following variables should be set to nullptr on object initialization

		uint32_t* p_vertex_binding_offsets;	//memory offsets within vertex buffer for the data corresponding to each of the cutting planes that form proxy geometry
		uint32_t* p_num_slice_vertices;		//stores number of generic vertices that form certain slice in proxy geometry


		//*********************************************************Raw OpenGL resources********************************************************
		GLuint ogl_vertex_attribute_object0;	//OpenGL identifier of vertex attribute object used by rendering modes based on ray casting
		GLuint ogl_vertex_buffer_object0;		//OpenGL identifier of vertex buffer object used by rendering modes based on ray casting
		GLuint ogl_index_buffer_object0;		//OpenGL identifier of index buffer object used by rendering modes based on ray casting

		GLuint ogl_vertex_attribute_object1;	//OpenGL identifier of vertex attribute object used by rendering modes based on proxy geometry
		GLuint ogl_vertex_buffer_object1;		//OpenGL identifier of vertex buffer object used by rendering modes based on proxy geometry
		//*************************************************************************************************************************************

		TextureSamplerReferenceCode medium_sampler_ref_code;		//sampler, which is used to interpolate media contained in the transparent box
		int should_use_rgb_channel;			//nonzero if RGB-channel of the 3D texture should contribute to the final color of the fragment
		vec3 medium_color;					//specifies "uniform color" of the media
		int should_use_colormap;			//nonzero if shading should use color maps

		//Reference code of the volumetric texture representing media contained in the transparent box. We assume that albedo of the media
		//is stored in the alpha-channel of the texture. The colors are as usual represented by the RGB-channel.
		TextureReferenceCode medium_texture_ref_code;

		//Reference code of the color map that can be enabled in order to "emphasize" certain features of the volumetric data
		TextureReferenceCode colormap_texture_ref_code;

		//Reference code and alias object of the texture used to accumulate attenuated light values
		ImmutableTexture2D light_buffer_texture;
		TextureReferenceCode light_buffer_texture_ref_code;

		//Alias object of the texture containing eye buffer data
		ImmutableTexture2D eye_buffer_texture;

		//Alias object of the texture containing bloom output of the eye buffer
		ImmutableTexture2D bloom_texture;

		//Alias object of the depth texture used by the framebuffer objects
		ImmutableTexture2D depth_texture;

		ShaderProgramReferenceCode rendering_program0_ref_code;		//shading program implementing algorithms based on ray casting
		ShaderProgramReferenceCode rendering_program1_ref_code;		//shading program implementing the light attenuation pass for TW_RENDERING_MODE_PROXY_GEOMETRY_ABSORBENT
		ShaderProgramReferenceCode rendering_program2_ref_code;		//shading program that implements eye blend-in pass for TW_RENDERING_MODE_PROXY_GEOMETRY_ABSORBENT

		uint32_t current_rendering_pass;	//current rendering pass of the rendering
		OrthogonalProjectingDevice lightview_projection;	//orthogonal projecting device describing directional light volume projection
		const AbstractProjectingDevice* p_renderer_projection;	//pointer to an instance of projecting device describing view and projection transforms used by the renderer
		AbstractRenderingDevice* p_render_target;		//rendering target where to draw the object


		//Render targets used to internally render the object
		Framebuffer eye_buffer;		//eye buffer
		Framebuffer light_buffer;	//light buffer


		//Full-screen rectangle needed to blend the final image into the scene
		SSFilter_HDRBloom canvas_filter;


		//Reference counter needed to monitor usage of resources, which might be shared by multiple instances of the object
		uint32_t* p_ref_counter;


		//Applies changes to the render target screen size
		void applyScreenSize(const uvec2& screen_size) override;
		bool configureRendering(AbstractRenderingDevice& render_target, uint32_t rendering_pass) override;
		void configureViewProjectionTransform(const AbstractProjectingDevice& projecting_device) override;
		bool configureRenderingFinalization() override;


		//Computes intersection between the given plane and the section AB. Returns pair with the first element equal to 'true' if such intersection has been found and the second element containing Cartesian coordinates of the
		//intersection. If the plane has no common points with the section or the whole section lies within the plane, the function returns pair with the first element equal to 'false' and the second element undefined
		inline std::pair<bool, vec3> compute_intersection_point(const vec4& Plane, const vec3& A, const vec3& B);

		void generate_proxy_geometry();	//computes proxy geometry based on location of the viewer and average light direction (proxy-geometry based volumetric rendering does not support multiple light sources)
		void update_vertex_data();		//updates vertex data based on the current settings for dimensions of the transparent box
		void init_transparent_box();	//takes care of initialization particulars
		void free_ogl_resources();	//releases raw OpenGL resources owned by the object

		//Helper functions for rendering modes based on ray tracing

		inline bool configureRendering_RT(AbstractRenderingDevice& render_target, uint32_t rendering_pass);
		inline bool render_RT();
		inline bool finalizeRendering_RT();


		//Helper functions for rendering modes based on proxy geometry

		inline bool configureRendering_PG(uint32_t rendering_pass);
		inline bool render_PG();	
		inline bool finalizeRendering_PG();

	public:
		//Constructor-Destructor infrastructure of the object:

		//Initializes new transparent box using given values for its dimensions
		TransparentBox(float width = 1.0f, float height = 1.0f, float depth = 1.0f);	

		//Initializes new transparent box using provided string name for weak identification and the given values for dimensions
		TransparentBox(std::string transparent_box_string_name, float width = 1.0f, float height = 1.0f, float depth = 1.0f);

		//Copy constructor
		TransparentBox(const TransparentBox& other);

		//Move constructor
		TransparentBox(TransparentBox&& other);

		//Destructor
		~TransparentBox();

		//Copy assignment operator
		TransparentBox& operator=(const TransparentBox& other);

		//Move assignment operator
		TransparentBox& operator=(TransparentBox&& other);


		//"Setters" and "getters" allowing to modify state of the object:

		void setDimensions(float width, float height, float depth);		//assigns new values for dimensions of the transparent box

		//Assigns new values for dimensions of the transparent box from a 3D vector. The meanings of components of the 3D vector are (width, height, depth)
		void setDimensions(const vec3& new_dimensions);	

		//Assigns new directional light to be taken into account when shading the transparent box geometry. Returns 'true' on success
		bool addLightSourceDirection(const DirectionalLight& directional_light);

		//Removes light source from object's registry based on the light's strong identifier. Returns 'true' on success
		bool removeLightSourceDirection(uint32_t directional_light_id);

		//Removes light source from object's registry based on the light's string name. If there are several lights sharing the same string name attached to the 
		//object, only the first one found that has the requested string name will be removed. The function returns 'true' on success and false if the light having
		//requested string name has not been found in the object's registry
		bool removeLightSourceDirection(std::string directional_light_string_name);

		//Removes all light sources for object's registry
		void removeAllLightSources();

		//Sets number of samples to be taken along each ray cast towards the observer
		void setNumberOfPrimarySamples(uint32_t nsamples);

		//Sets number of samples to be taken along each ray cast towards the light source(s) for approximation of scattering effects.
		//Note that this function has effect only for the optical modes that support light scattering
		void setNumberOfSecondarySamples(uint32_t nsamples);

		//Defines solid angle size in steradians for forward multiple scattering approximation
		void setSolidAngle(float fstangle);

		//Installs 3D texture, which simulates the medium contained in the transparent box
		void installPointCloud(const ImmutableTexture3D& _3d_point_cloud);

		//Defines color map 1D look-up texture to be used during shading.
		//Note that usage of the color map should be switched on separately
		void installColormap(const ImmutableTexture1D& colormap);

		//Specifies if RGB-channel of the 3D texture should contribute to shading of the transparent box
		void useRGBChannel(bool enable_rgb_scattering_state);

		//Specifies whether color map texture should be used to highlight "areas of interest" in the 3D volume.
		//Note that by default color map is undefined
		void useColormap(bool enable_colormap_state);

		//Sets "uniform color" of the media. Default value is (1, 1, 1). If RGB-channel contributes to the shading, the value set by this function is ignored
		void setMediumUniformColor(const vec3& color);


		vec3 getDimensions() const;		//returns valid dimensions of the transparent box as a 3D vector composed as (width, height, depth)

		uint32_t getNumberOfSamples() const;	//returns number of samples taken along each ray during shading

		float getSolidAngle() const;	//returns value of solid angle, within which resides the forward scattering model

		bool isRGBChannelInUse() const;	//Returns 'true' if the optical model takes into account RGB-channel of the 3D-texture

		bool isColormapInUse() const;	//Returns 'true' if color map is being used by shading

		//Returns current value of the "uniform color" of the media modeled by the 3D texture. This function returns valid setting of the uniform color
		//regardless of whether or not it is taken into account by the shading
		vec3 getMediumUniformColor() const;	

		

		//Rendering infrastructure of the object
		bool supportsRenderingMode(uint32_t rendering_mode) const override;
		uint32_t getNumberOfRenderingPasses(uint32_t rendering_mode) const override;
		bool render() override;
	};


}

#define TW__TRANSPARENT_BOX__
#endif