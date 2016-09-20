#ifndef TW__FULLSCREEN_RECTANGLE_H__

#include <functional>

#include "AbstractRenderableObject.h"
#include "ImmutableTexture2D.h"
#include "TextureUnitBlock.h"
#include "SeparateShaderProgram.h"

namespace tiny_world
{

//The following constants define allowed rendering modes for the full-screen rectangle object
#define TW_RENDERING_MODE_FILTER_AND_DYNAMICS	12	//both filtering in fragment shader and dynamical effects in geometry shader are allowed.
#define TW_RENDERING_MODE_SIMPLIFIED			13	//simplified rendering: filtering and dynamical effects are disabled.

	//Implements a flat rectangle that is drawn independently from the current user's position. 
	//Used for post-rendering filtering in combination with render-to-texture techniques.


	class FullscreenRectangle final : public AbstractRenderableObjectTextured, public AbstractRenderableObjectExtensionAggregator<>
	{
	private:
		static const std::string fixed_vp_name;	//string name of default vertex processing program
		static const std::string fixed_fp_name;	//string name of default fragment processing program
		static const short num_vertices = 4;	//number of vertices needed to draw the rectangle
		static std::array<vec4, num_vertices> vertices;	//vertices defining the rectangle
		static std::array<vec2, num_vertices> tex_coords;	//texture coordinates of the rectangle

		float width, height;	//width and height of the rectangle

		const AbstractProjectingDevice* p_projecting_device;	//projecting device applied to the full-screen rectangle

		//**************************************Shader pipeline infrastructure****************************************
		ShaderProgramReferenceCode fixed_vp_ref_code;		//default fixed vertex program used to render the rectangle
		ShaderProgramReferenceCode fixed_fp_ref_code;		//default fixed fragment program used to render the rectangle

		const SeparateShaderProgram *p_user_defined_geometry_shader;	//pointer to a user defined geometry shader

		//Function, which is called when user defined geometry shader gets activated. The function accepts only one parameter, which identifies the first texture unit 
		//vacant for bindings. It is required that the function returns 'true' on success and 'false' on failure.
		std::function<bool(const AbstractProjectingDevice&, const AbstractRenderingDevice&, int)> geometry_shader_setup_func;

		const SeparateShaderProgram *p_user_defined_fragment_shader;	//pointer to a user defined fragment shader

		//Function, which is called when user defined fragment shader gets activated. The function accepts only one parameter, which identifies the first texture unit 
		//vacant for bindings. It is required that the function returns 'true' on success and 'false' on failure.
		std::function<bool(const AbstractProjectingDevice&, const AbstractRenderingDevice&, int)> fragment_shader_setup_func;

		ProgramPipeline shader_pipeline;		//separate shader program pipeline


		//**************************************Texture management************************************************
		TextureReferenceCode texture_ref_code;
		TextureSamplerReferenceCode texture_sampler_ref_code;


		//*******************************************Raw OpenGL resources******************************************
		GLuint ogl_vertex_array_object_id;
		GLuint ogl_data_buf;
		//*************************************************************************************************************

		void applyScreenSize(const uvec2& screen_size) override;
		bool configureRendering(AbstractRenderingDevice& render_target, uint32_t rendering_pass) override;
		void configureViewProjectionTransform(const AbstractProjectingDevice& projecting_device) override;
		bool configureRenderingFinalization() override;

		void update_data();		//updates data stored in the OpenGL data buffer
		void setup_shader_pipeline(uint32_t mode);	//configures the shader pipeline
		void setup_object();	//performs initialization particulars

	public:
		FullscreenRectangle();	//Creates rectangles with unit width and height located in the origin of the scene
		FullscreenRectangle(float width, float height);	//Creates rectangle with given width and height located in the origin of the scene
		FullscreenRectangle(const FullscreenRectangle& other);		//Copy constructor
		FullscreenRectangle(FullscreenRectangle&& other);		//Move constructor

		FullscreenRectangle& operator=(const FullscreenRectangle& other);	//assignment operator
		FullscreenRectangle& operator=(FullscreenRectangle&& other);		//move-assignment operator

		~FullscreenRectangle();	//Destructor

		void setDimensions(float width, float height);		//Applies new dimensions for the rectangle
		std::pair<float, float> getDimensions() const;		//Returns width and height of the rectangle packed into a pair

		//Applies a 2D-texture onto the rectangle. Later on, this texture can be sampled in GLSL code using sampler2D variable named source0
		void installTexture(const ImmutableTexture2D& _2d_texture);

		//Defines sampler object to be used when sampling data from the texture applied to the rectangle
		void installSampler(const TextureSampler& sampler);
		
		//This function allows to apply a dynamic effect onto the rectangle. For instance, one may want to simulate a folding sheet of paper or
		//a breaking glass plate. Technically, the effect must be implemented by a geometry shader stage of  the rendering. If no user-defined
		//geometry shader is provided, then a trivial pass-through shader is used. See code examples for further details.
		void setDynamicEffect(const SeparateShaderProgram& geometry_shader, std::function<bool(const AbstractProjectingDevice&, const AbstractRenderingDevice&, int)> shader_setup_func);


		//This function allows to implement a post-process image filtering. For instance, one may want to add smooth appearance to the scene 
		//by implementing the Gaussian blur filter. The same technique should be used to implement Bloom rendering effects.
		//Technically this is done by a user defined fragment shader. If no user-defined shader is provided, the object will use default texturing.
		//See code examples for further details.
		//In order to operate properly, user-defined fragment shader needs to comply with the following requirements.
		//Requirement 1: the fragment shader declares input variable "in vec2 tex_coord" in order to access texture coordinates used by the full-screen rectangle.
		//Requirement 2: the fragment shader declares sampler uniform "uniform sampler2D source0", which is used to sample values from the texture applied 
		//to the full screen rectangle using installTexture(...).
		void setFilterEffect(const SeparateShaderProgram& fragment_shader, std::function<bool(const AbstractProjectingDevice&, const AbstractRenderingDevice&, int)> shader_setup_func);


		//*****************************************AbstractRenderableObject derived infrastructure*******************************************
		bool supportsRenderingMode(uint32_t rendering_mode) const override;
		uint32_t getNumberOfRenderingPasses(uint32_t rendering_mode) const override;
		bool render() override;
	};


}

#define TW__FULLSCREEN_RECTANGLE_H__
#endif