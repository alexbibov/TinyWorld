#ifndef TW__CUBE_H__

#include "AbstractRenderableObject.h"
#include "ImmutableTexture2D.h"

namespace tiny_world
{
	//Implements rendering of a simple cube with a single texture

	class Cube final : public AbstractRenderableObjectTextured, public AbstractRenderableObjectExtensionAggregator<>
	{
	private:
		static const std::string rendering_program_name;
		static char vp_source[];
		static char fp_source[];

		std::array<vec4, 36> vertices;
		std::array<vec2, 36> tex_coords;

		//Identifier of the cube rendering program
		ShaderProgramReferenceCode rendering_program_ref_code;

		float side_size;
		TextureReferenceCode main_texture_ref_code;
		TextureSamplerReferenceCode main_sampler_ref_code;

		GLuint ogl_array_buf_id;	//OpenGL identifier of data array buffer
		GLuint ogl_vertex_attribute_object_id;		//OpenGL identifier of vertex attribute object

		void applyScreenSize(const uvec2& screen_size) override;
		bool configureRendering(AbstractRenderingDevice& render_target, uint32_t rendering_pass) override;
		void configureViewProjectionTransform(const AbstractProjectingDevice& projecting_device) override;
		bool configureRenderingFinalization() override;

		void setup_object();		//initializes obligatory part of the object state: vertex positions, texture coordinates and rendering program
		void update_array_data();	//update data stored in vertex buffer object owned by Cube

	protected:
		using AbstractRenderableObjectExtensionAggregator<>::injectExtension;

	public:
		Cube();		//Default initialization: creates a unit cube
		explicit Cube(std::string cube_string_name);		//Default initialization using provided string name
		Cube(std::string cube_string_name, float side_size);		//Initializes cube with given size of its side
		Cube(std::string cube_string_name, float side_size, vec3 location);	//Initializes cube with given side size at given location
		Cube(std::string cube_string_name, float side_size, vec3 location, float z_rot_angle, float y_rot_angle, float x_rot_angle);	//Initializes cube and rotates it around z-, y- and x- axes by the given angles
		Cube(const Cube& other);	//Copy constructor
		Cube(Cube&& other);		//Move constructor
		~Cube();	//Destructor

		Cube& operator=(const Cube& other);		//Copy-assignment operator overload
		Cube& operator=(Cube&& other);	//Move-assignment operator overload

		float getSideSize() const;

		void setSideSize(float new_side_size);
		void installTexture(const ImmutableTexture2D& _2d_texture);


		bool supportsRenderingMode(uint32_t rendering_mode) const override;
		uint32_t getNumberOfRenderingPasses(uint32_t rendering_mode) const override;
		bool render() override;
	};


}

#define TW__CUBE_H__
#endif