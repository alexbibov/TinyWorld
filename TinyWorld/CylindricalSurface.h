//Implements cylindrical surface

#ifndef TW__CYLINDRICAL_SURFACE__

#include "AbstractRenderableObject.h"
#include "AbstractRenderableObjectLightEx.h"
#include "AbstractRenderableObjectHDRBloomEx.h"
#include "TextureUnitBlock.h"
#include "CompleteShaderProgram.h"


namespace tiny_world
{
	class CylindricalSurface final : virtual public AbstractRenderableObjectTextured, 
		public AbstractRenderableObjectExtensionAggregator<AbstractRenderableObjectLightEx, AbstractRenderableObjectHDRBloomEx>
	{
	private:
		static const std::string rendering_program0_name;

		uint32_t num_angular_base_nodes;	//number of angular steps in the tessellation billet
		uint32_t num_length_base_nodes;		//number of steps along the axis of the "cylinder" in the tessellation billet

		float texture_u_scale;		//texture scale in horizontal direction
		float texture_v_scale;		//texture scale in vertical direction

		float radius;	//radius of cylindrical surface
		float length;	//length of cylindrical surface

		TextureReferenceCode surface_map_reference_code;	//reference code of texture containing surface map data
		TextureReferenceCode side_texture_reference_code;	//reference code of the texture applied to the side surface of the cylindrical object
		TextureReferenceCode face_texture_reference_code;	//reference code of the texture applied to the faces of the cylindrical object
		TextureSamplerReferenceCode surface_map_sampler_ref_code;		//texture sampler of the surface map. The same sampler is used for the side surface texture. Face texture is sampled using the default sampler


		ImmutableTexture2D side_normal_map;		//side surface normal map
		ImmutableTexture2D side_specular_map;	//side surface specular map
		ImmutableTexture2D side_emission_map;	//side surface emission map

		ImmutableTexture2D face_normal_map;		//normal map applied to the object's faces
		ImmutableTexture2D face_specular_map;	//specular map applied to the object's faces
		ImmutableTexture2D face_emission_map;	//emission map applied to the object's faces


		ShaderProgramReferenceCode rendering_program0_ref_code;		//reference code of the shader program used to visualize the object


		//*********************************************Raw OpenGL resources*****************************************
		GLuint ogl_buffers[2];	//vertex and index buffers
		GLuint ogl_vertex_array_object;	//OpenGL object containing description of vertex attributes
		//**************************************************************************************************************


		AbstractRenderingDevice* p_render_target;	//rendering target of the drawable object
		uint32_t current_rendering_pass;	//currently active rendering pass

		bool is_surface_map_defined;	//equals 'true' if surface map has been defined


		void applyScreenSize(const uvec2& screen_size) override;
		bool configureRendering(AbstractRenderingDevice& render_target, uint32_t rendering_pass) override;
		void configureViewProjectionTransform(const AbstractProjectingDevice& projecting_device) override;
		bool configureRenderingFinalization() override;

		//Performs initialization particulars of cylindrical surface
		void init_cylindrical_surface();

		//This is a helper function, which defines texture coordinates for the faces of the surface based on their shape and aligns the side surface with its vertically averaged mass center
		//Note that the function updates the data provided in surface_map_data argument
		void process_surface_data(std::vector<vec2>& surface_map_data, uint32_t num_points_per_slice, uint32_t num_slices);

	public:
		//Default initializer
		CylindricalSurface(uint32_t num_angular_base_nodes = 360, uint32_t num_length_base_nodes = 100);

		//Initializes new cylindrical surface, assigns provided string name to it and populates it with data obtained from ASCII-table stored in supplied source file
		CylindricalSurface(const std::string& string_name, const std::string& source_file, 
			uint32_t num_angular_base_nodes = 360, uint32_t num_length_base_nodes = 100);

		//Initializes new cylindrical surface, assigns the given string name to it and populates it with data from supplied memory table
		CylindricalSurface(const std::string& string_name, const std::vector<vec2>& source_data, uint32_t num_points_per_slice, uint32_t num_slices,
			uint32_t num_angular_base_nodes = 360, uint32_t num_length_base_nodes = 100);

		//Copy constructor
		CylindricalSurface(const CylindricalSurface& other);

		//Move constructor
		CylindricalSurface(CylindricalSurface&& other);

		//Copy-assignment operator
		CylindricalSurface& operator= (const CylindricalSurface& other);

		//Move-assignment operator
		CylindricalSurface& operator=(CylindricalSurface&& other);

		//Destructor0.
		~CylindricalSurface();


		//Returns radius of the cylindrical surface
		float getRadius() const;

		//Sets radius of the cylindrical surface
		void setRadius(float new_radius);


		//Returns length of the cylindrical surface
		float getLength() const;

		//Sets radius of the cylindrical surface
		void setLength(float new_length);

		//Defines new scale for the side surface texture
		void setTextureScale(float u_scale, float v_scale);


		//Loads data to be represented by cylindrical surface from a source file, where the file must contain a rectangular ASCII-table with values.
		//The table must be organized as follows:
		//1) Cylindrical surface is cut by vertical slices with uniform stepping
		//2) In each slice the points at the intersection between the slice and the surface are extracted with uniform angular stepping (i.e. we extract a point from the intersection 
		//curve that has the polar angle of 0, then the one having the polar angle of A, then 2A, etc.)
		//3) x- and y- coordinates of the points are written into the table as two consequent rows (z- coordinate can always be extracted from the number of the slice).
		//Therefore, the first two rows of the table contains information about the first slice, the second two rows contain information about the second slice, and so forth.
		void defineSurface(const std::string& source_file);

		//Loads data for the cylindrical surface from a data table represented by an array of 2D vectors, where each vector contains x- and y- coordinates of a single point and
		//the vectors are ordered slice-by-slice, i.e. the vectors corresponding to the surface points from the first slice appear first in the array, the vectors corresponding to the
		//points from the second slice follow the second and so forth.
		void defineSurface(const std::vector<vec2>& source_data, uint32_t num_points_per_slice, uint32_t num_slices);

		//Assigns texture to the cylindrical object
		void installTexture(const ImmutableTexture2D& side_surface_texture, const ImmutableTexture2D& face_texture);

		//Assigns normal maps to the surface
		void applyNormalMapSourceTexture(const ImmutableTexture2D& side_surface_texture_NRM, const ImmutableTexture2D& face_texture_NRM);

		//Assigns specular maps to the surface
		void applySpecularMapSourceTexture(const ImmutableTexture2D& side_surface_texture_SPEC, const ImmutableTexture2D& face_texture_SPEC);

		//Assigns emission maps to the surface
		void applyEmissionMapSourceTexture(const ImmutableTexture2D& side_surface_texture_EMISSION, const ImmutableTexture2D& face_texture_EMISSION);


		//Standard infrastructure of a drawable object

		bool supportsRenderingMode(uint32_t rendering_mode) const override;
		uint32_t getNumberOfRenderingPasses(uint32_t rendering_mode) const override;
		bool render() override;
	};
}


#define TW__CYLINDRICAL_SURFACE__
#endif