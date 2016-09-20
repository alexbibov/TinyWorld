#ifndef TW__TEXTURE_SAMPLER__

#include <GL/glew.h>
#include <string>
#include <stdint.h>
#include "MatrixTypes.h"
#include "Entity.h"

namespace tiny_world{
	enum class SamplerWrappingMode : GLint{
		REPEAT = GL_REPEAT, MIRRORED_REPEAT = GL_MIRRORED_REPEAT,
		CLAMP_TO_EDGE = GL_CLAMP_TO_EDGE, CLAMP_TO_BORDER = GL_CLAMP_TO_BORDER
	};

	enum class SamplerMagnificationFilter : GLint{
		NEAREST = GL_NEAREST, LINEAR = GL_LINEAR
	};

	enum class SamplerMinificationFilter : GLint{
		NEAREST = GL_NEAREST, LINEAR = GL_LINEAR,
		NEAREST_MIPMAP_NEAREST = GL_NEAREST_MIPMAP_NEAREST,
		NEAREST_MIPMAP_LINEAR = GL_NEAREST_MIPMAP_LINEAR,
		LINEAR_MIPMAP_NEAREST = GL_LINEAR_MIPMAP_NEAREST,
		LINEAR_MIPMAP_LINEAR = GL_LINEAR_MIPMAP_LINEAR
	};

	struct SamplerWrapping{
		SamplerWrappingMode wrap_mode_s, wrap_mode_t, wrap_mode_r;
		bool operator ==(const SamplerWrapping& other) const;
	};


	//TODO: implement lazy OpenGL id instantiation for texture samplers


	//Implements core functionality of a texture sampler
	class TextureSampler_core : public Entity{
	private:
		SamplerMinificationFilter min_filter;	//Minification filter
		SamplerMagnificationFilter mag_filter;	//Magnification filter
		SamplerWrapping boundary_resolution_mode;	//Wrapping mode
		vec4 border_color;		//Border color used to blend texture values sampled from outside the region for GL_CLAMP_TO_BORDER wrapping mode
		GLuint ogl_sampler_id;		//OpenGL id of contained sampler object

		void init_sampler();	//performs preliminary particulars needed to initialize new sampler

	protected:
		GLuint getOpenGLId() const; //Returns OpenGL id of contained sampler

		//Constructor-Destructor infrastructure
		TextureSampler_core();	//Default constructor
		explicit TextureSampler_core(const std::string& sampler_string_name);	//Sampler initialized by a string name
		TextureSampler_core(const TextureSampler_core& other);	//Copy constructor
		TextureSampler_core(TextureSampler_core&& other);		//Move constructor

		virtual ~TextureSampler_core();	//Destructor

	public:
		//Information functions

		SamplerMinificationFilter getMinFilter() const;	//Returns minification filter
		SamplerMagnificationFilter getMagFilter() const; //Returns magnification filter
		SamplerWrapping getWrapping() const;	//Returns wrapping parameters used by the sampler for boundary values resolution
		vec4 getBorderColor() const;	//Returns the border color used by the sampler

		//Modifying functions: the functions that alter state of the object

		void setMinFilter(SamplerMinificationFilter min_filter);	//Sets minification filter used by contained sampler
		void setMagFilter(SamplerMagnificationFilter mag_filter);	//Sets magnification filter used by contained sampler
		void setWrapping(SamplerWrapping wrapping);		//Sets wrapping used by contained sampler to resolve boundary values of a texture
		void setBorderColor(const vec4& border_color);	//Sets new border color to be used with the sampler object
		

		TextureSampler_core& operator=(const TextureSampler_core& other);	//copy-assignment operator
		TextureSampler_core& operator=(TextureSampler_core&& other);	//move-assignment operator
		bool operator ==(const TextureSampler_core& other) const;		//comparison between two TextureSampler objects. Yields 'true' if compared samplers are equivalent.
	};


	class TextureUnitBlock;		//forward declaration of TextureUnitBlock class

	//Implements interface infrastructure of a texture sampler
	class TextureSampler final : public TextureSampler_core
	{
		friend class TextureUnitBlock;

	private:
		GLuint bind(GLuint texture_unit) const;	//Binds contained sampler to the specified texture unit. Returns OpenGL id of the sampler previously bound to this unit.

	public:
		//Constructor-Destructor infrastructure
		TextureSampler();	//Default constructor
		explicit TextureSampler(const std::string& sampler_string_name);	//Sampler initialized by a string name
		TextureSampler(const TextureSampler& other);	//Copy constructor
		TextureSampler(TextureSampler&& other);		//Move constructor

		TextureSampler& operator=(const TextureSampler& other);
		TextureSampler& operator=(TextureSampler&& other);

		virtual ~TextureSampler();	//Destructor
	};


}

#define TW__TEXTURE_SAMPLER__
#endif