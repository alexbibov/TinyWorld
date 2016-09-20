#ifndef TW__SHADER__

#include <GL/glew.h>
#include <cstdint>
#include <utility>
#include <string>
#include <map>

#include "Entity.h"

namespace tiny_world
{

	//The following class implements single standalone shader that can be used by itself and must be attached to a shader program to form correct executable

	enum class ShaderType : GLenum;		//forward declaration of enumeration enlisting allowed kinds of shader objects
	typedef std::pair<const char*, size_t> GLSLSourceCode;		//type representing GLSL source code. The first element contains the source as a string (not necessarily null-terminated) and the second element contains its length

	//Implements basic structure of a shader object
	class Shader final : public Entity
	{
	private:
		GLuint ogl_shader_id;				//OpenGL identifier of the shader object
		ShaderType shader_type;				//shader type (e.g. vertex shader, fragment shader etc.)
		uint32_t* ref_counter;				//reference counter

		GLuint compile_shader(const GLchar* glsl_source_raw_code, const char* glsl_source_name);		//compiles GLSL source code. Returns OpenGL identifier of the new shader object or 0 on failure.

	public:
		//Shader source gets compiled during initialization of containing object and can not be altered afterwards
		Shader(GLSLSourceCode glsl_source, ShaderType shader_type, const std::string& shader_string_name);	//initializes shader object using GLSL source code provided by the caller
		Shader(const std::string& file_source, ShaderType shader_type, const std::string& shader_string_name);	//initializes shader object using GLSL source code stored in the given file
		Shader(const Shader& other);	//copy constructor
		Shader(Shader&& other);		//move constructor
		~Shader();	//destructor
		Shader& operator=(const Shader& other);		//assignment operator
		Shader& operator=(Shader&& other);		//move-assignment operator

		ShaderType getShaderType() const;			//returns type of contained shader (e.g. vertex shader, fragment shader etc.)

		//Parses file source containing extended GLSL source code accepted by TinyWorld and returns raw GLSL source code that could be compiled by the driver.
		//If supplied source is not valid the function returns pair (false, "") and writes error message to "p_error_message". It is legal for "p_error_message" to be null, but
		//if function fails and "p_error_message" does not point to a valid string object the error message will not be written. 
		//The return value of this function is a pair with the first element indicating whether the function has succeeded and the second element containing the parsed source in case of success or an empty string in case of failure.
		static std::pair<bool, std::string> parseShaderSource(const std::string& file_source, std::string* p_error_message);

		operator GLuint() const;	//returns OpenGL identifier of contained shader object
	};



	enum class ShaderType : GLenum{
		VERTEX_SHADER = GL_VERTEX_SHADER, FRAGMENT_SHADER = GL_FRAGMENT_SHADER,
		TESS_CONTROL_SHADER = GL_TESS_CONTROL_SHADER, TESS_EVAL_SHADER = GL_TESS_EVALUATION_SHADER,
		GEOMETRY_SHADER = GL_GEOMETRY_SHADER, COMPUTE_SHADER = GL_COMPUTE_SHADER
	};

}

#define TW__SHADER__
#endif