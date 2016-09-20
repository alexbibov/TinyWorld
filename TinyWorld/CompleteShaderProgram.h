#ifndef TW__COMPLETE_SHADER_PROGRAM__

#include "ShaderProgram.h"

namespace tiny_world
{

#define COMPLETE_SHADER_PROGRAM_CAST(p_shader_program)\
	(*dynamic_cast<tiny_world::CompleteShaderProgram*>(p_shader_program))

	class CompleteShaderProgram final : public ShaderProgram
	{
	public:
		CompleteShaderProgram();	//Default initializer
		explicit CompleteShaderProgram(const std::string& program_string_name);		//initializes object using user-defined string name

		CompleteShaderProgram(const CompleteShaderProgram& other);	//copy constructor
		CompleteShaderProgram(CompleteShaderProgram&& other);	//move constructor

		~CompleteShaderProgram();	//destructor

		CompleteShaderProgram& operator=(CompleteShaderProgram&& other);	//move-assignment operator
		CompleteShaderProgram& operator=(const CompleteShaderProgram& other);	//copy-assignment operator

		CompleteShaderProgram(const std::string& program_string_name, std::vector<std::string> file_sources, std::vector<ShaderType> shader_type);	//Initializes program from a number of shader sources stored in files
		CompleteShaderProgram(const std::string& program_string_name, const std::vector<GLSLSourceCode>& glsl_sources, std::vector<ShaderType> shader_type); //Initializes program using a number of shader textual sources located in memory
		CompleteShaderProgram(const std::string& program_string_name, std::string shader_binary_source);	//initializes program object with previously compiled binary source
		CompleteShaderProgram(const std::string& program_string_name, const std::vector<Shader>& shaders);	//initializes program using provided set of shader objects

		bool activate() const;	//activates program
	    bool isSeparate() const override;	//returns 'false' for complete shader program
		ShaderProgram* clone() const override;	//clones shader program object
	};


}

#define TW__COMPLETE_SHADER_PROGRAM__
#endif
