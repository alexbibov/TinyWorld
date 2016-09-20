#ifndef TW__SEPARATE_SHADER_PROGRAM__

#include "ShaderProgram.h"
#include <map>

namespace tiny_world
{
#define SEPARATE_SHADER_PROGRAM_CAST(p_shader_program)\
	(*dynamic_cast<tiny_world::SeparateShaderProgram*>(p_shader_program))

	//Implements core part of the separate shader program
	class SeparateShaderProgram_Core : public ShaderProgram
	{
	protected:
		SeparateShaderProgram_Core();	//Default constructor
		explicit SeparateShaderProgram_Core(const std::string& program_string_name);		//initializes program using user-defined string name
		SeparateShaderProgram_Core(const SeparateShaderProgram_Core& other);		//copy constructor
		SeparateShaderProgram_Core(SeparateShaderProgram_Core&& other);				//move constructor
		SeparateShaderProgram_Core(const std::string& program_string_name, std::vector<std::string> file_sources, std::vector<ShaderType> shader_type);		//Initializes new shader object using file sources
		SeparateShaderProgram_Core(const std::string& program_string_name, const std::vector<GLSLSourceCode>& glsl_sources, std::vector<ShaderType> shader_type);		//Initialize new shader object using memory sources
		SeparateShaderProgram_Core(const std::string& program_string_name, std::string shader_binary_source);	//Initialize shader program from driver-specific binary representation
		SeparateShaderProgram_Core(const std::string& program_string_name, const std::vector<Shader>& shaders);	//Initialize shader program using provided set of shader objects

	public:	
		bool isSeparate() const override;		//returns 'true' for separate shader program object
	};


	//Implements interface part of the separate shader program that is able to communicate with program pipeline objects and must be used by the end-users
	class ProgramPipeline;		//forward declaration of the program pipeline object
	class SeparateShaderProgram final : public SeparateShaderProgram_Core
	{
		friend class ProgramPipeline;
	private:
		bool installToPipeline(GLuint ogl_pipeline_id) const;	//installs program object to the pipeline referred by ogl_pipeline_id
		bool activate(GLuint ogl_pipeline_id) const;	//makes program active on the pipeline identified by ogl_pipeline_id
	public:
		SeparateShaderProgram();
		explicit SeparateShaderProgram(const std::string& program_string_name);
		SeparateShaderProgram(const SeparateShaderProgram& other);
		SeparateShaderProgram(SeparateShaderProgram&& other);
		SeparateShaderProgram(const std::string& program_string_name, std::vector<std::string> file_sources, std::vector<ShaderType> shader_type);
		SeparateShaderProgram(const std::string& program_string_name, const std::vector<GLSLSourceCode>& glsl_sources, std::vector<ShaderType> shader_type);
		SeparateShaderProgram(const std::string& program_string_name, std::string shader_binary_source);
		SeparateShaderProgram(const std::string& program_string_name, const std::vector<Shader>& shaders);

		ShaderProgram* clone() const override;	//clones shader program object
	};


	class ProgramPipeline final : Entity{
	private:
		typedef std::map<PipelineStage, const SeparateShaderProgram*> pipeline_map;	//type defining mapping between shader stages and programs implementing those stages

		GLuint ogl_program_pipeline;		//OpenGL identifier of the program pipeline
		pipeline_map pipeline_state;	//state of the pipeline object

	public:
		ProgramPipeline();	//Default constructor
		ProgramPipeline(const ProgramPipeline& other);	//Copy constructor
		~ProgramPipeline(); //Destructor

		ProgramPipeline& attach(const SeparateShaderProgram& separate_shader_program);	//attaches separate shader program to the pipeline
		ProgramPipeline& operator+=(const SeparateShaderProgram& separate_shader_program);	//same as attach()
		ProgramPipeline& operator=(const ProgramPipeline& other);	//overloaded assignment operator

		bool bind() const;	//binds program pipeline
		bool isEnclosed() const;	//returns 'true' if the pipeline contains all obligatory program stages. For OpenGL 4.4 obligatory stages are vertex shader and fragment shader. Returns 'false' otherwise
		bool activate_program(long long program_id) const;  //makes one of the programs attached to the pipeline active
		bool activate_program(std::string program_string_name) const;	//the same, but the program being made active is identified by its string name
		void reset();	//reset the pipeline and remove all program attachments
	};

}

#define TW__SEPARATE_SHADER_PROGRAM__
#endif