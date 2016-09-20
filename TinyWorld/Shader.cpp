#include "Shader.h"

#include <fstream>
#include <sstream>
#include <list>
#include <regex>

using namespace tiny_world;


GLuint Shader::compile_shader(const GLchar* glsl_source_raw_code, const char* glsl_source_name)
{
	//Create new shader object
	ogl_shader_id = glCreateShader(static_cast<GLenum>(shader_type));
	if (!ogl_shader_id)
	{
		set_error_state(true);
		const char* err_msg = "Unable to create new shader object";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return 0;
	}

	//Add program source to the new shader object
	glShaderSource(ogl_shader_id, 1, &glsl_source_raw_code, NULL);

	//Compile shader object 
	glCompileShader(ogl_shader_id);

	//Check if the compilation was successful
	GLint shader_compile_status;
	glGetShaderiv(ogl_shader_id, GL_COMPILE_STATUS, &shader_compile_status);
	if (shader_compile_status == GL_FALSE)
	{
		//If compilation failed, get the info log
		GLint shader_info_log_length;
		glGetShaderiv(ogl_shader_id, GL_INFO_LOG_LENGTH, &shader_info_log_length);

		GLchar *compilation_info_log = new GLchar[shader_info_log_length];
		glGetShaderInfoLog(ogl_shader_id, shader_info_log_length, NULL, compilation_info_log);

		set_error_state(true);
		std::string err_msg = "Unable to compile shader object \"" + getStringName() + 
			"\" located in \"" + glsl_source_name + "\":\n" + std::string(compilation_info_log);
		delete[] compilation_info_log;
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return 0;
	}

	return ogl_shader_id;
}


Shader::Shader(GLSLSourceCode glsl_source, ShaderType shader_type, const std::string& shader_string_name) :
Entity("Shader", shader_string_name), shader_type{ shader_type }, ref_counter{ new uint32_t{ 1 } }
{
	//Initialize new buffer, which ensures that provided GLSL shader source code refers to a null-terminated C-style string
	GLchar *shader_source = new GLchar[glsl_source.second + 1];
	memcpy(shader_source, glsl_source.first, glsl_source.second);
	shader_source[glsl_source.second] = 0;
	compile_shader(shader_source, "<heap memory>");
	delete[] shader_source;
}

Shader::Shader(const std::string& file_source, ShaderType shader_type, const std::string& shader_string_name) :
Entity("Shader", shader_string_name), shader_type{ shader_type }, ref_counter{ new uint32_t{ 1 } }
{
	std::string parse_error;
	std::pair<bool, std::string> parse_result = parseShaderSource(file_source, &parse_error);
	if (!parse_result.first)
	{
		set_error_state(true);
		set_error_string(parse_error);
		call_error_callback(parse_error);
		return;
	}

	const GLchar *program_source_chars = parse_result.second.c_str();	//get C-string containing the source

	compile_shader(program_source_chars, file_source.c_str());
}

Shader::Shader(const Shader& other) : Entity(other), 
shader_type{ other.shader_type }, ref_counter{ other.ref_counter }, ogl_shader_id{ other.ogl_shader_id }
{
	++(*ref_counter);
}

Shader::Shader(Shader&& other) : Entity(std::move(other)),
shader_type{ std::move(other.shader_type) }, ref_counter{ other.ref_counter }, ogl_shader_id{ other.ogl_shader_id }
{
	(*ref_counter)++;
}

Shader::~Shader()
{
	--(*ref_counter);	//decrement reference counter

	//If reference counter reaches zero, delete contained shader object and release reference counter allocation
	if (!(*ref_counter))
	{
		glDeleteShader(ogl_shader_id);
		delete ref_counter;
	}
}

Shader& Shader::operator=(const Shader& other)
{
	//Take care of the special case of "assignment to itself"
	if (this == &other)
		return *this;

	Entity::operator=(other);

	//Increment reference counter of the source object
	++(*other.ref_counter);

	//Decrement reference counter of the destination object
	--(*ref_counter);

	//If reference counter of the destination object reaches zero, destroy contained shader
	if (!(*ref_counter))
	{
		glDeleteShader(ogl_shader_id);
		delete ref_counter;
	}

	//Copy state from the source object to the destination object
	shader_type = other.shader_type;
	ref_counter = other.ref_counter;
	ogl_shader_id = other.ogl_shader_id;

	return *this;
}

Shader& Shader::operator=(Shader&& other)
{
	//Take care of the special case of "assignment to itself"
	if (this == &other)
		return *this;

	Entity::operator=(std::move(other));

	
	//Move state from the source object to the destination object
	shader_type = std::move(other.shader_type);
	
	//Swap reference counters
	std::swap(ref_counter, other.ref_counter);

	//Swap OpenGL identifiers
	std::swap(ogl_shader_id, other.ogl_shader_id);

	return *this;
}

ShaderType Shader::getShaderType() const { return shader_type; }

Shader::operator GLuint() const{ return ogl_shader_id; }




//Removes space characters from beginning of string str
void remove_prefix_spases(std::string& str)
{
	size_t first_character = str.find_first_not_of(" \t");
	if (first_character != std::string::npos)
		str = str.substr(first_character, std::string::npos);
}

//Removes space characters from beginning and from the end of string str
void remove_dummy_spaces(std::string& str)
{
	size_t first_character = str.find_first_not_of(" \t");
	size_t last_character = str.find_last_not_of(" \t");
	if (first_character != std::string::npos)
		str = str.substr(first_character, last_character - first_character + 1);
}

//Removes prefixing in-line comments from beginning of string str (also removes prefixing space characters)
void remove_prefix_inline_comments(std::string& str)
{
	static std::regex inline_comment_regex{ R"(/\*.*\*/)" };
	static std::smatch match;

	remove_prefix_spases(str);
	if (std::regex_search(str, match, inline_comment_regex, std::regex_constants::match_continuous))
	{
		str = str.substr(match.length(), std::string::npos);
		remove_prefix_spases(str);
	}
}

//Returns "true" if provided string is an absolute path, returns "false" otherwise
bool is_absolute_path(const std::string& path)
{
	static std::regex absolute_path_regex{ R"([A-Z]:([\\/].*)*[\\/]?)" };
	return std::regex_match(path, absolute_path_regex);
}

//Returns "true" if provided string is a relative path, returns "false" otherwise
bool is_relative_path(const std::string& path)
{
	static std::regex relative_path_regex{ R"(.*([\\/].*)*[\\/]?)" };
	return std::regex_match(path, relative_path_regex);
}


std::pair<bool, std::string> Shader::parseShaderSource(const std::string& file_source, std::string* p_error_message)
{
	std::ifstream source_file_stream{ file_source.c_str(), std::ios_base::in };	//open new input file stream
	if (!source_file_stream.good())
	{
		std::string err_msg = "Unable to open shader source file \"" + file_source + "\"";
		if (p_error_message) *p_error_message = err_msg;
		return std::make_pair(false, "");
	}

	//Retrieve length of the source file
	source_file_stream.ignore(std::numeric_limits<std::streamsize>::max());
	std::streamsize source_file_length = source_file_stream.gcount();
	source_file_stream.clear();	//ensure that the input file stream is in a good shape after having encountered the end-of-file symbol during the last extraction on operation ignore()
	source_file_stream.seekg(0);

	char* p_raw_character_sequence = new char[static_cast<int>(source_file_length)];
	source_file_stream.read(p_raw_character_sequence, source_file_length);
	std::string source_code{ p_raw_character_sequence, static_cast<std::string::size_type>(source_file_length) };
	delete[] p_raw_character_sequence;

	source_file_stream.close();

	//Check if the source code contains #include directives
	std::regex include_string_literal{ R"((["<]).*\1)" };
	std::smatch match;
	const std::string include_directive_string{ "include" };
	std::istringstream source_code_explorer{ source_code };
	std::streampos current_source_code_position;
	std::streampos insertion_offset = 0;	//counts changes in the length of the source code due to #include directives
	std::list<std::pair<std::pair<std::streamsize, std::streamsize>, std::string>> included_sources_list;	//list of GLSL sources that have been inserted into the shader using #include directive
	uint32_t line_counter = 0;
	while (source_code_explorer.good())
	{
		//Read line of the source code
		current_source_code_position = source_code_explorer.tellg();
		source_code_explorer.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		if (!source_code_explorer.good()) source_code_explorer.clear();
		std::streamoff line_length = source_code_explorer.tellg() - current_source_code_position;
		source_code_explorer.seekg(current_source_code_position);

		char* raw_line = new char[static_cast<int>(line_length)+1];
		source_code_explorer.read(raw_line, line_length + 1);
		source_code_explorer.putback(raw_line[static_cast<int>(line_length)]);
		std::string line{ raw_line, static_cast<std::string::size_type>(line_length) };
		delete[] raw_line;

		//Remove space characters in the beginning and in the end of the line
		remove_dummy_spaces(line);
		++line_counter;

		//If the line is a pure comment, proceed to the next line
		if (line[0] == '/' && line[1] == '/') continue;

		//if the line has in-line comments in the beginning, remove all of them as well as the possible trailing space characters
		remove_prefix_inline_comments(line);

		//Check if the line defines a preprocessor directive
		if (line[0] != '#') continue;	//if the line is not a preprocessor directive, proceed to the next one
		line = line.substr(1, std::string::npos);	//consume "#" character

		//Again, remove all possible in-line comments and space characters that may appear between "#" and the name of the directive
		remove_prefix_inline_comments(line);

		//Check if what is left from the line is an #include directive
		if (line.substr(0, include_directive_string.length()).compare(include_directive_string) != 0) continue;

		//Consume the "include" key-word and again remove the possible occurrences of in-line comments
		line = line.substr(include_directive_string.length(), std::string::npos);
		remove_prefix_inline_comments(line);

		//Finally, check that what is left from the line begins with a string literal
		std::string include_path;
		if (std::regex_search(line, match, include_string_literal, std::regex_constants::match_continuous))
		{
			include_path = match.str();
			line = line.substr(match.length(), std::string::npos);
			remove_prefix_spases(line);
		}
		else
			continue;

		//The rest of the line must be either empty or contain an end-of-line comment'
		if (line.length() && line[0] != '\n' && (line[0] != '/' || line[1] != '/'))
		{
			std::string err_msg = "Unable to parse line " + std::to_string(line_counter) + " of shader source file \"" + file_source + "\". Invalid #include directive";
			if (p_error_message) *p_error_message = err_msg;
			return std::make_pair(false, "");
		}


		//Remove quotation marks from the string literal
		include_path = include_path.substr(1, include_path.length() - 2);


		//Process the string literal
		bool is_absolute = is_absolute_path(include_path);
		bool is_relative = is_relative_path(include_path);

		if (!is_absolute && !is_relative)
		{
			std::string err_msg = "Unable to parse shader source file \"" + file_source + "\". The path given for #include directive in line " + std::to_string(line_counter) + " is not valid";
			if (p_error_message) *p_error_message = err_msg;
			return std::make_pair(false, "");
		}

		if (is_relative)
		{
			//Remove the file name from the end of the path to the main shader source file
			size_t last_slash_character = file_source.find_last_of("\\/");

			//if path to the main shader file does not contain slashes, it means that it has only the file name in it
			//and there is nothing to be added to the include path recognized from the preprocessor directive
			if (last_slash_character != std::string::npos)
				include_path = file_source.substr(0, last_slash_character + 1) + include_path;
		}



		std::pair<bool, std::string> included_source_code = parseShaderSource(include_path, p_error_message);
		if (!included_source_code.first) return std::make_pair(false, "");	//if reading of included source code has failed, the calling reader returns an empty string
		included_sources_list.push_back(std::make_pair(std::make_pair(current_source_code_position + insertion_offset, current_source_code_position + insertion_offset + line_length), included_source_code.second));
		insertion_offset += included_source_code.second.length() - line_length;
	}

	std::for_each(included_sources_list.begin(), included_sources_list.end(),
		[&source_code](const std::pair<std::pair<std::streamsize, std::streamsize>, std::string>& elem)->void
	{
		source_code = std::string{ source_code.begin(), source_code.begin() + static_cast<std::string::size_type>(elem.first.first) } +elem.second +
			std::string{ source_code.begin() + static_cast<std::string::size_type>(elem.first.second), source_code.end() };
	}
	);

	return std::make_pair(true, source_code);
}