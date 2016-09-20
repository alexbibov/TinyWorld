#include "ShaderProgram.h"

#include <fstream>
#include <iterator>
#include <algorithm>
#include <tuple>
#include <cstring>

using namespace tiny_world;

long long ShaderProgram::active_program = 0;

std::string ShaderProgram::shader_base_catalog = "tw_shaders/";


void ShaderProgram::setShaderBaseCatalog(const std::string& new_catalog)
{
	ShaderProgram::shader_base_catalog = new_catalog;
}


std::string ShaderProgram::getShaderBaseCatalog()
{
	return ShaderProgram::shader_base_catalog;
}


void ShaderProgram::initialize_shader_program()
{
	ogl_program_id = glCreateProgram();
	if (!ogl_program_id)
	{
		set_error_state(true);
		const char* err_msg = "Unable to create program object.";
		set_error_string(err_msg);
		call_error_callback(err_msg);
	}
}


void ShaderProgram::relink()
{
	switch (linking)
	{
	case linking_type::normal_link:
	{
		bool b_error_state = getErrorState();
		set_error_state(false);
		needs_relink = true;
		link();

		set_error_state(b_error_state);
		break;
	}

	case linking_type::binary_link:
	{
		GLint binary_length = *static_cast<GLint*>(program_binary_buf);
		GLenum binary_format = *reinterpret_cast<GLenum*>(static_cast<char*>(program_binary_buf)+sizeof(GLint));
		GLvoid *binary_source = static_cast<GLvoid*>(static_cast<char*>(program_binary_buf)+sizeof(GLint)+sizeof(GLenum));
		glProgramBinary(ogl_program_id, binary_format, binary_source, binary_length);
		needs_relink = false;
		break;
	}
	}
}


ShaderProgram::ShaderProgram(const std::string& program_class_string_name) : 
Entity(program_class_string_name),
needs_relink(true), program_binary_buf(nullptr), 
allows_binary_representation(false), binary_representation_updated(false), linking(linking_type::no_link)
{
	initialize_shader_program();
}


ShaderProgram::ShaderProgram(const std::string& program_class_string_name, const std::string& program_string_name) : 
Entity(program_class_string_name, program_string_name),
needs_relink(true), program_binary_buf(nullptr), 
allows_binary_representation(false), binary_representation_updated(false), linking(linking_type::no_link)
{
	initialize_shader_program();
}


ShaderProgram::ShaderProgram(const ShaderProgram& other) :
Entity(other), needs_relink(other.needs_relink),
allows_binary_representation(other.allows_binary_representation), 
binary_representation_updated(other.binary_representation_updated), 
shader_objects(other.shader_objects), vertex_attribute_binding_map(other.vertex_attribute_binding_map), 
linking(other.linking)
{
	//note: create new internal_id
	//note: create new ogl_program_id
	//note: deep-copy index_based_uniform_assignment_map
	//note: deep-copy program_binary_buf

	initialize_shader_program();

	//Deep-copy contents of the program binary buffer, if the buffer exists in the source-copy object
	if (other.program_binary_buf)
	{
		GLint buf_size = *static_cast<GLint*>(other.program_binary_buf) + sizeof(GLint)+sizeof(GLenum);
		program_binary_buf = new char[buf_size];
		memcpy(program_binary_buf, other.program_binary_buf, buf_size);
	}
	else program_binary_buf = nullptr;

	//Copy program state

	//Check if the source of the copy is a separate program object and apply separate flag to the current context's object correspondingly
	//glProgramParameteri(ogl_program_id, GL_PROGRAM_SEPARABLE, is_separate);

	//Install shader objects from the source object to the current context's program
	std::for_each(shader_objects.begin(), shader_objects.end(),
		[this](Shader elem)->void
	{
		glAttachShader(ogl_program_id, elem);
	}
	);

	//Bind vertex attribute locations using binding relation from the source object
	std::for_each(vertex_attribute_binding_map.begin(), vertex_attribute_binding_map.end(),
		[this](std::pair<uint32_t, std::string> elem)->void
	{
		glBindAttribLocation(ogl_program_id, elem.first, elem.second.c_str());
	}
	);

	//If source object was linked, link the current object using linking parameters from the source
	if (!other.needs_relink) relink();

	//Deep-copy the contents of uniform assignment map.
	//Note, that this has to be done AFTER relinking the destination shader program object, otherwise the copy of location based
	//uniform selection map will get invalidated upon re-linking
	std::for_each(other.location_based_uniform_assignment_map.begin(), other.location_based_uniform_assignment_map.end(),
		[this](std::pair<uint32_t, AbstractQuery*> elem)->void
	{
		location_based_uniform_assignment_map.insert(std::make_pair(elem.first, elem.second->clone()));
	}
	);

	//Copy stage-location subroutine uniform selection map.
	//Note, that this also has to be done AFTER relinking. Otherwise, the uniform subroutine selection map will get invalidated
	stage_location_based_subroutine_uniform_selection_map = other.stage_location_based_subroutine_uniform_selection_map;
}

ShaderProgram::ShaderProgram(ShaderProgram&& other) :
Entity(std::move(other)), needs_relink(other.needs_relink),
allows_binary_representation(other.allows_binary_representation), 
binary_representation_updated(other.binary_representation_updated),
shader_objects(std::move(other.shader_objects)), vertex_attribute_binding_map(std::move(other.vertex_attribute_binding_map)), linking(other.linking),
stage_location_based_subroutine_uniform_selection_map(std::move(other.stage_location_based_subroutine_uniform_selection_map))
{
	//note: acquire internal_id of the object being moved
	//note: acquire ogl_program_id of the object being moved
	//note: swap data index_based_uniform_assignment_map
	//note: swap data program_binary_buf

	ogl_program_id = other.ogl_program_id;
	other.ogl_program_id = 0;	//make OpenGL program id of the move source object unspecified

	location_based_uniform_assignment_map = other.location_based_uniform_assignment_map;
	other.location_based_uniform_assignment_map.clear();

	program_binary_buf = other.program_binary_buf;
	other.program_binary_buf = nullptr;

	//Make sure that the object being left unspecified does not contain any state-related data 
	other.shader_objects.clear();	
	other.vertex_attribute_binding_map.clear();
	other.linking = linking_type::no_link;
}

ShaderProgram::ShaderProgram(const std::string& program_class_string_name, const std::string& program_string_name, std::vector<std::string> file_sources, std::vector<ShaderType> shader_type) : 
Entity(program_class_string_name, program_string_name),
needs_relink(true), program_binary_buf(nullptr), allows_binary_representation(false),
binary_representation_updated(false), linking(linking_type::no_link)
{
	if (file_sources.size() > shader_type.size())
		throw(std::logic_error("Not all of the given GLSL shader sources have corresponding shader type declaration provided"));
	
	initialize_shader_program();
	
	for (unsigned int i = 0; i < file_sources.size(); ++i)
		addShader(Shader{ file_sources[i], shader_type[i], "shader#" + std::to_string(i) + "(" + file_sources[i] + ")" });
}

ShaderProgram::ShaderProgram(const std::string& program_class_string_name, const std::string& program_string_name, const std::vector<GLSLSourceCode>& glsl_sources, std::vector<ShaderType> shader_type) : 
Entity(program_class_string_name, program_string_name),
needs_relink(true), program_binary_buf(nullptr), allows_binary_representation(false),
binary_representation_updated(false), linking(linking_type::no_link)
{
	if (glsl_sources.size() > shader_type.size())
		throw(std::range_error("Not all of the given GLSL shader sources have corresponding shader type declaration provided"));

	initialize_shader_program();

	for (unsigned int i = 0; i < glsl_sources.size(); ++i)
		addShader(Shader{ glsl_sources[i], shader_type[i], "shader#" + std::to_string(i) + "(<heap memory>)" });
};

ShaderProgram::ShaderProgram(const std::string& program_class_string_name, const std::string& program_string_name, const std::string& shader_binary_source) : 
Entity(program_class_string_name, program_string_name),
needs_relink(true), program_binary_buf(nullptr), allows_binary_representation(false),
binary_representation_updated(false), linking(linking_type::binary_link)
{
	initialize_shader_program();
	linkBinary(shader_binary_source);
}

ShaderProgram::ShaderProgram(const std::string& program_class_string_name, const std::string& program_string_name, const std::vector<Shader>& shaders) : 
Entity(program_class_string_name, program_string_name), 
needs_relink(true), program_binary_buf(nullptr), allows_binary_representation(false),
binary_representation_updated(false), linking(linking_type::no_link)
{
	initialize_shader_program();
	std::for_each(shaders.begin(), shaders.end(), [this](Shader elem)->void{addShader(elem); });
}

ShaderProgram::~ShaderProgram()
{
	//Remove contents of the location based uniform assignment map
	std::for_each(location_based_uniform_assignment_map.begin(), location_based_uniform_assignment_map.end(),
		[](std::pair<uint32_t, AbstractQuery*> elem)->void{delete elem.second; });

	//If binary buffer storage was ever used, release it
	if (program_binary_buf)
		delete[] program_binary_buf;


	//If program object has correct state, detach all shaders and destroy program
	if (ogl_program_id)
	{
		//Detach and destroy all shaders
		std::for_each(shader_objects.begin(), shader_objects.end(),
			[this](Shader elem)->void{glDetachShader(ogl_program_id, elem); });

		//Delete OpenGL program object
		glDeleteProgram(ogl_program_id);
	}

}

ShaderProgram& ShaderProgram::operator=(const ShaderProgram& other)
{
	//Account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	ErrorBehavioral::operator=(other);

	//Release buffer that stores program binary representation if the buffer exists
	if (program_binary_buf)
		delete[] program_binary_buf;

	//Copy contents of the binary buffer from the source. Set binary representation buffer pointer to nullptr if the source object does not contain the buffer
	if (other.program_binary_buf)
	{
		GLint buf_size = *static_cast<GLint*>(other.program_binary_buf) + sizeof(GLint)+sizeof(GLenum);
		program_binary_buf = new char[buf_size];
		memcpy(program_binary_buf, other.program_binary_buf, buf_size);
	}
	else program_binary_buf = nullptr;


	//If the source program object is specified and separable, make the destination object separable
	//is_separate = other.is_separate;
	//if (other.ogl_program_id)
	//	glProgramParameteri(ogl_program_id, GL_PROGRAM_SEPARABLE, is_separate);


	//Detach all shader objects from the destination program and destroy them
	std::for_each(shader_objects.begin(), shader_objects.end(),
		[this](Shader elem)->void{glDetachShader(ogl_program_id, elem); });

	//Attach shaders from the source object
	shader_objects = other.shader_objects;
	std::for_each(shader_objects.begin(), shader_objects.end(),
		[this](Shader elem)->void{glAttachShader(ogl_program_id, elem); });


	//Bind vertex attribute locations using vertex attribute binding relation from the source object
	vertex_attribute_binding_map = other.vertex_attribute_binding_map;
	std::for_each(vertex_attribute_binding_map.begin(), vertex_attribute_binding_map.end(),
		[this](std::pair<uint32_t, std::string> elem)->void
	{glBindAttribLocation(ogl_program_id, elem.first, elem.second.c_str()); });


	//Copy link status and error status from the source object and link the destination object's program if necessary
	linking = other.linking;
	needs_relink = other.needs_relink;
	if (!other.needs_relink) relink();

	//After relinking all data stored in the location based uniform assignment map gets automatically invalidated, 
	//that is there no need to erase it here manually

	//Copy contents of the location based uniform assignment map from the source object.
	//Note that even though the program has been relinked, since the destination program object is equivalent to the 
	//source program object, the location map should remain valid (at least for all known OpenGL implementations)
	std::for_each(other.location_based_uniform_assignment_map.begin(), other.location_based_uniform_assignment_map.end(),
		[this](std::pair<uint32_t, AbstractQuery*> elem)
	{
		location_based_uniform_assignment_map.insert(std::make_pair(elem.first, elem.second->clone()));
	}
	);

	//Copy contents of the stage-location subroutine uniform selection map from the source object.
	//Note that despite the object having been relinked the stage-location subroutine uniform selection map
	//from the source object should be valid for use in the destination object
	stage_location_based_subroutine_uniform_selection_map = other.stage_location_based_subroutine_uniform_selection_map;
	

	//If source object is unspecified, the destination object also becomes unspecified
	if (!other.ogl_program_id)
	{
		glDeleteProgram(ogl_program_id);
		ogl_program_id = 0;
	}

	binary_representation_updated = other.binary_representation_updated;
	allows_binary_representation = other.allows_binary_representation;
	return *this;
}


ShaderProgram& ShaderProgram::operator=(ShaderProgram&& other)
{
	//Account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	ErrorBehavioral::operator=(std::move(other));

	//Swap data between location based uniform assignment maps of the source and destination objects
	std::swap(location_based_uniform_assignment_map, other.location_based_uniform_assignment_map);

	//Swap data between stage-location based subroutine uniform selection maps of the source and destination objects
	std::swap(stage_location_based_subroutine_uniform_selection_map, other.stage_location_based_subroutine_uniform_selection_map);

	//Swap memory blocks that store program binary representation. The swap operation below is correct even if one of the objects taking part in 
	//the assignment operation does not have program binary buffer.
	std::swap(program_binary_buf, other.program_binary_buf);


	//Swap program state data

	//Swap shader object dictionaries
	std::swap(shader_objects, other.shader_objects);

	//Swap OpenGL and internal identifiers between source and destination objects
	std::swap(ogl_program_id, other.ogl_program_id);


	//Move the rest of state data
	vertex_attribute_binding_map = std::move(other.vertex_attribute_binding_map);
	//is_separate = other.is_separate;
	linking = other.linking;
	needs_relink = other.needs_relink;

	binary_representation_updated = other.binary_representation_updated;
	allows_binary_representation = other.allows_binary_representation;

	return *this;
}



bool ShaderProgram::addShader(const Shader& shader)
{
	//If object is in erroneous state ignore this call
	if (getErrorState()) return false;

	if (!shader)
	{
		set_error_state(true);
		set_error_string(shader.getErrorString());
		call_error_callback(shader.getErrorString());
		return false;
	}

	shader_objects.push_back(shader);
	glAttachShader(ogl_program_id, shader);

	needs_relink = true;	//after attaching new shader object to the program, the program needs to be relinked

	return true;
}

bool ShaderProgram::addShader(Shader&& shader)
{
	//if object is in an erroneous state ignore this call
	if (getErrorState()) return false;

	//If shader object being added is in erroneous state, put the program object to corresponding erroneous state
	if (!shader)
	{
		set_error_state(true);
		set_error_string(shader.getErrorString());
		call_error_callback(shader.getErrorString());
		return false;
	}

	shader_objects.push_back(std::move(shader));
	glAttachShader(ogl_program_id, shader);

	needs_relink = true;	//after attaching a new shader to the program, the program has to be relinked

	return true;
}

bool ShaderProgram::removeShader(const std::string& object_name)
{
	//if object is in an erroneous state ignore this call
	if (getErrorState()) return false;

	//Try to find the shader, which is getting removed
	auto object_to_remove =
		std::find_if(shader_objects.begin(), shader_objects.end(),
		[&object_name](const Shader& elem)->bool
	{
		return elem.getStringName() == object_name;
	}
	);

	if (object_to_remove == shader_objects.end())
		return false;
	else
	{
		needs_relink = true;	//after removing a shader from the program has to be relinked

		//Detach shader from the program
		glDetachShader(ogl_program_id, (*object_to_remove));
		shader_objects.erase(object_to_remove);

		return true;
	}
}

bool ShaderProgram::removeShader(uint32_t shader_id)
{
	//if object is in an erroneous state ignore this call
	if (getErrorState()) return false;

	//Attempt to retrieve the shader object, which is the subject for removal
	std::list<Shader>::const_iterator p_subject_for_removal;
	if ((p_subject_for_removal = std::find_if(shader_objects.begin(), shader_objects.end(),
		[shader_id](const Shader& elem) -> bool { return shader_id == elem.getId(); })) == shader_objects.end())
		return false;
	else
	{
		needs_relink = true;	//after removal of a shader object, the program has to be relinked

		//Detach shader object form the program
		glDetachShader(ogl_program_id, (*p_subject_for_removal));
		shader_objects.erase(p_subject_for_removal);

		return true;
	}

}

bool ShaderProgram::containsShader(const std::string& shader_string_name) const
{
	return std::find_if(shader_objects.begin(), shader_objects.end(),
		[&shader_string_name](const Shader& elem) -> bool {return shader_string_name == elem.getStringName(); }) != shader_objects.end();
}

bool ShaderProgram::containsShader(uint32_t shader_id) const
{
	return std::find_if(shader_objects.begin(), shader_objects.end(),
		[shader_id](const Shader& elem) -> bool { return shader_id == elem.getId(); }) != shader_objects.end();
}

const Shader* ShaderProgram::retrieveShader(const std::string& shader_string_name) const
{
	std::list<Shader>::const_iterator p_shader_to_retrieve;
	return (p_shader_to_retrieve = std::find_if(shader_objects.begin(), shader_objects.end(),
		[&shader_string_name](const Shader& elem) -> bool { return shader_string_name == elem.getStringName(); })) != shader_objects.end() ?
		&(*p_shader_to_retrieve) : nullptr;
}

const Shader* ShaderProgram::retrieveShader(uint32_t shader_id) const
{
	std::list<Shader>::const_iterator p_shader_to_retrieve;
	return (p_shader_to_retrieve = std::find_if(shader_objects.begin(), shader_objects.end(),
		[shader_id](const Shader& elem) -> bool { return shader_id == elem.getId(); })) != shader_objects.end() ?
		&(*p_shader_to_retrieve) : nullptr;
}

bool ShaderProgram::link()
{
	if (getErrorState()) return false;	//if object is in an erroneous state ignore this call
	if (!needs_relink) return true;	//if object does not need linking, just return 'true'

	//Link program
	glLinkProgram(ogl_program_id);

	//Get link status
	GLint program_link_status;
	glGetProgramiv(ogl_program_id,GL_LINK_STATUS,&program_link_status);
	//If linking has failed, get the error log
	if(program_link_status==GL_FALSE)
	{
		GLint program_info_log_length;
		glGetProgramiv(ogl_program_id,GL_INFO_LOG_LENGTH,&program_info_log_length);
	
		GLchar *linking_info_log=new GLchar[program_info_log_length];
		glGetProgramInfoLog(ogl_program_id,program_info_log_length,NULL,linking_info_log);

		set_error_state(true);
		std::string err_msg="Unable to link program: \n"+std::string(linking_info_log);
		delete [] linking_info_log;
		set_error_string(err_msg);
		call_error_callback(err_msg);

		return false;
	}

	binary_representation_updated = true;	//linking was successful, thus the binary representation stored in program_binary_buf should get updated on the next call of getBinary()
	needs_relink=false;
	linking = linking_type::normal_link;


	//Relinking the program invalidates previously defined locations of the uniform variables, thus the location based uniform assignment map and stage-location based subroutine uniform selection map must be emptied
	std::for_each(location_based_uniform_assignment_map.begin(), location_based_uniform_assignment_map.end(),
		[](std::pair < uint32_t, AbstractQuery*> map_entry) -> void{delete map_entry.second; });
	location_based_uniform_assignment_map.clear();
	stage_location_based_subroutine_uniform_selection_map.clear();

	return true;
}

bool tiny_world::ShaderProgram::linkBinary(const std::string& binary_source)
{
	std::basic_filebuf<char> input_file_buf;
	
	if (!input_file_buf.open(binary_source.c_str(), std::ios::in | std::ios::binary))
	{
		set_error_state(true);
		std::string err_msg = "Unable to open program binary representation source " + binary_source;
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return false;
	}

	union{
		GLint integer;
		char byte_array[sizeof(GLint)];
	}binary_representation_length;

	if (input_file_buf.sgetn(binary_representation_length.byte_array, sizeof(GLint)) < sizeof(GLint))
	{
		set_error_state(true);
		std::string err_msg = "Unable to determine length of binary representation. Binary representation file " +
			binary_source + " is damaged";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return false;
	}

	union{
		GLenum gl_enum;
		char byte_array[sizeof(GLenum)];
	}binary_representation_format;

	if (input_file_buf.sgetn(binary_representation_format.byte_array, sizeof(GLenum)) < sizeof(GLenum))
	{
		set_error_state(true);
		std::string err_msg = "Unable to determine storage format of binary representation. Binary representation file " +
			binary_source + " is damaged";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return false;
	}

	if (program_binary_buf)		
		delete[] program_binary_buf;
	program_binary_buf = new char[binary_representation_length.integer + sizeof(GLint)+sizeof(GLenum)];
	//Store information about the binary data length and about the driver-specific format of the binary data. These data gets stored in the beginning of the buffer
	*(static_cast<GLint*>(program_binary_buf)) = binary_representation_length.integer;	//first sizeof(GLint) bytes store length of the binary data
	*(reinterpret_cast<GLenum*>(static_cast<char*>(program_binary_buf)+sizeof(GLint))) = binary_representation_format.gl_enum;	//next sizeof(GLenum) bytes store driver-specific format of the binary


	if (input_file_buf.sgetn(static_cast<char*>(program_binary_buf)+sizeof(GLint)+sizeof(GLenum), binary_representation_length.integer) < binary_representation_length.integer)
	{
		set_error_state(true);
		std::string err_msg = "Unable to read binary representation from the source file " +
			binary_source + ". File is damaged";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return false;
	}

	if (!input_file_buf.close())
	{
		set_error_state(true);
		std::string err_msg = "Unable to close file" + binary_source;
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return false;
	}

	glProgramBinary(ogl_program_id, binary_representation_format.gl_enum,
		reinterpret_cast<GLvoid*>(static_cast<char*>(program_binary_buf)+sizeof(GLint)+sizeof(GLenum)),
		binary_representation_length.integer);

	binary_representation_updated = false;
	needs_relink = false;
	linking = linking_type::binary_link;


	//Relinking the program invalidates previously defined locations of the uniform variables, thus the location based uniform assignment map and stage-location based subroutine uniform selection map must be emptied
	std::for_each(location_based_uniform_assignment_map.begin(), location_based_uniform_assignment_map.end(),
		[](std::pair < uint32_t, AbstractQuery*> map_entry) -> void{delete map_entry.second; });
	location_based_uniform_assignment_map.clear();
	stage_location_based_subroutine_uniform_selection_map.clear();

	return true;
}

void ShaderProgram::allowBinaryRepresentation(bool binary_representation_flag)
{
	allows_binary_representation = binary_representation_flag;
	glProgramParameteri(ogl_program_id, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, allows_binary_representation);
	needs_relink = true;	//after setting the hint, program has to be relinked
}

void ShaderProgram::reset()
{
	//Clear location based uniform assignment map
	std::for_each(location_based_uniform_assignment_map.begin(), location_based_uniform_assignment_map.end(),
		[](std::pair<uint32_t, AbstractQuery*> elem)->void{delete elem.second; });
	location_based_uniform_assignment_map.clear();

	//Clear stage-location based subroutine uniform selection map
	stage_location_based_subroutine_uniform_selection_map.clear();

	//Clear binary representation buffer if exists
	if (program_binary_buf)
		delete[] program_binary_buf;
	program_binary_buf = nullptr;

	//Reset program separable status to default value: program is complete
	//is_separate = false;
	//glProgramParameteri(ogl_program_id, GL_PROGRAM_SEPARABLE, GL_FALSE);

	//Remove and destroy all shaders
	std::for_each(shader_objects.begin(), shader_objects.end(),
		[this](Shader elem)->void{glDetachShader(ogl_program_id, elem); });
	shader_objects.clear();

	//Clear attribute bindings
	vertex_attribute_binding_map.clear();


	//Reset the rest of the state
	allows_binary_representation = false;
	binary_representation_updated = false;
	registerErrorCallback([](const std::string& err_msg)->void{});
	resetErrorState();
	linking = linking_type::no_link;
	needs_relink = true;

	//If the program being reseted was active, deactivate it
	if (isActive())
	{
		ShaderProgram::active_program = 0;
		glUseProgram(0);
	}
		
}

bool ShaderProgram::needsRelink() const { return needs_relink; }

bool ShaderProgram::doesAllowBinary() const { return allows_binary_representation; }

GLuint ShaderProgram::getOpenGLProgramId() const { return ogl_program_id; }

ShaderProgram::operator GLuint() const { return ogl_program_id; }


const void* ShaderProgram::getBinary() const
{
	if (needs_relink)	//no binary can be returned if the program was not properly linked
		return nullptr;

	if (allows_binary_representation)
	{
		if (!binary_representation_updated)		//if program was not updated from the last call of this function, just return old binary representation
			return program_binary_buf;

		GLint program_binary_buf_length = 0;
		glGetProgramiv(ogl_program_id, GL_PROGRAM_BINARY_LENGTH, &program_binary_buf_length);

		if (program_binary_buf)
			delete[] program_binary_buf;

		program_binary_buf = new char[program_binary_buf_length + sizeof(GLint)+sizeof(GLenum)];
		GLenum binary_format;	//this variable will store driver-specific format of the binary representation of compiled shader

		glGetProgramBinary(ogl_program_id, program_binary_buf_length, NULL, &binary_format,
			reinterpret_cast<GLvoid*>(reinterpret_cast<char*>(program_binary_buf)+sizeof(GLint)+sizeof(GLenum)));

		//Store information about the binary data length and about the driver-specific format of the binary data. These data gets stored in the beginning of the buffer
		*(reinterpret_cast<GLint*>(program_binary_buf)) = program_binary_buf_length;	//first sizeof(GLint) bytes store length of the binary data
		*(reinterpret_cast<GLenum*>(reinterpret_cast<char*>(program_binary_buf)+sizeof(GLint))) = binary_format;	//next sizeof(GLenum) bytes store driver-specific format of the binary
		binary_representation_updated = false; //binary buffer now contains the most up-to-date binary representation
		
		return program_binary_buf;
	}
	else
		return nullptr;	//if binary representation is not allowed, return nullptr
}

bool ShaderProgram::serializeBinary(const std::string& file_name) const
{
	const GLvoid* binary_buf = getBinary();		//get binary buffer
	if (!binary_buf) return false;	//if binary representation is not available, return false

	GLint buffer_length = *(reinterpret_cast<const GLint*>(binary_buf)) + sizeof(GLint)+sizeof(GLenum);	//get full length of the buffer

	std::basic_filebuf<char> file_output_buffer;
	if (!file_output_buffer.open(file_name.c_str(), std::ios::out | std::ios::binary))
		return false;

	if (file_output_buffer.sputn(reinterpret_cast<const char*>(binary_buf), buffer_length) < buffer_length)
		return false;

	if (!file_output_buffer.close())
		return false;

	return true;
}

void ShaderProgram::made_active() const
{
	//Make uniform variable assignments
	std::for_each(location_based_uniform_assignment_map.begin(), location_based_uniform_assignment_map.end(), 
		[](std::pair<uint32_t,AbstractQuery*> elem) -> void
	{
		elem.second->bindData(elem.first);
	}
	);


	//Make subroutine uniform selection for each shader stage

	GLuint* vs_subroutine_indexes, *tcs_subroutine_indexes, *tes_subroutine_indexes, *gs_subroutine_indexes, *fs_subroutine_indexes;
	GLint vs_nsul, tcs_nsul, tes_nsul, gs_nsul, fs_nsul;	//these variables will hold information about number of active subroutine uniform locations in each shader stage

	//Get number of active subroutine uniform locations for each shader stage
	glGetProgramStageiv(ogl_program_id, GL_VERTEX_SHADER, GL_ACTIVE_SUBROUTINE_UNIFORM_LOCATIONS, &vs_nsul);
	glGetProgramStageiv(ogl_program_id, GL_TESS_CONTROL_SHADER, GL_ACTIVE_SUBROUTINE_UNIFORM_LOCATIONS, &tcs_nsul);
	glGetProgramStageiv(ogl_program_id, GL_TESS_EVALUATION_SHADER, GL_ACTIVE_SUBROUTINE_UNIFORM_LOCATIONS, &tes_nsul);
	glGetProgramStageiv(ogl_program_id, GL_GEOMETRY_SHADER, GL_ACTIVE_SUBROUTINE_UNIFORM_LOCATIONS, &gs_nsul);
	glGetProgramStageiv(ogl_program_id, GL_FRAGMENT_SHADER, GL_ACTIVE_SUBROUTINE_UNIFORM_LOCATIONS, &fs_nsul);

	//Specify subroutine index for each active subroutine uniform location
	vs_subroutine_indexes = new GLuint[vs_nsul]; memset(vs_subroutine_indexes, 0, sizeof(GLuint) * vs_nsul);
	tcs_subroutine_indexes = new GLuint[tcs_nsul]; memset(tcs_subroutine_indexes, 0, sizeof(GLuint) * tcs_nsul);
	tes_subroutine_indexes = new GLuint[tes_nsul]; memset(tes_subroutine_indexes, 0, sizeof(GLuint) * tes_nsul);
	gs_subroutine_indexes = new GLuint[gs_nsul]; memset(gs_subroutine_indexes, 0, sizeof(GLuint) * gs_nsul);
	fs_subroutine_indexes = new GLuint[fs_nsul]; memset(fs_subroutine_indexes, 0, sizeof(GLuint) * fs_nsul);

	bool vs_subroutines_assigned = false;
	bool tcs_subroutines_assigned = false;
	bool tes_subroutines_assigned = false;
	bool gs_subroutines_assigned = false;
	bool fs_subroutines_assigned = false;

	std::for_each(stage_location_based_subroutine_uniform_selection_map.begin(), stage_location_based_subroutine_uniform_selection_map.end(), 
		[vs_subroutine_indexes, tcs_subroutine_indexes, tes_subroutine_indexes, gs_subroutine_indexes, fs_subroutine_indexes,
		&vs_subroutines_assigned, &tcs_subroutines_assigned, &tes_subroutines_assigned, &gs_subroutines_assigned, &fs_subroutines_assigned]
	(std::pair<std::pair<PipelineStage, uint32_t>, uint32_t> map_entry) -> void
	{
		switch (map_entry.first.first)
		{
		case PipelineStage::VERTEX_SHADER:
			vs_subroutine_indexes[map_entry.first.second] = map_entry.second;
			vs_subroutines_assigned = true;
			break;

		case PipelineStage::TESS_CONTROL_SHADER:
			tcs_subroutine_indexes[map_entry.first.second] = map_entry.second;
			tcs_subroutines_assigned = true;
			break;

		case PipelineStage::TESS_EVAL_SHADER:
			tes_subroutine_indexes[map_entry.first.second] = map_entry.second;
			tes_subroutines_assigned = true;
			break;

		case PipelineStage::GEOMETRY_SHADER:
			gs_subroutine_indexes[map_entry.first.second] = map_entry.second;
			gs_subroutines_assigned = true;
			break;

		case PipelineStage::FRAGMENT_SHADER:
			fs_subroutine_indexes[map_entry.first.second] = map_entry.second;
			fs_subroutines_assigned = true;
			break;
		}
	});

	if (vs_nsul && vs_subroutines_assigned)
	{
		glUniformSubroutinesuiv(GL_VERTEX_SHADER, vs_nsul, vs_subroutine_indexes);
		delete[] vs_subroutine_indexes;
	}
	if (tcs_nsul && tcs_subroutines_assigned)
	{
		glUniformSubroutinesuiv(GL_TESS_CONTROL_SHADER, tcs_nsul, tcs_subroutine_indexes);
		delete[] tcs_subroutine_indexes;
	}
	if (tes_nsul && tes_subroutines_assigned)
	{
		glUniformSubroutinesuiv(GL_TESS_EVALUATION_SHADER, tes_nsul, tes_subroutine_indexes);
		delete[] tes_subroutine_indexes;
	}
	if (gs_nsul && gs_subroutines_assigned)
	{
		glUniformSubroutinesuiv(GL_GEOMETRY_SHADER, gs_nsul, gs_subroutine_indexes);
		delete[] gs_subroutine_indexes;
	}
	if (fs_nsul && fs_subroutines_assigned)
	{
		glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, fs_nsul, fs_subroutine_indexes);
		delete[] fs_subroutine_indexes;
	}


	ShaderProgram::active_program = getId();
}

bool ShaderProgram::isActive() const
{
	return ShaderProgram::active_program == getId();
}

GLbitfield ShaderProgram::getProgramStages() const
{
	GLbitfield rv{ 0 };
	std::for_each(shader_objects.begin(), shader_objects.end(),
		[&rv](Shader shader)->void
	{
		switch (shader.getShaderType())
		{
		case ShaderType::VERTEX_SHADER:
			rv |= GL_VERTEX_SHADER_BIT;
			break;

		case ShaderType::TESS_CONTROL_SHADER:
			rv |= GL_TESS_CONTROL_SHADER_BIT;
			break;

		case ShaderType::TESS_EVAL_SHADER:
			rv |= GL_TESS_EVALUATION_SHADER_BIT;
			break;

		case ShaderType::GEOMETRY_SHADER:
			rv |= GL_GEOMETRY_SHADER_BIT;
			break;

		case ShaderType::FRAGMENT_SHADER:
			rv |= GL_FRAGMENT_SHADER_BIT;
			break;

		case ShaderType::COMPUTE_SHADER:
			rv |= GL_COMPUTE_SHADER_BIT;
			break;
		}
	}
	);

	return rv;
}

bool ShaderProgram::containsStage(ShaderType shader_type) const
{
	switch (shader_type)
	{
	case ShaderType::VERTEX_SHADER:
		return (getProgramStages()&GL_VERTEX_SHADER_BIT) == GL_VERTEX_SHADER_BIT;
	case ShaderType::TESS_CONTROL_SHADER:
		return (getProgramStages()&GL_TESS_CONTROL_SHADER_BIT) == GL_TESS_CONTROL_SHADER_BIT;
	case ShaderType::TESS_EVAL_SHADER:
		return (getProgramStages()&GL_TESS_EVALUATION_SHADER_BIT) == GL_TESS_EVALUATION_SHADER_BIT;
	case ShaderType::GEOMETRY_SHADER:
		return (getProgramStages()&GL_GEOMETRY_SHADER_BIT) == GL_GEOMETRY_SHADER_BIT;
	case ShaderType::FRAGMENT_SHADER:
		return (getProgramStages()&GL_FRAGMENT_SHADER_BIT) == GL_FRAGMENT_SHADER_BIT;
	case ShaderType::COMPUTE_SHADER:
		return (getProgramStages()&GL_COMPUTE_SHADER_BIT) == GL_COMPUTE_SHADER_BIT;
	default:
		return false;
	}
}

bool ShaderProgram::containsStages(GLbitfield stage_bits) const
{
	return (stage_bits&getProgramStages()) == stage_bits;
}


void ShaderProgram::bindVertexAttributeId(const std::string& variable_name, uint32_t vertex_attribute_id)
{
	if (getErrorState()) return;
	glBindAttribLocation(ogl_program_id, vertex_attribute_id, variable_name.c_str());
	vertex_attribute_binding_map[vertex_attribute_id] = variable_name.c_str();
	needs_relink = true;	//when vertex attribute binding locations change, the program needs to be relinked
}

ShaderProgram::linking_type ShaderProgram::getLinkingType() const { return linking; }


void ShaderProgram::assignSubroutineUniform(uint32_t subroutine_uniform_location, PipelineStage pipeline_stage, const std::string& subroutine_string_name)
{
	if (needs_relink)
	{
		set_error_state(true);
		std::string err_msg = "Unable to assign subroutine \"" + subroutine_string_name + "\" to subroutine uniform with location " + std::to_string(subroutine_uniform_location) +
			". The program needs to be linked first";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	//Retrieve index of requested subroutine
	GLuint subroutine_index;
	if ((subroutine_index = glGetSubroutineIndex(ogl_program_id, static_cast<GLenum>(pipeline_stage), subroutine_string_name.c_str())) == GL_INVALID_INDEX)
	{
		set_error_state(true);
		std::string err_msg = "Unable to retrieve index of subroutine \"" + subroutine_string_name + "\". The subroutine might be inactive.";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	//Store assignment information to the stage-location based subroutine uniform selection map
	if (stage_location_based_subroutine_uniform_selection_map.find(std::make_pair(pipeline_stage, static_cast<uint32_t>(subroutine_uniform_location))) ==
		stage_location_based_subroutine_uniform_selection_map.end())
		stage_location_based_subroutine_uniform_selection_map.insert(std::make_pair(std::make_pair(pipeline_stage, static_cast<uint32_t>(subroutine_uniform_location)), subroutine_index));
	else
		stage_location_based_subroutine_uniform_selection_map.at(std::make_pair(pipeline_stage, static_cast<uint32_t>(subroutine_uniform_location))) = subroutine_index;
}


void ShaderProgram::assignSubroutineUniform(const std::string& subroutine_uniform_name, PipelineStage pipeline_stage, const std::string& subroutine_string_name)
{
	if (needs_relink)
	{
		set_error_state(true);
		std::string err_msg = "Unable to assign subroutine \"" + subroutine_string_name + "\" to subroutine uniform \"" + subroutine_uniform_name +
			"\". The program needs to be linked first";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	//Retrieve location of subroutine uniform
	GLint subroutine_uniform_location;
	if ((subroutine_uniform_location = glGetSubroutineUniformLocation(ogl_program_id, static_cast<GLenum>(pipeline_stage), subroutine_uniform_name.c_str())) == -1)
	{
		set_error_state(true);
		std::string err_msg = "Unable to retrieve location of subroutine uniform \"" + subroutine_uniform_name + "\". The subroutine uniform might be inactive";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	assignSubroutineUniform(subroutine_uniform_location, pipeline_stage, subroutine_string_name);
}


size_t tiny_world::ShaderProgram::getUniformBlockDataSize(const std::string& uniform_block_name) const
{
	if (needs_relink)
	{
		set_error_state(true);
		std::string err_msg = "Unable to retrieve size of uniform block \"" + uniform_block_name + "\". The program needs to be relinked";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return 0;
	}

	GLint uniform_block_index;
	if ((uniform_block_index = glGetProgramResourceIndex(ogl_program_id, GL_UNIFORM_BLOCK, uniform_block_name.c_str())) == -1)
	{
		set_error_state(true);
		std::string err_msg = "Unable to retrieve index value of uniform block \"" + uniform_block_name + "\". The requested uniform block can not be found";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return 0;
	}


	GLenum requested_properties[] = { GL_BUFFER_DATA_SIZE };
	GLint rv;
	glGetProgramResourceiv(ogl_program_id, GL_UNIFORM_BLOCK, uniform_block_index, 1, requested_properties, 1, NULL, &rv);

	return static_cast<size_t>(rv);
}


void ShaderProgram::assignUniformBlockToBuffer(const std::string& uniform_block_name, const std140UniformBuffer& std140_uniform_buffer) const
{
	assignUniformBlockToBuffer(uniform_block_name, std140_uniform_buffer.getBindingPoint());
}


void ShaderProgram::assignUniformBlockToBuffer(const std::string& uniform_block_name, uint32_t binding_point) const
{
	GLint uniform_block_index;
	if ((uniform_block_index = glGetProgramResourceIndex(ogl_program_id, GL_UNIFORM_BLOCK, uniform_block_name.c_str())) == -1)
	{
		set_error_state(true);
		std::string err_msg = "Unable to retrieve index value of uniform block \"" + uniform_block_name + "\". The requested uniform block can not be found";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}

	glUniformBlockBinding(ogl_program_id, uniform_block_index, binding_point);
}


//Base query interface implementation

ShaderProgram::AbstractQuery::AbstractQuery(uint32_t query_size, const GLvoid* query_data) :
query_size{ query_size }, needs_transpose{ false }
{
	data = new char[query_size];
	memcpy(data, query_data, query_size);
}

ShaderProgram::AbstractQuery::AbstractQuery(uint32_t query_size, const GLvoid* query_data, bool needs_transpose) :
query_size{ query_size }, needs_transpose{ needs_transpose }
{
	data = new char[query_size];
	memcpy(data, query_data, query_size);
}

ShaderProgram::AbstractQuery::AbstractQuery(const AbstractQuery& other)
{
	query_size = other.query_size;

	data = new char[query_size];
	memcpy(data, other.data, query_size);

	needs_transpose = other.needs_transpose;
}

ShaderProgram::AbstractQuery::AbstractQuery(AbstractQuery&& other)
{
	query_size = other.query_size;

	data = other.data;
	other.data = nullptr;

	needs_transpose = other.needs_transpose;
}

ShaderProgram::AbstractQuery::~AbstractQuery()
{
	if (data)
		delete[] data;
}

ShaderProgram::AbstractQuery& ShaderProgram::AbstractQuery::operator=(const AbstractQuery& other)
{
	query_size = other.query_size;

	delete[] data;
	data = new char [query_size];
	memcpy(data, other.data, query_size);

	needs_transpose = other.needs_transpose;
	return *this;
}

ShaderProgram::AbstractQuery& ShaderProgram::AbstractQuery::operator=(AbstractQuery&& other)
{
	query_size = other.query_size;

	GLvoid* aux = data;
	data = other.data;
	other.data = aux;
	
	needs_transpose = other.needs_transpose;
	return *this;
}
