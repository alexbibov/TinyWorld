#ifndef TW__SHADER_PROGRAM_H__

//Class ShaderProgram is a simple abstraction representing shader program compiled from a number of shaders

#include <string>
#include <vector>
#include <set>
#include <functional>
#include <stdint.h>
#include <map>
#include <list>

#include <GL/glew.h>

#include "Shader.h"
#include "MatrixTypes.h"
#include "Misc.h"
#include "std140UniformBuffer.h"
#include "ErrorBehavioral.h"

namespace tiny_world{

	//TODO: implement lazy OpenGL id instantiation for shader program object to alleviate for redundant dummy shader program objects creation


	typedef ShaderType PipelineStage;

	class ShaderProgram : public Entity{
	public:
		enum class linking_type : short{
			normal_link = 1, binary_link = 2, no_link = 3
		};

	private:
		__declspec(thread) static long long active_program;	//id of the program object, which is currently active on the calling thread

		std::list<Shader> shader_objects;	//double-linked-list of shader objects
		bool needs_relink;			//set if contained program has to be relinked before further use
		mutable bool binary_representation_updated;		//set if binary representation of the program was updated during the last call of link()
		bool allows_binary_representation;	//set if contained program allows binary representation
		mutable GLvoid *program_binary_buf;		//pointer that refers to program binary buffer
		GLuint ogl_program_id;	//internal OpenGL id of the program
		linking_type linking;	//determines currently active linking type
		//bool is_separate;		//equals 'true' if program is separate and is intended to be attached to a program pipeline
		
		//Class implementing abstract data binding query (all queries eventually get processed by function made_active())
		//Each query "knows" how to make corresponding data binding, which is achieved through run-time polymorphisms
		class AbstractQuery{
		protected:
			bool needs_transpose;	//used by matrix uniform queries: 'true' if the matrix needs to be transposed before getting assigned to shader uniform
			uint32_t query_size;	//size of the data in query
			GLvoid *data;			//pointer to data
		public:
			AbstractQuery(uint32_t query_size, const GLvoid* query_data);
			AbstractQuery(uint32_t query_size, const GLvoid* query_data, bool needs_transpose);

			AbstractQuery(const AbstractQuery& other);
			AbstractQuery(AbstractQuery&& other);
			virtual ~AbstractQuery();

			AbstractQuery& operator=(const AbstractQuery& other);
			AbstractQuery& operator=(AbstractQuery&& other);

			virtual void bindData(uint32_t location) const = 0;		//binds contained data
			virtual AbstractQuery* clone() const = 0;	//clones "this" object
		};
		
		//The following template defines uniform assignment query
		template<typename T> struct uniform_assignment_query;

		//Scalar uniform queries
		template<>
		struct uniform_assignment_query<GLint> : public AbstractQuery
		{
			typedef GLint type_value;

			uniform_assignment_query(uint32_t count, const type_value* raw_data) :
				AbstractQuery{ sizeof(type_value)*count, raw_data } {}

			void bindData(uint32_t location) const override
			{
				glUniform1iv(location, query_size / sizeof(type_value), static_cast<type_value*>(data));
			}

			AbstractQuery* clone() const override
			{ 
				return new uniform_assignment_query{ query_size / sizeof(type_value), static_cast<type_value*>(data) }; 
			}
		};

		template<>
		struct uniform_assignment_query<GLuint> : public AbstractQuery
		{
			typedef GLuint type_value;

			uniform_assignment_query(uint32_t count, const type_value* raw_data) :
				AbstractQuery{ sizeof(type_value)*count, raw_data } {}

			void bindData(uint32_t location) const override
			{
				glUniform1uiv(location, query_size / sizeof(type_value), static_cast<type_value*>(data));
			}

			AbstractQuery* clone() const override
			{ 
				return new uniform_assignment_query{ query_size / sizeof(type_value), static_cast<type_value*>(data) }; 
			}
		};
		
		template<>
		struct uniform_assignment_query<GLfloat> : public AbstractQuery
		{
			typedef GLfloat type_value;

			uniform_assignment_query(uint32_t count, const type_value* raw_data) :
				AbstractQuery{ sizeof(type_value)*count, raw_data } {}

			void bindData(uint32_t location) const override
			{
				glUniform1fv(location, query_size / sizeof(type_value), static_cast<type_value*>(data));
			}

			AbstractQuery* clone() const override
			{ 
				return new uniform_assignment_query{ query_size / sizeof(type_value), static_cast<type_value*>(data) }; 
			}
		};

		template<>
		struct uniform_assignment_query<GLdouble> : public AbstractQuery
		{
			typedef GLdouble type_value;

			uniform_assignment_query(uint32_t count, const type_value* raw_data) :
				AbstractQuery{ sizeof(type_value)*count, raw_data } {}

			void bindData(uint32_t location) const override
			{
				glUniform1dv(location, query_size / sizeof(type_value), static_cast<type_value*>(data));
			}

			AbstractQuery* clone() const override
			{ 
				return new uniform_assignment_query{ query_size / sizeof(type_value), static_cast<type_value*>(data) }; 
			}
		};





		//Vector uniform queries
		template<>
		struct uniform_assignment_query<ivec4> : public AbstractQuery
		{
			typedef ivec4 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::dimension*count, raw_data } {}

			void bindData(uint32_t location) const override
			{
				glUniform4iv(location, query_size / (sizeof(type_value::value_type)*type_value::dimension),
					static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{ 
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::dimension), 
					static_cast<type_value::value_type*>(data) }; 
			}
		};

		template<>
		struct uniform_assignment_query<uvec4> : public AbstractQuery
		{
			typedef uvec4 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::dimension*count, raw_data } {}

			void bindData(uint32_t location) const override
			{
				glUniform4uiv(location, query_size / (sizeof(type_value::value_type)*type_value::dimension),
					static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::dimension),
					static_cast<type_value::value_type*>(data) };
			}
		};

		template<>
		struct uniform_assignment_query<vec4> : public AbstractQuery
		{
			typedef vec4 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::dimension*count, raw_data } {}

			void bindData(uint32_t location) const override
			{
				glUniform4fv(location, query_size / (sizeof(type_value::value_type)*type_value::dimension),
					static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::dimension),
					static_cast<type_value::value_type*>(data) };
			}
		};
		
		template<>
		struct uniform_assignment_query<dvec4> : public AbstractQuery
		{
			typedef dvec4 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::dimension*count, raw_data } {}

			void bindData(uint32_t location) const override
			{
				glUniform4dv(location, query_size / (sizeof(type_value::value_type)*type_value::dimension),
					static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::dimension),
					static_cast<type_value::value_type*>(data) };
			}
		};



		template<>
		struct uniform_assignment_query<ivec3> : public AbstractQuery
		{
			typedef ivec3 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::dimension*count, raw_data } {}

			void bindData(uint32_t location) const override
			{
				glUniform3iv(location, query_size / (sizeof(type_value::value_type)*type_value::dimension),
					static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::dimension),
					static_cast<type_value::value_type*>(data) };
			}
		};

		template<>
		struct uniform_assignment_query<uvec3> : public AbstractQuery
		{
			typedef uvec3 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::dimension*count, raw_data } {}

			void bindData(uint32_t location) const override
			{
				glUniform3uiv(location, query_size / (sizeof(type_value::value_type)*type_value::dimension),
					static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::dimension),
					static_cast<type_value::value_type*>(data) };
			}
		};

		template<>
		struct uniform_assignment_query<vec3> : public AbstractQuery
		{
			typedef vec3 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::dimension*count, raw_data } {}

			void bindData(uint32_t location) const override
			{
				glUniform3fv(location, query_size / (sizeof(type_value::value_type)*type_value::dimension),
					static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::dimension),
					static_cast<type_value::value_type*>(data) };
			}
		};

		template<>
		struct uniform_assignment_query<dvec3> : public AbstractQuery
		{
			typedef dvec3 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::dimension*count, raw_data } {}

			void bindData(uint32_t location) const override
			{
				glUniform3dv(location, query_size / (sizeof(type_value::value_type)*type_value::dimension),
					static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::dimension),
					static_cast<type_value::value_type*>(data) };
			}
		};



		template<>
		struct uniform_assignment_query<ivec2> : public AbstractQuery
		{
			typedef ivec2 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::dimension*count, raw_data } {}

			void bindData(uint32_t location) const override
			{
				glUniform2iv(location, query_size / (sizeof(type_value::value_type)*type_value::dimension),
					static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::dimension),
					static_cast<type_value::value_type*>(data) };
			}
		};

		template<>
		struct uniform_assignment_query<uvec2> : public AbstractQuery
		{
			typedef uvec2 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::dimension*count, raw_data } {}

			void bindData(uint32_t location) const override
			{
				glUniform2uiv(location, query_size / (sizeof(type_value::value_type)*type_value::dimension),
					static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::dimension),
					static_cast<type_value::value_type*>(data) };
			}
		};

		template<>
		struct uniform_assignment_query<vec2> : public AbstractQuery
		{
			typedef vec2 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::dimension*count, raw_data } {}

			void bindData(uint32_t location) const override
			{
				glUniform2fv(location, query_size / (sizeof(type_value::value_type)*type_value::dimension),
					static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::dimension),
					static_cast<type_value::value_type*>(data) };
			}
		};

		template<>
		struct uniform_assignment_query<dvec2> : public AbstractQuery
		{
			typedef dvec2 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::dimension*count, raw_data } {}

			void bindData(uint32_t location) const override
			{
				glUniform2dv(location, query_size / (sizeof(type_value::value_type)*type_value::dimension),
					static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::dimension),
					static_cast<type_value::value_type*>(data) };
			}
		};


		//Matrix uniform queries
		template<>
		struct uniform_assignment_query<mat4> : public AbstractQuery
		{ 
			typedef mat4 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data } {}
			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data, bool needs_transpose) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data, needs_transpose } {}

			void bindData(uint32_t location) const override
			{
				glUniformMatrix4fv(location, query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					needs_transpose, static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					static_cast<type_value::value_type*>(data), needs_transpose };
			}
		};
		
		template<>
		struct uniform_assignment_query<dmat4> : public AbstractQuery
		{
			typedef dmat4 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data } {}
			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data, bool needs_transpose) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data, needs_transpose } {}

			void bindData(uint32_t location) const override
			{
				glUniformMatrix4dv(location, query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					needs_transpose, static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					static_cast<type_value::value_type*>(data), needs_transpose };
			}
		};



		template<>
		struct uniform_assignment_query<mat3> : public AbstractQuery
		{
			typedef mat3 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data } {}
			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data, bool needs_transpose) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data, needs_transpose } {}

			void bindData(uint32_t location) const override
			{
				glUniformMatrix3fv(location, query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					needs_transpose, static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					static_cast<type_value::value_type*>(data), needs_transpose };
			}
		};

		template<>
		struct uniform_assignment_query<dmat3> : public AbstractQuery
		{
			typedef dmat3 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data } {}
			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data, bool needs_transpose) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data, needs_transpose } {}

			void bindData(uint32_t location) const override
			{
				glUniformMatrix3dv(location, query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					needs_transpose, static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					static_cast<type_value::value_type*>(data), needs_transpose };
			}
		};



		template<>
		struct uniform_assignment_query<mat2> : public AbstractQuery
		{
			typedef mat2 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data } {}
			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data, bool needs_transpose) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data, needs_transpose } {}

			void bindData(uint32_t location) const override
			{
				glUniformMatrix2fv(location, query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					needs_transpose, static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					static_cast<type_value::value_type*>(data), needs_transpose };
			}
		};

		template<>
		struct uniform_assignment_query<dmat2> : public AbstractQuery
		{
			typedef dmat2 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data } {}
			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data, bool needs_transpose) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data, needs_transpose } {}

			void bindData(uint32_t location) const override
			{
				glUniformMatrix2dv(location, query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					needs_transpose, static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					static_cast<type_value::value_type*>(data), needs_transpose };
			}
		};



		template<>
		struct uniform_assignment_query<mat4x3> : public AbstractQuery
		{
			typedef mat4x3 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data } {}
			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data, bool needs_transpose) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data, needs_transpose } {}

			void bindData(uint32_t location) const override
			{
				glUniformMatrix4x3fv(location, query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					needs_transpose, static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					static_cast<type_value::value_type*>(data), needs_transpose };
			}
		};

		template<>
		struct uniform_assignment_query<dmat4x3> : public AbstractQuery
		{
			typedef dmat4x3 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data } {}
			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data, bool needs_transpose) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data, needs_transpose } {}

			void bindData(uint32_t location) const override
			{
				glUniformMatrix4x3dv(location, query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					needs_transpose, static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					static_cast<type_value::value_type*>(data), needs_transpose };
			}
		};



		template<>
		struct uniform_assignment_query<mat4x2> : public AbstractQuery
		{
			typedef mat4x2 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data } {}
			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data, bool needs_transpose) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data, needs_transpose } {}

			void bindData(uint32_t location) const override
			{
				glUniformMatrix4x2fv(location, query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					needs_transpose, static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					static_cast<type_value::value_type*>(data), needs_transpose };
			}
		};

		template<>
		struct uniform_assignment_query<dmat4x2> : public AbstractQuery
		{
			typedef dmat4x2 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data } {}
			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data, bool needs_transpose) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data, needs_transpose } {}

			void bindData(uint32_t location) const override
			{
				glUniformMatrix4x2dv(location, query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					needs_transpose, static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					static_cast<type_value::value_type*>(data), needs_transpose };
			}
		};



		template<>
		struct uniform_assignment_query<mat3x4> : public AbstractQuery
		{
			typedef mat3x4 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data } {}
			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data, bool needs_transpose) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data, needs_transpose } {}

			void bindData(uint32_t location) const override
			{
				glUniformMatrix3x4fv(location, query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					needs_transpose, static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					static_cast<type_value::value_type*>(data), needs_transpose };
			}
		};

		template<>
		struct uniform_assignment_query<dmat3x4> : public AbstractQuery
		{
			typedef dmat3x4 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data } {}
			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data, bool needs_transpose) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data, needs_transpose } {}

			void bindData(uint32_t location) const override
			{
				glUniformMatrix3x4dv(location, query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					needs_transpose, static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					static_cast<type_value::value_type*>(data), needs_transpose };
			}
		};



		template<>
		struct uniform_assignment_query<mat2x4> : public AbstractQuery
		{
			typedef mat2x4 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data } {}
			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data, bool needs_transpose) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data, needs_transpose } {}

			void bindData(uint32_t location) const override
			{
				glUniformMatrix2x4fv(location, query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					needs_transpose, static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					static_cast<type_value::value_type*>(data), needs_transpose };
			}
		};

		template<>
		struct uniform_assignment_query<dmat2x4> : public AbstractQuery
		{
			typedef dmat2x4 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data } {}
			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data, bool needs_transpose) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data, needs_transpose } {}

			void bindData(uint32_t location) const override
			{
				glUniformMatrix2x4dv(location, query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					needs_transpose, static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					static_cast<type_value::value_type*>(data), needs_transpose };
			}
		};



		template<>
		struct uniform_assignment_query<mat3x2> : public AbstractQuery
		{
			typedef mat3x2 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data } {}
			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data, bool needs_transpose) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data, needs_transpose } {}

			void bindData(uint32_t location) const override
			{
				glUniformMatrix3x2fv(location, query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					needs_transpose, static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					static_cast<type_value::value_type*>(data), needs_transpose };
			}
		};

		template<>
		struct uniform_assignment_query<dmat3x2> : public AbstractQuery
		{
			typedef dmat3x2 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data } {}
			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data, bool needs_transpose) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data, needs_transpose } {}

			void bindData(uint32_t location) const override
			{
				glUniformMatrix3x2dv(location, query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					needs_transpose, static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					static_cast<type_value::value_type*>(data), needs_transpose };
			}
		};



		template<>
		struct uniform_assignment_query<mat2x3> : public AbstractQuery
		{
			typedef mat2x3 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data } {}
			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data, bool needs_transpose) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data, needs_transpose } {}

			void bindData(uint32_t location) const override
			{
				glUniformMatrix2x3fv(location, query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					needs_transpose, static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					static_cast<type_value::value_type*>(data), needs_transpose };
			}
		};

		template<>
		struct uniform_assignment_query<dmat2x3> : public AbstractQuery
		{
			typedef dmat2x3 type_value;

			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data } {}
			uniform_assignment_query(uint32_t count, const type_value::value_type* raw_data, bool needs_transpose) :
				AbstractQuery{ sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns*count, raw_data, needs_transpose } {}

			void bindData(uint32_t location) const override
			{
				glUniformMatrix2x3dv(location, query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					needs_transpose, static_cast<type_value::value_type*>(data));
			}

			AbstractQuery* clone() const override
			{
				return new uniform_assignment_query{ query_size / (sizeof(type_value::value_type)*type_value::num_rows*type_value::num_columns),
					static_cast<type_value::value_type*>(data), needs_transpose };
			}
		};

		std::map<uint32_t, AbstractQuery*> location_based_uniform_assignment_map;		//uniform assignment map based on locations of uniform program variables
		std::map<uint32_t, std::string> vertex_attribute_binding_map;	//map determining binding relation between generic vertex attribute ids and string names of variables declared in the vertex shader of the program
		std::map<std::pair<PipelineStage, uint32_t>, uint32_t> stage_location_based_subroutine_uniform_selection_map;	//subroutine selection map based on program stage and subroutine uniform location within the stage


		inline void initialize_shader_program();	//initializes new shader program object. Used by object constructors
		void relink();		//re-links program using parameters from the last linking. This function guarantees that a re-link attempt will be made regardless of current values of error_state and needs_relink

	protected:
		static std::string shader_base_catalog;	//catalog name of the shader base

		GLuint getOpenGLProgramId() const;	//returns internal OpenGL program id
		void made_active() const;	//performs necessary routines that need to be done right AFTER the program object has become active (generic vertex attribute bindings, uniform assignments etc.)
	
		//Constructor-destructor infrastructure is protected since ShaderProgram can not exist on its own

		explicit ShaderProgram(const std::string& program_class_string_name);	//Default initializer	
		ShaderProgram(const std::string& program_class_string_name, const std::string& program_string_name);		//Initializes program using user-defined string name
		ShaderProgram(const ShaderProgram& other);		//Copy constructor
		ShaderProgram(ShaderProgram&& other);	//Move constructor

		//initializes program from a number of GLSL shader sources stored in files
		ShaderProgram(const std::string& program_class_string_name, const std::string& program_string_name, std::vector<std::string> file_sources, std::vector<ShaderType> shader_type);

		//initializes program using a number of shader GLSL textual sources located in memory
		ShaderProgram(const std::string& program_class_string_name, const std::string& program_string_name, const std::vector<GLSLSourceCode>& glsl_sources, std::vector<ShaderType> shader_type);
		
		//initializes program object with previously compiled binary source
		ShaderProgram(const std::string& program_class_string_name, const std::string& program_string_name, const std::string& shader_binary_source);
		
		//initializes program object using provided set of shader objects
		ShaderProgram(const std::string& program_class_string_name, const std::string& program_string_name, const std::vector<Shader>& shaders);

		ShaderProgram& operator=(const ShaderProgram& other);	//assigns "other" to "this" shader program
		ShaderProgram& operator=(ShaderProgram&& other);	//move assignment

	public:
		virtual ~ShaderProgram();	//Destructor

		//The following functions allow to modify and retrieve active shader look-up preferences
		static void setShaderBaseCatalog(const std::string& new_catalog);		//sets new directory to look for the shader sources
		static std::string getShaderBaseCatalog();	//returns the directory where the engine currently looks for the shaders

		//The following functions alter object's state. All boolean functions return 'true' on successful operation and 'false' otherwise

		//Adds new shader object to the program and returns 'true' on success
		bool addShader(const Shader& shader);

		//Moves shader into the program object and returns 'true' on success
		bool addShader(Shader&& shader);

		//Removes shader object from the program based on provided string name. If  program contains several shader objects having the same string name, only the first found will be removed.
		//The function returns 'true' on successful removal and 'false' if no shader objects with the given string name have been attached to the program.
		bool removeShader(const std::string& shader_string_name);	

		//Removes shader object from the program based on the given strong identifier. Note: if shader object has been added to the program by copying, then the copy contained in the program
		//would have its own strong identifier different from the original 's
		bool removeShader(uint32_t shader_id);

		//Returns 'true' if program contains at least one shader object having the given string name
		bool containsShader(const std::string& shader_string_name) const;

		//Returns 'true' if program contains shader object with the given strong identifier
		bool containsShader(uint32_t shader_id) const;

		//Returns pointer to the shader object owned by the program, which has the given string name. If program has several shader objects having the requested name, then
		//the pointer to the first one found is returned. If program does not contain shader objects with the given string name, the function returns nullptr
		const Shader* retrieveShader(const std::string& shader_string_name) const;	

		//Returns pointer to the shader object contained in the program, which has the given strong identifier. If requested shader object can not be retrieved, the function returns nullptr
		const Shader* retrieveShader(uint32_t shader_id) const;


		//Bind vertex attribute id to vertex shader variable referred by "variable_name". NOTE: vertex attribute bindings should be performed BEFORE the program gets linked.
		void bindVertexAttributeId(const std::string& variable_name, uint32_t vertex_attribute_id);

		//NOTE: If program was initialized from precompiled binary or linked to a binary source, it is still possible to use usual addShader()/removeShader()/restoreShader() and link() functionality.
		//Nevertheless, usage of link() and linkBinary() is mutualy exclusive in the sence that these commands when called completely replace the effect of the previously called counterpart.
		//In other words, if linkBinary() was called after a call of link() the code stored in the program object is effectively replaced by the binary representation referrenced by linkBinary().
		//The same rule applies when link() is called after linkBinary(): in this case program gets composed of the shaders previously attached by addShader()/restoreShader() and any code activated
		//by any previous call of linkBinary() gets replaced.

		bool link();	//Links the program
		bool linkBinary(const std::string& binary_source);	//links program using previously compiled binary. This function does not get affected by the current error state of the object.

		void allowBinaryRepresentation(bool binary_representation_flag);	//Allows binary representation for the shader. This function does not get affected by the current error state of the object.

		void reset();	//resets object to its initial state. Reset is possible regardless of the current error state.

		//Assigns subroutine having provided string name to subroutine uniform variable with given location at given graphical pipeline stage. Be aware, however, that this function has no ability to check, whether 
		//the subroutine being assigned to the subroutine uniform is compatible with this uniform on the code level of the shader. Therefore, the caller is obliged to ensure that the program defines uniforms in a consistent manner.
		void assignSubroutineUniform(uint32_t subroutine_uniform_location, PipelineStage pipeline_stage, const std::string& subroutine_string_name);

		//Assigns subroutine having provided string name to subroutine uniform variable with given string name at given graphical pipeline stage. Be aware, however, that this function has no ability to check, whether 
		//the subroutine being assigned to the subroutine uniform is compatible with this uniform on the code level of the shader. Therefore, the caller is obliged to ensure that the program defines uniforms in a consistent manner.
		void assignSubroutineUniform(const std::string& subroutine_uniform_name, PipelineStage pipeline_stage, const std::string& subroutine_string_name);

		//Returns data size of uniform block with the given name. The ShaderProgram object is put into an erroneous state if uniform block with requested name is not defined in the program
		size_t getUniformBlockDataSize(const std::string& uniform_block_name) const;

		//Assigns uniform block with given name to std140 uniform buffer object. Note, that it is important that the uniform block being assigned to the buffer object must have been declared
		//using std140 layout qualifier. The function does not put the ShaderProgram object into an erroneous state if std140 layout qualifier was missing from declaration of the 
		//uniform block. However, the data that will populate members of such block is undefined.
		void assignUniformBlockToBuffer(const std::string& uniform_block_name, const std140UniformBuffer& std140_uniform_buffer) const;

		//Assigns uniform block with given string name to a uniform binding point. To later on populate this uniform block with data, one has to bind a uniform buffer object to the given binding point.
		void assignUniformBlockToBuffer(const std::string& uniform_block_name, uint32_t binding_point) const;

		virtual ShaderProgram* clone() const = 0;	//Clones shader program object

		//Helper functions: allow to gather information about the object's state. These functions can be called regardless of the current error state of the object.

		bool needsRelink() const;	//returns 'true' if the program has to be relinked
		bool doesAllowBinary() const;	//returns 'true' if contained program allows binary representation, which is usually used to avoid repetitive compilation of the shaders on the next program run
		const void* getBinary() const; //returns binary representation of contained program or nullptr if binary representation is not allowed or is not available (program is not compiled or not linked)
		bool serializeBinary(const std::string& file_name) const; //saves binary representation of contained program to disk. Returns 'true' on success and 'false' otherwise.
		bool isActive() const;	//returns 'true' if contained program is currently in use as a complete program object or if it is attached to currently bound program pipeline
		GLbitfield getProgramStages() const;	//returns shader stages used in contained program as bitwise OR product of the corresponding shader bits (GL_*_SHADER_BIT)
		bool containsStage(ShaderType shader_type) const;	//checks if shader of type shader_type is included into the program and returns 'true' in case of success. If shader of the given type is missing from the program returns 'false'
		bool containsStages(GLbitfield stage_bits) const;   //checks whether the program contains shader objects corresponding to the pipeline stages defined by the bitfield stage_bits.
		virtual bool isSeparate() const = 0;	//returns true if program object is a separate program that should be attached to a program pipeline
		linking_type getLinkingType() const;	//returns linkage type of the program (binary linking, linking from compiled shader objects or no linking)

		operator GLuint() const; //returns internal OpenGL id of contained program. Needed to allow mixing of TinyWorld and pure OpenGL code

		//The following functions allow to pass data to the shader program

		//uniform scalar assignments
		template<typename scalar_type>
		void assignUniformScalar(uint32_t location, scalar_type scalar)
		{
			if (needs_relink)
			{
				set_error_state(true);
				std::string err_msg = "Can not assigned value to uniform scalar variable located at index \"" +
					std::to_string(location) + "\" in program \"" + getStringName() + "\". The program was not properly linked.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return;
			}
				

			typedef ogl_type_mapper<scalar_type>::ogl_type gl_type;
			gl_type raw_data = static_cast<gl_type>(scalar);

			if (location_based_uniform_assignment_map.find(location) == location_based_uniform_assignment_map.end())
				location_based_uniform_assignment_map.insert(std::make_pair(location,
				new typename uniform_assignment_query<gl_type>(1, &raw_data)));
			else
				*dynamic_cast<uniform_assignment_query<gl_type>*>(location_based_uniform_assignment_map.at(location)) =
				uniform_assignment_query<gl_type>{1, &raw_data };

			if (isActive())
				location_based_uniform_assignment_map.at(location)->bindData(location);
		}


		template<typename scalar_type>
		void assignUniformScalar(const std::string& name, scalar_type scalar)
		{
			if (needs_relink)
			{
				set_error_state(true);
				std::string err_msg = "Can not assign value to uniform scalar variable \"" +
					name + "\" in program \"" + getStringName() + "\". The program was not properly linked.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return;
			}
				
			
			GLint location;
			if ((location = glGetUniformLocation(ogl_program_id, name.c_str())) == -1)
			{
				set_error_state(true);
				std::string err_msg = "Unable to retrieve location of uniform scalar variable \"" + name + "\". The variable might be inactive.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return;
			}
			assignUniformScalar(location, scalar);
		}

		

		//uniform vector assignments
		template<typename glsl_vector_type>
		void assignUniformVector(uint32_t location, glsl_vector_type vector)
		{
			if (needs_relink)
			{
				set_error_state(true);
				std::string err_msg = "Can not assigned value to uniform vector variable located at index \"" +
					std::to_string(location) + "\" in program \"" + getStringName() + "\". The program was not properly linked.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return;
			}
			
			typename glsl_vector_type::value_type raw_data[typename glsl_vector_type::dimension];

			for (int i = 0; i < glsl_vector_type::dimension; ++i)
				raw_data[i] = vector[i];

			if (location_based_uniform_assignment_map.find(location) == location_based_uniform_assignment_map.end())
				location_based_uniform_assignment_map.insert(std::make_pair(location,
				new typename uniform_assignment_query<glsl_vector_type>(1, raw_data)));
			else
				*dynamic_cast<uniform_assignment_query<glsl_vector_type>*>(location_based_uniform_assignment_map.at(location)) =
				uniform_assignment_query<glsl_vector_type>{1, raw_data };

			if (isActive())
				location_based_uniform_assignment_map.at(location)->bindData(location);
		}


		template<typename glsl_vector_type>
		void assignUniformVector(const std::string& name, glsl_vector_type vector)
		{
			if (needs_relink)
			{
				set_error_state(true);
				std::string err_msg = "Can not assign value to uniform vector variable \"" +
					name + "\" in program \"" + getStringName() + "\". The program was not properly linked.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return;
			}
				

			GLint location;
			if ((location = glGetUniformLocation(ogl_program_id, name.c_str())) == -1)
			{
				set_error_state(true);
				std::string err_msg = "Unable to retrieve location of uniform vector variable \"" + name + "\". The variable might be inactive.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return;
			}
			assignUniformVector(location, vector);
		}



		//uniform matrix assignments
		template<typename glsl_matrix_type>
		void assignUniformMatrix(uint32_t location, glsl_matrix_type matrix, bool transpose = false)
		{
			if (needs_relink)
			{
				set_error_state(true);
				std::string err_msg = "Can not assigned value to uniform matrix variable located at index \"" +
					std::to_string(location) + "\" in program \"" + getStringName() + "\". The program was not properly linked.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return;
			}
				

			const glsl_matrix_type::value_type* raw_data = matrix.getRawData();

			if (location_based_uniform_assignment_map.find(location) == location_based_uniform_assignment_map.end())
				location_based_uniform_assignment_map.insert(std::make_pair(location,
				new typename uniform_assignment_query<glsl_matrix_type>(1, raw_data, transpose)));
			else
				*dynamic_cast<uniform_assignment_query<glsl_matrix_type>*>(location_based_uniform_assignment_map.at(location)) =
				uniform_assignment_query<glsl_matrix_type>{1, raw_data, transpose };

			if (isActive())
				location_based_uniform_assignment_map.at(location)->bindData(location);
		}


		template<typename glsl_matrix_type>
		void assignUniformMatrix(const std::string& name, glsl_matrix_type matrix, bool transpose = false)
		{
			if (needs_relink)
			{
				set_error_state(true);
				std::string err_msg = "Can not assign value to uniform matrix variable \"" +
					name + "\" in program \"" + getStringName() + "\". The program was not properly linked.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return;
			}
				
			
			GLint location;
			if ((location = glGetUniformLocation(ogl_program_id, name.c_str())) == -1)
			{
				set_error_state(true);
				std::string err_msg = "Unable to retrieve location of uniform matrix variable \"" + name + "\". The variable might be inactive.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return;
			}
			assignUniformMatrix(location, matrix, transpose);
		}



		//uniform scalar assignments: array forms
		template<typename scalar_type>
		void assignUniformScalar(uint32_t location, const std::vector<scalar_type>& scalars)
		{
			if (needs_relink)
			{
				set_error_state(true);
				std::string err_msg = "Can not assign value to uniform scalar array located at index \"" +
					std::to_string(location) + "\" in program \"" + getStringName() + "\". The program was not properly linked.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return;
			}
				

			typedef ogl_type_mapper<scalar_type>::ogl_type gl_type;

			const gl_type* raw_data = static_cast<gl_type*>(scalars.data());

			if (location_based_uniform_assignment_map.find(location) == location_based_uniform_assignment_map.end())
				location_based_uniform_assignment_map.insert(std::make_pair(location,
				new typename uniform_assignment_query<gl_type>(scalars.size(), raw_data)));
			else
				*dynamic_cast<uniform_assignment_query<gl_type>*>(location_based_uniform_assignment_map.at(location)) =
				uniform_assignment_query<gl_type>{scalars.size(), raw_data };

			if (isActive())
				location_based_uniform_assignment_map.at(location)->bindData(location);
		}


		template<typename scalar_type>
		void assignUniformScalar(const std::string& name, const std::vector<scalar_type>& scalars)
		{
			if (needs_relink)
			{
				set_error_state(true);
				std::string err_msg = "Can not assign value to uniform scalar array \"" +
					name + "[]\" in program \"" + getStringName() + "\". The program was not properly linked.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return;
			}
				

			GLint location;
			if ((location = glGetUniformLocation(ogl_program_id, name.c_str())) == -1)
			{
				set_error_state(true);
				std::string err_msg = "Unable to retrieve location of uniform scalar array variable \"" + name + "\". The variable might be inactive.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return;
			}
			assignUniformScalar(location, scalars);
		}
		


		//uniform vector assignments: array forms
		template<typename glsl_vector_type>
		void assignUniformVector(uint32_t location, const std::vector<glsl_vector_type>& vectors)
		{
			if (needs_relink)
			{
				set_error_state(true);
				std::string err_msg = "Can not assign value to uniform vector array located at index \"" +
					std::to_string(location) + "\" in program \"" + getStringName() + "\". The program was not properly linked.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return;
			}
				

			typename glsl_vector_type::value_type *raw_data =
				new typename glsl_vector_type::value_type[vectors.size()*glsl_vector_type::dimension];

			for (unsigned int i = 0; i < vectors.size(); ++i)
				for (int j = 0; j < glsl_vector_type::dimension; ++j)
					raw_data[glsl_vector_type::dimension * i + j] = vectors[i][j];

			if (location_based_uniform_assignment_map.find(location) == location_based_uniform_assignment_map.end())
				location_based_uniform_assignment_map.insert(std::make_pair(location,
				new typename uniform_assignment_query<glsl_vector_type>(vectors.size(), raw_data)));
			else
				*dynamic_cast<uniform_assignment_query<glsl_vector_type>*>(location_based_uniform_assignment_map.at(location)) =
				uniform_assignment_query<glsl_vector_type>{vectors.size(), raw_data };

			if (isActive())
				location_based_uniform_assignment_map.at(location)->bindData(location);

			delete[] raw_data;
		}


		template<typename glsl_vector_type>
		void assignUniformVector(const std::string& name, const std::vector<glsl_vector_type>& vectors)
		{
			if (needs_relink)
			{
				set_error_state(true);
				std::string err_msg = "Can not assign value to uniform vector array \"" +
					name + "[]\" in program \"" + getStringName() + "\". The program was not properly linked.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return;
			}
				

			GLint location;
			if ((location = glGetUniformLocation(ogl_program_id, name.c_str())) == -1)
			{
				set_error_state(true);
				std::string err_msg = "Unable to retrieve location of uniform vector array variable \"" + name + "\". The variable might be inactive.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return;
			}
			assignUniformVector(location, vectors);
		}
		
		

		//uniform matrix assignments: array forms
		template<typename glsl_matrix_type>
		void assignUniformMatrix(uint32_t location, std::vector<glsl_matrix_type>& matrices, bool transpose = false)
		{
			if (needs_relink)
			{
				set_error_state(true);
				std::string err_msg = "Can not assign value to uniform matrix array located at index \"" +
					std::to_string(location) + "\" in program \"" + getStringName() + "\". The program was not properly linked.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return;
			}
				

			typename glsl_matrix_type::value_type* raw_data =
				new typename glsl_matrix_type::value_type[matrices.size()*glsl_matrix_type::num_rows*glsl_matrix_type::num_columns];

			for (unsigned int i = 0; i < matrices.size(); ++i)
				memcpy(&raw_data[i*glsl_matrix_type::num_rows*glsl_matrix_type::num_columns],
				matrices[i].getRawData(),
				glsl_matrix_type::num_rows*glsl_matrix_type::num_columns*sizeof(typename glsl_matrix_type::value_type));

			if (location_based_uniform_assignment_map.find(location) == location_based_uniform_assignment_map.end())
				location_based_uniform_assignment_map.insert(std::make_pair(location,
				new typename uniform_assignment_query<glsl_matrix_type>(matrices.size(), raw_data, transpose)));
			else
				*dynamic_cast<uniform_assignment_query<glsl_matrix_type>*>(location_based_uniform_assignment_map.at(location)) =
				uniform_assignment_query<glsl_matrix_type>{matrices.size(), raw_data, transpose };

			if (isActive())
				location_based_uniform_assignment_map.at(location)->bindData(location);

			delete[] raw_data;
		}


		template<typename glsl_matrix_type>
		void assignUniformMatrix(const std::string& name, std::vector<glsl_matrix_type>& matrices, bool transpose = false)
		{
			if (needs_relink)
			{
				set_error_state(true);
				std::string err_msg = "Can not assign value to uniform matrix array \"" +
					name + "[]\" in program \"" + getStringName() + "\". The program was not properly linked.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return;
			}
				

			GLint location;
			if ((location = glGetUniformLocation(ogl_program_id, name.c_str())) == -1)
			{
				set_error_state(true);
				std::string err_msg = "Unable to retrieve location of uniform matrix array variable \"" + name + "\". The variable might be inactive.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return;
			}
			assignUniformMatrix(location, matrices, transpose);
		}



		//the following functionality allows to retrieve data from uniform variables defined in shader program
		template<typename T> struct gl_get_uniform_forward_call;	//forward call for glGetUniform*v functions

		template<>
		struct gl_get_uniform_forward_call<GLint>
		{
			typedef GLint call_type;

			static void getUniform(GLuint program, GLint location, GLint* params)
			{
				glGetUniformiv(program, location, params);
			}

		};

		template<>
		struct gl_get_uniform_forward_call<GLuint>
		{
			typedef GLuint call_type;

			static void getUniform(GLuint program, GLint location, GLuint* params)
			{
				glGetUniformuiv(program, location, params);
			}

		};

		template<>
		struct gl_get_uniform_forward_call<GLfloat>
		{
			typedef GLfloat call_type;

			static void getUniform(GLuint program, GLint location, GLfloat* params)
			{
				glGetUniformfv(program, location, params);
			}

		};

		template<>
		struct gl_get_uniform_forward_call<GLdouble>
		{
			typedef GLdouble call_type;

			static void getUniform(GLuint program, GLint location, GLdouble* params)
			{
				glGetUniformdv(program, location, params);
			}

		};

		//scalar uniforms retrieval 
		template<typename T>
		T& getUniformScalarValue(uint32_t location, T* value) const
		{
			if (needs_relink)
			{
				set_error_state(true);
				std::string err_msg = "Unable to retrieve value of uniform scalar variable located at index \"" +
					std::to_string(location) + "\" in program \"" + program_string_name + "\". The program was not properly linked.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return T{};
			}
				

			gl_get_uniform_forward_call<typename ogl_type_mapper<T>::ogl_type>::getUniform(ogl_program_id, location, static_cast<typename ogl_type_mapper<T>::ogl_type*>(value));
			return *value;
		}

		template<typename T>
		T& getUniformScalarValue(const std::string& name, T* value) const
		{
			if (needs_relink)
			{
				set_error_state(true);
				std::string err_msg = "Unable to retrieve value of uniform scalar variable \"" +
					name + "\" in program \"" + program_string_name + "\". The program was not properly linked.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return T{};
			}
				

			GLint location;
			if ((location = glGetUniformLocation(ogl_program_id, name.c_str())) == -1)
			{
				set_error_state(true);
				std::string err_msg = "Unable to retrieve location of uniform scalar variable \"" + name + "\". The variable might be inactive.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return;
			}
			gl_get_uniform_forward_call<typename ogl_type_mapper<T>::ogl_type>::getUniform(ogl_program_id, location, static_cast<typename ogl_type_mapper<T>::ogl_type*>(value));
			return *value;
		}



		//vector uniform retrieval
		template<typename glsl_vector_type>
		glsl_vector_type& getUniformVectorValue(uint32_t location, glsl_vector_type* value) const
		{
			if (needs_relink)
			{
				set_error_state(true);
				std::string err_msg = "Unable to retrieve value of uniform vector variable located at index \"" +
					std::to_string(location) + "\" in program \"" + program_string_name + "\". The program was not properly linked.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return glsl_vector_type{};
			}
				

			typedef typename glsl_vector_type::value_type gl_type;
			gl_type* params = new gl_type[glsl_vector_type::dimension];

			gl_get_uniform_forward_call<glsl_vector_type::value_type>::getUniform(ogl_program_id, location, params);

			for (int i = 0; i < glsl_vector_type::dimension; ++i)
				(*value)[i] = params[i];

			delete[] params;
			return *value;
		}

		template<typename glsl_vector_type>
		glsl_vector_type& getUniformVectorValue(const std::string& name, glsl_vector_type* value) const
		{
			if (needs_relink)
			{
				set_error_state(true);
				std::string err_msg = "Unable to retrieve value of uniform vector variable \"" +
					name + "\" in program \"" + program_string_name + "\". The program was not properly linked.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return glsl_vector_type{};
			}
				

			GLint location;
			if ((location = glGetUniformLocation(ogl_program_id, name.c_str())) == -1)
			{
				set_error_state(true);
				std::string err_msg = "Unable to retrieve location of uniform scalar variable \"" + name + "\". The variable might be inactive.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return;
			}
			getUniformVectorValue(location, value);
			return *value;
		}



		//matrix uniform retrieval
		template<typename glsl_matrix_type>
		glsl_matrix_type& getUniformMatrixValue(uint32_t location, glsl_matrix_type* value) const
		{
			if (needs_relink)
			{
				set_error_state(true);
				std::string err_msg = "Unable to retrieve value of uniform matrix variable located at index \"" +
					std::to_string(location) + "\" in program \"" + program_string_name + "\". The program was not properly linked.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return glsl_matrix_type{};
			}
				

			typedef typename glsl_matrix_type::value_type gl_type;
			const auto num_rows = glsl_matrix_type::num_rows;
			const auto num_columns = glsl_matrix_type::num_columns;

			gl_type* params = new gl_type[num_rows*num_columns];
			gl_get_uniform_forward_call<gl_type>::getUniform(ogl_program_id, location, params);

			memcpy(value->getRawData(), params, num_rows*num_columns*sizeof(gl_type));

			delete[] params;
			return *value;
		}

		template<typename glsl_matrix_type>
		glsl_matrix_type& getUniformMatrixValue(const std::string& name, glsl_matrix_type* value) const
		{
			if (needs_relink)
			{
				set_error_state(true);
				std::string err_msg = "Unable to retrieve value of uniform scalar variable \"" +
					name + "\" in program \"" + program_string_name + "\". The program was not properly linked.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return glsl_matrix_type{};
			}
				

			GLint location;
			if ((location = glGetUniformLocation(ogl_program_id, name.c_str())) == -1)
			{
				set_error_state(true);
				std::string err_msg = "Unable to retrieve location of uniform scalar variable \"" + name + "\". The variable might be inactive.";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return;
			}
			getUniformMatrixValue(location, value);
			return *value;
		}
		
	};

}

#define TW__SHADER_PROGRAM_H__
#endif