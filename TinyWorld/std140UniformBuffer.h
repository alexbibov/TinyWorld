//Implements wrapper over OpenGL uniform buffer defined using std140 layout

#ifndef TW__std140UniformBuffer__

#include <vector>
#include <cstdarg>

#include "Buffer.h"
#include "MatrixTypes.h"
#include "VectorTypes.h"


namespace tiny_world
{


	class std140UniformBuffer : public Buffer
	{
	private:
		ptrdiff_t offset;
		uint32_t binding_point;

		//The following templates provide alignment information for all base GLSL types and for arrays of such types
		template<typename T> struct alignment_traits;


		//*****************************************************************************************GLSL base types*****************************************************************************************
		template<typename T, bool is_double_precision> struct glsl_scalar_types_alignment_traits;

		template<typename T>
		struct glsl_scalar_types_alignment_traits < T, false >
		{
			const static unsigned char base_alignment = sizeof(T);
			const static unsigned char log2_base_alignment = 2;
			const static bool is_array = false;
			const static unsigned int element_size = sizeof(T);
		};

		template<typename T>
		struct glsl_scalar_types_alignment_traits < T, true >
		{
			const static unsigned char base_alignment = sizeof(T);
			const static unsigned char log2_base_alignment = 3;
			const static bool is_array = false;
			const static unsigned int element_size = sizeof(T);
		};

		template<typename T>
		struct glsl_scalar_types_alignment_traits < std::vector<T>, false >
		{
			const static unsigned char base_alignment = 16;
			const static unsigned char log2_base_alignment = 4;
			const static bool is_array = true;
			const static unsigned int element_size = 16;	//for arrays element_size represents size of a single array "bucket"
		};



		template<> struct alignment_traits<GLint> : public glsl_scalar_types_alignment_traits < GLint, false > {};
		template<> struct alignment_traits<GLuint> : public glsl_scalar_types_alignment_traits < GLuint, false >{};
		template<> struct alignment_traits<GLboolean> : public glsl_scalar_types_alignment_traits < GLboolean, false >{};
		template<> struct alignment_traits<GLfloat> : public glsl_scalar_types_alignment_traits < GLfloat, false >{};
		template<> struct alignment_traits<GLdouble> : public glsl_scalar_types_alignment_traits < GLdouble, true >{};



		template<> struct alignment_traits<std::vector<GLint>> : public glsl_scalar_types_alignment_traits < std::vector<GLint>, false >{};
		template<> struct alignment_traits<std::vector<GLuint>> : public glsl_scalar_types_alignment_traits < std::vector<GLuint>, false >{};
		template<> struct alignment_traits<std::vector<GLboolean>> : public glsl_scalar_types_alignment_traits < std::vector<GLboolean>, false >{};
		template<> struct alignment_traits<std::vector<GLfloat>> : public glsl_scalar_types_alignment_traits < std::vector<GLfloat>, false >{};
		template<> struct alignment_traits<std::vector<GLdouble>> : public glsl_scalar_types_alignment_traits < std::vector<GLdouble>, false >{};	//is_double_precesion here is set to 'false' since base alignment for arrays of scalar doubles is the same as for arrays of scalar 4-byte elements (int, float, etc.)



		//*****************************************************************************************GLSL vector types*****************************************************************************************
		template<typename T, bool is_double_precision, unsigned int dimension> struct glsl_vector_types_alignment_traits;

		template<typename T>
		struct glsl_vector_types_alignment_traits < T, false, 2 >
		{
			const static unsigned char base_alignment = 8;
			const static unsigned char log2_base_alignment = 3;
			const static bool is_array = false;
			const static unsigned int element_size = sizeof(typename T::value_type) * 2;
		};

		template<typename T>
		struct glsl_vector_types_alignment_traits < T, false, 3 >
		{
			const static unsigned char base_alignment = 16;
			const static unsigned char log2_base_alignment = 4;
			const static bool is_array = false;
			const static unsigned int element_size = sizeof(typename T::value_type) * 3;
		};

		template<typename T>
		struct glsl_vector_types_alignment_traits < T, false, 4 >
		{
			const static unsigned char base_alignment = 16;
			const static unsigned char log2_base_alignment = 4;
			const static bool is_array = false;
			const static unsigned int element_size = sizeof(typename T::value_type) * 4;
		};

		template<typename T>
		struct glsl_vector_types_alignment_traits < T, true, 2 >
		{
			const static unsigned char base_alignment = 16;
			const static unsigned char log2_base_alignment = 4;
			const static bool is_array = false;
			const static unsigned int element_size = sizeof(typename T::value_type) * 2;
		};

		template<typename T>
		struct glsl_vector_types_alignment_traits < T, true, 3 >
		{
			const static unsigned char base_alignment = 32;
			const static unsigned char log2_base_alignment = 5;
			const static bool is_array = false;
			const static unsigned int element_size = sizeof(typename T::value_type) * 3;
		};

		template<typename T>
		struct glsl_vector_types_alignment_traits < T, true, 4 >
		{
			const static unsigned char base_alignment = 32;
			const static unsigned char log2_base_alignment = 5;
			const static bool is_array = false;
			const static unsigned int element_size = sizeof(typename T::value_type) * 4;
		};



		template<typename T>
		struct glsl_vector_types_alignment_traits < std::vector<T>, false, 2 >
		{
			const static unsigned char base_alignment = 16;
			const static unsigned char log2_base_alignment = 4;
			const static bool is_array = true;
			const static unsigned int element_size = 16;	//for arrays element_size represents size of a single array "bucket"
		};

		template<typename T>
		struct glsl_vector_types_alignment_traits < std::vector<T>, false, 3 >
		{
			const static unsigned char base_alignment = 16;
			const static unsigned char log2_base_alignment = 4;
			const static bool is_array = true;
			const static unsigned int element_size = 16;	//for arrays element_size represents size of a single array "bucket"
		};

		template<typename T>
		struct glsl_vector_types_alignment_traits < std::vector<T>, false, 4 >
		{
			const static unsigned char base_alignment = 16;
			const static unsigned char log2_base_alignment = 4;
			const static bool is_array = true;
			const static unsigned int element_size = 16;	//for arrays element_size represents size of a single array "bucket"
		};

		template<typename T>
		struct glsl_vector_types_alignment_traits < std::vector<T>, true, 2 >
		{
			const static unsigned char base_alignment = 16;
			const static unsigned char log2_base_alignment = 4;
			const static bool is_array = true;
			const static unsigned int element_size = 16;	//for arrays element_size represents size of a single array "bucket"
		};

		template<typename T>
		struct glsl_vector_types_alignment_traits < std::vector<T>, true, 3 >
		{
			const static unsigned char base_alignment = 32;
			const static unsigned char log2_base_alignment = 5;
			const static bool is_array = true;
			const static unsigned int element_size = 32;	//for arrays element_size represents size of a single array "bucket"
		};

		template<typename T>
		struct glsl_vector_types_alignment_traits < std::vector<T>, true, 4 >
		{
			const static unsigned char base_alignment = 32;
			const static unsigned char log2_base_alignment = 5;
			const static bool is_array = true;
			const static unsigned int element_size = 32;	//for arrays element_size represents size of a single array "bucket"
		};



		template<> struct alignment_traits<ivec2> : public glsl_vector_types_alignment_traits < ivec2, false, 2 >{};
		template<> struct alignment_traits<uvec2> : public glsl_vector_types_alignment_traits < uvec2, false, 2 >{};
		template<> struct alignment_traits<bvec2> : public glsl_vector_types_alignment_traits < bvec2, false, 2 >{};
		template<> struct alignment_traits<vec2> : public glsl_vector_types_alignment_traits < vec2, false, 2 >{};
		template<> struct alignment_traits<dvec2> : public glsl_vector_types_alignment_traits < dvec2, true, 2 >{};

		template<> struct alignment_traits<ivec3> : public glsl_vector_types_alignment_traits < ivec3, false, 3 > {};
		template<> struct alignment_traits<uvec3> : public glsl_vector_types_alignment_traits < uvec3, false, 3 > {};
		template<> struct alignment_traits<bvec3> : public glsl_vector_types_alignment_traits < bvec3, false, 3 > {};
		template<> struct alignment_traits<vec3> : public glsl_vector_types_alignment_traits < vec3, false, 3 > {};
		template<> struct alignment_traits<dvec3> : public glsl_vector_types_alignment_traits < dvec3, true, 3 > {};

		template<> struct alignment_traits<ivec4> : public glsl_vector_types_alignment_traits < ivec4, false, 4 > {};
		template<> struct alignment_traits<uvec4> : public glsl_vector_types_alignment_traits < uvec4, false, 4 > {};
		template<> struct alignment_traits<bvec4> : public glsl_vector_types_alignment_traits < bvec4, false, 4 > {};
		template<> struct alignment_traits<vec4> : public glsl_vector_types_alignment_traits < vec4, false, 4 > {};
		template<> struct alignment_traits<dvec4> : public glsl_vector_types_alignment_traits < dvec4, true, 4 > {};



		template<> struct alignment_traits<std::vector<ivec2>> : public glsl_vector_types_alignment_traits < std::vector<ivec2>, false, 2 >{};
		template<> struct alignment_traits<std::vector<uvec2>> : public glsl_vector_types_alignment_traits < std::vector<uvec2>, false, 2 >{};
		template<> struct alignment_traits<std::vector<bvec2>> : public glsl_vector_types_alignment_traits < std::vector<bvec2>, false, 2 >{};
		template<> struct alignment_traits<std::vector<vec2>> : public glsl_vector_types_alignment_traits < std::vector<vec2>, false, 2 >{};
		template<> struct alignment_traits<std::vector<dvec2>> : public glsl_vector_types_alignment_traits < std::vector<dvec2>, true, 2 >{};

		template<> struct alignment_traits<std::vector<ivec3>> : public glsl_vector_types_alignment_traits < std::vector<ivec3>, false, 3 > {};
		template<> struct alignment_traits<std::vector<uvec3>> : public glsl_vector_types_alignment_traits < std::vector<uvec3>, false, 3 > {};
		template<> struct alignment_traits<std::vector<bvec3>> : public glsl_vector_types_alignment_traits < std::vector<bvec3>, false, 3 > {};
		template<> struct alignment_traits<std::vector<vec3>> : public glsl_vector_types_alignment_traits < std::vector<vec3>, false, 3 > {};
		template<> struct alignment_traits<std::vector<dvec3>> : public glsl_vector_types_alignment_traits < std::vector<dvec3>, true, 3 > {};

		template<> struct alignment_traits<std::vector<ivec4>> : public glsl_vector_types_alignment_traits < std::vector<ivec4>, false, 4 > {};
		template<> struct alignment_traits<std::vector<uvec4>> : public glsl_vector_types_alignment_traits < std::vector<uvec4>, false, 4 > {};
		template<> struct alignment_traits<std::vector<bvec4>> : public glsl_vector_types_alignment_traits < std::vector<bvec4>, false, 4 > {};
		template<> struct alignment_traits<std::vector<vec4>> : public glsl_vector_types_alignment_traits < std::vector<vec4>, false, 4 > {};
		template<> struct alignment_traits<std::vector<dvec4>> : public glsl_vector_types_alignment_traits < std::vector<dvec4>, true, 4 > {};



		//*****************************************************************************************GLSL matrix types*****************************************************************************************
		template<typename T, bool is_double_precision, unsigned int nrows> struct glsl_matrix_types_alignment_traits;

		template<typename T>
		struct glsl_matrix_types_alignment_traits < T, false, 2 >
		{
			const static unsigned char base_alignment = 16;
			const static unsigned char log2_base_alignment = 4;
			const static bool is_array = false;
			const static unsigned int element_size = 16 * T::num_columns;	//for matrices element_size contains number of bytes consumed by a properly (i.e. in accordance with std140 rules) packed matrix
		};

		template<typename T>
		struct glsl_matrix_types_alignment_traits < T, false, 3 >
		{
			const static unsigned char base_alignment = 16;
			const static unsigned char log2_base_alignment = 4;
			const static bool is_array = false;
			const static unsigned int element_size = 16 * T::num_columns;	//for matrices element_size contains number of bytes consumed by a properly (i.e. in accordance with std140 rules) packed matrix
		};

		template<typename T>
		struct glsl_matrix_types_alignment_traits < T, false, 4 >
		{
			const static unsigned char base_alignment = 16;
			const static unsigned char log2_base_alignment = 4;
			const static bool is_array = false;
			const static unsigned int element_size = 16 * T::num_columns; //for matrices element_size contains number of bytes consumed by a properly (i.e. in accordance with std140 rules) packed matrix
		};

		template<typename T>
		struct glsl_matrix_types_alignment_traits < T, true, 2 >
		{
			const static unsigned char base_alignment = 16;
			const static unsigned char log2_base_alignment = 4;
			const static bool is_array = false;
			const static unsigned int element_size = 16 * T::num_columns;	//for matrices element_size contains number of bytes consumed by a properly (i.e. in accordance with std140 rules) packed matrix
		};

		template<typename T>
		struct glsl_matrix_types_alignment_traits < T, true, 3 >
		{
			const static unsigned char base_alignment = 32;
			const static unsigned char log2_base_alignment = 5;
			const static bool is_array = false;
			const static unsigned int element_size = 32 * T::num_columns;	//for matrices element_size contains number of bytes consumed by a properly (i.e. in accordance with std140 rules) packed matrix
		};

		template<typename T>
		struct glsl_matrix_types_alignment_traits < T, true, 4 >
		{
			const static unsigned char base_alignment = 32;
			const static unsigned char log2_base_alignment = 5;
			const static bool is_array = false;
			const static unsigned int element_size = 32 * T::num_columns;	//for matrices element_size contains number of bytes consumed by a properly (i.e. in accordance with std140 rules) packed matrix
		};



		template<typename T>
		struct glsl_matrix_types_alignment_traits < std::vector<T>, false, 2 >
		{
			const static unsigned char base_alignment = 16;
			const static unsigned char log2_base_alignment = 4;
			const static bool is_array = true;
			const static unsigned int element_size = 16 * T::num_columns;	//for matrices element_size contains number of bytes consumed by a properly (i.e. in accordance with std140 rules) packed matrix
		};

		template<typename T>
		struct glsl_matrix_types_alignment_traits < std::vector<T>, false, 3 >
		{
			const static unsigned char base_alignment = 16;
			const static unsigned char log2_base_alignment = 4;
			const static bool is_array = true;
			const static unsigned int element_size = 16 * T::num_columns;	//for matrices element_size contains number of bytes consumed by a properly (i.e. in accordance with std140 rules) packed matrix
		};
		
		template<typename T>
		struct glsl_matrix_types_alignment_traits < std::vector<T>, false, 4 >
		{
			const static unsigned char base_alignment = 16;
			const static unsigned char log2_base_alignment = 4;
			const static bool is_array = true;
			const static unsigned int element_size = 16 * T::num_columns;	//for matrices element_size contains number of bytes consumed by a properly (i.e. in accordance with std140 rules) packed matrix
		};

		template<typename T>
		struct glsl_matrix_types_alignment_traits < std::vector<T>, true, 2 >
		{
			const static unsigned char base_alignment = 16;
			const static unsigned char log2_base_alignment = 4;
			const static bool is_array = true;
			const static unsigned int element_size = 16 * T::num_columns;	//for matrices element_size contains number of bytes consumed by a properly (i.e. in accordance with std140 rules) packed matrix
		};

		template<typename T>
		struct glsl_matrix_types_alignment_traits < std::vector<T>, true, 3 >
		{
			const static unsigned char base_alignment = 32;
			const static unsigned char log2_base_alignment = 5;
			const static bool is_array = true;
			const static unsigned int element_size = 32 * T::num_columns;	//for matrices element_size contains number of bytes consumed by a properly (i.e. in accordance with std140 rules) packed matrix
		};

		template<typename T>
		struct glsl_matrix_types_alignment_traits < std::vector<T>, true, 4 >
		{
			const static unsigned char base_alignment = 32;
			const static unsigned char log2_base_alignment = 5;
			const static bool is_array = true;
			const static unsigned int element_size = 32 * T::num_columns;	//for matrices element_size contains number of bytes consumed by a properly (i.e. in accordance with std140 rules) packed matrix
		};



		template<> struct alignment_traits<mat2x2> : public glsl_matrix_types_alignment_traits < mat2x2, false, 2 >{};
		template<> struct alignment_traits<mat2x3> : public glsl_matrix_types_alignment_traits < mat2x3, false, 2 >{};
		template<> struct alignment_traits<mat2x4> : public glsl_matrix_types_alignment_traits < mat2x4, false, 2 >{};

		template<> struct alignment_traits<mat3x2> : public glsl_matrix_types_alignment_traits < mat3x2, false, 3 >{};
		template<> struct alignment_traits<mat3x3> : public glsl_matrix_types_alignment_traits < mat3x3, false, 3 >{};
		template<> struct alignment_traits<mat3x4> : public glsl_matrix_types_alignment_traits < mat3x4, false, 3 >{};

		template<> struct alignment_traits<mat4x2> : public glsl_matrix_types_alignment_traits < mat4x2, false, 4 >{};
		template<> struct alignment_traits<mat4x3> : public glsl_matrix_types_alignment_traits < mat4x3, false, 4 >{};
		template<> struct alignment_traits<mat4x4> : public glsl_matrix_types_alignment_traits < mat4x4, false, 4 >{};


		template<> struct alignment_traits<dmat2x2> : public glsl_matrix_types_alignment_traits < dmat2x2, true, 2 >{};
		template<> struct alignment_traits<dmat2x3> : public glsl_matrix_types_alignment_traits < dmat2x3, true, 2 >{};
		template<> struct alignment_traits<dmat2x4> : public glsl_matrix_types_alignment_traits < dmat2x4, true, 2 >{};

		template<> struct alignment_traits<dmat3x2> : public glsl_matrix_types_alignment_traits < dmat3x2, true, 3 >{};
		template<> struct alignment_traits<dmat3x3> : public glsl_matrix_types_alignment_traits < dmat3x3, true, 3 >{};
		template<> struct alignment_traits<dmat3x4> : public glsl_matrix_types_alignment_traits < dmat3x4, true, 3 >{};

		template<> struct alignment_traits<dmat4x2> : public glsl_matrix_types_alignment_traits < dmat4x2, true, 4 >{};
		template<> struct alignment_traits<dmat4x3> : public glsl_matrix_types_alignment_traits < dmat4x3, true, 4 >{};
		template<> struct alignment_traits<dmat4x4> : public glsl_matrix_types_alignment_traits < dmat4x4, true, 4 >{};



		template<> struct alignment_traits<std::vector<mat2x2>> : public glsl_matrix_types_alignment_traits < std::vector<mat2x2>, false, 2 >{};
		template<> struct alignment_traits<std::vector<mat2x3>> : public glsl_matrix_types_alignment_traits < std::vector<mat2x3>, false, 2 >{};
		template<> struct alignment_traits<std::vector<mat2x4>> : public glsl_matrix_types_alignment_traits < std::vector<mat2x4>, false, 2 >{};

		template<> struct alignment_traits<std::vector<mat3x2>> : public glsl_matrix_types_alignment_traits < std::vector<mat3x2>, false, 3 > {};
		template<> struct alignment_traits<std::vector<mat3x3>> : public glsl_matrix_types_alignment_traits < std::vector<mat3x3>, false, 3 > {};
		template<> struct alignment_traits<std::vector<mat3x4>> : public glsl_matrix_types_alignment_traits < std::vector<mat3x4>, false, 3 > {};

		template<> struct alignment_traits<std::vector<mat4x2>> : public glsl_matrix_types_alignment_traits < std::vector<mat4x2>, false, 4 > {};
		template<> struct alignment_traits<std::vector<mat4x3>> : public glsl_matrix_types_alignment_traits < std::vector<mat4x3>, false, 4 > {};
		template<> struct alignment_traits<std::vector<mat4x4>> : public glsl_matrix_types_alignment_traits < std::vector<mat4x4>, false, 4 > {};


		template<> struct alignment_traits<std::vector<dmat2x2>> : public glsl_matrix_types_alignment_traits < std::vector<dmat2x2>, true, 2 > {};
		template<> struct alignment_traits<std::vector<dmat2x3>> : public glsl_matrix_types_alignment_traits < std::vector<dmat2x3>, true, 2 > {};
		template<> struct alignment_traits<std::vector<dmat2x4>> : public glsl_matrix_types_alignment_traits < std::vector<dmat2x4>, true, 2 > {};

		template<> struct alignment_traits<std::vector<dmat3x2>> : public glsl_matrix_types_alignment_traits < std::vector<dmat3x2>, true, 3 > {};
		template<> struct alignment_traits<std::vector<dmat3x3>> : public glsl_matrix_types_alignment_traits < std::vector<dmat3x3>, true, 3 > {};
		template<> struct alignment_traits<std::vector<dmat3x4>> : public glsl_matrix_types_alignment_traits < std::vector<dmat3x4>, true, 3 > {};

		template<> struct alignment_traits<std::vector<dmat4x2>> : public glsl_matrix_types_alignment_traits < std::vector<dmat4x2>, true, 4 > {};
		template<> struct alignment_traits<std::vector<dmat4x3>> : public glsl_matrix_types_alignment_traits < std::vector<dmat4x3>, true, 4 > {};
		template<> struct alignment_traits<std::vector<dmat4x4>> : public glsl_matrix_types_alignment_traits < std::vector<dmat4x4>, true, 4 > {};



		//*****************************************************************************************Helper templates*****************************************************************************************
		template<typename T> 
		struct remove_reference
		{
			typedef T value_type;
		};
		
		template<typename T>
		struct remove_reference < T& >
		{
			typedef T value_type;
		};

		template<typename T>
		struct remove_reference < T&& >
		{
			typedef T value_type;
		};


		template<typename T> 
		struct remove_const
		{
			typedef T value_type;
		};

		template<typename T> 
		struct remove_const<const T>
		{
			typedef T value_type;
		};


		//predicate is a helper template that evaluates to val1 if boolean_expression=true and to val2 otherwise
		template<typename T, T val1, T val2, bool boolean_expression> struct predicate;

		template<typename T, T val1, T val2>
		struct predicate<T, val1, val2, true>
		{
			const static T result_value = val1;
		};

		template<typename T, T val1, T val2>
		struct predicate < T, val1, val2, false >
		{
			const static T result_value = val2;
		};

		//maximum is a helper alias template that allows to compute maximum of two given values based on template 'predicate'
		template<typename T, T val1, T val2>
		using maximum = predicate < T, val1, val2, (val1 > val2) > ;


		//This helper template allows to check if given type is a scalar GLSL type or an array of GLSL scalars
		template<typename T> struct is_scalar_type{ static const bool result_value = false; };


		template<> struct is_scalar_type < int > { static const bool result_value = true; };
		template<> struct is_scalar_type < unsigned int > { static const bool result_value = true; };
		template<> struct is_scalar_type < bool > { static const bool result_value = true; };
		template<> struct is_scalar_type < float > { static const bool result_value = true; };
		template<> struct is_scalar_type < double > { static const bool result_value = true; };

		template<> struct is_scalar_type < std::vector<int> > { static const bool result_value = true; };
		template<> struct is_scalar_type < std::vector<unsigned int> > { static const bool result_value = true; };
		template<> struct is_scalar_type < std::vector<bool> > { static const bool result_value = true; };
		template<> struct is_scalar_type < std::vector<float> > { static const bool result_value = true; };
		template<> struct is_scalar_type < std::vector<double> > { static const bool result_value = true; };


		//This helper template allows to check if given type is a GLSL vector type or an array of GLSL vectors
		template<typename T> struct is_glsl_vector_type{ static const bool result_value = false; };


		template<> struct is_glsl_vector_type < ivec2 > { static const bool result_value = true; };
		template<> struct is_glsl_vector_type < uvec2 > { static const bool result_value = true; };
		template<> struct is_glsl_vector_type < bvec2 > { static const bool result_value = true; };
		template<> struct is_glsl_vector_type < vec2 > { static const bool result_value = true; };
		template<> struct is_glsl_vector_type < dvec2 > { static const bool result_value = true; };

		template<> struct is_glsl_vector_type < ivec3 > { static const bool result_value = true; };
		template<> struct is_glsl_vector_type < uvec3 > { static const bool result_value = true; };
		template<> struct is_glsl_vector_type < bvec3 > { static const bool result_value = true; };
		template<> struct is_glsl_vector_type < vec3 > { static const bool result_value = true; };
		template<> struct is_glsl_vector_type < dvec3 > { static const bool result_value = true; };

		template<> struct is_glsl_vector_type < ivec4 > { static const bool result_value = true; };
		template<> struct is_glsl_vector_type < uvec4 > { static const bool result_value = true; };
		template<> struct is_glsl_vector_type < bvec4 > { static const bool result_value = true; };
		template<> struct is_glsl_vector_type < vec4 > { static const bool result_value = true; };
		template<> struct is_glsl_vector_type < dvec4 > { static const bool result_value = true; };


		template<> struct is_glsl_vector_type < std::vector<ivec2> > { static const bool result_value = true; };
		template<> struct is_glsl_vector_type < std::vector<uvec2> > { static const bool result_value = true; };
		template<> struct is_glsl_vector_type < std::vector<bvec2> > { static const bool result_value = true; };
		template<> struct is_glsl_vector_type < std::vector<vec2> > { static const bool result_value = true; };
		template<> struct is_glsl_vector_type < std::vector<dvec2> > { static const bool result_value = true; };

		template<> struct is_glsl_vector_type < std::vector<ivec3> > { static const bool result_value = true; };
		template<> struct is_glsl_vector_type < std::vector<uvec3> > { static const bool result_value = true; };
		template<> struct is_glsl_vector_type < std::vector<bvec3> > { static const bool result_value = true; };
		template<> struct is_glsl_vector_type < std::vector<vec3> > { static const bool result_value = true; };
		template<> struct is_glsl_vector_type < std::vector<dvec3> > { static const bool result_value = true; };

		template<> struct is_glsl_vector_type < std::vector<ivec4> > { static const bool result_value = true; };
		template<> struct is_glsl_vector_type < std::vector<uvec4> > { static const bool result_value = true; };
		template<> struct is_glsl_vector_type < std::vector<bvec4> > { static const bool result_value = true; };
		template<> struct is_glsl_vector_type < std::vector<vec4> > { static const bool result_value = true; };
		template<> struct is_glsl_vector_type < std::vector<dvec4> > { static const bool result_value = true; };


		//This helper template allows to check if given type is one of GLSL matrix types or an array of GLSL matrices
		template<typename T> struct is_glsl_matrix_type{ static const bool result_value = false; };


		template<> struct is_glsl_matrix_type < mat2x2 > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < mat2x3 > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < mat2x4 > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < mat3x2 > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < mat3x3 > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < mat3x4 > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < mat4x2 > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < mat4x3 > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < mat4x4 > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < dmat2x2 > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < dmat2x3 > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < dmat2x4 > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < dmat3x2 > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < dmat3x3 > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < dmat3x4 > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < dmat4x2 > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < dmat4x3 > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < dmat4x4 > { static const bool result_value = true; };


		template<> struct is_glsl_matrix_type < std::vector<mat2x2> > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < std::vector<mat2x3> > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < std::vector<mat2x4> > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < std::vector<mat3x2> > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < std::vector<mat3x3> > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < std::vector<mat3x4> > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < std::vector<mat4x2> > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < std::vector<mat4x3> > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < std::vector<mat4x4> > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < std::vector<dmat2x2> > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < std::vector<dmat2x3> > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < std::vector<dmat2x4> > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < std::vector<dmat3x2> > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < std::vector<dmat3x3> > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < std::vector<dmat3x4> > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < std::vector<dmat4x2> > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < std::vector<dmat4x3> > { static const bool result_value = true; };
		template<> struct is_glsl_matrix_type < std::vector<dmat4x4> > { static const bool result_value = true; };


		//This helper template allows to check whether the given type is an array of GLSL atomic elements
		template<typename T> struct is_glsl_array_type{ static const bool result_value = false; };

		template<> struct is_glsl_array_type < std::vector<int> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<unsigned int> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<bool> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<float> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<double> > { static const bool result_value = true; };

		template<> struct is_glsl_array_type < std::vector<ivec2> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<uvec2> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<bvec2> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<vec2> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<dvec2> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<ivec3> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<uvec3> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<bvec3> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<vec3> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<dvec3> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<ivec4> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<uvec4> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<bvec4> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<vec4> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<dvec4> > { static const bool result_value = true; };

		template<> struct is_glsl_array_type < std::vector<mat2x2> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<mat2x3> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<mat2x4> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<mat3x2> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<mat3x3> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<mat3x4> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<mat4x2> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<mat4x3> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<mat4x4> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<dmat2x2> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<dmat2x3> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<dmat2x4> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<dmat3x2> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<dmat3x3> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<dmat3x4> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<dmat4x2> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<dmat4x3> > { static const bool result_value = true; };
		template<> struct is_glsl_array_type < std::vector<dmat4x4> > { static const bool result_value = true; };

		
		//*******************************************************************************std140 structure template definitions*******************************************************************************
		template<typename... Args> struct tag_std140Structure;

		template<> 
		struct tag_std140Structure<>
		{
			const static unsigned int number_of_elements = 0;
		};

		template<typename Head, typename... Tail>
		struct tag_std140Structure<Head, Tail...> : public tag_std140Structure<Tail...>
		{
			template<typename... Args> friend struct std140StructureMemberPusher;
			template<unsigned int member_id, typename... Args> friend struct std140StructureMemberAccessAdapter;
		private:
			typedef Head value_type;

			Head member;

		public:
			const static unsigned int number_of_elements = tag_std140Structure<Tail...>::number_of_elements + 1;


			//Default initialization
			tag_std140Structure() : member(), tag_std140Structure<Tail...>()
			{

			}

			//Provides initial values for all members in the structure
			tag_std140Structure(Head structure_member, Tail... others) : member(structure_member), tag_std140Structure<Tail...>(others...)
			{

			}
		};



		template<unsigned int member_id, typename... Args> struct std140StructureMemberAccessAdapter;

		template<typename Head, typename... Args>
		struct std140StructureMemberAccessAdapter<0, Head, Args...>
		{
			typedef Head member_value_type;

			static member_value_type get(const tag_std140Structure<Head, Args...>& std140structure)
			{
				return std140structure.member;
			}

			static void set(tag_std140Structure<Head, Args...>& std140structure, member_value_type member_value)
			{
				std140structure.member = member_value;
			}
		};

		template<unsigned int member_id, typename Head, typename... Args>
		struct std140StructureMemberAccessAdapter<member_id, Head, Args...>
		{
			typedef typename std140StructureMemberAccessAdapter<member_id - 1, Args...>::member_value_type member_value_type;

			static member_value_type get(const tag_std140Structure<Head, Args...>& std140structure)
			{
				return std140StructureMemberAccessAdapter<member_id - 1, Args...>::get(std140structure);
			}

			static void set(tag_std140Structure<Head, Args...>& std140structure, member_value_type member_value)
			{
				std140StructureMemberAccessAdapter<member_id - 1, Args...>::set(std140structure, member_value);
			}
		};
		


		template<typename... Args> struct std140StructureAlignmentGetter;

		template<> 
		struct std140StructureAlignmentGetter<>
		{
			static const unsigned char base_alignment = 0;
			static const unsigned char log2_base_alignment = 0;
			static const unsigned char element_size = 0;
		};

		template<typename Head, typename... Args>
		struct std140StructureAlignmentGetter<Head, Args...>
		{
			static const unsigned char base_alignment =
				maximum<unsigned char, maximum<unsigned char, alignment_traits<Head>::base_alignment, std140StructureAlignmentGetter<Args...>::base_alignment>::result_value, 16U>::result_value;

			static const unsigned char log2_base_alignment =
				maximum<unsigned char, maximum<unsigned char, alignment_traits<Head>::log2_base_alignment, std140StructureAlignmentGetter<Args...>::log2_base_alignment>::result_value, 4U>::result_value;

			static const unsigned int element_size =
				(((alignment_traits<Head>::element_size + std140StructureAlignmentGetter<Args...>::element_size) >> log2_base_alignment) << log2_base_alignment) + base_alignment;
		};

		template<typename Head, typename... Args>
		struct alignment_traits < tag_std140Structure<Head, Args...> >
		{
			static const unsigned char base_alignment = std140StructureAlignmentGetter<Head, Args...>::base_alignment;
			static const unsigned char log2_base_alignment = std140StructureAlignmentGetter<Head, Args...>::log2_base_alignment;
			static const bool is_array = false;
			static const unsigned int element_size = std140StructureAlignmentGetter<Head, Args...>::element_size;
		};



		//This helper template allows to check, whether supplied type represents an std140-aligned structure
		template<typename T> struct is_std140_structure_type { static const bool result_value = false; };

		template<typename... Args> struct is_std140_structure_type < tag_std140Structure<Args...> > { static const bool result_value = true; };


		//The following template is a helper, which allows to decide between the families of overloaded functions designed to push different kinds of elements into the buffer
		//(in other words, one needs to know whether to call pushScalar(), pushVector(), pushMatrix(), or pushStructure() for an element of general type T)
		template<bool is_scalar, bool is_glsl_vector, bool is_glsl_matrix, bool is_std140_structure> struct std140StructureMemberPusher_selector;

		template<>
		struct std140StructureMemberPusher_selector < true, false, false, false >
		{
			template<typename T>
			static void pushMember(std140UniformBuffer& std140buffer, T member){ std140buffer.pushScalar(member); }

			template<typename T>
			static void skipMember(std140UniformBuffer& std140buffer, T member){ std140buffer.skipScalar<T>(1, false); }

			template<typename T>
			static void skipMember(std140UniformBuffer& std140buffer, std::vector<T> member){ std140buffer.skipScalar<T>(static_cast<uint32_t>(member.size()), true); }
		};

		template<>
		struct std140StructureMemberPusher_selector < false, true, false, false >
		{
			template<typename T>
			static void pushMember(std140UniformBuffer& std140buffer, T member){ std140buffer.pushVector(member); }

			template<typename T>
			static void skipMember(std140UniformBuffer& std140buffer, T member){ std140buffer.skipVector<T>(1, false); }

			template<typename T>
			static void skipMember(std140UniformBuffer& std140buffer, std::vector<T> member){ std140buffer.skipVector<T>(static_cast<uint32_t>(member.size()), true); }
		};

		template<>
		struct std140StructureMemberPusher_selector < false, false, true, false >
		{
			template<typename T>
			static void pushMember(std140UniformBuffer& std140buffer, T member){ std140buffer.pushMatrix(member); }

			template<typename T>
			static void skipMember(std140UniformBuffer& std140buffer, T member){ std140buffer.skipMatrix<T>(1, false); }

			template<typename T>
			static void skipMember(std140UniformBuffer& std140buffer, std::vector<T> member){ std140buffer.skipMatrix<T>(static_cast<uint32_t>(member.size()), true); }
		};

		template<>
		struct std140StructureMemberPusher_selector < false, false, false, true >
		{
			template<typename T>
			static void pushMember(std140UniformBuffer& std140buffer, T member){ std140buffer.pushStructure(member); }

			template<typename T>
			static void skipMember(std140UniformBuffer& std140buffer, T member){ std140buffer.skipStructure(member); }
		};



		template<typename... Args> struct std140StructureMemberPusher;

		template<> 
		struct std140StructureMemberPusher<>
		{
			static void push_members(std140UniformBuffer& std140buffer, const tag_std140Structure<>& std140struct){};
			static void skip_members(std140UniformBuffer& std140buffer, const tag_std140Structure<>& std140struct){};
		};

		template<typename Head, typename... Args>
		struct std140StructureMemberPusher<Head, Args...>
		{
			//While dissecting "the first" Head we assume that field 'offset' of std140UniformBuffer is aligned in accordance with where the structure should begin

			static void push_members(std140UniformBuffer& std140buffer, const tag_std140Structure<Head, Args...>& std140struct)
			{
				std140StructureMemberPusher_selector < is_scalar_type<Head>::result_value, is_glsl_vector_type<Head>::result_value,
					is_glsl_matrix_type<Head>::result_value, is_std140_structure_type<Head>::result_value > ::pushMember(std140buffer, std140struct.member);

				std140StructureMemberPusher<Args...>::push_members(std140buffer, std140struct);
			}

			static void skip_members(std140UniformBuffer& std140buffer, const tag_std140Structure<Head, Args...>& std140struct)
			{
				std140StructureMemberPusher_selector < is_scalar_type<Head>::result_value, is_glsl_vector_type<Head>::result_value,
					is_glsl_matrix_type<Head>::result_value, is_std140_structure_type<Head>::result_value > ::skipMember(std140buffer, std140struct.member);

				std140StructureMemberPusher<Args...>::skip_members(std140buffer, std140struct);
			}
		};
		

		//****************************************************************************************************************************************************************************************************


	public:
		template<typename... Args>
		using std140Structure = tag_std140Structure < Args... > ;

		//Default initialization. By default new std140 uniform buffer is assigned to the 0 binding index
		std140UniformBuffer();

		//Initializes new std140 uniform buffer and assigns it to the given binding point
		std140UniformBuffer(uint32_t binding_point);

		//Initializes new std140 uniform buffer with storage that has requested capacity
		std140UniformBuffer(size_t buffer_size, uint32_t binding_point);
		


		//The following set of overloaded template functions are responsible for pushing sole variables or arrays into the buffer. 
		//The variables are added to the buffer so that OpenGL std140 uniform buffer pack alignment rules are satisfied. 
		//Note however, that in order to end up with proper alignment, the variables should be added in EXACTLY SAME order
		//as they appear in their GLSL uniform block definition

		//Pushes scalar type into the buffer
		template<typename T>
		void pushScalar(T scalar)
		{
			typename ogl_type_mapper<T>::ogl_type converted_scalar =
				static_cast<typename ogl_type_mapper<T>::ogl_type>(scalar);

			const unsigned char base_alignemnt = alignment_traits<ogl_type_mapper<T>::ogl_type>::base_alignment;
			const unsigned char log2_base_alignment = alignment_traits<ogl_type_mapper<T>::ogl_type>::log2_base_alignment;
			const ptrdiff_t aligned_offset = offset%base_alignemnt ? ((offset >> log2_base_alignment) << log2_base_alignment) + base_alignemnt : offset;
			const size_t scalar_size = alignment_traits<ogl_type_mapper<T>::ogl_type>::element_size;

			setSubData(aligned_offset, scalar_size, &converted_scalar);
			offset = aligned_offset + scalar_size;
		}

		//Pushes scalar boolean into the buffer
		void pushScalar(bool boolean_scalar);
		

		//Pushes array of variables into the buffer
		template<typename T>
		void pushScalar(const std::vector<T>& scalars)
		{
			typedef typename ogl_type_mapper<T>::ogl_type glsl_scalar_type;

			const unsigned char base_alignment = alignment_traits<std::vector<glsl_scalar_type>>::base_alignment;
			const unsigned char log2_base_alignment = alignment_traits<std::vector<glsl_scalar_type>>::log2_base_alignment;
			const ptrdiff_t aligned_offset = offset%base_alignment ? ((offset >> log2_base_alignment) << log2_base_alignment) + base_alignment : offset;
			const size_t array_size = alignment_traits<std::vector<glsl_scalar_type>>::element_size * scalars.size();
			const unsigned char type_array_stride = 16 / sizeof(glsl_scalar_type);

			void* std140_array = malloc(array_size);
			for (unsigned int i = 0; i < scalars.size(); ++i)
			{
				*(static_cast<glsl_scalar_type*>(std140_array)+type_array_stride * i) = static_cast<glsl_scalar_type>(scalars[i]);
			}

			setSubData(aligned_offset, array_size, std140_array);
			free(std140_array);
			offset = aligned_offset + array_size;
		}

		//Pushes array of boolean variables into the buffer
		void pushScalar(const std::vector<bool>& boolean_scalars);



		//Pushes GLSL vector into the buffer
		template<typename glsl_vector_type>
		void pushVector(const glsl_vector_type& vector)
		{
			const unsigned char base_alignment = alignment_traits<glsl_vector_type>::base_alignment;
			const unsigned char log2_base_alignment = alignment_traits<glsl_vector_type>::log2_base_alignment;
			ptrdiff_t aligned_offset = offset%base_alignment ? ((offset >> log2_base_alignment) << log2_base_alignment) + base_alignment : offset;
			size_t vector_size = alignment_traits<glsl_vector_type>::element_size;

			setSubData(aligned_offset, vector_size, vector.getDataAsArray());
			offset = aligned_offset + vector_size;
		}

		//Pushes GLSL boolean vector into the buffer
		void pushVector(const bvec4& vector);

		//Pushes GLSL boolean vector into the buffer
		void pushVector(const bvec3& vector);

		//Pushes GLSL boolean vector into the buffer
		void pushVector(const bvec2& vector);


		//Pushes array of GLSL vectors into the buffer
		template<typename glsl_vector_type>
		void pushVector(const std::vector<glsl_vector_type>& vectors)
		{
			const unsigned char base_alignment = alignment_traits<std::vector<glsl_vector_type>>::base_alignment;
			const unsigned char log2_base_alignment = alignment_traits<std::vector<glsl_vector_type>>::log2_base_alignment;
			const unsigned char type_array_stride = base_alignment / sizeof(typename glsl_vector_type::value_type);

			const ptrdiff_t aligned_offset = offset%base_alignment ? ((offset >> log2_base_alignment) << log2_base_alignment) + base_alignment : offset;
			size_t array_of_vectors_size = alignment_traits<std::vector<glsl_vector_type>>::element_size * vectors.size();

			void* std140_array_of_vectors = malloc(array_of_vectors_size);
			for (unsigned int i = 0; i < vectors.size(); ++i)
			{
				for (unsigned int j = 0; j < glsl_vector_type::dimension; ++j)
					*(static_cast<typename glsl_vector_type::value_type*>(std140_array_of_vectors)+type_array_stride*i + j) = (vectors[i])[j];
			}


			setSubData(aligned_offset, array_of_vectors_size, std140_array_of_vectors);
			free(std140_array_of_vectors);
			offset = aligned_offset + array_of_vectors_size;
		}

		//Pushes array of boolean 4D-vectors into the buffer
		void pushVector(const std::vector<bvec4>& vectors);

		//Pushed array of boolean 3D-vectors into the buffer
		void pushVector(const std::vector<bvec3>& vectors);

		//Pushes array of boolean 2D-vectors into the buffer
		void pushVector(const std::vector<bvec2>& vectors);

	

		//Pushes GLSL matrix into the buffer
		template<typename glsl_matrix_type>
		void pushMatrix(const glsl_matrix_type& matrix)
		{
			const unsigned char base_alignment = alignment_traits<glsl_matrix_type>::base_alignment;
			const unsigned char log2_base_alignment = alignment_traits<glsl_matrix_type>::log2_base_alignment;
			const unsigned char type_array_stride = base_alignment / sizeof(typename glsl_matrix_type::value_type);

			const ptrdiff_t aligned_offset = offset%base_alignment ? ((offset >> log2_base_alignment) << log2_base_alignment) + base_alignment : offset;
			size_t matrix_size = alignment_traits<glsl_matrix_type>::element_size;

			void* std140_matrix = malloc(matrix_size);
			for (unsigned int j = 0; j < glsl_matrix_type::num_columns; ++j)
			{
				for (unsigned int i = 0; i < glsl_matrix_type::num_rows; ++i)
					*(static_cast<typename glsl_matrix_type::value_type*>(std140_matrix)+j*type_array_stride + i) = matrix[i][j];
			}


			setSubData(aligned_offset, matrix_size, std140_matrix);
			free(std140_matrix);
			offset = aligned_offset + matrix_size;
		}


		//Pushes array of GLSL matrices into the buffer
		template<typename glsl_matrix_type>
		void pushMatrix(const std::vector<glsl_matrix_type>& matrices)
		{
			const unsigned char base_alignment = alignment_traits<std::vector<glsl_matrix_type>>::base_alignment;
			const unsigned char log2_base_alignment = alignment_traits<std::vector<glsl_matrix_type>>::log2_base_alignment;
			const unsigned char type_array_stride = base_alignment / sizeof(typename glsl_matrix_type::value_type);

			const ptrdiff_t aligned_offset = offset%base_alignment ? ((offset >> log2_base_alignment) << log2_base_alignment) + base_alignment : offset;
			size_t array_of_matrices_size = alignment_traits<std::vector<glsl_matrix_type>>::element_size * matrices.size();

			void* std140_array_of_matrices = malloc(array_of_matrices_size);

			for (unsigned int j = 0; j < glsl_matrix_type::num_columns * matrices.size(); ++j)
			{
				for (unsigned int i = 0; i < glsl_matrix_type::num_rows; ++i)
					*(static_cast<typename glsl_matrix_type::value_type*>(std140_array_of_matrices)+j*type_array_stride + i) =
					(matrices[j / glsl_matrix_type::num_columns])[i][j % glsl_matrix_type::num_columns];
			}


			setSubData(aligned_offset, array_of_matrices_size, std140_array_of_matrices);
			free(std140_array_of_matrices);
			offset = aligned_offset + array_of_matrices_size;
		}


		//Pushes structure (or single member of array of structures) into the buffer
		template<typename... Args>
		void pushStructure(const std140Structure<Args...>& std140structure)
		{
			const unsigned char base_alignment = alignment_traits<std140Structure<Args...>>::base_alignment;
			const unsigned char log2_base_alignment = alignment_traits<std140Structure<Args...>>::log2_base_alignment;

			offset = offset%base_alignment ? ((offset >> log2_base_alignment) << log2_base_alignment) + base_alignment : offset;
			std140StructureMemberPusher<Args...>::push_members(*this, std140structure);
			offset = offset%base_alignment ? ((offset >> log2_base_alignment) << log2_base_alignment) + base_alignment : offset;
		}



		//Returns value of a member of std140Structure based on the given member identifier. Identifiers are assigned to members 
		//of std140Structure in the order of their appearance in definition beginning from 0. 
		template<unsigned int member_id, typename... Args>
		static typename std140StructureMemberAccessAdapter<member_id, Args...>::member_value_type std140StructureGetMember(const std140Structure<Args...>& std140structure)
		{
			return std140StructureMemberAccessAdapter<member_id, Args...>::get(std140structure);
		}


		//Assigns new value to a member of std140Structure based on its identifier. Identifiers are assigned to members 
		//of std140Structure in the order of their appearance in definition beginning from 0. 
		template<unsigned int member_id, typename... Args>
		static void std140StructureSetMember(std140Structure<Args...>& std140structure, typename std140StructureMemberAccessAdapter<member_id, Args...>::member_value_type member_value)
		{
			std140StructureMemberAccessAdapter<member_id, Args...>::set(std140structure, member_value);
		}




		//Resets the offset counter of std140 uniform buffer. Any elements pushed into the buffer following a call of this function should be pushed in the order as if the data was being defined
		//from the very beginning. In other words, if the offset counter has been reset, all the data previously pushed into the buffer must be neglected in order to avoid possible alignment conflicts
		void resetOffsetCounter();

		

		//The following functions allow to traverse the buffer forward without actually writing any data into it. The current writing position is modified in accordance with provided
		//data type and std140 alignment rules. The position can not go outside the allowed range determined by the size of the buffer, i.e. position is always clamped the buffer size.
		//Due to alignment rules the buffer can only be traversed in forward direction. Use resetOffsetCounter() in order to move the writing carret back to the 0 offset.
		

		//Adds to the current writing offset of the buffer the amount of bytes needed to store scalar or array of scalars. Function allows to skip several scalars of the same type  
		//in a row if num_elements_to_skip is greater than 1. In addition, if is_array is set to true, the function assumes that the elements, which it has to traverse through are packed 
		//into an array, and applies the corresponding pack alignment rules. Note, that this function does not allow to skip several arrays in a row. If this is what is needed, then 
		//the function must be called repeatedly. Note also, that if is_array is 'true' the array pack alignment rules are applied even if num_elements_to_skip is equal to 1.
		template<typename T>
		void skipScalar(uint32_t num_elements_to_skip, bool is_array)
		{
			typedef typename ogl_type_mapper<T>::ogl_type glsl_scalar_type;

			if (is_array)
			{
				const unsigned char base_alignment = alignment_traits<std::vector<glsl_scalar_type>>::base_alignment;
				const unsigned char log2_base_alignment = alignment_traits<std::vector<glsl_scalar_type>>::log2_base_alignment;
				const ptrdiff_t aligned_offset = offset%base_alignment ? ((offset >> log2_base_alignment) << log2_base_alignment) + base_alignment : offset;
				offset = aligned_offset + alignment_traits<std::vector<glsl_scalar_type>>::element_size * num_elements_to_skip;
			}
			else
			{
				const unsigned char base_alignment = alignment_traits<glsl_scalar_type>::base_alignment;
				const unsigned char log2_base_alignment = alignment_traits<glsl_scalar_type>::log2_base_alignment;
				
				for (unsigned int i = 0; i < num_elements_to_skip; ++i)
				{
					const ptrdiff_t aligned_offset = offset%base_alignment ? ((offset >> log2_base_alignment) << log2_base_alignment) + base_alignment : offset;
					offset = aligned_offset + alignment_traits<glsl_scalar_type>::element_size;
				}
			}

			offset = std::min(offset, static_cast<ptrdiff_t>(this->getSize()));
		}

		template<> void skipScalar<bool>(uint32_t num_elements_to_skip, bool is_array){ skipScalar<int>(num_elements_to_skip, is_array); }


		//Adds to the current writing offset of the buffer the amount of bytes needed to store single vector or an array of vectors. In addition if num_elements_to_skip is greater then 1 and
		//is_array is set to 'false'  the function allows to skip several vector elements of the same type in a row. If is_array is 'true' then the function applies pack alignment rules used
		//for arrays of the given GLSL vector type. Note, that in this case array pack alignment rules are applied even if num_elements_to_skip is 1.
		template<typename glsl_vector_type>
		void skipVector(uint32_t num_elements_to_skip, bool is_array)
		{
			if (is_array)
			{
				const unsigned char base_alignment = alignment_traits<std::vector<glsl_vector_type>>::base_alignment;
				const unsigned char log2_base_alignment = alignment_traits<std::vector<glsl_vector_type>>::log2_base_alignment;
				const ptrdiff_t aligned_offset = offset%base_alignment ? ((offset >> log2_base_alignment) << log2_base_alignment) + base_alignment : offset;
				offset = aligned_offset + alignment_traits<std::vector<glsl_vector_type>>::element_size * num_elements_to_skip;
			}
			{
				const unsigned char base_alignment = alignment_traits<glsl_vector_type>::base_alignment;
				const unsigned char log2_base_alignment = alignment_traits<glsl_vector_type>::log2_base_alignment;
				
				for (unsigned int i = 0; i < num_elements_to_skip; ++i)
				{
					const ptrdiff_t aligned_offset = offset%base_alignment ? ((offset >> log2_base_alignment) << log2_base_alignment) + base_alignment : offset;
					offset = aligned_offset + alignment_traits<glsl_vector_type>::element_size;
				}
			}

			offset = std::min(offset, static_cast<ptrdiff_t>(this->getSize()));
		}

		template<> void skipVector<bvec2>(uint32_t num_elements_to_skip, bool is_array) { skipVector<ivec2>(num_elements_to_skip, is_array); }
		template<> void skipVector<bvec3>(uint32_t num_elements_to_skip, bool is_array) { skipVector<ivec3>(num_elements_to_skip, is_array); }
		template<> void skipVector<bvec4>(uint32_t num_elements_to_skip, bool is_array) { skipVector<ivec4>(num_elements_to_skip, is_array); }


		//Adds to the current writing offset of the buffer the amount of bytes needed to store single matrix or an array of matrices of the given type. In addition, if num_elements_to_skip is 
		//greater then 1 and is_array is set to 'false' , the function allows to skip several matrix elements of the given type in a row. If is_array is 'true' then the function applies pack
		//alignment rules used for arrays of the given GLSL matrix type. Note, that in this case array pack alignment rules are applied even if num_elements_to_skip is 1.
		template<typename glsl_matrix_type>
		void skipMatrix(uint32_t num_elements_to_skip, bool is_array)
		{
			if (is_array)
			{
				const unsigned char base_alignment = alignment_traits<glsl_matrix_type>::base_alignment;
				const unsigned char log2_base_alignment = alignment_traits<glsl_matrix_type>::log2_base_alignment;
				const ptrdiff_t aligned_offset = offset%base_alignment ? ((offset >> log2_base_alignment) << log2_base_alignment) + base_alignment : offset;
				offset = aligned_offset + alignment_traits<glsl_matrix_type>::element_size*num_elements_to_skip;
			}
			else
			{
				const unsigned char base_alignment = alignment_traits<glsl_matrix_type>::base_alignment;
				const unsigned char log2_base_alignment = alignment_traits<glsl_matrix_type>::log2_base_alignment;
				
				for (unsigned int i = 0; i < num_elements_to_skip; ++i)
				{
					const ptrdiff_t aligned_offset = offset%base_alignment ? ((offset >> log2_base_alignment) << log2_base_alignment) + base_alignment : offset;
					offset = aligned_offset + alignment_traits<glsl_matrix_type>::element_size;
				}
			}

			offset = std::min(offset, static_cast<ptrdiff_t>(this->getSize()));
		}


		//Adds to the current writing offset of the buffer the amount of bytes needed to store the provided structure object without actually storing this
		//object into the buffer. Note, that the function is only able to compute correct offset change if the structure provided is filled with data. This
		//is induced by dynamic nature of array types that have undefined size until they are populated with data. Note also, that in order to skip
		//an array of structures it is enough to call this function repeatedly with the same input as many times as the number of elements residing 
		//in the array, which should be traversed
		template<typename... Args>
		void skipStructure(const std140Structure<Args...>& std140structure)
		{
			const unsigned char base_alignment = alignment_traits<std140Structure<Args...>>::base_alignment;
			const unsigned char log2_base_alignment = alignment_traits<std140Structure<Args...>>::log2_base_alignment;

			offset = offset%base_alignment ? ((offset >> log2_base_alignment) << log2_base_alignment) + base_alignment : offset;
			std140StructureMemberPusher<Args...>::skip_members(*this, std140structure);
			offset = offset%base_alignment ? ((offset >> log2_base_alignment) << log2_base_alignment) + base_alignment : offset;

			offset = std::min(offset, static_cast<ptrdiff_t>(this->getSize()));
		}



		//Sets binding point used by the buffer object
		void setBindingPoint(uint32_t binding_point);

		//Returns binding point used by the buffer object
		uint32_t getBindingPoint() const;

		//Binds the buffer to the uniform storage buffer indexed target
		void bind() const;



		//Allows to calculate minimal capacity of the buffer required to store fields with provided data types
		//following std140 layout rules. Since std::vector types themselves do not determine their lengths, the lengths
		//should be listed separately in the run-time argument list of the function. The order in which the lengths
		//are listed should be the same as the enumeration order of the std::vector types included into the variadic
		//argument of the template
		template<typename... Args>
		static uint32_t getMinimalStorageCapacity(uint32_t first = 0, ...)
		{
			const uint32_t num_vector_types = get_number_of_vector_types<Args...>::result_value;
			const uint32_t num_variadic_args = static_cast<uint32_t>(maximum<int32_t, 0, num_vector_types - 1>::result_value);

			uint32_t array_lengths[maximum<uint32_t, 1U, num_vector_types>::result_value];
			array_lengths[0] = first;

			va_list vl;
			va_start(vl, first);
			for (uint32_t i = 0; i < num_variadic_args; ++i)
				array_lengths[i + 1] = va_arg(vl, uint32_t);
			va_end(vl);


			return getMinimalStorageCapacity<Args...>(0, array_lengths);
		}



		private:

		//Helper function. Takes into account the capacity computed so far
		template<typename Head, typename... Tail>
		static uint32_t getMinimalStorageCapacity(uint32_t offset, uint32_t array_lengths[])
		{
			uint32_t base_alignment = alignment_traits<Head>::base_alignment;
			uint32_t log2_base_alignment = alignment_traits<Head>::log2_base_alignment;
			uint32_t aligned_offset = offset % base_alignment ? ((offset >> log2_base_alignment) << log2_base_alignment) + base_alignment : offset;
			uint32_t data_size = alignment_traits<Head>::element_size * process_data_field<Head>::get_number_of_elements(array_lengths);

			return getMinimalStorageCapacity<Tail...>(aligned_offset + data_size, array_lengths + process_data_field<Head>::array_lengths_shift);
		}


		//Overrided template of getMinimalStorageCapaity() specialized for the case where all data fields have already been processed
		template<int = 0>
		static uint32_t getMinimalStorageCapacity(uint32_t offset, uint32_t array_lengths[])
		{
			return offset;
		}


		//Helper template structure allowing to count number of vector-typed elements in provided template argument pack
		template<typename... Args> struct get_number_of_vector_types;

		template<> struct get_number_of_vector_types<>
		{
			static const uint32_t result_value = 0U;
		};

		template<typename Head, typename... Tail>
		struct get_number_of_vector_types<Head, Tail...>
		{
			static const uint32_t result_value = get_number_of_vector_types<Tail...>::result_value;
		};

		template<typename T, typename... Tail>
		struct get_number_of_vector_types<std::vector<T>, Tail...>
		{
			static const uint32_t result_value = 1U + get_number_of_vector_types<Tail...>::result_value;
		};


		//Helper template structure providing capabilities to process data fields depending on whether they
		//represent a scalar or a vector data type
		template<typename T>
		struct process_data_field
		{
			static uint32_t get_number_of_elements(const uint32_t[]){ return 1U; }
			static const unsigned char array_lengths_shift = 0U;
		};

		template<typename T>
		struct process_data_field<std::vector<T>>
		{
			static uint32_t get_number_of_elements(const uint32_t array_lengths[]){ return array_lengths[0]; }
			static const unsigned char array_lengths_shift = 1U;
		};
	};
}

#define TW__std140UniformBuffer__
#endif