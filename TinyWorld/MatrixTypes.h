#ifndef TW__MATRIX_TYPES__

#include "VectorTypes.h"

#include <stdint.h>
#include <array>
#include <utility>
#include <cmath>

namespace tiny_world
{
	//Forward declaration of the template needed to represent GLSL matrix types
	template<typename T, uint32_t nrows, uint32_t ncolumns> class glsl_matrix_type;

	template<typename T, uint32_t nrows, uint32_t ncolumns>
	class abstract_matrix
	{
		template<typename P, uint32_t, uint32_t> friend class abstract_matrix;
	public:
		typedef std::array<T, nrows*ncolumns> data_storage_type;
		typedef T value_type;
		static const uint32_t num_rows = nrows;
		static const uint32_t num_columns = ncolumns;

	protected:
		data_storage_type data;


		//Default constructor
		abstract_matrix()
		{
			data.fill(T{});
		}

		//Constructor using array initializer
		explicit abstract_matrix(data_storage_type data) : data(data) {}

		//Initializes matrix to diagonal form val*I
		abstract_matrix(T val)
		{
			data.fill(0);
			for (int i = 0; i < static_cast<signed int>(std::min(nrows, ncolumns)); ++i)
				data[i*nrows + i] = val;
		}

		//Copy constructor
		abstract_matrix(const abstract_matrix& other) : data(other.data)
		{

		}

		//Move constructor
		abstract_matrix(abstract_matrix&& other) : data(std::move(other.data))
		{

		}

		//Copy assignment operator
		abstract_matrix& operator=(const abstract_matrix& other)
		{
			if (this == &other)
				return *this;

			data = other.data;
			return *this;
		}

		//Move assignment operator
		abstract_matrix& operator=(abstract_matrix&& other)
		{
			if (this == &other)
				return *this;

			data = std::move(other.data);
			return *this;
		}

		//destructor
		virtual ~abstract_matrix() {}


		inline std::pair<T, uint32_t> get_max_column_elem(data_storage_type aux, uint32_t column_index) const
		{
			T max_val = aux[column_index*nrows];
			uint32_t max_elem_row_index = 0;
			for (int i = 1; i < nrows; ++i)
			{
				T cval = aux[column_index*nrows + i];
				if (std::abs(cval)>std::abs(max_val))
				{
					max_val = cval;
					max_elem_row_index = i;
				}
			}

			return std::make_pair(max_val, max_elem_row_index);
		}

		inline std::pair<T, uint32_t> get_max_row_elem(data_storage_type aux, uint32_t row_index) const
		{
			T max_val = aux[row_index];
			uint32_t max_elem_column_index = 0;
			for (int i = 1; i < ncolumns; ++i)
			{
				T cval = aux[i*nrows + row_index];
				if (std::abs(cval)>std::abs(max_val))
				{
					max_val = cval;
					max_elem_column_index = i;
				}
			}

			return std::make_pair(max_val, max_elem_column_index);
		}

	public:
		class abstract_matrix_row
		{
		private:
			uint32_t row_offset;
			abstract_matrix* p_matrix_object;	//pointer to requested matrix object valid in non-const context  
			const abstract_matrix* p_const_matrix_object;	//pointer to requested matrix object valid in constant context

		public:
			abstract_matrix_row(uint32_t row_number, abstract_matrix* p_matrix_object) :
				row_offset(row_number), p_matrix_object(p_matrix_object), p_const_matrix_object(p_matrix_object) {}

			abstract_matrix_row(uint32_t row_number, const abstract_matrix* p_const_matrix_object) :
				row_offset(row_number), p_matrix_object(const_cast<abstract_matrix*>(p_const_matrix_object)), 
				p_const_matrix_object(p_const_matrix_object) {}

			//extracts elements from a matrix row using zero-based indexing
			T& operator[](uint32_t index)
			{
				if (index >= ncolumns)
					throw(std::out_of_range("attempt to access matrix column, which is out of range"));

				return p_matrix_object->data[index*nrows + row_offset];
			}

			//extracts elements from a matrix row using zero-based indexing for read-only
			const T& operator[](uint32_t index) const
			{
				if (index >= ncolumns)
					throw(std::out_of_range("attempt to access matrix column, which is out of range"));

				return p_const_matrix_object->data[index*nrows + row_offset];
			}
		};

		//extracts row from contained matrix using zero-based indexing
		abstract_matrix_row operator[] (uint32_t index)
		{
			if (index >= nrows)
				throw(std::out_of_range("attempt to access matrix row, which is out of range"));

			return abstract_matrix_row(index, this);
		}

		//extracts read-only row structure from contained matrix using zero-based indexing
		const abstract_matrix_row operator[] (uint32_t index) const
		{
			if (index >= nrows)
				throw(std::out_of_range("attempt to access matrix row, which is out of range"));

			return abstract_matrix_row(index, this);
		}

		//computes rank of contained matrix
		uint32_t rank() const
		{
			data_storage_type aux = data;
			uint32_t rv = 0;
			for (int i = 0; i < ncolumns; ++i)
			{
				uint32_t max_elem_row_index;
				std::pair<T, uint32_t> max_elem = get_max_column_elem(aux, i);

				if (!max_elem.first) continue;

				++rv;

				for (int j = i + 1; j < ncolumns - 1; ++j)
				{
					T alpha = aux[j*nrows + max_elem.second] / max_elem.first;
					for (int k = 0; k < nrows; ++k)
						aux[j*nrows + k] -= alpha*aux[i*nrows + k];
					aux[j*nrows + max_elem.second] = 0;	//ensure that elements below the leading column vanish
				}
			}

			return rv;
		}

		const T* getRawData() const
		{
			return data.data();
		}

		T* getRawData()
		{
			return data.data();
		}

		//adds two matrices
		template<typename P>
		auto operator+(const abstract_matrix<P, nrows, ncolumns>& other) const ->
			glsl_matrix_type < decltype(T{}+P{}), nrows, ncolumns>
		{
			typedef decltype(T{}+P{}) return_value_type;
			std::array<return_value_type, nrows*ncolumns> res_data;
			for (int i = 0; i < nrows*ncolumns; ++i)
				res_data[i] = data[i] + other.data[i];

			return glsl_matrix_type < return_value_type, nrows, ncolumns>(res_data);
		}

		//adds two matrices and assigns the result to the left operand
		template<typename P>
		auto operator+=(const abstract_matrix<P, nrows, ncolumns>& other) ->
			glsl_matrix_type<T, nrows, ncolumns>&
		{
			for (int i = 0; i < nrows*ncolumns; ++i)
				data[i] += other.data[i];
			return *dynamic_cast<glsl_matrix_type<T, nrows, ncolumns>*>(this);
		}

		//subtracts two matrices
		template<typename P>
		auto operator-(const abstract_matrix<P, nrows, ncolumns>& other) const ->
			glsl_matrix_type < decltype(T{}-P{}), nrows, ncolumns>
		{
			typedef decltype(T{}-P{}) return_value_type;
			std::array<return_value_type, nrows*ncolumns> res_data;
			for (int i = 0; i < nrows*ncolumns; ++i)
				res_data[i] = data[i] - other.data[i];

			return glsl_matrix_type < return_value_type, nrows, ncolumns>(res_data);
		}

		//subtracts two matrices and assigns the result to the left operand
		template<typename P>
		auto operator-=(const abstract_matrix<P, nrows, ncolumns>& other) ->
			glsl_matrix_type<T, nrows, ncolumns>&
		{
			for (int i = 0; i < nrows*ncolumns; ++i)
				data[i] -= other.data[i];
			return *dynamic_cast<glsl_matrix_type<T, nrows, ncolumns>*>(this);
		}

		//multiplies matrix by scalar
		template<typename P>
		auto operator*(P alpha) const->
			glsl_matrix_type < decltype(T{}*P{}), nrows, ncolumns>
		{
			typedef decltype(T{}*P{}) return_value_type;
			std::array<return_value_type, nrows*ncolumns> res_data;

			for (int i = 0; i < nrows*ncolumns; ++i)
				res_data[i] = data[i] * alpha;

			return glsl_matrix_type<return_value_type, nrows, ncolumns>(res_data);
		}

		//multiplies matrix by scalar and assigns the result of multiplication to the left operand (i.e. to the matrix operand)
		template<typename P>
		auto operator*=(P alpha) ->
			glsl_matrix_type<T, nrows, ncolumns>&
		{
			for (int i = 0; i < nrows*ncolumns; ++i)
				data[i] *= alpha;
			return *dynamic_cast<glsl_matrix_type<T, nrows, ncolumns>*>(this);
		}

		//multiplies two matrices
		template<typename P, uint32_t ncolumns2>
		auto operator*(const glsl_matrix_type<P, ncolumns, ncolumns2>& other) const ->
			glsl_matrix_type < decltype(T{}*P{}), nrows, ncolumns2>
		{
			typedef decltype(T{}*P{}) return_value_type;
			std::array<return_value_type, nrows*ncolumns2> res_data;

			for (int j = 0; j < ncolumns2; ++j)
				for (int i = 0; i < nrows; ++i)
				{
					res_data[j*nrows + i] = 0;
					for (int k = 0; k < ncolumns; ++k)
						res_data[j*nrows + i] += data[k*nrows + i] * other.data[j*ncolumns + k];
				}

			return glsl_matrix_type < return_value_type, nrows, ncolumns2>(res_data);
		}

		//multiplies two matrices and assigns the result to the left operand of multiplication
		template<typename P>
		auto operator*=(const glsl_matrix_type<P, ncolumns, ncolumns>& other) ->
			glsl_matrix_type<T, nrows, ncolumns>&
		{
			for (int i = 0; i < nrows; ++i)
			{
				std::array<decltype(T{}*P{}), ncolumns> aux;
				for (int j = 0; j < ncolumns; ++j)
				{
					aux[j] = 0;
					for (int k = 0; k < ncolumns; ++k)
						aux[j] += data[k*nrows + i] + other.data[j*ncolumns + k];
				}

				for (int j = 0; j < ncolumns; ++j)
					data[j*nrows + i] = static_cast<T>(aux[j]);
			}

			return *dynamic_cast<glsl_matrix_type<T, nrows, ncolumns>*>(this);
		}

		//computes additive-inverse matrix (i.e. multiplies every matrix entry by -1)
		glsl_matrix_type<T, nrows, ncolumns> operator-() const
		{
			std::array<T, nrows*ncolumns> res_data;
			for (int i = 0; i < nrows*ncolumns; ++i)
				res_data[i] = -data[i];

			return glsl_matrix_type<T, nrows, ncolumns>(res_data);
		}

		//returns transpose matrix
		glsl_matrix_type<T, ncolumns, nrows> transpose() const
		{
			std::array<T, nrows*ncolumns> res_data;

			for (int j = 0; j < nrows; ++j)
				for (int i = 0; i < ncolumns; ++i)
					res_data[j*ncolumns + i] = data[i*nrows + j];

			return glsl_matrix_type<T, ncolumns, nrows>(res_data);
		}

		//converts given matrix to a GLSL matrix with provided base type
		template<typename P>
		explicit operator glsl_matrix_type<P, nrows, ncolumns>() const
		{
			std::array<P, nrows*ncolumns> res_data;
			for (int i = 0; i < nrows*ncolumns; ++i)
				res_data[i] = static_cast<P>(data[i]);

			return glsl_matrix_type<P, nrows, ncolumns>(res_data);
		}
	};

	//Multiplies scalar by matrix
	template<typename T, uint32_t nrows, uint32_t ncolumns>
	auto operator*(float alpha, const glsl_matrix_type<T, nrows, ncolumns>& matrix) ->
		glsl_matrix_type < decltype(float{}*T{}), nrows, ncolumns >
	{
		typedef decltype(float{}*T{}) return_value_type;
		std::array<return_value_type, nrows*ncolumns> res_data;

		const T* p_data = matrix.getRawData();

		for (int i = 0; i < nrows*ncolumns; ++i)
			res_data[i] = p_data[i] * alpha;

		return glsl_matrix_type < return_value_type, nrows, ncolumns>(res_data);
	}

	template<typename T, uint32_t nrows, uint32_t ncolumns>
	auto operator*(double alpha, const glsl_matrix_type<T, nrows, ncolumns>& matrix) ->
		glsl_matrix_type < decltype(double{}*T{}), nrows, ncolumns >
	{
		typedef decltype(double{}*T{}) return_value_type;
		std::array<return_value_type, nrows*ncolumns> res_data;

		const T* p_data = matrix.getRawData();

		for (int i = 0; i < nrows*ncolumns; ++i)
			res_data[i] = p_data[i] * alpha;

		return glsl_matrix_type < return_value_type, nrows, ncolumns>(res_data);
	}



	//General matrix declaration
	template<typename T, uint32_t nrows, uint32_t ncolumns>
	class concrete_matrix : public abstract_matrix<T, nrows, ncolumns>
	{
	public:
		concrete_matrix() : abstract_matrix() {}
		explicit concrete_matrix(data_storage_type data) : abstract_matrix(data) {}
		concrete_matrix(T val) : abstract_matrix(val) {}
	};

	//Specialization used by square matrices
	template<typename T, uint32_t dim>
	class concrete_matrix<T, dim, dim> :public abstract_matrix<T, dim, dim>
	{
	public:
		concrete_matrix() : abstract_matrix() {}
		explicit concrete_matrix(data_storage_type data) : abstract_matrix(data) {}
		concrete_matrix(T val) : abstract_matrix(val) {}

		//computes determinant of the matrix
		T determinant() const
		{
			data_storage_type aux = data;
			std::array<uint32_t, dim> permutation;
			T rv = 1;

			for (int i = 0; i < dim ; ++i)
			{
				std::pair<T, uint32_t> max_elem = get_max_column_elem(aux, i);
				if (!max_elem.first) return 0;

				permutation[i] = max_elem.second;
				rv = rv*aux[i*dim + max_elem.second];

				for (int j = i + 1; j < dim; ++j)
				{
					T alpha = aux[j*dim + max_elem.second] / max_elem.first;

					for (int k = 0; k < dim; ++k)
						aux[j*dim + k] -= alpha*aux[i*dim + k];
					aux[j*dim + max_elem.second] = 0;
				}
			}

			//Compute sign of determinant
			int sign = 1;
			for (int i = 0; i < dim; ++i)
				for (int j = i + 1; j < dim; ++j)
					if (permutation[i]>permutation[j])
						sign *= -1;


			return rv*sign;
		}

		//computes inverse matrix
		glsl_matrix_type <decltype(1.0f / T{ 1 }), dim, dim> inverse() const
		{

			typedef signed decltype(1.0f / T{ 1 }) return_value_type;
			data_storage_type aux = data;
			data_storage_type inv_matrix_data = {};
			std::array<uint32_t, dim> permutation;

			for (int i = 0; i < dim; ++i)
				inv_matrix_data[i*dim+i] = 1;

			for (int i = 0; i < dim; ++i)
			{
				std::pair<T, uint32_t> max_elem = get_max_row_elem(aux, i);

				if (!max_elem.first)
					throw(std::runtime_error("matrix does not have inverse"));

				permutation[i] = max_elem.second;

				for (int k = 0; k < dim; ++k)
				{
					aux[k*dim + i] /= max_elem.first;
					inv_matrix_data[k*dim + i] /= max_elem.first;
				}

				for (int j = 0; j < dim; ++j)
				{
					if (j == i) continue;

					T alpha = aux[max_elem.second*dim + j] / aux[max_elem.second*dim + i];
					for (int k = 0; k < dim; ++k)
					{
						aux[k*dim + j] -= alpha*aux[k*dim + i];
						inv_matrix_data[k*dim + j] -= alpha*inv_matrix_data[k*dim + i];
					}
				}
			}

			data_storage_type inv_matrix_data_permuted;
			for (int i = 0; i < dim; ++i)
			{
				for (int k = 0; k < dim; ++k)
					inv_matrix_data_permuted[k * dim + permutation[i]] = inv_matrix_data[k*dim +i];
			}

			return glsl_matrix_type<return_value_type, dim, dim>(inv_matrix_data_permuted);

		}
	};

	//mat4 specialization
	template<typename T>
	class glsl_matrix_type<T, 4, 4> : public concrete_matrix<T, 4, 4>
	{
	public:
		glsl_matrix_type() : concrete_matrix() {}

		explicit glsl_matrix_type(data_storage_type data) : concrete_matrix(data) {}

		glsl_matrix_type(T val) : concrete_matrix(val) {}

		glsl_matrix_type(const tag_vec4<T>& col1, const tag_vec4<T>& col2, const tag_vec4<T>& col3, const tag_vec4<T>& col4) : 
			concrete_matrix(data_storage_type{col1.x, col1.y, col1.z, col1.w,
			col2.x, col2.y, col2.z, col2.w,
			col3.x, col3.y, col3.z, col3.w,
			col4.x, col4.y, col4.z, col4.w })
		{
		}

		glsl_matrix_type(T m11, T m21, T m31, T m41, T m12, T m22, T m32, T m42, T m13, T m23, T m33, T m43, T m14, T m24, T m34, T m44) :
			concrete_matrix(data_storage_type{ m11, m21, m31, m41, m12, m22, m32, m42, m13, m23, m33, m43, m14, m24, m34, m44 })
		{
		}

		//matrix-vector multiplication
		template<typename P>
		auto operator *(const tag_vec4<P>& vector) const->tag_vec4 < decltype(T{}*P{})>
		{
			typedef decltype(T{}*P{}) return_type;
			return_type res_data[4];
			for (int i = 0; i < 4; ++i)
			{
				res_data[i] = 0;
				for (int k = 0; k < 4; ++k)
					res_data[i] += data[k*num_rows + i] * vector[k];
			}
				
			return tag_vec4<return_type>(res_data[0], res_data[1], res_data[2], res_data[3]);
		}

		using abstract_matrix<T, 4, 4>::operator*;	//inherited multiplication must be overloaded, not shadowed!
	};

	//mat3 specialization
	template<typename T>
	class glsl_matrix_type<T, 3, 3> : public concrete_matrix<T, 3, 3>
	{
	public:
		glsl_matrix_type() : concrete_matrix() {}

		explicit glsl_matrix_type(data_storage_type data) : concrete_matrix(data) {}

		glsl_matrix_type(T val) : concrete_matrix(val) {}

		glsl_matrix_type(const tag_vec3<T>& col1, const tag_vec3<T>& col2, const tag_vec3<T>& col3) :
			concrete_matrix(data_storage_type{ col1.x, col1.y, col1.z,
			col2.x, col2.y, col2.z,
			col3.x, col3.y, col3.z }) {}

		glsl_matrix_type(T m11, T m21, T m31, T m12, T m22, T m32, T m13, T m23, T m33) :
			concrete_matrix(data_storage_type{ m11, m21, m31, m12, m22, m32, m13, m23, m33 }) {}

		//matrix-vector multiplication
		template<typename P>
		auto operator *(const tag_vec3<P>& vector) const->tag_vec3 < decltype(T{}*P{})>
		{
			typedef decltype(T{}*P{}) return_type;
			return_type res_data[3];
			for (int i = 0; i < num_rows; ++i)
			{
				res_data[i] = 0;
				for (int k = 0; k < num_columns; ++k)
					res_data[i] += data[k*num_rows + i] * vector[k];
			}

			return tag_vec3<return_type>(res_data[0], res_data[1], res_data[2]);
		}

		using abstract_matrix<T, 3, 3>::operator*;		//multiplication operator should be inherited and overridden, not shadowed
	};

	//mat2 specialization
	template<typename T>
	class glsl_matrix_type<T, 2, 2> : public concrete_matrix<T, 2, 2>
	{
	public:
		glsl_matrix_type() : concrete_matrix() {}

		explicit glsl_matrix_type(data_storage_type data) : concrete_matrix(data) {}

		glsl_matrix_type(T val) : concrete_matrix(val) {}

		glsl_matrix_type(const tag_vec2<T>& col1, const tag_vec2<T>& col2) :
			concrete_matrix(data_storage_type{ col1.x, col1.y, col2.x, col2.y }) {}

		glsl_matrix_type(T m11, T m21, T m12, T m22) :
			concrete_matrix(data_storage_type{ m11, m21, m12, m22 }) {}

		//matrix-vector multiplication
		template<typename P>
		auto operator *(const tag_vec2<P>& vector) const->tag_vec2 < decltype(T{}*P{})>
		{
			typedef decltype(T{}*P{}) return_type;
			return_type res_data[2];

			for (int i = 0; i < num_rows; ++i)
			{
				res_data[i] = 0;
				for (int k = 0; k < num_columns; ++k)
					res_data[i] += data[k*num_rows + i] * vector[k];
			}

			return tag_vec2<return_type>(res_data[0], res_data[1]);
		}

		using abstract_matrix<T, 2, 2>::operator*;		//multiplication operator should be inherited and overridden, not shadowed
	};

	//mat4x3 specialization
	template<typename T>
	class glsl_matrix_type<T, 4, 3> : public concrete_matrix<T, 4, 3>
	{
	public:
		glsl_matrix_type() : concrete_matrix() {}

		explicit glsl_matrix_type(data_storage_type data) : concrete_matrix(data) {}

		glsl_matrix_type(T val) : concrete_matrix(val) {}

		glsl_matrix_type(const tag_vec4<T>& col1, const tag_vec4<T>& col2, const tag_vec4<T>& col3) :
			concrete_matrix(data_storage_type{ col1.x, col1.y, col1.z, col1.w,
			col2.x, col2.y, col2.z, col2.w,
			col3.x, col3.y, col3.z, col3.w }) {}

		glsl_matrix_type(T m11, T m21, T m31, T m41, T m12, T m22, T m32, T m42, T m13, T m23, T m33, T m43) :
			concrete_matrix(data_storage_type{ m11, m21, m31, m41, m12, m22, m32, m42, m13, m23, m33, m43 })
		{}

		//matrix-vector multiplication
		template<typename P>
		auto operator *(const tag_vec3<P>& vector) const->tag_vec4 < decltype(T{}*P{})>
		{
			typedef decltype(T{}*P{}) return_type;
			return_type res_data[4];

			for (int i = 0; i < num_rows; ++i)
			{
				res_data[i] = 0;
				for (int k = 0; k < num_columns; ++k)
					res_data[i] += data[k*num_rows + i] * vector[k];
			}

			return tag_vec4<return_type>(res_data[0], res_data[1], res_data[2], res_data[3]);
		}

		using abstract_matrix<T, 4, 3>::operator*;		//multiplication operator should be inherited and overridden, not shadowed
	};

	//mat3x4 specialization
	template<typename T>
	class glsl_matrix_type<T, 3, 4> : public concrete_matrix<T, 3, 4>
	{
	public:
		glsl_matrix_type() : concrete_matrix() {}

		explicit glsl_matrix_type(data_storage_type data) : concrete_matrix(data) {}

		glsl_matrix_type(T val) : concrete_matrix(val) {}

		glsl_matrix_type(const tag_vec3<T>& col1, const tag_vec3<T>& col2, const tag_vec3<T>& col3, const tag_vec3<T>& col4) :
			concrete_matrix(data_storage_type{ col1.x, col1.y, col1.z,
			col2.x, col2.y, col2.z,
			col3.x, col3.y, col3.z,
			col4.x, col4.y, col4.z }) {}

		glsl_matrix_type(T m11, T m21, T m31, T m12, T m22, T m32, T m13, T m23, T m33, T m14, T m24, T m34) :
			concrete_matrix(data_storage_type{ m11, m21, m31, m12, m22, m32, m13, m23, m33, m14, m24, m34 }) {}

		//matrix-vector multiplication
		template<typename P>
		auto operator *(const tag_vec4<P>& vector) const->tag_vec3 < decltype(T{}*P{})>
		{
			typedef decltype(T{}*P{}) return_type;
			return_type res_data[3];

			for (int i = 0; i < num_rows; ++i)
			{
				res_data[i] = 0;
				for (int k = 0; k < num_columns; ++k)
					res_data[i] += data[k*num_rows + i] * vector[k];
			}

			return tag_vec3<return_type>(res_data[0], res_data[1], res_data[2]);
		}

		using abstract_matrix<T, 3, 4>::operator*;		//multiplication operator should be inherited and overridden, not shadowed
	};

	//mat4x2 specialization
	template<typename T>
	class glsl_matrix_type<T, 4, 2> : public concrete_matrix<T, 4, 2>
	{
	public:
		glsl_matrix_type() : concrete_matrix() {}

		explicit glsl_matrix_type(data_storage_type data) : concrete_matrix(data) {}

		glsl_matrix_type(T val) : concrete_matrix(val) {}

		glsl_matrix_type(const tag_vec4<T>& col1, const tag_vec4<T>& col2) :
			concrete_matrix(data_storage_type{ col1.x, col1.y, col1.z, col1.w,
			col2.x, col2.y, col2.z, col2.w }) {}

		glsl_matrix_type(T m11, T m21, T m31, T m41, T m12, T m22, T m32, T m42) :
			concrete_matrix(data_storage_type{ m11, m21, m31, m41, m12, m22, m32, m42 }) {}

		//matrix-vector multiplication
		template<typename P>
		auto operator *(const tag_vec2<P>& vector) const->tag_vec4 < decltype(T{}*P{})>
		{
			typedef decltype(T{}*P{}) return_type;
			return_type res_data[4];

			for (int i = 0; i < num_rows; ++i)
			{
				res_data[i] = 0;
				for (int k = 0; k < num_columns; ++k)
					res_data[i] += data[k*num_rows + i] * vector[k];
			}

			return tag_vec4<return_type>(res_data[0], res_data[1], res_data[2], res_data[3]);
		}

		using abstract_matrix<T, 4, 2>::operator*;		//multiplication operator should be inherited and overridden, not shadowed
	};

	//mat2x4 specialization
	template<typename T>
	class glsl_matrix_type<T, 2, 4> : public concrete_matrix<T, 2, 4>
	{
	public:
		glsl_matrix_type() : concrete_matrix() {}

		explicit glsl_matrix_type(data_storage_type data) : concrete_matrix(data) {}

		glsl_matrix_type(T val) : concrete_matrix(val) {}

		glsl_matrix_type(const tag_vec2<T>& col1, const tag_vec2<T>& col2, const tag_vec2<T>& col3, const tag_vec2<T>& col4) : 
			concrete_matrix(data_storage_type{ col1.x, col1.y,
			col2.x, col2.y,
			col3.x, col3.y,
			col4.x, col4.y }) {}

		glsl_matrix_type(T m11, T m21, T m12, T m22, T m13, T m23, T m14, T m24) :
			concrete_matrix(data_storage_type{ m11, m21, m12, m22, m13, m23, m14, m24 }) {}

		//matrix-vector multiplication
		template<typename P>
		auto operator *(const tag_vec4<P>& vector) const->tag_vec2 < decltype(T{}*P{})>
		{
			typedef decltype(T{}*P{}) return_type;
			return_type res_data[2];

			for (int i = 0; i < num_rows; ++i)
			{
				res_data[i] = 0;
				for (int k = 0; k < num_columns; ++k)
					res_data[i] += data[k*num_rows + i] * vector[k];
			}

			return tag_vec2<return_type>(res_data[0], res_data[1]);
		}

		using abstract_matrix<T, 2, 4>::operator*;		//multiplication operator should be inherited and overridden, not shadowed
	};

	//mat3x2 specialization
	template<typename T>
	class glsl_matrix_type<T, 3, 2> : public concrete_matrix<T, 3, 2>
	{
	public:
		glsl_matrix_type() : concrete_matrix() {}

		explicit glsl_matrix_type(data_storage_type data) : concrete_matrix(data) {}

		glsl_matrix_type(T val) : concrete_matrix(val) {}

		glsl_matrix_type(const tag_vec3<T>& col1, const tag_vec3<T>& col2) :
			concrete_matrix(data_storage_type{ col1.x, col1.y, col1.z,
			col2.x, col2.y, col2.z }) {}

		glsl_matrix_type(T m11, T m21, T m31, T m12, T m22, T m32) :
			concrete_matrix(data_storage_type{ m11, m21, m31, m12, m22, m32 }) {}

		//matrix-vector multiplication
		template<typename P>
		auto operator *(const tag_vec2<P>& vector) const->tag_vec3 < decltype(T{}*P{})>
		{
			typedef decltype(T{}*P{}) return_type;
			return_type res_data[3];

			for (int i = 0; i < num_rows; ++i)
			{
				res_data[i] = 0;
				for (int k = 0; k < num_columns; ++k)
					res_data[i] += data[k*num_rows + i] * vector[k];
			}

			return tag_vec3<return_type>(res_data[0], res_data[1], res_data[2]);
		}

		using abstract_matrix<T, 3, 2>::operator*;		//multiplication operator should be inherited and overridden, not shadowed
	};

	//mat2x3 specialization
	template<typename T>
	class glsl_matrix_type<T, 2, 3> : public concrete_matrix<T, 2, 3>
	{
	public:
		glsl_matrix_type() : concrete_matrix() {}

		explicit glsl_matrix_type(data_storage_type data) : concrete_matrix(data) {}

		glsl_matrix_type(T val) : concrete_matrix(val) {}

		glsl_matrix_type(const tag_vec2<T>& col1, const tag_vec2<T>& col2, const tag_vec2<T>& col3) :
			concrete_matrix(data_storage_type{ col1.x, col1.y,
			col2.x, col2.y,
			col3.x, col3.y }) {}

		glsl_matrix_type(T m11, T m21, T m12, T m22, T m13, T m23) :
			concrete_matrix(data_storage_type{ m11, m21, m12, m22, m13, m23 }) {}

		//matrix-vector multiplication
		template<typename P>
		auto operator *(const tag_vec3<P>& vector) const->tag_vec2 < decltype(T{}*P{})>
		{
			typedef decltype(T{}*P{}) return_type;
			return_type res_data[2];

			for (int i = 0; i < num_rows; ++i)
			{
				res_data[i] = 0;
				for (int k = 0; k < num_columns; ++k)
					res_data[i] += data[k*num_rows + i] * vector[k];
			}

			return tag_vec2<return_type>(res_data[0], res_data[1]);
		}

		using abstract_matrix<T, 2, 3>::operator*;	//multiplication operator should be inherited and overridden, not shadowed
	};


	//GLSL matrix types typedefs
	typedef glsl_matrix_type<GLfloat, 4, 4> mat4;
	typedef mat4 mat4x4;
	typedef glsl_matrix_type<GLfloat, 3, 3> mat3;
	typedef mat3 mat3x3;
	typedef glsl_matrix_type<GLfloat, 2, 2> mat2;
	typedef mat2 mat2x2;
	typedef glsl_matrix_type<GLfloat, 4, 3> mat4x3;
	typedef glsl_matrix_type<GLfloat, 4, 2> mat4x2;
	typedef glsl_matrix_type<GLfloat, 3, 4> mat3x4;
	typedef glsl_matrix_type<GLfloat, 2, 4> mat2x4;
	typedef glsl_matrix_type<GLfloat, 3, 2> mat3x2;
	typedef glsl_matrix_type<GLfloat, 2, 3> mat2x3;

	typedef glsl_matrix_type<GLdouble, 4, 4> dmat4;
	typedef dmat4 dmat4x4;
	typedef glsl_matrix_type<GLdouble, 3, 3> dmat3;
	typedef dmat3 dmat3x3;
	typedef glsl_matrix_type<GLdouble, 2, 2> dmat2;
	typedef dmat2 dmat2x2;
	typedef glsl_matrix_type<GLdouble, 4, 3> dmat4x3;
	typedef glsl_matrix_type<GLdouble, 4, 2> dmat4x2;
	typedef glsl_matrix_type<GLdouble, 3, 4> dmat3x4;
	typedef glsl_matrix_type<GLdouble, 2, 4> dmat2x4;
	typedef glsl_matrix_type<GLdouble, 3, 2> dmat3x2;
	typedef glsl_matrix_type<GLdouble, 2, 3> dmat2x3;
}

#define TW__MATRIX_TYPES__
#endif