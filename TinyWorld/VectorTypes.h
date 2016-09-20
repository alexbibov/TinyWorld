#ifndef TW__VECTOR_TYPES__

#include <initializer_list>

#include "Misc.h"

//NOTE: there is small mistake in design of these classes, which leads to necessity to repeat vector-vector and matrix-vector
//operation overriding. Possible way to alleviate this problem is to inherit vector types from a common type, which comprises
//common operations, and design each concrete vector type as an inherited template specialization

namespace tiny_world
{
	//forward declarations of tag_vec3 and tag_vec2
	template<typename T> struct tag_vec2;
	template<typename T> struct tag_vec3;

	template<typename T>
	struct tag_vec4{
	private:
		mutable T data[4];		//this variable is only used to return data stored in vector as an array. This becomes handy for uniform assignments.

	public:
		typedef T value_type;
		static const char dimension = 4;

		T x;
		T y;
		T z;
		T w;

		tag_vec4(T x, T y, T z, T w) : x{ x }, y{ y }, z{ z }, w{ w } {}
		tag_vec4(T x, T y, T z) : x{ x }, y{ y }, z{ z }, w{ 0 } {}
		tag_vec4(T x, T y) : x{ x }, y{ y }, z{ 0 }, w{ 0 } {}
		tag_vec4(T x) : x{ x }, y{ x }, z{ x }, w{ x }{}
		tag_vec4() : x{}, y{}, z{}, w{} {}

		tag_vec4(const tag_vec3<T>& v3, T w) : x{ v3.x }, y{ v3.y }, z{ v3.z }, w{ w } {}
		tag_vec4(T x, const tag_vec3<T>& v3) : x{ x }, y{ v3.x }, z{ v3.y }, w{ v3.z } {}
		tag_vec4(const tag_vec2<T>& v2_1, const tag_vec2<T>& v2_2) : x{ v2_1.x }, y{ v2_1.y }, z{ v2_2.x }, w{ v2_2.y } {}
		tag_vec4(const tag_vec2<T>& v2, T z, T w) : x{ v2.x }, y{ v2.y }, z{ z }, w{ w } {}
		tag_vec4(T x, const tag_vec2<T>& v2, T w) : x{ x }, y{ v2.x }, z{ v2.y }, w{ w } {}
		tag_vec4(T x, T y, const tag_vec2<T>& v2) : x{ x }, y{ y }, z{ v2.x }, w{ v2.y } {}

		//Copy constructor
		tag_vec4(const tag_vec4& other) : x(other.x), y(other.y),
			z(other.z), w(other.w) 
		{

		}

		//returns data stored in vector packed into an array
		const T* getDataAsArray() const 
		{
			data[0] = x; data[1] = y; data[2] = z; data[3] = w;
			return data;
		}

		//returns norm of contained vector
		decltype(1.0f / T{ 1 }) norm() const
		{
			return std::sqrt(x*x + y*y + z*z + w*w);
		}

		//returns normalized version of contained vector
		tag_vec4<decltype(1.0f / T{ 1 })> get_normalized() const
		{
			typedef decltype(1.0f / T{ 1 }) result_type;
			result_type norm_factor = norm();
			return tag_vec4 < result_type > {x, y, z, w} / norm_factor;
		}

		//returns dot product of two vectors
		T dot_product(const tag_vec4& other) const
		{
			return x*other.x + y*other.y + z*other.z + w*other.w;
		}

		//component wise multiplication by -1
		tag_vec4<T> operator -() const
		{
			return tag_vec4<T>{-x, -y, -z, -w};
		}

		//element access for read-only via indexing operator
		const T& operator[](const int index) const
		{
			switch (index)
			{
			case 0:
				return x;
			case 1:
				return y;
			case 2:
				return z;
			case 3:
				return w;
			default:
				throw(std::range_error("Out of range: vec4 only allows element access for indexes from 0 to 3"));
			}
		}

		//element access via indexing operator for read-write
		T& operator[](const int index)
		{
			return const_cast<T&>(const_cast<const tag_vec4*>(this)->operator [](index));
		}

		//Template comparison operator
		template<typename P>
		bool operator==(const tag_vec4<P>& other) const
		{
			return x == other.x && y == other.y && z == other.z && w == other.w;
		}

		//Copy assignment operator
		tag_vec4& operator=(const tag_vec4& other)
		{
			if (this == &other)
				return *this;

			x = other.x;
			y = other.y;
			z = other.z;
			w = other.w;
			return *this;
		}

		//Conversion between vectors with different base value types
		template<typename P>
		explicit operator tag_vec4<P>() const
		{
			return tag_vec4 < P > {static_cast<P>(x), static_cast<P>(y), static_cast<P>(z), static_cast<P>(w)};
		}

		//vector addition
		template<typename P>
		auto operator +(const tag_vec4<P>& other) const->tag_vec4<decltype(T{} +P{})>
		{
			return tag_vec4 < decltype(T{} +P{}) > {x + other.x, y + other.y, z + other.z, w + other.w};
		}

		//Template addition-assignment operator
		template<typename P>
		tag_vec4<T>& operator+=(const tag_vec4<P>& other)
		{
			x += other.x;
			y += other.y;
			z += other.z;
			w += other.w;

			return *this;
		}

		//vector subtraction
		template<typename P>
		auto operator -(const tag_vec4<P>& other) const->tag_vec4<decltype(T{} -P{})>
		{
			return tag_vec4<decltype(T{} -P{})>{x - other.x, y - other.y, z - other.z, w - other.w};
		}

		//Template subtraction-assignment operator
		template<typename P>
		tag_vec4<T>& operator-=(const tag_vec4<P>& other)
		{
			x -= other.x;
			y -= other.y;
			z -= other.z;
			w -= other.w;

			return *this;
		}

		//component-wise vector multiplication
		template<typename P>
		auto operator *(const tag_vec4<P>& other) const->tag_vec4<decltype(T{} *P{})>
		{
			return tag_vec4<decltype(T{} *P{})>{x * other.x, y * other.y, z * other.z, w * other.w};
		}

		//Template component-wise vector multiplication-assignment operator
		template<typename P>
		tag_vec4<T>& operator*=(const tag_vec4<P>& other)
		{
			x *= other.x;
			y *= other.y;
			z *= other.z;
			w *= other.w;

			return *this;
		}

		//component-wise vector division
		template<typename P>
		auto operator /(const tag_vec4<P>& other) const->tag_vec4 < decltype(T{ 1 } / P{ 1 }) >
		{
			return tag_vec4 < decltype(T{ 1 } / P{ 1 }) > {x / other.x, y / other.y, z / other.z, w / other.w};
		}

		//Template component-wise vector division-assignment operator
		template<typename P>
		tag_vec4<T>& operator/=(const tag_vec4<P>& other)
		{
			x /= other.x;
			y /= other.y;
			z /= other.z;
			w /= other.w;

			return *this;
		}

		//multiplies vector by scalar
		template<typename P>
		auto operator *(P alpha) const->tag_vec4 < decltype(T{}*P{}) >
		{
			return tag_vec4 < decltype(T{} *P{}) > {x*alpha, y*alpha, z*alpha, w*alpha};
		}

		//Template vector-scalar multiplication-assignment operator
		template<typename P>
		tag_vec4<T>& operator*=(P alpha)
		{
			x *= alpha;
			y *= alpha;
			z *= alpha;
			w *= alpha;

			return *this;
		}

		//divides vector by scalar
		template<typename P>
		auto operator /(P alpha) const->tag_vec4 < decltype(T{ 1 } / P{ 1 }) >
		{
			return tag_vec4 < decltype(T{ 1 } / P{ 1 }) > {x / alpha, y / alpha, z / alpha, w / alpha};
		}

		//Template vector-scalar division-assignment operator
		template<typename P>
		tag_vec4<T>& operator /=(P alpha)
		{
			x /= alpha;
			y /= alpha;
			z /= alpha;
			w /= alpha;

			return *this;
		}
	};

	//multiplies scalar by vector
	template<typename T>
	tag_vec4<decltype(float{}*T{})> operator*(float alpha, const tag_vec4<T>& vector)
	{
		return tag_vec4 < decltype(float{}*T{}) > {alpha*vector.x, alpha*vector.y, alpha*vector.z, alpha*vector.w};
	}

	template<typename T>
	tag_vec4<decltype(double{}*T{})> operator*(double alpha, const tag_vec4<T>& vector)
	{
		return tag_vec4 < decltype(double{}*T{}) > {alpha*vector.x, alpha*vector.y, alpha*vector.z, alpha*vector.w};
	}



	template<typename T>
	struct tag_vec3
	{
	private:
		mutable T data[3];		//this variable is only used to return data stored in vector as an array. This becomes handy for uniform assignments.

	public:
		typedef T value_type;
		static const char dimension = 3;

		T x;
		T y;
		T z;

		tag_vec3(T x, T y, T z) : x{ x }, y{ y }, z{ z } {}
		tag_vec3(T x, T y) : x{ x }, y{ y }, z{ 0 } {}
		tag_vec3(T x) : x{ x }, y{ x }, z{ x } {}
		tag_vec3() : x{}, y{}, z{} {}

		tag_vec3(const tag_vec2<T>& v2, T z) : x{ v2.x }, y{ v2.y }, z{ z } {}
		tag_vec3(T x, const tag_vec2<T>& v2) : x{ x }, y{ v2.x }, z{ v2.y } {}

		//Copy constructor
		tag_vec3(const tag_vec3& other) : x(other.x), y(other.y), z(other.z) 
		{

		}

		//returns data stored in vector represented by an array
		const T* getDataAsArray() const
		{
			data[0] = x; data[1] = y; data[2] = z;
			return data;
		}

		//returns norm of contained vector
		decltype(1.0f / T{ 1 }) norm() const
		{
			return std::sqrt(x*x + y*y + z*z);
		}

		//returns normalized version of contained vector
		tag_vec3<decltype(1.0f / T{ 1 })> get_normalized() const
		{
			typedef decltype(1.0f / T{ 1 }) result_type;
			result_type norm_factor = norm();
			return tag_vec3 < result_type > {x, y, z} / norm_factor;
		}

		//returns dot product of two vectors
		T dot_product(const tag_vec3& other) const
		{
			return x*other.x + y*other.y + z*other.z;
		}

		//returns cross product of two vectors
		tag_vec3 cross_product(const tag_vec3& other) const
		{
			return tag_vec3<T>{y*other.z - z*other.y, -x*other.z + z*other.x, x*other.y - y*other.x};
		}

		//converts vec3 defined in 3D Euclidean space to the corresponding vec4 defined in 4D Homogeneous space
		template<typename P>
		explicit operator tag_vec4<P>() const
		{
			return tag_vec4<P>{static_cast<P>(x), static_cast<P>(y), static_cast<P>(z), P{ 1 }};
		}

		//Conversion between templates with different value types
		template<typename P>
		explicit operator tag_vec3<P>() const
		{
			return tag_vec3<P>{static_cast<P>(x), static_cast<P>(y), static_cast<P>(z)};
		}

		//component wise multiplication by -1
		tag_vec3<T> operator -() const
		{
			return tag_vec3<T>{-x, -y, -z};
		}

		//element access for read-only via indexing operator
		const T& operator[](const int index) const
		{
			switch (index)
			{
			case 0:
				return x;
			case 1:
				return y;
			case 2:
				return z;
			default:
				throw(std::range_error("Out of range: vec3 only allows element access for indexes from 0 to 2"));
			}
		}

		//element access via indexing operator for read-write
		T& operator[](const int index)
		{
			return const_cast<T&>(const_cast<const tag_vec3*>(this)->operator [](index));
		}

		//Template comparison operator
		template<typename P>
		bool operator==(const tag_vec3<P>& other) const
		{
			return x == other.x && y == other.y && z == other.z;
		}

		//Copy assignment operator
		tag_vec3& operator=(const tag_vec3& other)
		{
			if (this == &other)
				return *this;

			x = other.x;
			y = other.y;
			z = other.z;
			return *this;
		}


		//vector addition
		template<typename P>
		auto operator +(const tag_vec3<P>& other) const->tag_vec3<decltype(T{} +P{})>
		{
			return tag_vec3<decltype(T{} +P{})>{x + other.x, y + other.y, z + other.z};
		}

		//Template addition-assignment operator
		template<typename P>
		tag_vec3<T>& operator+=(const tag_vec3<P>& other)
		{
			x += other.x;
			y += other.y;
			z += other.z;

			return *this;
		}

		//vector subtraction
		template<typename P>
		auto operator -(const tag_vec3<P>& other) const->tag_vec3<decltype(T{} -P{})>
		{
			return tag_vec3<decltype(T{} -P{})>{x - other.x, y - other.y, z - other.z};
		}

		//Template subtraction-assignment operator
		template<typename P>
		tag_vec3<T>& operator-=(const tag_vec3<P>& other)
		{
			x -= other.x;
			y -= other.y;
			z -= other.z;

			return *this;
		}

		//component-wise vector multiplication
		template<typename P>
		auto operator *(const tag_vec3<P>& other) const->tag_vec3<decltype(T{} *P{})>
		{
			return tag_vec3<decltype(T{} *P{})>{x * other.x, y * other.y, z * other.z};
		}

		//Template component-wise vector multiplication-assignment operator
		template<typename P>
		tag_vec3<T>& operator*=(const tag_vec3<P>& other)
		{
			x *= other.x;
			y *= other.y;
			z *= other.z;

			return *this;
		}

		//component-wise vector division
		template<typename P>
		auto operator /(const tag_vec3<P>& other) const->tag_vec3 < decltype(T{ 1 } / P{ 1 }) >
		{
			return tag_vec3 < decltype(T{ 1 } / P{ 1 }) > {x / other.x, y / other.y, z / other.z};
		}

		//Template component-wise vector division-assignment operator
		template<typename P>
		tag_vec3<T>& operator/=(const tag_vec3<P>& other)
		{
			x /= other.x;
			y /= other.y;
			z /= other.z;

			return *this;
		}

		//multiplies vector by scalar
		template<typename P>
		auto operator *(P alpha) const->tag_vec3 < decltype(T{}*P{}) >
		{
			return tag_vec3 < decltype(T{}*P{}) > {x*alpha, y*alpha, z*alpha};
		}

		//Template vector-scalar multiplication-assignment operator
		template<typename P>
		tag_vec3<T>& operator*=(P alpha)
		{
			x *= alpha;
			y *= alpha;
			z *= alpha;

			return *this;
		}

		//divides vector by scalar
		template<typename P>
		auto operator /(P alpha) const->tag_vec3 < decltype(T{ 1 } / P{ 1 }) >
		{
			return tag_vec3 < decltype(T{ 1 } / P{ 1 }) > {x / alpha, y / alpha, z / alpha};
		}

		//Template vector-scalar division-assignment operator
		template<typename P>
		tag_vec3<T>& operator/=(P alpha)
		{
			x /= alpha;
			y /= alpha;
			z /= alpha;
			return *this;
		}
	};

	//multiplies scalar by vector
	template<typename T>
	tag_vec3<decltype(float{}*T{})> operator*(float alpha, const tag_vec3<T>& vector)
	{
		return tag_vec3 < decltype(float{}*T{}) > {alpha*vector.x, alpha*vector.y, alpha*vector.z};
	}

	template<typename T>
	tag_vec3<decltype(double{}*T{})> operator*(double alpha, const tag_vec3<T>& vector)
	{
		return tag_vec3 < decltype(double{}*T{}) > {alpha*vector.x, alpha*vector.y, alpha*vector.z};
	}



	template<typename T>
	struct tag_vec2
	{
	private:
		mutable T data[2];		//this variable is only used to return data stored in vector as an array. This becomes handy for uniform assignments.

	public:
		typedef T value_type;
		static const char dimension = 2;

		T x;
		T y;

		tag_vec2(T x, T y) : x{ x }, y{ y } {}
		tag_vec2(T x) : x{ x }, y{ x } {}
		tag_vec2() : x{}, y{} {}

		//Copy constructor
		tag_vec2(const tag_vec2& other) : x(other.x), y(other.y)
		{
		
		}

		//returns data stored in vector represented by an array
		const T* getDataAsArray() const
		{
			data[0] = x; data[1] = y;
			return data;
		}

		//returns norm of contained vector
		decltype(1.0f / T{ 1 }) norm() const
		{
			return std::sqrt(x*x + y*y);
		}

		//returns normalized version of contained vector
		tag_vec2<decltype(1.0f / T{ 1 })> get_normalized() const
		{
			typedef decltype(1.0f / T{ 1 }) result_type;
			result_type norm_factor = norm();
			return tag_vec2 < result_type > {x, y} / norm_factor;
		}

		//returns dot product of two vectors
		T dot_product(const tag_vec2& other) const
		{
			return x*other.x + y*other.y;
		}

		//converts vec2 to vec3
		template<typename P>
		explicit operator tag_vec3<P>() const
		{
			return tag_vec3<P>{static_cast<P>(x), static_cast<P>(y), P{ 0 }};
		}

		//converts vec2 represented in 2D Euclidean space to the corresponding vec4 represented in 4D Homogeneous space
		template<typename P>
		explicit operator tag_vec4<P>() const
		{
			return tag_vec4<P>{static_cast<P>(x), static_cast<P>(y), static_cast<P>(0), P{ 1 }};
		}

		//Conversion between templates with different value types
		template<typename P>
		explicit operator tag_vec2<P>() const
		{
			return tag_vec2<P>{static_cast<P>(x), static_cast<P>(y)};
		}

		//component wise multiplication by -1
		tag_vec2<T> operator -() const
		{
			return tag_vec2<T>{-x, -y};
		}

		//element access for read-only via indexing operator
		const T& operator[](const int index) const
		{
			switch (index)
			{
			case 0:
				return x;
			case 1:
				return y;
			default:
				throw(std::range_error("Out of range: vec2 only allows element access for indexes from 0 to 1"));
			}
		}

		//element access via indexing operator for read-write
		T& operator[](const int index)
		{
			return const_cast<T&>(const_cast<const tag_vec2*>(this)->operator [](index));
		}

		//Template comparison operator
		template<typename P>
		bool operator==(const tag_vec2<P>& other) const
		{
			return x == other.x && y == other.y;
		}

		//Copy assignment operator
		tag_vec2& operator=(const tag_vec2& other)
		{
			if (this == &other)
				return *this;

			x = other.x;
			y = other.y;
			return *this;
		}

		//vector addition
		template<typename P>
		auto operator +(const tag_vec2<P>& other) const->tag_vec2<decltype(T{} +P{})>
		{
			return tag_vec2<decltype(T{} +P{})>{x + other.x, y + other.y};
		}

		//Template addition-assignment operator
		template<typename P>
		tag_vec2<T>& operator+=(const tag_vec2<P>& other)
		{
			x += other.x;
			y += other.y;
			return *this;
		}

		//vector subtraction
		template<typename P>
		auto operator -(const tag_vec2<P>& other) const->tag_vec2<decltype(T{} -P{})>
		{
			return tag_vec2<decltype(T{} -P{})>{x - other.x, y - other.y};
		}

		//Template subtraction-assignment operator
		template<typename P>
		tag_vec2<T>& operator-=(const tag_vec2<P>& other)
		{
			x -= other.x;
			y -= other.y;
			return *this;
		}

		//component-wise vector multiplication
		template<typename P>
		auto operator *(const tag_vec2<P>& other) const->tag_vec2<decltype(T{} *P{})>
		{
			return tag_vec2<decltype(T{} *P{})>{x * other.x, y * other.y};
		}

		//Template component-wise vector multiplication-assignment operator
		template<typename P>
		tag_vec2<T>& operator*=(const tag_vec2<P>& other)
		{
			x *= other.x;
			y *= other.y;
			return *this;
		}

		//component-wise vector division
		template<typename P>
		auto operator /(const tag_vec2<P>& other) const->tag_vec2 < decltype(T{ 1 } / P{ 1 }) >
		{
			return tag_vec2 < decltype(T{ 1 } / P{ 1 }) > {x / other.x, y / other.y};
		}

		//Template component-wise vector division-assignment operator
		template<typename P>
		tag_vec2<T>& operator/=(const tag_vec2<P>& other)
		{
			x /= other.x;
			y /= other.y;
			return *this;
		}

		//multiplies vector by scalar
		template<typename P>
		auto operator *(P alpha) const->tag_vec2<decltype(T{} *P{})>
		{
			return tag_vec2 < decltype(T{} *P{})>{x*alpha, y*alpha};
		}

		//Template vector-scalar multiplication-assignment operator
		template<typename P>
		tag_vec2<T>& operator*=(P alpha)
		{
			x *= alpha;
			y *= alpha;
			return *this;
		}

		//divides vector by scalar
		template<typename P>
		auto operator /(P alpha) const->tag_vec2 < decltype(T{ 1 } / P{ 1 }) >
		{
			return tag_vec2 < decltype(T{ 1 } / P{ 1 }) > {x / alpha, y / alpha};
		}

		//Template vector-scalar division-assignment operator
		template<typename P>
		tag_vec2<T>& operator/=(P alpha)
		{
			x /= alpha;
			y /= alpha;
			return *this;
		}
	};

	//multiplies scalar by vector
	template<typename T>
	tag_vec2<decltype(float{}*T{})> operator*(float alpha, const tag_vec2<T>& vector)
	{
		return tag_vec2 < decltype(float{}*T{}) > {alpha*vector.x, alpha*vector.y};
	}

	template<typename T>
	tag_vec2<decltype(double{}*T{})> operator*(double alpha, const tag_vec2<T>& vector)
	{
		return tag_vec2 < decltype(double{}*T{}) > {alpha*vector.x, alpha*vector.y};
	}



#include <GL/glew.h>

	typedef tag_vec4<ogl_type_mapper<int>::ogl_type> ivec4;
	typedef tag_vec4<ogl_type_mapper<float>::ogl_type> vec4;
	typedef tag_vec4<ogl_type_mapper<double>::ogl_type> dvec4;
	typedef tag_vec4<ogl_type_mapper<unsigned int>::ogl_type> uvec4;
	typedef tag_vec4<ogl_type_mapper<bool>::ogl_type> bvec4;

	typedef tag_vec3<ogl_type_mapper<int>::ogl_type> ivec3;
	typedef tag_vec3<ogl_type_mapper<float>::ogl_type> vec3;
	typedef tag_vec3<ogl_type_mapper<double>::ogl_type> dvec3;
	typedef tag_vec3<ogl_type_mapper<unsigned int>::ogl_type> uvec3;
	typedef tag_vec3<ogl_type_mapper<bool>::ogl_type> bvec3;

	typedef tag_vec2<ogl_type_mapper<int>::ogl_type> ivec2;
	typedef tag_vec2<ogl_type_mapper<float>::ogl_type> vec2;
	typedef tag_vec2<ogl_type_mapper<double>::ogl_type> dvec2;
	typedef tag_vec2<ogl_type_mapper<unsigned int>::ogl_type> uvec2;
	typedef tag_vec2<ogl_type_mapper<bool>::ogl_type> bvec2;

}

#define  TW__VECTOR_TYPES__
#endif