#ifndef TW__QUATERNION_TYPES__

#include "VectorTypes.h"
#include "MatrixTypes.h"

namespace tiny_world
{

	//Describes Hamiltonian quaternion and related operations
	template<typename T>
	class Quaternion
	{
		template<typename P> friend class Quaternion;
	private:
		tag_vec4<T> internal_representation;	


	public:
		typedef T value_type;

		//Default initializer
		Quaternion() : internal_representation{} 
		{

		}

		//Copy constructor
		Quaternion(const Quaternion& other) : internal_representation{ other.internal_representation }
		{
		
		}

		//Move constructor
		Quaternion(Quaternion&& other) : internal_representation{ std::move(other.internal_representation) }
		{

		}

		//Copy assignment operator
		Quaternion& operator=(const Quaternion& other)
		{
			if (this == &other)
				return *this;

			internal_representation = other.internal_representation;
			return *this;
		}

		//Move assignment operator
		Quaternion& operator=(Quaternion&& other)
		{
			if (this == &other)
				return *this;

			internal_representation = std::move(other.internal_representation);
			return *this;
		}

		//Destructor
		~Quaternion()
		{
			 
		}



		//Initializes quaternion given its scalar and vector parts
		Quaternion(T scalar_part, const tag_vec3<T>& vector_part) : 
			internal_representation{ scalar_part, vector_part }
		{

		}

		//Initializes quaternion using provided 4D-vector
		explicit Quaternion(const tag_vec4<T>& vector) : internal_representation{ vector }
		{
			 
		}

		//Initializes quaternion given its elements
		Quaternion(T w, T x, T y, T z) : 
			internal_representation{ w, x, y, z }
		{

		}


		//Multiplies quaternion by scalar
		template<typename P>
		Quaternion<decltype(T{}*P{})> operator*(P other) const
		{
			return Quaternion < decltype(T{}*P{}) > {internal_representation*other};
		}
		
		//Multiplies quaternion by scalar and assigns the result to the left operand
		template<typename P>
		Quaternion& operator*=(P other)
		{
			internal_representation *= other;
			return *this;
		}


		//Divides quaternion by scalar
		template<typename P>
		Quaternion<decltype(T{ 1 } / P{ 1 })> operator/(P other) const
		{
			return Quaternion < decltype(T{ 1 } / P{ 1 }) > {internal_representation / other};
		}

		//Divides quaternion by scalar and assigns the result to the left operand of division
		template<typename P>
		Quaternion& operator/=(P other)
		{
			internal_representation /= other;
			return *this;
		}


		//Implements multiplication of two quaternions
		template<typename P>
		Quaternion<decltype(T{}*P{})> operator*(const Quaternion<P>& other) const
		{
			typedef decltype(T{}*P{}) result_type;

			tag_vec3<result_type> v1{ internal_representation[1], internal_representation[2], internal_representation[3] };
			tag_vec3<result_type> v2{ other.internal_representation[1], other.internal_representation[2], other.internal_representation[3] };

			result_type scalar_part = internal_representation[0] * other.internal_representation[0] - v1.dot_product(v2);
			tag_vec3<result_type> vector_part = internal_representation[0] * v2 + other.internal_representation[0] * v1 + v1.cross_product(v2);

			return Quaternion < result_type > {scalar_part, vector_part};
		}

		//Multiplies two quaternions and assigns the result of multiplication to the left operand
		template<typename P>
		Quaternion& operator*=(const Quaternion<P>& other)
		{
			typedef decltype(T{}*P{}) widest_type;

			tag_vec3<widest_type> v1{ internal_representation[1], internal_representation[2], internal_representation[3] };
			tag_vec3<widest_type> v2{ other.internal_representation[1], other.internal_representation[2], other.internal_representation[3] };

			widest_type scalar_part = internal_representation[0] * other.internal_representation[0] - v1.dot_product(v2);
			tag_vec3<widest_type> vector_part = internal_representation[0] * v2 + other.internal_representation[0] * v1 + v1.cross_product(v2);

			internal_representation[0] = scalar_part;
			internal_representation[1] = vector_part[0];
			internal_representation[2] = vector_part[1];
			internal_representation[3] = vector_part[2];

			return *this;
		}


		//Implements multiplication of quaternion by a 4D vector
		template<typename P>
		Quaternion<decltype(T{}*P{})> operator*(const tag_vec4<P>& other) const
		{
			return (*this)*Quaternion < P > {other};
		}

		//Implements multiplication of quaternion by a 4D vector followed by assignment of the result to the left operand (i.e. to the quaternion object) of multiplication
		template<typename P>
		Quaternion& operator*=(const tag_vec4<P>& other)
		{
			return (*this) *= Quaternion < P > {other};
		}


		//Performs comparison of two quaternions (possibly with different base types)
		template<typename P>
		bool operator==(const Quaternion<P>& other) const
		{
			return internal_representation[0] == other.internal_representation[0] &&
				internal_representation[1] == other.internal_representation[1] &&
				internal_representation[2] == other.internal_representation[2] &&
				internal_representation[3] == other.internal_representation[3];
		}


		//Returns quaternion's conjugate
		Quaternion<decltype(1.0f / T{ 1 })> conjugate() const
		{
			typedef decltype(1.0f / T{ 1 }) result_type;
			return Quaternion < result_type > { static_cast<result_type>(internal_representation[0]),
				-static_cast<result_type>(internal_representation[1]),
				-static_cast<result_type>(internal_representation[2]),
				-static_cast<result_type>(internal_representation[3])};
		}

		//Returns the norm of quaternion
		decltype(1.0f / T{ 1 }) norm() const
		{
			typedef decltype(1.0f / T{ 1 }) result_type;
			result_type w = static_cast<result_type>(internal_representation[0]);
			result_type x = static_cast<result_type>(internal_representation[1]);
			result_type y = static_cast<result_type>(internal_representation[2]);
			result_type z = static_cast<result_type>(internal_representation[3]);

			return std::sqrt(w*w + x*x + y*y + z*z);
		}

		//Returns quaternion's reciprocal
		Quaternion<decltype(1.0f / T{ 1 })> reciprocal() const
		{
			decltype(1.0f / T{ 1 }) quaternion_norm = norm();
			return conjugate() / (quaternion_norm * quaternion_norm);
		}


		//Implements division of two quaternions
		template<typename P>
		auto operator/(const Quaternion<P>& other) -> decltype(Quaternion < T > {}*other.reciprocal()) const
		{
			return (*this)*other.reciprocal();
		}

		//Divides the given quaternion by other quaternion and assigns the result to the left operand of the division
		template<typename P>
		Quaternion& operator/=(const Quaternion<P>& other)
		{
			return (*this) *= other.reciprocal();
		}


		//Divides quaternion by a 4D-vector
		template<typename P>
		Quaternion<decltype(T{ 1 } / P{ 1 })> operator/(const tag_vec4<P>& other) const
		{
			return (*this) / Quaternion < P > {other};
		}

		//Divides quaternion by a 4D-vector and assigns result of the division to back to the left operand (i.e. to the quaternion object)
		template<typename P>
		Quaternion& operator/=(const tag_vec4<P>& other)
		{
			return (*this) /= Quaternion < P > {other};
		}


		//Adds two quaternions
		template<typename P>
		Quaternion<decltype(T{}+P{})> operator+(const Quaternion<P>& other)
		{
			return Quaternion < decltype(T{}+P{}) > {internal_representation + other.internal_representation};
		}

		//Adds two quaternions and assigns the result to the left operand of addition
		template<typename P>
		Quaternion& operator+=(const Quaternion<P>& other)
		{
			internal_representation += other, internal_representation;
			return *this;
		}


		//Subtracts quaternions
		template<typename P>
		Quaternion<decltype(T{}-P{})> operator-(const Quaternion<P>& other)
		{
			return Quaternion < decltype(T{}+P{}) > {internal_representation - other.internal_representation};
		}

		//Subtracts quaternions and assigns the result of subtraction to the left operand
		template<typename P>
		Quaternion& operator-(const Quaternion<P>& other)
		{
			internal_representation -= other.internal_representation;
			return *this;
		}

		//Computes additive-inverse of the quaternion
		Quaternion<T>& operator-()
		{
			return Quaternion < T > {-internal_representation};
		}


		//Conversion between quaternion objects with different base types
		template<typename P>
		explicit operator Quaternion<P>() const
		{
			tag_vec4<P> converted_internal_representation = static_cast<tag_vec4<P>>(internal_representation);
			return Quaternion < P > {converted_internal_representation};
		}

		//Implements conversion of a quaternion into a 4D vector
		template<typename P>
		explicit operator tag_vec4<P>() const
		{
			return static_cast<tag_vec4<P>>(internal_representation);
		}


		//Returns rotation matrix corresponding to the quaternion
		glsl_matrix_type<T, 3, 3> getRotationMatrix() const
		{
			tag_vec4<T> aux = internal_representation.get_normalized();
			T w = aux[0];
			T x = aux[1];
			T y = aux[2];
			T z = aux[3];

			return glsl_matrix_type < T, 3, 3 > {1 - 2 * y*y - 2 * z*z, 2 * x*y + 2 * w*z, 2 * x*z - 2 * w*y,
				2 * x*y - 2 * w*z, 1 - 2 * x*x - 2 * z*z, 2 * y*z + 2 * w*x,
				2 * x*z + 2 * w*y, 2 * y*z - 2 * w*x, 1 - 2 * x*x - 2 * y*y};
		}

		//Converts quaternion to a 3-by-3 rotation matrix
		explicit operator glsl_matrix_type<T, 3, 3>() const
		{
			return getRotationMatrix();
		}
	};


	typedef Quaternion<float> quaternion;
	typedef Quaternion<double> dquaternion;

}

#define  TW__QUATERNION_TYPES__
#endif