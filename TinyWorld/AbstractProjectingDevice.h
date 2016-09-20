#ifndef TW__ABSTRACT_PROJECTING_DEVICE__

#include <string>
#include <stdint.h>
//#include <numeric>

#include "Misc.h"
#include "VectorTypes.h"
#include "MatrixTypes.h"
#include "Entity.h"

namespace tiny_world
{
	//This class implements virtual projecting device: an abstraction that defines projection of a 3D-world onto a 2D-rectangular area
	class AbstractProjectingDevice : public Entity{
	private:
		vec3 location;				//center of projection (location of projecting device in the world space)
		vec3 eye_direction;			//principal axis of projection described in world space coordinates (viewing direction)
		vec3 up_vector;				//up-vector of projecting device described in world space coordinates. This tells, where projecting device (e.g. virtual "camera") has its "up side" with respect to the world space. 
		//NOTE: the up-vector always gets automatically rotated within the plane defined by itself and by viewing direction so that it makes a right angle with the viewing direction 

		mat3 camera_transform;		//camera transformation matrix

		virtual void setup_view_transform();	//initializes view transform matrix using currently active settings

	protected:
		//The following values define boundaries of 2D rectangular area carved by projection volume out of projection (i.e. "focal") plane
		float left, right;			//left and right boundaries of 2D projection rectangular area defined in coordinate system attached to projecting device
		float bottom, top;			//bottom and top boundaries of 2D projection rectangular area defined in coordinate system attached to projecting device
		float near;					//distance from the center of projection to projection ("focal") plane
		float far;					//distance from the center of projection to the far cutting plane (anything beyond the far cutting plane gets culled by vertex fetching)
		mat4  projection_transform;	//projection transformation matrix

		virtual void setup_projection_transform() = 0;	//initializes projection transform. Should be overridden by derived class

		//Default initializer. Default projecting device is located in the origin of world space, its viewing direction is aligned with negative direction of the world space's z-axis and
		//its up-vector is aligned with positive direction of world space's y-axis. The default projection rectangle is defined by [-1,1]x[-1,1]. The points projected outside of this 
		//rectangle are getting clipped by vertex fetching stage of OpenGL. The default near cutting plane is 1.0f, default far cutting plane is 1000.0f
		explicit AbstractProjectingDevice(const std::string& projecting_device_class_string_name);


		//Initializer that creates default projecting device and allows to set a user-defined string name, which will be used as its weak identifier
		explicit AbstractProjectingDevice(const std::string& projecting_device_class_string_name, const std::string& projecting_device_string_name);


		//Simplified initialization. Projecting device gets located and oriented by default location and orientation settings. However, projection rectangle is defined by [left,right]x[bottom,top],
		//and near and far cutting planes are defined explicitly.
		AbstractProjectingDevice(const std::string& projecting_device_class_string_name, const std::string& projecting_device_string_name, 
			float left, float right, float bottom, float top, float near, float far);

		//Simplified initialization. Projection device gets default location and orientation, but projection rectangle is defined explicitly by [-width/2, width/2]x[-height/2, height/2].
		//The near and far cutting planes are given explicitly.
		AbstractProjectingDevice(const std::string& projecting_device_class_string_name, const std::string& projecting_device_string_name, 
			float width, float height, float near, float far);

		//Full initialization with projection rectangle defined by [-width/2, width/2]x[-height/2, height/2]
		AbstractProjectingDevice(const std::string& projecting_device_class_string_name, const std::string& projecting_device_string_name, 
			float width, float height, float near, float far,
			const vec3& location, const vec3& target, const vec3& up_vector);

		//Full user-defined initialization with all parameters given explicitly
		AbstractProjectingDevice(const std::string& projecting_device_class_string_name, const std::string& projecting_device_string_name, 
			float left, float right, float bottom, float top, float near, float far,
			const vec3& location, const vec3& target, const vec3& up_vector);

		//Copy constructor
		AbstractProjectingDevice(const AbstractProjectingDevice& other);

		//Move constructor
		AbstractProjectingDevice(AbstractProjectingDevice&& other);

	public:
		~AbstractProjectingDevice();
		AbstractProjectingDevice& operator=(const AbstractProjectingDevice& other);
		AbstractProjectingDevice& operator=(AbstractProjectingDevice&& other);

		//Getters
		vec3 getLocation() const;	//returns current location of projecting device represented in world space
		vec3 getViewDirection() const;	//returns viewing direction of projecting device (principal projection axis) represented in world space coordinates
		vec3 getUpVector() const;	//returns up-vector of projecting device represented in world space coordinates
		mat4 getViewTransform() const;	//returns currently active view transformation matrix of projecting device

		void getProjectionVolume(float* left, float* right, float* bottom, float* top, float* near, float* far) const;		//returns information defining projection volume	
		
		//The following are convenience functions to extract near and far clipping planes
		
		float getNearClipPlane() const;	//returns near clipping plane
		float getFarClipPlane() const;	//returns far clipping plane

		mat4 getProjectionTransform() const;	//returns currently active projection transform of projecting device

		//Setters
		void setLocation(const vec3& new_location);		//updates location of projecting device
		void setTarget(const vec3& target);		//updates viewing target of projecting device by setting new viewing direction
		void setProjectionVolume(float left, float right,
			float bottom, float top, float near, float far);	//updates projection volume of projecting device

		//Transforms (all the angles below are assumed to be given in radians)
		void rotateX(float angle, RotationFrame frame);				//rotates projecting device around X-axis for a given angle
		void rotateY(float angle, RotationFrame frame);				//rotates projecting device around Y-axis for a given angle
		void rotateZ(float angle, RotationFrame frame);				//rotates projecting device around Z-axis for a given angle
		void rotate(const vec3& axis, float angle, RotationFrame frame);		//rotates projecting device around an arbitrary axis represented by vector "axis" for a given angle 
		void translate(const vec3& translation);		//moves projecting device by adding vector "translation" to the current location of the projecting device
		void mirrorXY();	//mirrors projection along XY-plane of projection device
		void mirrorYZ();	//mirrors projection along YZ-plane of projection device
		void mirrorXZ();	//mirrors projection along XZ-plane of projection device
	};




	//Implements viewing frustum
	class PerspectiveProjectingDevice final: public AbstractProjectingDevice
	{
	protected:
		virtual void setup_projection_transform() override;

	public:
		//NOTE: perspective projection can be defined so that the far cutting plane is located infinitely far from location of projecting device
		const float value_of_infinity = std::numeric_limits<float>::infinity();

		//Default initializer
		PerspectiveProjectingDevice();

		//Default initializer that allows to set a user-defined string name, which will be used as a weak identifier for the perspective projecting device
		explicit PerspectiveProjectingDevice(const std::string& perspective_projecting_device_string_name);

		//Simplified initialization. Creates projecting device with default location and orientation and with user-defined settings of viewing frustum.
		PerspectiveProjectingDevice(const std::string& perspective_projecting_device_string_name, 
			float left, float right, float bottom, float top, float near, float far);

		//Simplified initialization. Uses default settings for location and orientation. Projection screen is defined by [-width/2, width/2]x[-height/2, height/2]
		PerspectiveProjectingDevice(const std::string& perspective_projecting_device_string_name, 
			float width, float height, float near, float far);

		//Full initialization with projection screen defined by [-width/2, width/2]x[-height/2, height/2]
		PerspectiveProjectingDevice(const std::string& perspective_projecting_device_string_name,
			float width, float height, float near, float far,
			const vec3& location, const vec3& target, const vec3& up_vector);

		//Full custom initialization
		PerspectiveProjectingDevice(const std::string& perspective_projecting_device_string_name,
			float left, float right, float bottom, float top, float near, float far,
			const vec3& location, const vec3& target, const vec3& up_vector);

		PerspectiveProjectingDevice& operator=(const PerspectiveProjectingDevice& other);	//copy-assignment operator (must be redefined due to the presence of constant field)
		PerspectiveProjectingDevice& operator=(PerspectiveProjectingDevice&& other);	//move-assignment operator (must be redefined due to the presence of constant filed)
	};



	//Implements viewing cube (orthogonal projection)
	class OrthogonalProjectingDevice final: public AbstractProjectingDevice
	{
	protected:
		virtual void setup_projection_transform() override;

	public:
		
		//Default initializer
		OrthogonalProjectingDevice();

		//Initializer that allows to create default orthogonal projecting device and attach a user-defined string name, 
		//which will be used as a weak identifier for the newly created device
		explicit OrthogonalProjectingDevice(const std::string& orthogonal_projecting_device);

		//Simplified initialization. Creates projecting device with default location and orientation and with user-defined settings of viewing frustum.
		OrthogonalProjectingDevice(const std::string& orthogonal_projecting_device,
			float left, float right, float bottom, float top, float near, float far);

		//Simplified initialization. Uses default settings for location and orientation. Projection screen is defined by [-width/2, width/2]x[-height/2, height/2]
		OrthogonalProjectingDevice(const std::string& orthogonal_projecting_device,
			float width, float height, float near, float far);

		//Full initialization with projection screen defined by [-width/2, width/2]x[-height/2, height/2]
		OrthogonalProjectingDevice(const std::string& orthogonal_projecting_device,
			float width, float height, float near, float far,
			const vec3& location, const vec3& target, const vec3& up_vector);

		//Full custom initialization
		OrthogonalProjectingDevice(const std::string& orthogonal_projecting_device,
			float left, float right, float bottom, float top, float near, float far,
			const vec3& location, const vec3& target, const vec3& up_vector);
	};
}


#define TW__ABSTRACT_PROJECTING_DEVICE__
#endif