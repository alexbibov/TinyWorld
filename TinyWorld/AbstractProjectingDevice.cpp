#include "AbstractProjectingDevice.h"
#include <cmath>

using namespace tiny_world;



void AbstractProjectingDevice::setup_view_transform()
{
	//Modify up-vector so that it becomes orthogonal to the viewing direction
	float alpha = -up_vector.dot_product(eye_direction) / eye_direction.dot_product(eye_direction);
	up_vector = up_vector + eye_direction*alpha;

	//Normalize up-vector
	up_vector = up_vector.get_normalized();

	//Compute normalized side-vector of the camera
	vec3 side_vector = eye_direction.cross_product(up_vector);

	//Define camera-view transformation matrix
	camera_transform = mat3{ side_vector, up_vector, -eye_direction }.transpose();
}



AbstractProjectingDevice::AbstractProjectingDevice(const std::string& projecting_device_class_string_name) : 
Entity{ projecting_device_class_string_name },
location{ 0 }, eye_direction{ 0, 0, -1 }, up_vector{ 0, 1, 0 },
left{ -1.0f }, right{ 1.0f }, bottom{ -1.0f }, top{ 1.0f }, near{ 1.0f }, far{ 1000.0f }
{
	setup_view_transform();
}


AbstractProjectingDevice::AbstractProjectingDevice(const std::string& projecting_device_class_string_name, const std::string& projecting_device_string_name) :
Entity{ projecting_device_class_string_name, projecting_device_string_name },
location{ 0 }, eye_direction{ 0, 0, -1 }, up_vector{ 0, 1, 0 },
left{ -1.0f }, right{ 1.0f }, bottom{ -1.0f }, top{ 1.0f }, near{ 1.0f }, far{ 1000.0f }
{
	setup_view_transform();
}

AbstractProjectingDevice::AbstractProjectingDevice(const std::string& projecting_device_class_string_name, const std::string& projecting_device_string_name,
	float left, float right, float bottom, float top, float near, float far) :
	Entity{ projecting_device_class_string_name, projecting_device_string_name }, 
	location{ 0 }, eye_direction{ 0, 0, -1 }, up_vector{ 0, 1, 0 },
	left{ left }, right{ right }, bottom{ bottom }, top{ top }, near{ near }, far{ far }
{
	setup_view_transform();
}


AbstractProjectingDevice::AbstractProjectingDevice(const std::string& projecting_device_class_string_name, const std::string& projecting_device_string_name,
	float width, float height, float near, float far) :
	Entity{ projecting_device_class_string_name, projecting_device_string_name },
	location{ 0 }, eye_direction{ 0, 0, -1 }, up_vector{ 0, 1, 0 },
	left{ -width / 2 }, right{ width / 2 }, bottom{ -height / 2 }, top{ height / 2 },
	near{ near }, far{ far }
{
	setup_view_transform();
}


AbstractProjectingDevice::AbstractProjectingDevice(const std::string& projecting_device_class_string_name, const std::string& projecting_device_string_name,
	float width, float height, float near, float far,
	const vec3& location, const vec3& target, const vec3& up_vector) :
	Entity{ projecting_device_class_string_name, projecting_device_string_name },
	location{ location }, eye_direction{ (target - location).get_normalized() }, up_vector{ up_vector },
	left{ -width / 2 }, right{ width / 2 }, bottom{ -height / 2 }, top{ height / 2 },
	near{ near }, far{ far }
{
	setup_view_transform();
}


AbstractProjectingDevice::AbstractProjectingDevice(const std::string& projecting_device_class_string_name, const std::string& projecting_device_string_name,
	float left, float right, float bottom, float top,
	float near, float far, const vec3& location, const vec3& target, const vec3& up_vector) :
	Entity{ projecting_device_class_string_name, projecting_device_string_name },
	location{ location }, eye_direction{ (target - location).get_normalized() }, up_vector{ up_vector },
	left{ left }, right{ right }, bottom{ bottom }, top{ top }, near{ near }, far{ far }
{
	setup_view_transform();
}

AbstractProjectingDevice::AbstractProjectingDevice(const AbstractProjectingDevice& other) :
Entity{ other },
location{ other.location }, eye_direction{ other.eye_direction }, up_vector{ other.up_vector },
camera_transform{ other.camera_transform }, left{ other.left }, right{ other.right }, bottom{ other.bottom }, top{ other.top }, near{ other.near }, far{ other.far },
projection_transform{ other.projection_transform }
{
	setup_view_transform();
}

AbstractProjectingDevice::AbstractProjectingDevice(AbstractProjectingDevice&& other) :
Entity{ std::move(other) }, 
location{ std::move(other.location) }, eye_direction{ std::move(other.eye_direction) }, up_vector{ std::move(other.up_vector) },
camera_transform{ std::move(other.camera_transform) }, left{ other.left }, right{ other.right }, bottom{ other.bottom }, top{ other.top },
near{ other.near }, far{ other.far }, projection_transform{ std::move(other.projection_transform) }
{

}

AbstractProjectingDevice::~AbstractProjectingDevice() {}

AbstractProjectingDevice& AbstractProjectingDevice::operator=(const AbstractProjectingDevice& other)
{
	//Account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	Entity::operator=(other);

	//Copy state variables
	location = other.location;
	eye_direction = other.eye_direction;
	up_vector = other.up_vector;
	camera_transform = other.camera_transform;

	left = other.left;
	right = other.right;
	bottom = other.bottom;
	top = other.top;
	near = other.near;
	far = other.far;
	projection_transform = other.projection_transform;

	return *this;
}

AbstractProjectingDevice& AbstractProjectingDevice::operator=(AbstractProjectingDevice&& other)
{
	//Account for the special case of assignment to itself
	if (this == &other)
		return *this;

	Entity::operator=(std::move(other));

	//Move state variables
	location = std::move(other.location);
	eye_direction = std::move(other.eye_direction);
	up_vector = std::move(other.up_vector);
	camera_transform = std::move(other.camera_transform);

	left = other.left;
	right = other.right;
	bottom = other.bottom;
	top = other.top;
	near = other.near;
	far = other.far;
	projection_transform = std::move(other.projection_transform);

	return *this;
}

vec3 AbstractProjectingDevice::getLocation() const { return location; }

vec3 AbstractProjectingDevice::getViewDirection() const { return eye_direction; }

vec3 AbstractProjectingDevice::getUpVector() const { return up_vector; }

mat4 AbstractProjectingDevice::getViewTransform() const 
{ 
	const mat3 &C = camera_transform;
	vec3 transformed_location = C * location;

	mat4 V{ C[0][0], C[1][0], C[2][0], 0,
		C[0][1], C[1][1], C[2][1], 0,
		C[0][2], C[1][2], C[2][2], 0,
		-transformed_location[0], -transformed_location[1], -transformed_location[2], 1 };
	return V;
}

void AbstractProjectingDevice::getProjectionVolume(float* left, float* right, float* bottom, float* top,
	float* near, float* far) const
{
	*left = this->left;
	*right = this->right;
	*bottom = this->bottom;
	*top = this->top;
	*near = this->near;
	*far = this->far;
}

float AbstractProjectingDevice::getNearClipPlane() const { return near; }

float AbstractProjectingDevice::getFarClipPlane() const { return far; }

mat4 AbstractProjectingDevice::getProjectionTransform() const { return projection_transform; }

void AbstractProjectingDevice::setLocation(const vec3& new_location){ location = new_location; }

void AbstractProjectingDevice::setTarget(const vec3& target)
{
	vec3 new_eye_direction = camera_transform*(target - location).get_normalized();
	vec3 norm_eye_direction = camera_transform*eye_direction.get_normalized();
	float angle = std::acos(new_eye_direction.dot_product(norm_eye_direction));
	vec3 rot_axis = angle != 0 ? norm_eye_direction.cross_product(new_eye_direction) : vec3(1.0f, 0.0f, 0.0f);

	if (rot_axis.norm() < 0.5f) rot_axis = vec3(0.0f, 1.0f, 0.0f);

	rotate(rot_axis, angle, RotationFrame::LOCAL);
}

void AbstractProjectingDevice::setProjectionVolume(float left, float right, float bottom, float top,
	float near, float far)
{
	this->left = left;
	this->right = right;
	this->bottom = bottom;
	this->top = top;
	this->near = near;
	this->far = far;
	setup_projection_transform();
}

void AbstractProjectingDevice::rotateX(float angle, RotationFrame frame)
{
	float c = std::cos(angle);
	float s = std::sin(angle);

	mat3 rx_transform{ 1, 0, 0,
		0, c, -s,
		0, s, c };

	switch (frame)
	{
	case RotationFrame::LOCAL:
		camera_transform = rx_transform * camera_transform;
		break;
	case RotationFrame::GLOBAL:
		camera_transform = camera_transform * rx_transform.transpose();
		break;
	}

	up_vector = vec3{ camera_transform[1][0], camera_transform[1][1], camera_transform[1][2] };
	eye_direction = -vec3{ camera_transform[2][0], camera_transform[2][1], camera_transform[2][2] };
}

void AbstractProjectingDevice::rotateY(float angle, RotationFrame frame)
{
	float c = std::cos(angle);
	float s = std::sin(angle);

	mat3 ry_transform{ c, 0, s,
		0, 1, 0,
		-s, 0, c };

	switch (frame)
	{
	case RotationFrame::LOCAL:
		camera_transform = ry_transform * camera_transform;
		break;
	case RotationFrame::GLOBAL:
		camera_transform = camera_transform * ry_transform.transpose();
		break;
	}

	up_vector = vec3{ camera_transform[1][0], camera_transform[1][1], camera_transform[1][2] };
	eye_direction = -vec3{ camera_transform[2][0], camera_transform[2][1], camera_transform[2][2] };
}

void AbstractProjectingDevice::rotateZ(float angle, RotationFrame frame)
{
	float c = std::cos(angle);
	float s = std::sin(angle);
	
	mat3 rz_transform{ c, -s, 0,
		s, c, 0,
		0, 0, 1 };

	switch (frame)
	{
	case RotationFrame::LOCAL:
		camera_transform = rz_transform * camera_transform;
		break;
	case RotationFrame::GLOBAL:
		camera_transform = camera_transform * rz_transform.transpose();
		break;
	}

	up_vector = vec3{ camera_transform[1][0], camera_transform[1][1], camera_transform[1][2] };
	eye_direction = -vec3{ camera_transform[2][0], camera_transform[2][1], camera_transform[2][2] };
}

void AbstractProjectingDevice::rotate(const vec3& axis, float angle, RotationFrame frame)
{
	float c = std::cos(angle);
	float s = std::sin(angle);
	vec3 naxis = axis.get_normalized();		//normalize input axis

	mat3 ra_transform{ c + (1 - c)*naxis.x*naxis.x, (1 - c)*naxis.x*naxis.y - s*naxis.z, (1 - c)*naxis.x*naxis.z + s*naxis.y,
		(1 - c)*naxis.x*naxis.y + s*naxis.z, c + (1 - c)*naxis.y*naxis.y, (1 - c)*naxis.y*naxis.z - s*naxis.x,
		(1 - c)*naxis.x*naxis.z - s*naxis.y, (1 - c)*naxis.y*naxis.z + s*naxis.x, c + (1 - c)*naxis.z*naxis.z };

	switch (frame)
	{
	case RotationFrame::LOCAL:
		camera_transform = ra_transform * camera_transform;
		break;
	case RotationFrame::GLOBAL:
		camera_transform = camera_transform * ra_transform.transpose();
		break;
	}

	up_vector = vec3{ camera_transform[1][0], camera_transform[1][1], camera_transform[1][2] };
	eye_direction = -vec3{ camera_transform[2][0], camera_transform[2][1], camera_transform[2][2] };
}

void AbstractProjectingDevice::translate(const vec3& translation)
{ 
	location = location + camera_transform.transpose() * translation; 
}

void AbstractProjectingDevice::mirrorXY()
{
	camera_transform[0][0] *= -1; camera_transform[0][1] *= -1; camera_transform[0][2] *= -1;
	camera_transform[1][0] *= -1; camera_transform[1][1] *= -1; camera_transform[1][2] *= -1;
	up_vector *= -1;
}

void AbstractProjectingDevice::mirrorYZ()
{
	camera_transform[1][0] *= -1; camera_transform[1][1] *= -1; camera_transform[1][2] *= -1;
	camera_transform[2][0] *= -1; camera_transform[2][1] *= -1; camera_transform[2][2] *= -1;
	up_vector *= -1;
	eye_direction *= -1;
}

void AbstractProjectingDevice::mirrorXZ()
{
	camera_transform[0][0] *= -1; camera_transform[0][1] *= -1; camera_transform[0][2] *= -1;
	camera_transform[2][0] *= -1; camera_transform[2][1] *= -1; camera_transform[2][2] *= -1;
	eye_direction *= -1;
}




void PerspectiveProjectingDevice::setup_projection_transform()
{
	if (std::abs(far) == value_of_infinity)	//case of infinitely distant far cutting plane
		projection_transform = mat4{ vec4{ 2 * near / (right - left), 0, 0, 0 },
		vec4{ 0, 2 * near / (top - bottom), 0, 0 },
		vec4{ (right + left) / (right - left), (top + bottom) / (top - bottom), -1, -1 },
		vec4{ 0, 0, -2 * near, 0 } };
	else	//case of the usual viewing frustum
		projection_transform = mat4{ vec4{ 2 * near / (right - left), 0, 0, 0 },
		vec4{ 0, 2 * near / (top - bottom), 0, 0 },
		vec4{ (right + left) / (right - left), (top + bottom) / (top - bottom), -(far + near) / (far - near), -1 },
		vec4{ 0, 0, -2 * near*far / (far - near), 0 } };
}

PerspectiveProjectingDevice::PerspectiveProjectingDevice() : AbstractProjectingDevice("PerspectiveProjectingDevice")
{
	setup_projection_transform();
}

PerspectiveProjectingDevice::PerspectiveProjectingDevice(const std::string& perspective_projecting_device_string_name) :
AbstractProjectingDevice("PerspectiveProjectingDevice", perspective_projecting_device_string_name)
{
	setup_projection_transform();
}

PerspectiveProjectingDevice::PerspectiveProjectingDevice(const std::string& perspective_projecting_device_string_name,
	float left, float right, float bottom, float top, float near, float far) :
	AbstractProjectingDevice("PerspectiveProjectingDevice", perspective_projecting_device_string_name, left, right, bottom, top, near, far)
{
	setup_projection_transform();
}

PerspectiveProjectingDevice::PerspectiveProjectingDevice(const std::string& perspective_projecting_device_string_name,
	float width, float height, float near, float far) :
	AbstractProjectingDevice("PerspectiveProjectingDevice", perspective_projecting_device_string_name, width, height, near, far)
{
	setup_projection_transform();
}

PerspectiveProjectingDevice::PerspectiveProjectingDevice(const std::string& perspective_projecting_device_string_name,
	float width, float height, float near, float far,
	const vec3& location, const vec3& target, const vec3& up_vector) :
	AbstractProjectingDevice("PerspectiveProjectingDevice", perspective_projecting_device_string_name, width, height, near, far, location, target, up_vector)
{
	setup_projection_transform();
}

PerspectiveProjectingDevice::PerspectiveProjectingDevice(const std::string& perspective_projecting_device_string_name,
	float left, float right, float bottom, float top, float near, float far,
	const vec3& location, const vec3& target, const vec3& up_vector) :
	AbstractProjectingDevice("PerspectiveProjectingDevice", perspective_projecting_device_string_name, left, right, bottom, top, near, far, location, target, up_vector)
{
	setup_projection_transform();
}

PerspectiveProjectingDevice& PerspectiveProjectingDevice::operator=(const PerspectiveProjectingDevice& other)
{
	//Account for the possibility of "assignment to itself"
	if (this == &other)
		return *this;

	AbstractProjectingDevice::operator=(other);

	return *this;
}

PerspectiveProjectingDevice& PerspectiveProjectingDevice::operator=(PerspectiveProjectingDevice&& other)
{
	//Account for the possibility of "assignment to itself"
	if (this == &other)
		return *this;

	AbstractProjectingDevice::operator=(std::move(other));

	return *this;
}




void OrthogonalProjectingDevice::setup_projection_transform()
{
	projection_transform = mat4{ vec4{ 2 / (right - left), 0, 0, 0 },
		vec4{ 0, 2 / (top - bottom), 0, 0 },
		vec4{ 0, 0, -2 / (far - near), 0 },
		vec4{ -(right + left) / (right - left), -(top + bottom) / (top - bottom), -(far + near) / (far - near), 1 } };
}

OrthogonalProjectingDevice::OrthogonalProjectingDevice() :
AbstractProjectingDevice("OrthogonalProjectingDevice")
{
	setup_projection_transform();
}

OrthogonalProjectingDevice::OrthogonalProjectingDevice(const std::string& orthogonal_projecting_device) :
AbstractProjectingDevice("OrthogonalProjectingDevice", orthogonal_projecting_device)
{
	setup_projection_transform();
}

OrthogonalProjectingDevice::OrthogonalProjectingDevice(const std::string& orthogonal_projecting_device,
	float left, float right, float bottom, float top, float near, float far) :
	AbstractProjectingDevice("OrthogonalProjectingDevice", orthogonal_projecting_device, left, right, bottom, top, near, far)
{
	setup_projection_transform();
}

OrthogonalProjectingDevice::OrthogonalProjectingDevice(const std::string& orthogonal_projecting_device,
	float width, float height, float near, float far) :
	AbstractProjectingDevice("OrthogonalProjectingDevice", orthogonal_projecting_device, width, height, near, far)
{
	setup_projection_transform();
}

OrthogonalProjectingDevice::OrthogonalProjectingDevice(const std::string& orthogonal_projecting_device, 
	float width, float height, float near, float far,
	const vec3& location, const vec3& target, const vec3& up_vector) :
	AbstractProjectingDevice("OrthogonalProjectingDevice", orthogonal_projecting_device, width, height, near, far, location, target, up_vector)
{
	setup_projection_transform();
}

OrthogonalProjectingDevice::OrthogonalProjectingDevice(const std::string& orthogonal_projecting_device, 
	float left, float right, float bottom, float top, float near, float far,
	const vec3& location, const vec3& target, const vec3& up_vector) :
	AbstractProjectingDevice("OrthogonalProjectingDevice", orthogonal_projecting_device, left, right, bottom, top, near, far, location, target, up_vector)
{
	setup_projection_transform();
}