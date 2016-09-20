#ifndef VLOG_HEADER

#include <string>
#include <vector>
#include <cstdint>

//Describes X-Ray source
struct XRAY_SOURCE
{
	double location_x, location_y, location_z;		//location of x-ray source
	double direction_x, direction_y, direction_z;	//direction of x-ray source
	double span;		//emission span of the source
	double intencity;	//X-Ray source intencity
};


//Describes single pixel of an X-Ray detector (pixel is modeled by rectangular area)
struct XRAY_DETECTOR_PIXEL
{
	double c_x, c_y, c_z;	//position of the center of detector pixel
	double r_x, r_y, r_z;	//location of central point of the pixel's right edge
	double t_x, t_y, t_z;	//location of central point of the pixel's top edge
};


//Describes XINPUT-format
class XRAY_DOMAIN
{
private:
	size_t count;		//number of allocated elements of size sizeof(double)
	double* defined_data;		//defined domain data (full data block size is defined_width*defined_height*defined_depth*sizeof(float))

public:
	//Virtual domain is the domain where reconstruction and forward projections simulation take place.
	//Virtual domain is neither discretized, nor gets it stored in memory. Its only purpose is to define the virtual "playground" for the process.
	//The origin of the main coordinate system is always located in the center point of the virtual domain.
	double virtual_width;		//width of the virtual domain
	double virtual_height;	//height of the virtual domain
	double virtual_depth;		//depth of the virtual domain

	//Active domain is a cubical area with the center point aligned with the origin. Active domain is the part of virtual domain, which is divided into voxels.
	double active_width_fraction;			//the portion of width of virtual domain, which gets discretized during ray tracing
	double active_height_fraction;		//the portion of height of virtual domain, which gets discretized during ray tracing
	double active_depth_fraction;			//the portion of depth of virtual domain, which gets discretized during ray tracing
	//More precisely, the part of virtual domain that is discretized has each dimension defined as active_X_fraction*virtual_X, where X can be "width", "height" or "depth".
	//The center point of active domain is always aligned with the central point of the virtual domain

	//Defined domain is a part of virtual domain, for which data are actually defined and get loaded into memory
	double defined_width;		//defined domain width
	double defined_height;		//defined domain height
	double defined_depth;		//defined domain depth
	double defined_cx, defined_cy, defined_cz;			//coordinates of defined domain's center point represented in the main frame
	double defined_vx, defined_vy, defined_vz;			//size of defined domain's voxel along X-, Y- and Z- axes

	//uint32_t num_xray_sources;	//number of X-Ray sources in the installation
	//uint32_t num_xray_detector_pixels;		//number of X-Ray detector pixels in the installation

	//XRAY_SOURCE *xray_sources;	//Block of X-Ray source descriptors
	//XRAY_DETECTOR_PIXEL *xray_detector_pixels;	//Block of X-Ray detector pixel descriptors

	XRAY_DOMAIN() : virtual_width(0), virtual_height(0), virtual_depth(0),
		active_width_fraction(1.0), active_height_fraction(1.0), active_depth_fraction(1.0),
		defined_width(0), defined_height(0), defined_depth(0), defined_cx(0), defined_cy(0), defined_cz(0),
		defined_vx(1.0), defined_vy(1.0), defined_vz(1.0), count(0),
		defined_data(NULL) {}

	XRAY_DOMAIN(const XRAY_DOMAIN& other) : virtual_width(other.virtual_width), virtual_height(other.virtual_height), virtual_depth(other.virtual_depth),
		active_width_fraction(other.active_width_fraction), active_height_fraction(other.active_height_fraction), active_depth_fraction(other.active_depth_fraction),
		defined_width(other.defined_width), defined_height(other.defined_height), defined_depth(other.defined_depth), defined_cx(other.defined_cx), defined_cy(other.defined_cy), defined_cz(other.defined_cz),
		defined_vx(other.defined_vx), defined_vy(other.defined_vy), defined_vz(other.defined_vz), count(other.count), defined_data(NULL)
	{
		if (other.defined_data)
		{
			defined_data = new double[count];
			memcpy(defined_data, other.defined_data, count * sizeof(double));
		}
	}

	XRAY_DOMAIN& operator=(const XRAY_DOMAIN& other)
	{
		virtual_width = other.virtual_width;
		virtual_height = other.virtual_height;
		virtual_depth = other.virtual_depth;

		active_width_fraction = other.active_width_fraction;
		active_height_fraction = other.active_height_fraction;
		active_depth_fraction = other.active_depth_fraction;

		defined_width = other.defined_width;
		defined_height = other.defined_height;
		defined_depth = other.defined_depth;
		defined_cx = other.defined_cx;
		defined_cy = other.defined_cy;
		defined_cz = other.defined_cz;
		defined_vx = other.defined_vx;
		defined_vy = other.defined_vy;
		defined_vz = other.defined_vz;

		count = other.count;
		if (other.defined_data)
		{
			defined_data = new double[count];
			memcpy(defined_data, other.defined_data, count * sizeof(double));
		}
		else
			defined_data = NULL;

		return *this;
	}

	~XRAY_DOMAIN(){ if (defined_data) delete[] defined_data; }

	double* allocate_defined_data_buf(size_t count)
	{
		this->count = count;
		return (defined_data = new double[count]);
	}
	const double* get_defined_data_ptr() const { return defined_data; }
};


//This type is needed to represent integral data types in raw binary format when writing to or reading from a stream
template<typename T> union field{
	T val;
	char bytes[sizeof(T)];
	field(T val) : val(val) {}
	field() : val{} {}
	operator const char*() const { return bytes; }
	operator char*() { return bytes; }
	operator T() const { return val; }
};



//loads data stored in XINPUT binary format. Returns 'true' on success and 'false' on failure
bool load_vlog_data(std::string xinput_file_name, XRAY_DOMAIN* p_domain_info, std::vector<XRAY_SOURCE>* p_xray_sources, std::vector<XRAY_DETECTOR_PIXEL>* p_xray_detector_pixels);


//Defines new dimensions for the given virtual log 3D point density cloud and performs trilinear interpolation of the data onto the new discretization grid.
//This function can be used for both up-sampling and down-sampling of the source data. The function returns 'true' on success and 'false' on failure
bool upsample_vlog_data(const double* original_data, uint32_t original_width, uint32_t original_height, uint32_t original_depth,
	uint32_t target_width, uint32_t target_height, uint32_t target_depth, double* interpolated_data);

#define VLOG_HEADER
#endif
