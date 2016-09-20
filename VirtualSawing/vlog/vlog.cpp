#include "vlog.h"

#include <fstream>
#include <cstdint>
#include <algorithm>

template<typename T>
inline T extract_xinput_field(std::filebuf* xinput_data_streambuf, T* out_val)
{
	field<T> aux_data_buf;
	xinput_data_streambuf->sgetn(aux_data_buf, sizeof(T));
	*out_val = aux_data_buf;
	return aux_data_buf;
}


//Loads VLOG-file using the given path. Returns 'true' on success and 'false' on failure
bool load_vlog_data(std::string xinput_file_name, XRAY_DOMAIN* p_domain_info, std::vector<XRAY_SOURCE>* p_xray_sources, std::vector<XRAY_DETECTOR_PIXEL>* p_xray_detector_pixels)
{
	std::ifstream xinput_data_stream(xinput_file_name, std::ios_base::in | std::ios_base::binary);
	if (!xinput_data_stream) return false;

	std::filebuf* xinput_data_streambuf = xinput_data_stream.rdbuf();

	field<double> aux_double_data_buf;

	//Read domain parameters
	extract_xinput_field(xinput_data_streambuf, &p_domain_info->virtual_width);
	extract_xinput_field(xinput_data_streambuf, &p_domain_info->virtual_height);
	extract_xinput_field(xinput_data_streambuf, &p_domain_info->virtual_depth);

	extract_xinput_field(xinput_data_streambuf, &p_domain_info->active_width_fraction);
	extract_xinput_field(xinput_data_streambuf, &p_domain_info->active_height_fraction);
	extract_xinput_field(xinput_data_streambuf, &p_domain_info->active_depth_fraction);

	extract_xinput_field(xinput_data_streambuf, &p_domain_info->defined_width);
	extract_xinput_field(xinput_data_streambuf, &p_domain_info->defined_height);
	extract_xinput_field(xinput_data_streambuf, &p_domain_info->defined_depth);
	extract_xinput_field(xinput_data_streambuf, &p_domain_info->defined_cx);
	extract_xinput_field(xinput_data_streambuf, &p_domain_info->defined_cy);
	extract_xinput_field(xinput_data_streambuf, &p_domain_info->defined_cz);
	extract_xinput_field(xinput_data_streambuf, &p_domain_info->defined_vx);
	extract_xinput_field(xinput_data_streambuf, &p_domain_info->defined_vy);
	extract_xinput_field(xinput_data_streambuf, &p_domain_info->defined_vz);

	//Read defined domain data
	std::streamsize defined_data_size =
		static_cast<std::streamsize>(std::floor(p_domain_info->defined_width / p_domain_info->defined_vx + 0.5) *
		std::floor(p_domain_info->defined_height / p_domain_info->defined_vy + 0.5) *
		std::floor(p_domain_info->defined_depth / p_domain_info->defined_vz + 0.5));
	double *defined_domain_data_ptr = p_domain_info->allocate_defined_data_buf(defined_data_size);
	xinput_data_streambuf->sgetn(reinterpret_cast<char*>(defined_domain_data_ptr), defined_data_size * sizeof(double));


	//Read parameters of  X-Ray sources and detector pixels
	uint32_t xray_sources_num;
	uint32_t xray_detector_pixels_num;
	extract_xinput_field(xinput_data_streambuf, &xray_sources_num);
	extract_xinput_field(xinput_data_streambuf, &xray_detector_pixels_num);


	for (int i = 0; i < static_cast<int>(xray_sources_num); ++i)
	{
		XRAY_SOURCE xray_source;
		extract_xinput_field(xinput_data_streambuf, &xray_source.location_x);
		extract_xinput_field(xinput_data_streambuf, &xray_source.location_y);
		extract_xinput_field(xinput_data_streambuf, &xray_source.location_z);
		extract_xinput_field(xinput_data_streambuf, &xray_source.direction_x);
		extract_xinput_field(xinput_data_streambuf, &xray_source.direction_y);
		extract_xinput_field(xinput_data_streambuf, &xray_source.direction_z);
		extract_xinput_field(xinput_data_streambuf, &xray_source.span);
		extract_xinput_field(xinput_data_streambuf, &xray_source.intencity);

		p_xray_sources->push_back(xray_source);
	}

	for (int i = 0; i < static_cast<int>(xray_detector_pixels_num); ++i)
	{
		XRAY_DETECTOR_PIXEL xray_detector_pixel;
		extract_xinput_field(xinput_data_streambuf, &xray_detector_pixel.c_x);
		extract_xinput_field(xinput_data_streambuf, &xray_detector_pixel.c_y);
		extract_xinput_field(xinput_data_streambuf, &xray_detector_pixel.c_z);
		extract_xinput_field(xinput_data_streambuf, &xray_detector_pixel.r_x);
		extract_xinput_field(xinput_data_streambuf, &xray_detector_pixel.r_y);
		extract_xinput_field(xinput_data_streambuf, &xray_detector_pixel.r_z);
		extract_xinput_field(xinput_data_streambuf, &xray_detector_pixel.t_x);
		extract_xinput_field(xinput_data_streambuf, &xray_detector_pixel.t_y);
		extract_xinput_field(xinput_data_streambuf, &xray_detector_pixel.t_z);

		p_xray_detector_pixels->push_back(xray_detector_pixel);
	}
	return true;
}



//Defines new dimensions for the given virtual log 3D point density cloud and performs trilinear interpolation of the data onto the new discretization grid.
//This function can be used for both up-sampling and down-sampling of the source data. The function returns 'true' on success and 'false' on failure
bool upsample_vlog_data(const double* original_data, uint32_t original_width, uint32_t original_height, uint32_t original_depth,
	uint32_t target_width, uint32_t target_height, uint32_t target_depth, double* interpolated_data)
{
	//Check if pointers to the source and to the destination data are "correct" (i.e. non-zero deducible)
	if (!original_data || !interpolated_data) return false;

	//Check if target dimensions coincide with original ones
	if (target_width == original_width && target_height == original_height && target_depth == original_depth)
	{
		memcpy(interpolated_data, original_data, sizeof(double)*original_width*original_height*original_depth);
		return true;
	}


	//We assume that all the data lives in a unit cube for the reasons of convenience
	for (int i = 0; i < static_cast<int>(target_width); ++i)
		for (int j = 0; j < static_cast<int>(target_height); ++j)
			for (int k = 0; k < static_cast<int>(target_depth); ++k)
			{
				//Compute coordinates of the point currently being interpolated
				double x = static_cast<double>(i) / (target_width - 1.0);
				double y = static_cast<double>(j) / (target_height - 1.0);
				double z = static_cast<double>(k) / (target_depth - 1.0);

				//Next, interpolation is done on 3D integer grid
				x *= (original_width - 1.0);
				y *= (original_height - 1.0);
				z *= (original_depth - 1.0);

				//Locate vertices of the interpolation cube
				double xm1 = std::max(std::floor(x), 0.0), xp1 = std::min(std::ceil(x), original_width - 1.0);
				double ym1 = std::max(std::floor(y), 0.0), yp1 = std::min(std::ceil(y), original_height - 1.0);
				double zm1 = std::max(std::floor(z), 0.0), zp1 = std::min(std::ceil(z), original_depth - 1.0);

				//Compute distances from the point coordinates being interpolated to the corresponding lower boundaries
				double xd = x - xm1, yd = y - ym1, zd = z - zm1;

				//Perform linear interpolation along the x-axis
				double c00 = original_data[static_cast<uint32_t>(original_width * original_height * zm1 + original_height * ym1 + xm1)] * (1.0 - xd) +
					original_data[static_cast<uint32_t>(original_width * original_height * zm1 + original_height * ym1 + xp1)] * xd;
				double c10 = original_data[static_cast<uint32_t>(original_width * original_height * zm1 + original_height * yp1 + xm1)] * (1.0 - xd) +
					original_data[static_cast<uint32_t>(original_width * original_height * zm1 + original_height * yp1 + xp1)] * xd;
				double c01 = original_data[static_cast<uint32_t>(original_width * original_height * zp1 + original_height * ym1 + xm1)] * (1.0 - xd) +
					original_data[static_cast<uint32_t>(original_width * original_height * zp1 + original_height * ym1 + xp1)] * xd;
				double c11 = original_data[static_cast<uint32_t>(original_width * original_height * zp1 + original_height * yp1 + xm1)] * (1.0 - xd) +
					original_data[static_cast<uint32_t>(original_width * original_height * zp1 + original_height * yp1 + xp1)] * xd;

				//Perform linear interpolation along the y-axis
				double c0 = c00 * (1.0 - yd) + c10 * yd;
				double c1 = c01 * (1.0 - yd) + c11 * yd;

				//Perform linear interpolation along the z-axis and write the resulting value to the target data set
				interpolated_data[static_cast<uint32_t>(target_width*target_height*k + target_height*j + i)] = c0 * (1.0 - zd) + c1 * zd;

			}

	return true;
}