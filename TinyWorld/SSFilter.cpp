#include "SSFilter.h"

#include <set>

using namespace tiny_world;




AbstractSSFilter::AbstractSSFilter(const std::string& ssfilter_class_name) : Entity(ssfilter_class_name) {}




CascadeFilterLevel::CascadeFilterLevel() : Entity("CascadeFilterLevel"),
is_initialized{ false }, output_color_width{ ColorWidth::_16bit }
{
	setOutputResolution(uvec2{ 1024U, 768U });
	setOutputColorComponents(true, true, true, true);
}

CascadeFilterLevel::CascadeFilterLevel(const uvec2 output_resolution,
	bool has_red_channel /* = true */, bool has_green_channel /* = true */, bool has_blue_channel /* = true */, bool has_alpha_channel /* = true */,
	ColorWidth output_color_width /* = color_width::_16bit */) : Entity("CascadeFilterLevel"),
	is_initialized{ false }, output_color_width{ output_color_width }
{
	setOutputResolution(output_resolution);
	setOutputColorComponents(has_red_channel, has_green_channel, has_blue_channel, has_alpha_channel);
}

void CascadeFilterLevel::setOutputResolution(const uvec2& output_resolution)
{
	uv2OutputResolution = output_resolution;
	auxiliary_framebuffer.defineViewport(Rectangle{ 0.0f, 0.0f, static_cast<float>(uv2OutputResolution.x), static_cast<float>(uv2OutputResolution.y) });

	is_initialized = false;
}

uvec2 CascadeFilterLevel::getOutputResolution() const { return uv2OutputResolution; }

void CascadeFilterLevel::setOutputColorComponents(bool has_red_channel, bool has_green_channel, bool has_blue_channel, bool has_alpha_channel)
{
	if (has_red_channel) output_color_bits = 1;
	if (has_green_channel) output_color_bits = 3;
	if (has_blue_channel) output_color_bits = 7;
	if (has_alpha_channel) output_color_bits = 0xF;

	auxiliary_framebuffer.setColorBufferMask(has_red_channel, has_green_channel, has_blue_channel, has_alpha_channel);

	is_initialized = false;
}

bvec4 CascadeFilterLevel::getOutputColorComponents() const
{
	bvec4 rv;
	rv.x = output_color_bits.test(0);
	rv.y = output_color_bits.test(1);
	rv.z = output_color_bits.test(2);
	rv.w = output_color_bits.test(3);

	return rv;
}

void CascadeFilterLevel::setOutputColorBitWidth(ColorWidth out_color_width)
{
	output_color_width = out_color_width;
	is_initialized = false;
}

CascadeFilterLevel::ColorWidth CascadeFilterLevel::getOutputColorBitWidth() const
{
	return output_color_width;
}


void CascadeFilterLevel::addFilter(AbstractSSFilter* p_filter)
{
	level_filter_list.push_back(p_filter);
	is_initialized = false;
}

void CascadeFilterLevel::removeFirstFilter()
{
	level_filter_list.pop_front();
	is_initialized = false;
}

void CascadeFilterLevel::removeLastFilter()
{
	level_filter_list.pop_back();
	is_initialized = false;
}

AbstractSSFilter* CascadeFilterLevel::retrieveFilter(uint32_t index)
{
	if (index >= level_filter_list.size())
	{
		set_error_state(true);
		std::string err_msg = "Unable to retrieve filter from cascade filter level: index " + std::to_string(index) + " is out of the allowed range (0, " +
			std::to_string(level_filter_list.size()) + ")";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return nullptr;
	}

	uint32_t i = 0;
	std::list<AbstractSSFilter*>::iterator p_filter_iterator = level_filter_list.begin();
	while (i < index)
	{
		++i;
		++p_filter_iterator;
	}

	return *p_filter_iterator;
}

const AbstractSSFilter* CascadeFilterLevel::retrieveFilter(uint32_t index) const
{
	return const_cast<CascadeFilterLevel*>(this)->retrieveFilter(index);
}

uint32_t CascadeFilterLevel::getNumberOfFilters() const { return static_cast<uint32_t>(level_filter_list.size()); }



void CascadeFilterLevel::initialize()
{
	if (is_initialized) return;


	//Initialize all filters attached to the cascade level
	{
		AbstractSSFilter* p_current_filter;
		std::list<AbstractSSFilter*>::iterator p_current_filter_iterator;
		int filter_index = 0;
		for (p_current_filter_iterator = level_filter_list.begin(), p_current_filter = level_filter_list.front(), filter_index = 0;
			p_current_filter_iterator != level_filter_list.end(); ++p_current_filter_iterator, ++filter_index)
		{
			p_current_filter = *p_current_filter_iterator;

			p_current_filter->initialize();
			if (p_current_filter->getErrorState())
			{
				set_error_state(true);
				std::string err_msg = "Error when initializing filter #" + std::to_string(filter_index) + ":" + p_current_filter->getErrorString();
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return;
			}
		}
	}


	//Determine, what storage format shell be used for cascade output textures
	InternalPixelFormat output_storage_format;
	switch (output_color_bits.to_ulong())
	{
	case 1:
		output_storage_format = output_color_width == tag_color_width::_16bit ?
			InternalPixelFormat::SIZED_FLOAT_R16 : InternalPixelFormat::SIZED_FLOAT_R32;
		break;

	case 3:
		output_storage_format = output_color_width == tag_color_width::_16bit ?
			InternalPixelFormat::SIZED_FLOAT_RG16 : InternalPixelFormat::SIZED_FLOAT_RG32;
		break;

	case 7:
		output_storage_format = output_color_width == tag_color_width::_16bit ?
			InternalPixelFormat::SIZED_FLOAT_RGB16 : InternalPixelFormat::SIZED_FLOAT_RGB32;
		break;

	case 15:
		output_storage_format = output_color_width == tag_color_width::_16bit ?
			InternalPixelFormat::SIZED_FLOAT_RGBA16 : InternalPixelFormat::SIZED_FLOAT_RGBA32;
		break;
	}


	//Create output textures
	output_textures.clear();	//first, ensure that all of the previously created textures are destroyed
	for (unsigned int i = 0; i < level_filter_list.size(); ++i)
	{
		ImmutableTexture2D _2d_texture{ "CascadeFilterLevel::output_texture#" + std::to_string(i) };
		_2d_texture.allocateStorage(1, 1, TextureSize{ uv2OutputResolution.x, uv2OutputResolution.y, 0 }, output_storage_format);
		output_textures.push_back(_2d_texture);
	}

	auxiliary_framebuffer.applyOpenGLContextSettings();

	is_initialized = true;
}


bool CascadeFilterLevel::isInitialized() const
{
	return is_initialized;
}


bool CascadeFilterLevel::pass(const AbstractProjectingDevice& projecting_device)
{
	if (!is_initialized)
	{
		set_error_state(true);
		std::string err_msg = "Unable to execute cascade filter level. The level has not been initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return false;
	}

	//Perform rendering for each filter in this cascade level
	{
		AbstractSSFilter* p_filter;
		std::list<AbstractSSFilter*>::iterator filter_iterator;
		std::list<ImmutableTexture2D>::iterator out_texture_iterator;
		int filter_index;
		for (filter_iterator = level_filter_list.begin(), p_filter = level_filter_list.front(), filter_index = 0, out_texture_iterator = output_textures.begin();
			filter_iterator != level_filter_list.end();
			++filter_iterator, ++filter_index, ++out_texture_iterator)
		{
			p_filter = *filter_iterator;

			auxiliary_framebuffer.attachTexture(FramebufferAttachmentPoint::COLOR0, FramebufferAttachmentInfo{ 0, 0, &(*out_texture_iterator) });
			bool rv = p_filter->pass(projecting_device, auxiliary_framebuffer);

			//Check if filter has worked without errors
			if (!rv)
			{
				set_error_state(true);
				std::string err_msg = "Unable to run filter#" + std::to_string(filter_index) + " in cascade filter level: " + p_filter->getErrorString();
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return false;
			}
		}
	}

	return true;
}


ImmutableTexture2D CascadeFilterLevel::getTextureOutput(uint32_t index)
{
	if (index >= output_textures.size())
	{
		set_error_state(true);
		std::string err_msg = "Unable to retrieve output texture from cascade filter level: index " + std::to_string(index) + " is out of the allowed range (0, " +
			std::to_string(output_textures.size()) + ")";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return ImmutableTexture2D{};
	}

	uint32_t i = 0;
	std::list<ImmutableTexture2D>::iterator output_textures_iterator = output_textures.begin();
	while (i < index)
	{
		++output_textures_iterator;
		++i;
	}

	return *output_textures_iterator;
}




CascadeFilter::tag_DataFlow::tag_DataFlow() : input_channel_index{ 0 }, output_channel_level{ 0 }, output_channel_index{ 0 }
{

}


CascadeFilter::tag_DataFlow::tag_DataFlow(uint32_t in_channel, uint32_t out_level, uint32_t out_channel) :
input_channel_index{ in_channel }, output_channel_level{ out_level }, output_channel_index{ out_channel }
{

}


void CascadeFilter::setTextureSource(uint32_t index, const ImmutableTexture2D& _2dtexture)
{
	set_error_state(true);
	const char* err_msg = "Attempt to access texture source of a cascade filter: cascade filters do not have own texture sources";
	set_error_string(err_msg);
	call_error_callback(err_msg);
}

ImmutableTexture2D CascadeFilter::getTextureSource(uint32_t index) const
{
	set_error_state(true);
	const char* err_msg = "Attempt to access texture source of a cascade filter: cascade filters do not have own texture sources";
	set_error_string(err_msg);
	call_error_callback(err_msg);

	return ImmutableTexture2D{};
}


CascadeFilter::CascadeFilter() : AbstractSSFilter("CascadeFilter"),
levels_use_common_output_reolution{ false },
uv2CommonLevelOutputResolution{ 1024U, 768U }, is_initialized{ false }
{

}


CascadeFilter::CascadeFilter(const CascadeFilterLevel& base_cascade_filter) :
AbstractSSFilter("CascadeFilter"),
levels_use_common_output_reolution{ false },
uv2CommonLevelOutputResolution{ 1024U, 768U }, is_initialized{ false }
{
	levels.push_back(base_cascade_filter);
}


void CascadeFilter::addBaseLevel(const CascadeFilterLevel& base_filter_level)
{
	if (!levels.size()) levels.push_back(base_filter_level);
}


void CascadeFilter::addLevel(const CascadeFilterLevel& filter_level, const std::list<DataFlow>& data_redirection_descriptor)
{
	if (!levels.size()) return;

	levels.push_back(filter_level);
	level_data_flow_redirection_tables.push_back(data_redirection_descriptor);
	is_initialized = false;
}

void CascadeFilter::useCommonOutputResolution(bool flag)
{
	levels_use_common_output_reolution = flag;
	is_initialized = false;
}

bool CascadeFilter::isCommonOutputResolutionInUse() const { return levels_use_common_output_reolution; }

void CascadeFilter::setCommonOutputResolutionValue(const uvec2& resolution)
{
	uv2CommonLevelOutputResolution = resolution;
	is_initialized = false;
}

uvec2 CascadeFilter::getCommonOutputResolutionValue() const { return uv2CommonLevelOutputResolution; }

void CascadeFilter::initialize()
{
	if (is_initialized) return;

	//Firstly, initialize the base cascade level
	if (levels_use_common_output_reolution)levels[0].setOutputResolution(uv2CommonLevelOutputResolution);
	levels[0].initialize();

	//Check that initialization has been accomplished without errors
	if (!levels[0])
	{
		set_error_state(true);
		std::string err_msg = std::string{ "Base cascade level error (" } +levels[0].getErrorString() + ")";
		set_error_string(err_msg.c_str());
		call_error_callback(err_msg.c_str());
		return;
	}


	{
		std::vector<CascadeFilterLevel>::iterator level_iterator;
		int level_idx;
		for (level_iterator = ++levels.begin(), level_idx = 1; level_iterator != levels.end(); ++level_iterator, ++level_idx)
		{
			if (levels_use_common_output_reolution) level_iterator->setOutputResolution(uv2CommonLevelOutputResolution);

			//Apply data flow redirection settings
			std::for_each(level_data_flow_redirection_tables[level_idx-1].begin(), level_data_flow_redirection_tables[level_idx-1].end(),
				[this, level_iterator](DataFlow data_redirection_descriptor) -> void
			{
				uint32_t input_channel = data_redirection_descriptor.input_channel_index;

				uint32_t input_channel_offset = 0;
				uint32_t filter_idx = 0;	//this variable will store offset of the filter to which input channel with the given enumeration index belongs
				while (input_channel_offset + level_iterator->retrieveFilter(filter_idx)->getNumberOfInputChannels() <= input_channel)
				{
					input_channel_offset += level_iterator->retrieveFilter(filter_idx)->getNumberOfInputChannels();
					++filter_idx;
				}

				level_iterator->retrieveFilter(filter_idx)->setTextureSource(input_channel - input_channel_offset,
					levels[data_redirection_descriptor.output_channel_level].getTextureOutput(data_redirection_descriptor.output_channel_index));
			});

			//Initialize cascade level
			level_iterator->initialize();

			//Check that initialization has completed without errors
			if (!(*level_iterator))
			{
				set_error_state(true);
				std::string err_msg = "Error initializing cascade filter level#" + std::to_string(level_idx) + "(" + level_iterator->getErrorString() + ")";
				set_error_string(err_msg.c_str());
				call_error_callback(err_msg.c_str());
				return;
			}
		}
	}


	//We always use first output channel from the last cascade level as the final result of cascade filtering
	result_canvas.installTexture(levels[levels.size() - 1].getTextureOutput(0));

	is_initialized = true;
}

bool CascadeFilter::pass(const AbstractProjectingDevice& projecting_device, AbstractRenderingDevice& render_target)
{
	if (!is_initialized)
	{
		set_error_state(true);
		std::string err_msg = "Unable to execute cascade filter. The filter has not been initialized";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return false;
	}

	//Execute each cascade level
	for (uint32_t level_idx = 0; level_idx < levels.size(); ++level_idx)
	{
		levels[level_idx].pass(projecting_device);
		if (!levels[level_idx])
		{
			set_error_state(true);
			std::string err_msg = "Error while executing cascade filter level#" + std::to_string(level_idx) + " (" +
				levels[level_idx].getErrorString() + ")";
			set_error_string(err_msg.c_str());
			call_error_callback(err_msg.c_str());
			return false;
		}
	}



	//Get information about projection device being in use and setup the canvas accordingly
	float left, right, bottom, top, near, far;
	projecting_device.getProjectionVolume(&left, &right, &bottom, &top, &near, &far);
	result_canvas.setLocation(vec3{ (left + right) / 2.0f, (bottom + top) / 2.0f, -near });
	result_canvas.setDimensions(right - left, top - bottom);
	result_canvas.applyViewProjectionTransform(projecting_device);

	//Draw canvas to the output render target
	for (uint32_t rendering_pass = 0; rendering_pass < result_canvas.getNumberOfRenderingPasses(result_canvas.getActiveRenderingMode()); ++rendering_pass)
	{
		if (!result_canvas.prepareRendering(render_target, rendering_pass))
		{
			set_error_state(true);
			std::string err_msg = "Error in prepareRendering() when redirecting output from cascade filter (error occurred at rendering pass " + std::to_string(rendering_pass) + ")";
			set_error_string(err_msg.c_str());
			call_error_callback(err_msg.c_str());
			return false;
		}

		if (!result_canvas.render())
		{
			set_error_state(true);
			std::string err_msg = "Error in render() when redirecting output from cascade filter (error occurred at rendering pass " + std::to_string(rendering_pass) + ")";
			set_error_string(err_msg.c_str());
			call_error_callback(err_msg.c_str());
			return false;
		}

		if (!result_canvas.finalizeRendering())
		{
			set_error_state(true);
			std::string err_msg = "Error in finalizeRendering() when redirecting output from cascade filter (error occurred at rendering pass " + std::to_string(rendering_pass) + ")";
			set_error_string(err_msg.c_str());
			call_error_callback(err_msg.c_str());
			return false;
		}
	}

	return true;
}

uint32_t CascadeFilter::getNumberOfInputChannels() const
{
	//For each level of the cascade compute number of unused input channels. These channels are identified as the inputs of the cascade

	uint32_t num_inputs = 0;
	for (unsigned int filter_idx = 0; filter_idx < levels[0].getNumberOfFilters(); ++filter_idx)
		num_inputs += levels[0].retrieveFilter(filter_idx)->getNumberOfInputChannels();

	uint32_t num_busy_inputs = 0;
	for (unsigned int level_idx = 1; level_idx < levels.size(); ++level_idx)
	{
		std::set<uint32_t> busy_inputs;
		std::for_each(level_data_flow_redirection_tables[level_idx-1].begin(),
			level_data_flow_redirection_tables[level_idx-1].end(),
			[&busy_inputs](const DataFlow& data_rediection_entry) -> void
		{
			busy_inputs.insert(data_rediection_entry.input_channel_index);
		});
		num_busy_inputs += static_cast<uint32_t>(busy_inputs.size());

		for (unsigned int filter_idx = 0; filter_idx < levels[level_idx].getNumberOfFilters(); ++filter_idx)
			num_inputs += levels[level_idx].retrieveFilter(filter_idx)->getNumberOfInputChannels();
	}

	return num_inputs - num_busy_inputs;
}

bool CascadeFilter::isInitialized() const { return is_initialized; }