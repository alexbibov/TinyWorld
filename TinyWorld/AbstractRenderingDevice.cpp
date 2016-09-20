#include "AbstractRenderingDevice.h"

#include <algorithm>

using namespace tiny_world;

long long AbstractRenderingDevice::active_device = 0;

void AbstractRenderingDevice::applyOpenGLContextSettings()
{
	if (!isActive()) return;	//if the rendering device is not active on the calling thread, do nothing


	//Apply viewport settings
	std::for_each(viewports.begin(), viewports.end(),
		[](std::map<uint32_t, Rectangle>::value_type elem) -> void
	{glViewportIndexedf(elem.first, elem.second.x, elem.second.y, elem.second.w, elem.second.h); });


	//Apply scissor rectangle data
	std::for_each(scissor_rectangles.begin(), scissor_rectangles.end(),
		[](std::map<uint32_t, Rectangle>::value_type elem) -> void
	{glScissorIndexed(static_cast<GLsizei>(elem.first), static_cast<GLsizei>(elem.second.x), static_cast<GLsizei>(elem.second.y), 
	static_cast<GLsizei>(elem.second.w), static_cast<GLenum>(elem.second.h)); });
	

	//Apply cull test settings
	glCullFace(static_cast<GLenum>(context_front.face_to_cull));
	if (context_front.cull_test_enabled)
		glEnable(GL_CULL_FACE);
	else
		glDisable(GL_CULL_FACE);


	//Apply clear buffer settings
	glClearColor(context_front.clear_color.x, context_front.clear_color.y, context_front.clear_color.z, context_front.clear_color.w);
	glClearDepth(context_front.clear_depth);
	glClearStencil(context_front.clear_stencil);


	//Apply scissor test settings
	if (context_front.scissor_test_enabled)
		glEnable(GL_SCISSOR_TEST);
	else
		glDisable(GL_SCISSOR_TEST);


	//Apply stencil test settings
	if (context_front.scissor_test_enabled)
		glEnable(GL_STENCIL_TEST);
	else
		glDisable(GL_STENCIL_TEST);

	glStencilFuncSeparate(GL_FRONT, static_cast<GLenum>(context_front.stencil_test_pass_func[0].first),
		context_front.stencil_test_pass_func[0].second, context_front.stencil_test_pass_func[0].third.to_ulong());
	glStencilFuncSeparate(GL_BACK, static_cast<GLenum>(context_front.stencil_test_pass_func[1].first),
		context_front.stencil_test_pass_func[1].second, context_front.stencil_test_pass_func[1].third.to_ulong());

	glStencilOpSeparate(GL_FRONT, static_cast<GLenum>(context_front.stencil_test_op_stfail[0]),
		static_cast<GLenum>(context_front.stencil_test_op_dtfail[0]), static_cast<GLenum>(context_front.stencil_test_op_dtpass[0]));
	glStencilOpSeparate(GL_BACK, static_cast<GLenum>(context_front.stencil_test_op_stfail[1]),
		static_cast<GLenum>(context_front.stencil_test_op_dtfail[1]), static_cast<GLenum>(context_front.stencil_test_op_dtpass[1]));

	glStencilMaskSeparate(GL_FRONT, context_front.stencil_mask[0].to_ulong());
	glStencilMaskSeparate(GL_BACK, context_front.stencil_mask[1].to_ulong());



	//Apply depth test settings
	if (context_front.depth_test_enabled)
		glEnable(GL_DEPTH_TEST);
	else
		glDisable(GL_DEPTH_TEST);

	glDepthFunc(static_cast<GLenum>(context_front.depth_test_pass_func));
	glDepthMask(context_front.depth_test_update_flag);
	
	if (context_front.depth_test_clamp_flag)
		glEnable(GL_DEPTH_CLAMP);
	else
		glDisable(GL_DEPTH_CLAMP);



	//Apply color blending settings
	if (context_front.color_blend_enabled)
		glEnable(GL_BLEND);
	else
		glDisable(GL_BLEND);

	glBlendColor(context_front.blend_color.x, context_front.blend_color.y, context_front.blend_color.z, context_front.blend_color.w);	//set constant blending color

	glBlendFuncSeparate(static_cast<GLenum>(context_back.g_rgb_source), static_cast<GLenum>(context_back.g_rgb_destination),
		static_cast<GLenum>(context_back.g_alpha_source), static_cast<GLenum>(context_back.g_alpha_destination));		//set global blending factors
	glBlendEquationSeparate(static_cast<GLenum>(context_back.g_rgb_blending_eq), static_cast<GLenum>(context_back.g_alpha_blending_eq));	//set global blending equations



	//Apply color masking settings
	glColorMask(context_back.g_color_buffer_mask.x, context_back.g_color_buffer_mask.y, 
		context_back.g_color_buffer_mask.z, context_back.g_color_buffer_mask.w);	//apply global color mask



	//Apply multisampling functionality
	if (context_front.multisampling_enabled)
		glEnable(GL_MULTISAMPLE);
	else
		glDisable(GL_MULTISAMPLE);



	//Apply primitive restart functionality
	if (context_front.primitive_restart_enabled)
		glEnable(GL_PRIMITIVE_RESTART);
	else
		glDisable(GL_PRIMITIVE_RESTART);
	glPrimitiveRestartIndex(static_cast<GLuint>(context_front.primitive_restart_index));



	//Apply pixel pack parameters
	glPixelStorei(GL_PACK_SWAP_BYTES, static_cast<GLint>(context_front.swap_bytes));
	glPixelStorei(GL_PACK_LSB_FIRST, static_cast<GLint>(context_front.lsb_first));
	glPixelStorei(GL_PACK_ALIGNMENT, static_cast<GLint>(context_front.pack_alignment));
}


void AbstractRenderingDevice::makeActive()
{
	AbstractRenderingDevice::active_device = getId();
}


void AbstractRenderingDevice::makeActiveForDrawing()
{
	AbstractRenderingDevice::active_device = getId();
}


void AbstractRenderingDevice::init_device()
{
	context_front.clear_color = vec4{ 0.0f };
	context_front.clear_depth = 1.0;
	context_front.clear_stencil = 0;

	context_front.cull_test_enabled = false;
	context_front.face_to_cull = Face::BACK;
	context_front.scissor_test_enabled = false;
	context_front.stencil_test_enabled = false;
	context_front.depth_test_enabled = false;
	context_front.color_blend_enabled = false;
	context_front.multisampling_enabled = false;

	//NOTE: Initially, rendering device has no viewports (and hence, it does not have scissor rectangles)

	//Initial parameters for stencil test
	context_front.stencil_test_pass_func[0] = StencilTestPassCriterion{ StencilTestPassFunction::ALWAYS, 0, 0xFFFFFFFF };
	context_front.stencil_test_pass_func[1] = StencilTestPassCriterion{ StencilTestPassFunction::ALWAYS, 0, 0xFFFFFFFF };

	context_front.stencil_test_op_stfail[0] = StencilBufferUpdateOperation::KEEP;
	context_front.stencil_test_op_stfail[1] = StencilBufferUpdateOperation::KEEP;

	context_front.stencil_test_op_dtfail[0] = StencilBufferUpdateOperation::KEEP;
	context_front.stencil_test_op_dtfail[1] = StencilBufferUpdateOperation::KEEP;

	context_front.stencil_test_op_dtpass[0] = StencilBufferUpdateOperation::KEEP;
	context_front.stencil_test_op_dtpass[1] = StencilBufferUpdateOperation::KEEP;

	context_front.stencil_mask[0].set();
	context_front.stencil_mask[1].set();


	//Initial parameters for depth test
	context_front.depth_test_pass_func = DepthTestPassFunction::LESS_OR_EQUAL;
	context_front.depth_test_update_flag = true;
	context_front.depth_test_clamp_flag = false;


	//Initial parameters for primitive restart functionality
	context_front.primitive_restart_enabled = false;
	context_front.primitive_restart_index = 0xFFFFFFFF;


	//Initial parameters for color blending
	context_front.blend_color = vec4{ 0, 0, 0, 0 };
	context_back.g_rgb_source = ColorBlendFactor::ONE;
	context_back.g_alpha_source = ColorBlendFactor::ONE;
	context_back.g_rgb_destination = ColorBlendFactor::ZERO;
	context_back.g_alpha_destination = ColorBlendFactor::ZERO;
	context_back.g_rgb_blending_eq = ColorBlendEquation::ADD;
	context_back.g_alpha_blending_eq = ColorBlendEquation::ADD;


	//Initial parameters for color masking
	context_back.g_color_buffer_mask = bvec4{ 1, 1, 1, 1 };


	//Initial parameters for pixel packing
	context_front.swap_bytes = false;
	context_front.lsb_first = false;
	context_front.pack_alignment = TextureStorageAlignment::_4BYTE;
}


AbstractRenderingDevice::AbstractRenderingDevice(const std::string& rendering_device_class_string_name) : Entity(rendering_device_class_string_name)
{
	init_device();
}


AbstractRenderingDevice::AbstractRenderingDevice(const std::string& rendering_device_class_string_name, const std::string& rendering_device_string_name) : 
Entity(rendering_device_class_string_name, rendering_device_string_name)
{
	init_device();
}


AbstractRenderingDevice::AbstractRenderingDevice(const AbstractRenderingDevice& other) : 
Entity(other),
viewports(other.viewports), scissor_rectangles(other.scissor_rectangles), 
context_front(other.context_front),
context_back(other.context_back)
{

}


AbstractRenderingDevice::AbstractRenderingDevice(AbstractRenderingDevice&& other) : 
Entity(std::move(other)), 
viewports(std::move(other.viewports)), scissor_rectangles(std::move(other.scissor_rectangles)), 
context_front(std::move(other.context_front)),
context_back(std::move(other.context_back))
{

}


AbstractRenderingDevice::~AbstractRenderingDevice()
{

}

AbstractRenderingDevice& AbstractRenderingDevice::operator=(const AbstractRenderingDevice& other)
{
	//Handle the case where the object gets assigned "to itself"
	if (this == &other)
		return *this;

	Entity::operator=(other);

	viewports = other.viewports;
	scissor_rectangles = other.scissor_rectangles;

	context_front = other.context_front;
	
	context_back = other.context_back;

	return *this;
}


AbstractRenderingDevice& AbstractRenderingDevice::operator=(AbstractRenderingDevice&& other)
{
	//Handle the case where the object gets assigned "to itself"
	if (this == &other)
		return *this;

	Entity::operator=(std::move(other));

	viewports = std::move(other.viewports);
	scissor_rectangles = std::move(other.scissor_rectangles);

	context_front = std::move(other.context_front);

	context_back = std::move(other.context_back);

	return *this;
}

bool AbstractRenderingDevice::isActive() const { return AbstractRenderingDevice::active_device == getId(); }

long long AbstractRenderingDevice::getActiveDevice() { return active_device; }

void AbstractRenderingDevice::pushOpenGLContextSettings()
{
	context_stack.push_back(std::make_pair(ContextSnapshotFront{ context_front }, ContextSnapshotBack{ context_back }));
}

void AbstractRenderingDevice::popOpenGLContextSettings()
{
	std::pair<ContextSnapshotFront, ContextSnapshotBack> snapshot;

	//Check if stack is not empty
	if (context_stack.size())
	{
		//Retrieve context snapshot from the context settings stack
		snapshot = context_stack.back();
		context_stack.pop_back();
	}
	else
	{
		//if stack is empty, use default settings for the context
		ContextSnapshotFront snapshot_front;

		snapshot_front.cull_test_enabled = false;
		snapshot_front.face_to_cull = Face::BACK;

		snapshot_front.clear_color = vec4{ 0.0f };
		snapshot_front.clear_depth = 1.0f;
		snapshot_front.clear_stencil = 0;

		snapshot_front.scissor_test_enabled = false;

		snapshot_front.stencil_test_enabled = false;
		snapshot_front.stencil_test_pass_func =
			std::array < StencilTestPassCriterion, 2 > {StencilTestPassCriterion{ StencilTestPassFunction::ALWAYS, 0, 0xFFFFFFFF },
			StencilTestPassCriterion{ StencilTestPassFunction::ALWAYS, 0, 0xFFFFFFFF }};
		snapshot_front.stencil_test_op_stfail = std::array < StencilBufferUpdateOperation, 2 > {StencilBufferUpdateOperation::KEEP, StencilBufferUpdateOperation::KEEP};
		snapshot_front.stencil_test_op_dtfail = std::array < StencilBufferUpdateOperation, 2 > {StencilBufferUpdateOperation::KEEP, StencilBufferUpdateOperation::KEEP};
		snapshot_front.stencil_test_op_dtpass = std::array < StencilBufferUpdateOperation, 2 > {StencilBufferUpdateOperation::KEEP, StencilBufferUpdateOperation::KEEP};
		snapshot_front.stencil_mask[0].set(); snapshot_front.stencil_mask[1].set();

		snapshot_front.depth_test_enabled = false;
		snapshot_front.depth_test_pass_func = DepthTestPassFunction::LESS;
		snapshot_front.depth_test_update_flag = true;
		snapshot_front.depth_test_clamp_flag = false;

		snapshot_front.multisampling_enabled = false;

		snapshot_front.color_blend_enabled = false;
		snapshot_front.blend_color = vec4{ 0.0f };

		snapshot_front.primitive_restart_enabled = false;
		snapshot_front.primitive_restart_index = 0xFFFFFFFF;

		snapshot_front.swap_bytes = false;
		snapshot_front.lsb_first = false;
		snapshot_front.pack_alignment = TextureStorageAlignment::_4BYTE;



		ContextSnapshotBack snapshot_back;

		snapshot_back.g_rgb_source = ColorBlendFactor::ZERO;
		snapshot_back.g_alpha_source = ColorBlendFactor::ZERO;
		snapshot_back.g_rgb_destination = ColorBlendFactor::ONE;
		snapshot_back.g_alpha_destination = ColorBlendFactor::ONE;
		snapshot_back.g_rgb_blending_eq = ColorBlendEquation::ADD;
		snapshot_back.g_alpha_blending_eq = ColorBlendEquation::ADD;

		snapshot_back.g_color_buffer_mask = bvec4{ true };


		snapshot.first = snapshot_front;
		snapshot.second = snapshot_back;
	}

	context_front = snapshot.first;
	context_back = snapshot.second;
	
	AbstractRenderingDevice::applyOpenGLContextSettings();
}



void AbstractRenderingDevice::clearBuffers(BufferClearTarget clear_target)
{
	if (isActive()) glClear(static_cast<GLbitfield>(clear_target));
}



void AbstractRenderingDevice::defineViewport(Rectangle viewport)
{
	viewports[0] = viewport;
}

void AbstractRenderingDevice::defineViewport(float x, float y, float width, float height)
{
	viewports[0] = Rectangle{ x, y, width, height };
}

void AbstractRenderingDevice::defineViewport(uint32_t viewport_index, Rectangle viewport)
{
	viewports[viewport_index] = viewport;
}

void AbstractRenderingDevice::defineViewport(uint32_t viewport_index, float x, float y, float width, float height)
{
	viewports[viewport_index] = Rectangle{ x, y, width, height };
}

Rectangle AbstractRenderingDevice::getViewportRectangle(uint32_t viewport_index) const
{
	auto viewport = viewports.find(viewport_index);

	if (viewport != viewports.end())
		return (*viewport).second;
	else
		throw(std::out_of_range{ "View port located at index = " + std::to_string(viewport_index) + " was not defined" });
}



void AbstractRenderingDevice::setScissorTestEnableState(bool enabled)
{
	context_front.scissor_test_enabled = enabled;
}

bool AbstractRenderingDevice::getScissorTestEnableState() const { return context_front.scissor_test_enabled; }

void AbstractRenderingDevice::setScissorTestPassRectangle(uint32_t viewport_index, Rectangle scissor_test_pass_area)
{
	scissor_rectangles[viewport_index] = scissor_test_pass_area;
}

Rectangle AbstractRenderingDevice::getScissorTestPassRectangle(uint32_t viewport_index) const
{
	auto viewport = viewports.find(viewport_index);

	if (viewport == viewports.end())
		throw(std::out_of_range{ "View port located at index = " + std::to_string(viewport_index) + " was not defined" });
	else
	{
		auto scissor_passed_area = scissor_rectangles.find(viewport_index);
		if (scissor_passed_area == scissor_rectangles.end())
			return (*viewport).second;
		else
			return (*scissor_passed_area).second;
	}

}



void AbstractRenderingDevice::setCullTestMode(Face face) { context_front.face_to_cull = face; }


Face AbstractRenderingDevice::getCullTestMode() const { return context_front.face_to_cull; }


void AbstractRenderingDevice::setCullTestEnableState(bool enabled) { context_front.cull_test_enabled = enabled; }


bool AbstractRenderingDevice::getCullTestEnableState() const { return context_front.cull_test_enabled; }



void AbstractRenderingDevice::setClearColor(const vec4& clear_color){ context_front.clear_color = clear_color; }


void AbstractRenderingDevice::setClearColor(float r, float g, float b, float a){ setClearColor(vec4{ r, g, b, a }); }


vec4 AbstractRenderingDevice::getClearColor() const { return context_front.clear_color; }


void AbstractRenderingDevice::setClearDepth(double clear_depth){ context_front.clear_depth = clear_depth; }


double AbstractRenderingDevice::getClearDepth() const { return context_front.clear_depth; }


void AbstractRenderingDevice::setClearStencil(int clear_stencil){ context_front.clear_stencil = clear_stencil; }


int AbstractRenderingDevice::getClearStencil() const { return context_front.clear_stencil; }



void AbstractRenderingDevice::setStencilTestEnableState(bool enabled)
{
	context_front.stencil_test_enabled = enabled;
}


bool AbstractRenderingDevice::getStencilTestEnableState() const { return context_front.stencil_test_enabled; }


void AbstractRenderingDevice::setStencilTestPassFunction(Face face, StencilTestPassFunction func, uint32_t refval, bitmask32 refval_mask)
{
	switch (face)
	{
	case Face::FRONT:
		context_front.stencil_test_pass_func[0] = StencilTestPassCriterion{ func, refval, refval_mask };
		break;
	case Face::BACK:
		context_front.stencil_test_pass_func[1] = StencilTestPassCriterion{ func, refval, refval_mask };
		break;
	case Face::FRONT_AND_BACK:
		context_front.stencil_test_pass_func[0] = StencilTestPassCriterion{ func, refval, refval_mask };
		context_front.stencil_test_pass_func[1] = StencilTestPassCriterion{ func, refval, refval_mask };
		break;
	}
}


StencilTestPassCriterion AbstractRenderingDevice::getStencilTestPassFunction(Face face) const
{
	switch (face)
	{
	case Face::FRONT:
		return context_front.stencil_test_pass_func[0];
	case Face::BACK:
		return context_front.stencil_test_pass_func[1];
	case Face::FRONT_AND_BACK:
		throw(std::logic_error{ "Value \"tiny_world::Face::FRONT_AND_BACK\" is not supported by parameter \"face\" of function getStencilTestPassFunction()" });
	}

	return context_front.stencil_test_pass_func[0];
}


void AbstractRenderingDevice::setStencilTestFailStencilOperation(Face face, StencilBufferUpdateOperation op)
{
	switch (face)
	{
	case Face::FRONT:
		context_front.stencil_test_op_stfail[0] = op;
		break;
	case Face::BACK:
		context_front.stencil_test_op_stfail[1] = op;
		break;
	case Face::FRONT_AND_BACK:
		context_front.stencil_test_op_stfail[0] = op;
		context_front.stencil_test_op_stfail[1] = op;
		break;
	}
}


StencilBufferUpdateOperation AbstractRenderingDevice::getStencilTestFailStencilOperation(Face face) const
{
	switch (face)
	{
	case Face::FRONT:
		return context_front.stencil_test_op_stfail[0];
	case Face::BACK:
		return context_front.stencil_test_op_stfail[1];
	case Face::FRONT_AND_BACK:
		throw(std::logic_error{ "Value \"tiny_world::Face::FRONT_AND_BACK\" is not supported by parameter \"face\" of function getStencilTestFailStencilOperation()" });
	}

	return context_front.stencil_test_op_stfail[0];
}



void AbstractRenderingDevice::setDepthTestFailStencilOperation(Face face, StencilBufferUpdateOperation op)
{
	switch (face)
	{
	case Face::FRONT:
		context_front.stencil_test_op_dtfail[0] = op;
		break;
	case Face::BACK:
		context_front.stencil_test_op_dtfail[1] = op;
		break;
	case Face::FRONT_AND_BACK:
		context_front.stencil_test_op_dtfail[0] = op;
		context_front.stencil_test_op_dtfail[1] = op;
		break;
	}
}


StencilBufferUpdateOperation AbstractRenderingDevice::getDepthTestFailStencilOperation(Face face) const
{
	switch (face)
	{
	case Face::FRONT:
		return context_front.stencil_test_op_dtfail[0];
	case Face::BACK:
		return context_front.stencil_test_op_dtfail[1];
	case Face::FRONT_AND_BACK:
		throw(std::logic_error{ "Value \"tiny_world::Face::FRONT_AND_BACK\" is not supported by parameter \"face\" of function getDepthTestFailStencilOperation()" });
	}

	return context_front.stencil_test_op_dtfail[0];
}


void AbstractRenderingDevice::setDepthTestPassStencilOperation(Face face, StencilBufferUpdateOperation op)
{
	switch (face)
	{
	case Face::FRONT:
		context_front.stencil_test_op_dtpass[0] = op;
		break;
	case Face::BACK:
		context_front.stencil_test_op_dtpass[1] = op;
		break;
	case Face::FRONT_AND_BACK:
		context_front.stencil_test_op_dtpass[0] = op;
		context_front.stencil_test_op_dtpass[1] = op;
		break;
	}
}


StencilBufferUpdateOperation AbstractRenderingDevice::getDepthTestPassStencilOperation(Face face) const
{
	switch (face)
	{
	case Face::FRONT:
		return context_front.stencil_test_op_dtpass[0];
	case Face::BACK:
		return context_front.stencil_test_op_dtpass[1];
	case Face::FRONT_AND_BACK:
		throw(std::logic_error{ "Value \"tiny_world::Face::FRONT_AND_BACK\" is not supported by parameter \"face\" of function getDepthTestPassStencilOperation()" });
	}

	return context_front.stencil_test_op_dtpass[0];
}



void AbstractRenderingDevice::setStencilMask(Face face, bitmask32 mask)
{
	switch (face)
	{
	case Face::FRONT:
		context_front.stencil_mask[0] = mask;
		break;
	case Face::BACK:
		context_front.stencil_mask[1] = mask;
		break;
	case Face::FRONT_AND_BACK:
		context_front.stencil_mask[0] = mask;
		context_front.stencil_mask[1] = mask;
		break;
	}
}


bitmask32 AbstractRenderingDevice::getStencilMask(Face face) const
{
	switch (face)
	{
	case Face::FRONT:
		return context_front.stencil_mask[0];
	case Face::BACK:
		return context_front.stencil_mask[1];
	case Face::FRONT_AND_BACK:
		throw(std::logic_error{ "Value \"tiny_world::Face::FRONT_AND_BACK\" is not supported by parameter \"face\" of function getStencilMask()" });
	}

	return context_front.stencil_mask[0];
}



void AbstractRenderingDevice::setDepthTestEnableState(bool enabled)
{
	context_front.depth_test_enabled = enabled;
}


bool AbstractRenderingDevice::getDepthTestEnableState() const { return context_front.depth_test_enabled; }


void AbstractRenderingDevice::setDepthTestPassFunction(DepthTestPassFunction func)
{
	context_front.depth_test_pass_func = func;
}


DepthTestPassFunction AbstractRenderingDevice::getDepthTestPassFunction() const { return context_front.depth_test_pass_func; }


void AbstractRenderingDevice::setDepthBufferUpdateFlag(bool depth_mask)
{
	context_front.depth_test_update_flag = depth_mask;
}


bool AbstractRenderingDevice::getDepthBufferUpdateFlag() const { return context_front.depth_test_update_flag; }


void AbstractRenderingDevice::setDepthBufferClampFlag(bool clamp_flag)
{
	context_front.depth_test_clamp_flag = clamp_flag;
}


bool AbstractRenderingDevice::getDepthBufferClampFlag() const { return context_front.depth_test_clamp_flag; }



void AbstractRenderingDevice::setColorBlendEnableState(bool enabled)
{
	context_front.color_blend_enabled = enabled;
}


bool AbstractRenderingDevice::getColorBlendEnableState() const { return context_front.color_blend_enabled; }


void AbstractRenderingDevice::setBlendConstantColor(vec4 constant_blend_color)
{
	context_front.blend_color = constant_blend_color;
}


void AbstractRenderingDevice::setBlendConstantColor(float red, float green, float blue, float alpha)
{
	context_front.blend_color = vec4{ red, green, blue, alpha };
}


vec4 AbstractRenderingDevice::getBlendConstantColor() const { return context_front.blend_color; }


void AbstractRenderingDevice::setRGBSourceBlendFactor(ColorBlendFactor rgb_source_bf)
{
	context_back.g_rgb_source = rgb_source_bf;
}


void AbstractRenderingDevice::setAlphaSourceBlendFactor(ColorBlendFactor alpha_source_bf)
{
	context_back.g_alpha_source = alpha_source_bf;
}


void AbstractRenderingDevice::setSourceBlendFactor(ColorBlendFactor source_bf)
{
	context_back.g_rgb_source = source_bf;
	context_back.g_alpha_source = source_bf;
}


void AbstractRenderingDevice::setRGBDestinationBlendFactor(ColorBlendFactor rgb_destination_bf)
{
	context_back.g_rgb_destination = rgb_destination_bf;
}


void AbstractRenderingDevice::setAlphaDestinationBlendFactor(ColorBlendFactor alpha_destination_bf)
{
	context_back.g_alpha_destination = alpha_destination_bf;
}


void AbstractRenderingDevice::setDestinationBlendFactor(ColorBlendFactor destination_bf)
{
	context_back.g_rgb_destination = destination_bf;
	context_back.g_alpha_destination = destination_bf;
}


void AbstractRenderingDevice::setRGBBlendEquation(ColorBlendEquation rgb_blend_eq)
{
	context_back.g_rgb_blending_eq = rgb_blend_eq;
}


void AbstractRenderingDevice::setAlphaBlendEquation(ColorBlendEquation alpha_blend_eq)
{
	context_back.g_alpha_blending_eq = alpha_blend_eq;
}


void AbstractRenderingDevice::setBlendEquation(ColorBlendEquation blend_eq)
{
	context_back.g_alpha_blending_eq = blend_eq;
	context_back.g_rgb_blending_eq = blend_eq;
}


void AbstractRenderingDevice::setColorBufferMask(bool red, bool green, bool blue, bool alpha)
{
	context_back.g_color_buffer_mask = bvec4{ static_cast<GLboolean>(red), static_cast<GLboolean>(green),
		static_cast<GLboolean>(blue), static_cast<GLboolean>(alpha) };
}


void AbstractRenderingDevice::setColorBufferMask(bvec4 mask)
{
	context_back.g_color_buffer_mask = mask;
}


void AbstractRenderingDevice::setMultisamplingEnableState(bool enabled)
{
	context_front.multisampling_enabled = enabled;
}


bool AbstractRenderingDevice::getMultisamplingEnableState() const
{
	return context_front.multisampling_enabled;
}


void AbstractRenderingDevice::setPrimitiveRestartEnableState(bool enabled) { context_front.primitive_restart_enabled = enabled; }


bool AbstractRenderingDevice::getPrimitiveRestartEnableState() const { return context_front.primitive_restart_enabled; }


void AbstractRenderingDevice::setPrimitiveRestartIndexValue(uint32_t value) { context_front.primitive_restart_index = value; }


uint32_t AbstractRenderingDevice::getPrimitiveRestartIndexValue() const { return context_front.primitive_restart_index; }


void AbstractRenderingDevice::LSBFirst(bool flag) { context_front.lsb_first = flag; }


bool AbstractRenderingDevice::isLSBFirst() const { return context_front.lsb_first; }


void AbstractRenderingDevice::swapBytes(bool flag) { context_front.swap_bytes = flag; }


bool AbstractRenderingDevice::doesSwapBytes() const { return context_front.swap_bytes; }


void AbstractRenderingDevice::setPackPadding(TextureStorageAlignment new_pack_padding) { context_front.pack_alignment = new_pack_padding; }


TextureStorageAlignment AbstractRenderingDevice::getPackPadding() const { return context_front.pack_alignment; }


void AbstractRenderingDevice::readPixels(RenderingColorBuffer source_color_buffer, int x, int y, int width, int height, PixelReadLayout pixel_layout, PixelDataType pixel_type, void* data) const
{
	//Store previously active read framebuffer
	GLint current_read_framebuffer;
	glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &current_read_framebuffer);

	//Make this framebuffer active for reading
	makeActiveForReading();

	//Set source for the pixel read operation
	glReadBuffer(static_cast<GLenum>(source_color_buffer));

	//Read pixels into the buffer
	glReadPixels(static_cast<GLint>(x), static_cast<GLint>(y), static_cast<GLint>(width), static_cast<GLint>(height), static_cast<GLenum>(pixel_layout), static_cast<GLenum>(pixel_type), data);

	//Restore the old binding
	glBindFramebuffer(GL_READ_FRAMEBUFFER, current_read_framebuffer);
}


void AbstractRenderingDevice::readPixelsIntoTexture(RenderingColorBuffer source_color_buffer, const ImmutableTexture1D& _1d_texture, uint32_t mipmap_level, uint32_t layer, uint32_t xoffset, uint32_t x, uint32_t y, size_t width, size_t num_layers_to_modify /* = 1 */) const
{
	if (layer + num_layers_to_modify - 1 >= _1d_texture.getNumberOfArrayLayers())
	{
		set_error_state(true);
		std::string err_msg = "Unable to read pixels into layer " + std::to_string(std::max(_1d_texture.getNumberOfArrayLayers(), layer)) + " of 1D texture " + _1d_texture.getStringName() +
			". The layer must be in range [0, " + std::to_string(_1d_texture.getNumberOfArrayLayers() - 1) + "]";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}


	//Store previously active read framebuffer
	GLint current_read_framebuffer;
	glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &current_read_framebuffer);

	//Make this framebuffer active for reading
	makeActiveForReading();

	//Set source for the pixel read operation
	glReadBuffer(static_cast<GLenum>(source_color_buffer));

	//Bind the target texture 
	GLuint old_texture_binding = _1d_texture.bind();

	if (_1d_texture.isArrayTexture())
	{
		glCopyTexSubImage2D(GL_TEXTURE_1D_ARRAY, static_cast<GLint>(mipmap_level), static_cast<GLint>(xoffset), static_cast<GLint>(layer),
			static_cast<GLint>(x), static_cast<GLint>(y), static_cast<GLsizei>(width), static_cast<GLsizei>(num_layers_to_modify));
		glBindTexture(GL_TEXTURE_1D_ARRAY, old_texture_binding);	//restore the old texture binding
	}
	else
	{
		glCopyTexSubImage1D(GL_TEXTURE_1D, static_cast<GLint>(mipmap_level), static_cast<GLint>(xoffset), static_cast<GLint>(x), static_cast<GLint>(y), static_cast<GLsizei>(width));
		glBindTexture(GL_TEXTURE_1D, old_texture_binding);	//restore the old texture binding
	}

	//Restore the old framebuffer binding
	glBindFramebuffer(GL_READ_FRAMEBUFFER, current_read_framebuffer);
}


void AbstractRenderingDevice::readPixelsIntoTexture(RenderingColorBuffer source_color_buffer, const ImmutableTexture2D& _2d_texture, uint32_t mipmap_level, uint32_t layer, uint32_t xoffset, uint32_t yoffset,
	uint32_t x, uint32_t y, size_t width, size_t height) const
{
	if (layer >= _2d_texture.getNumberOfArrayLayers())
	{
		set_error_state(true);
		std::string err_msg = "Unable to read pixels into layer " + std::to_string(layer) + " of 2D texture " + _2d_texture.getStringName() +
			". The layer must be in range [0, " + std::to_string(_2d_texture.getNumberOfArrayLayers() - 1) + "]";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}


	//Store previously active read framebuffer
	GLint current_read_framebuffer;
	glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &current_read_framebuffer);


	//Make this framebuffer active for reading
	makeActiveForReading();

	//Set source for the pixel read operation
	glReadBuffer(static_cast<GLenum>(source_color_buffer));

	//Bind the target texture 
	GLuint old_texture_binding = _2d_texture.bind();

	if (_2d_texture.isArrayTexture())
	{
		glCopyTexSubImage3D(GL_TEXTURE_2D_ARRAY, static_cast<GLint>(mipmap_level), static_cast<GLint>(xoffset), static_cast<GLint>(yoffset), static_cast<GLint>(layer),
			static_cast<GLint>(x), static_cast<GLint>(y), static_cast<GLsizei>(width), static_cast<GLsizei>(height));
		glBindTexture(GL_TEXTURE_2D_ARRAY, old_texture_binding);	//restore the old texture binding
	}
	else
	{
		glCopyTexSubImage2D(GL_TEXTURE_2D, static_cast<GLint>(mipmap_level), static_cast<GLint>(xoffset), static_cast<GLint>(yoffset), 
			static_cast<GLint>(x), static_cast<GLint>(y), static_cast<GLsizei>(width), static_cast<GLsizei>(height));
		glBindTexture(GL_TEXTURE_2D, old_texture_binding);	//restore the old texture binding
	}


	//Restore the old framebuffer binding
	glBindFramebuffer(GL_READ_FRAMEBUFFER, current_read_framebuffer);
}


void AbstractRenderingDevice::readPixelsIntoTexture(RenderingColorBuffer source_color_buffer, const ImmutableTexture3D& _3d_texture, uint32_t mipmap_level, uint32_t xoffset, uint32_t yoffset, uint32_t zoffset, uint32_t x, uint32_t y, size_t width, size_t height) const
{
	//Store previously active read framebuffer
	GLint current_read_framebuffer;
	glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &current_read_framebuffer);

	makeActiveForReading();	//make this framebuffer active for reading
	glReadBuffer(static_cast<GLenum>(source_color_buffer));	//set source for the pixel read operation
	GLuint old_texture_binding = _3d_texture.bind();	//bind the target texture 


	glCopyTexSubImage3D(GL_TEXTURE_3D, static_cast<GLint>(mipmap_level), static_cast<GLint>(xoffset), static_cast<GLint>(yoffset), static_cast<GLint>(zoffset),
		static_cast<GLint>(x), static_cast<GLint>(y), static_cast<GLsizei>(width), static_cast<GLsizei>(height));


	glBindTexture(GL_TEXTURE_3D, old_texture_binding);	//restore the old texture binding
	glBindFramebuffer(GL_READ_FRAMEBUFFER, current_read_framebuffer);	//restore the old framebuffer binding
}


void AbstractRenderingDevice::readPixelsIntoTexture(RenderingColorBuffer source_color_buffer, const ImmutableTextureCubeMap& cubemap_texture, uint32_t mipmap_level, uint32_t layer, CubemapFace face, uint32_t xoffset, uint32_t yoffset, uint32_t x, uint32_t y, size_t width, size_t height) const
{
	if (layer >= cubemap_texture.getNumberOfArrayLayers())
	{
		set_error_state(true);
		std::string err_msg = "Unable to read pixels into layer " + std::to_string(layer) + " of 2D texture " + cubemap_texture.getStringName() +
			". The layer must be in range [0, " + std::to_string(cubemap_texture.getNumberOfArrayLayers() - 1) + "]";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return;
	}


	//Store previously active read framebuffer
	GLint current_read_framebuffer;
	glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &current_read_framebuffer);

	makeActiveForReading();	//make this framebuffer active for reading
	glReadBuffer(static_cast<GLenum>(source_color_buffer));	//set source for the pixel read operation
	GLuint old_texture_binding = cubemap_texture.bind();	//bind the target texture 

	if (cubemap_texture.isArrayTexture())
	{
		glCopyTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, static_cast<GLint>(mipmap_level), static_cast<GLint>(xoffset), static_cast<GLint>(yoffset), static_cast<GLint>(6 * layer + static_cast<GLenum>(face)-GL_TEXTURE_CUBE_MAP_POSITIVE_X),
			static_cast<GLint>(x), static_cast<GLint>(y), static_cast<GLsizei>(width), static_cast<GLsizei>(height));
		glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, old_texture_binding);	//restore the old texture binding
	}
	else
	{
		glCopyTexSubImage2D(static_cast<GLenum>(face), static_cast<GLint>(mipmap_level), static_cast<GLint>(xoffset), static_cast<GLint>(yoffset),
			static_cast<GLint>(x), static_cast<GLint>(y), static_cast<GLsizei>(width), static_cast<GLsizei>(height));
		glBindTexture(GL_TEXTURE_CUBE_MAP, old_texture_binding);	//restore the old texture binding
	}

	glBindFramebuffer(GL_READ_FRAMEBUFFER, current_read_framebuffer);	//restore the old framebuffer binding
}