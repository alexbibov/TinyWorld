#include "Framebuffer.h"

using namespace tiny_world;


Framebuffer::Framebuffer() : AbstractRenderingDevice{"Framebuffer"},
ogl_framebuffer_id{}, stencil_attachment{}, depth_attachment{}, stencil_depth_attachment{},
color_attachments({}), renderer{ [](Framebuffer&) -> void {} }
{
	glGenFramebuffers(1, &ogl_framebuffer_id);
	//TO BE EXTENDED
}

Framebuffer::Framebuffer(const std::string& framebuffer_string_name) : AbstractRenderingDevice{"Framebuffer", framebuffer_string_name},
ogl_framebuffer_id{ ogl_framebuffer_id }, stencil_attachment{}, depth_attachment{},
stencil_depth_attachment{}, color_attachments({}), renderer{ [](Framebuffer&) -> void{} }
{ 
	glGenFramebuffers(1, &ogl_framebuffer_id);
	//TO BE EXTENDED
}

Framebuffer::Framebuffer(const Framebuffer& other) : AbstractRenderingDevice{ other },
ogl_framebuffer_id{}, stencil_attachment{}, depth_attachment{}, stencil_depth_attachment{}, 
color_attachments({}), context_framebuffer(other.context_framebuffer), renderer(other.renderer)
{
	glGenFramebuffers(1, &ogl_framebuffer_id);


	if (other.stencil_attachment)
		attachTexture(FramebufferAttachmentPoint::STENCIL, other.stencil_attachment);
		
	if (other.depth_attachment)
		attachTexture(FramebufferAttachmentPoint::DEPTH, other.depth_attachment);

	if (other.stencil_depth_attachment)
		attachTexture(FramebufferAttachmentPoint::STENCIL_DEPTH, other.stencil_depth_attachment);

	for (int i = 0; i < static_cast<int>(other.color_attachments.size()); ++i)
	{
		if (other.color_attachments[i])
		{
			FramebufferAttachmentPoint color_attachment_point =
				static_cast<FramebufferAttachmentPoint>(static_cast<GLenum>(FramebufferAttachmentPoint::COLOR0) + i);
			attachTexture(color_attachment_point, other.color_attachments[i]);
		}
	}

}

Framebuffer::Framebuffer(Framebuffer&& other) : 
AbstractRenderingDevice{std::move(other)},
ogl_framebuffer_id{ other.ogl_framebuffer_id },
stencil_attachment{ std::move(other.stencil_attachment) },
depth_attachment{ std::move(other.depth_attachment) },
stencil_depth_attachment{ std::move(other.stencil_depth_attachment) },
color_attachments({}),
context_framebuffer(std::move(other.context_framebuffer)),
renderer{ std::move(other.renderer) }
{
	other.ogl_framebuffer_id = 0;
	for (int i = 0; i < static_cast<int>(other.color_attachments.size()); ++i)
	{
		if (other.color_attachments[i])
			color_attachments[i] = std::move(other.color_attachments[i]);
	}
}

Framebuffer::~Framebuffer()
{
	if (ogl_framebuffer_id) glDeleteFramebuffers(1, &ogl_framebuffer_id);
}


void Framebuffer::applyOpenGLContextSettings()
{
	if (!isActive()) return;	//if Framebuffer is not active on the calling thread, do nothing

	AbstractRenderingDevice::applyOpenGLContextSettings();

	//apply individually defined blending factors for the color buffers that have such individual definitions provided
	std::for_each(context_framebuffer.rgb_sources.begin(), context_framebuffer.rgb_sources.end(),
		[this](std::map<uint32_t, ColorBlendFactor>::value_type elem) -> void
	{
		glBlendFuncSeparatei(elem.first, static_cast<GLenum>(elem.second), static_cast<GLenum>(context_framebuffer.rgb_destinations.at(elem.first)),
			static_cast<GLenum>(context_framebuffer.alpha_sources.at(elem.first)), static_cast<GLenum>(context_framebuffer.alpha_destinations.at(elem.first)));
	}
	);

	//apply individually defined blending equations for the color buffers that have them
	std::for_each(context_framebuffer.rgb_blending_eqs.begin(), context_framebuffer.rgb_blending_eqs.end(),
		[this](std::map<uint32_t, ColorBlendEquation>::value_type elem) -> void
	{
		glBlendEquationSeparatei(elem.first, static_cast<GLenum>(elem.second), static_cast<GLenum>(context_framebuffer.alpha_blending_eqs.at(elem.first)));
	}
	);

	//apply individual color mask for the color buffers that have such definition
	std::for_each(context_framebuffer.color_buffer_masks.begin(), context_framebuffer.color_buffer_masks.end(),
		[](std::map<uint32_t, bvec4>::value_type elem) -> void
	{
		glColorMaski(elem.first, elem.second.x, elem.second.y, elem.second.z, elem.second.w);
	}
	);
}


void Framebuffer::pushOpenGLContextSettings()
{
	AbstractRenderingDevice::pushOpenGLContextSettings();
	context_framebuffer_stack.push_back(context_framebuffer);
}


void Framebuffer::popOpenGLContextSettings()
{
	//Pop base settings of the context
	AbstractRenderingDevice::popOpenGLContextSettings();


	//Check if framebuffer context settings stack is not empty
	if (context_framebuffer_stack.size())
	{
		//Pop up settings from the top of the framebuffer settings stack
		 ContextSnapshotFramebuffer context_framebuffer_snapshot = context_framebuffer_stack.back();
		context_framebuffer_stack.pop_back();

		//For each element of the retrieved snapshot that has been present in the previously active settings bundle and is not present in the retrieved bundle,
		//assign default set of parameters, so that default values will get restored on the next activation of the rendering device (framebuffer)
		std::for_each(context_framebuffer.rgb_sources.begin(), context_framebuffer.rgb_sources.end(),
			[&context_framebuffer_snapshot, this](std::pair<uint32_t, ColorBlendFactor> elem) -> void
		{
			if (context_framebuffer_snapshot.rgb_sources.insert(elem).second)
				context_framebuffer_snapshot.rgb_sources.at(elem.first) = context_back.g_rgb_source;
		});

		std::for_each(context_framebuffer.alpha_sources.begin(), context_framebuffer.alpha_sources.end(),
			[&context_framebuffer_snapshot, this](std::pair<uint32_t, ColorBlendFactor> elem) -> void
		{
			if (context_framebuffer_snapshot.alpha_sources.insert(elem).second)
				context_framebuffer_snapshot.alpha_sources.at(elem.first) = context_back.g_alpha_source;
		});

		std::for_each(context_framebuffer.rgb_destinations.begin(), context_framebuffer.rgb_destinations.end(),
			[&context_framebuffer_snapshot, this](std::pair<uint32_t, ColorBlendFactor> elem) -> void
		{
			if (context_framebuffer_snapshot.rgb_destinations.insert(elem).second)
				context_framebuffer_snapshot.rgb_destinations.at(elem.first) = context_back.g_rgb_destination;
		});

		std::for_each(context_framebuffer.alpha_destinations.begin(), context_framebuffer.alpha_destinations.end(),
			[&context_framebuffer_snapshot, this](std::pair<uint32_t, ColorBlendFactor> elem) -> void
		{
			if (context_framebuffer_snapshot.alpha_destinations.insert(elem).second)
				context_framebuffer_snapshot.alpha_destinations.at(elem.first) = context_back.g_alpha_destination;
		});

		std::for_each(context_framebuffer.color_buffer_masks.begin(), context_framebuffer.color_buffer_masks.end(),
			[&context_framebuffer_snapshot, this](std::pair<uint32_t, bvec4> elem) -> void
		{
			if (context_framebuffer_snapshot.color_buffer_masks.insert(elem).second)
				context_framebuffer_snapshot.color_buffer_masks.at(elem.first) = context_back.g_color_buffer_mask;
		});


		//Apply new settings to the context:
		context_framebuffer = context_framebuffer_snapshot;


		if (isActive())
		{
			//Apply blending functions
			std::for_each(context_framebuffer.rgb_sources.begin(), context_framebuffer.rgb_sources.end(),
				[this](std::pair<uint32_t, ColorBlendFactor> elem) -> void
			{
				glBlendFuncSeparatei(elem.first, static_cast<GLenum>(elem.second), static_cast<GLenum>(context_framebuffer.rgb_destinations.at(elem.first)),
					static_cast<GLenum>(context_framebuffer.alpha_sources.at(elem.first)), static_cast<GLenum>(context_framebuffer.alpha_destinations.at(elem.first)));
			});

			//Apply blending equations
			std::for_each(context_framebuffer.rgb_blending_eqs.begin(), context_framebuffer.rgb_blending_eqs.end(),
				[this](std::pair<uint32_t, ColorBlendEquation> elem) -> void
			{
				glBlendEquationSeparatei(elem.first, static_cast<GLenum>(elem.second), static_cast<GLenum>(context_framebuffer.alpha_blending_eqs.at(elem.first)));
			});

			//Apply color masks
			std::for_each(context_framebuffer.color_buffer_masks.begin(), context_framebuffer.color_buffer_masks.end(),
				[](std::pair<uint32_t, bvec4> elem) -> void
			{
				glColorMaski(elem.first, elem.second.x, elem.second.y, elem.second.z, elem.second.w);
			});
		}
	}
	else
	{
		//If framebuffer context settings stack is empty, we need to restore the defaults
		std::for_each(context_framebuffer.rgb_sources.begin(), context_framebuffer.rgb_sources.end(),
			[this](std::pair<uint32_t, ColorBlendFactor> elem) -> void
		{
			elem.second = context_back.g_rgb_source;
			context_framebuffer.alpha_sources.at(elem.first) = context_back.g_alpha_source;
			context_framebuffer.rgb_destinations.at(elem.first) = context_back.g_rgb_destination;
			context_framebuffer.alpha_destinations.at(elem.first) = context_back.g_alpha_destination;

			if (isActive())
				glBlendFuncSeparatei(elem.first, static_cast<GLenum>(context_back.g_rgb_source), static_cast<GLenum>(context_back.g_rgb_destination),
				static_cast<GLenum>(context_back.g_alpha_source), static_cast<GLenum>(context_back.g_alpha_destination));
		});

		std::for_each(context_framebuffer.rgb_blending_eqs.begin(), context_framebuffer.rgb_blending_eqs.end(),
			[this](std::pair<uint32_t, ColorBlendEquation> elem) -> void
		{
			elem.second = context_back.g_rgb_blending_eq;
			context_framebuffer.alpha_blending_eqs.at(elem.first) = context_back.g_alpha_blending_eq;

			if (isActive())
				glBlendEquationSeparatei(elem.first, static_cast<GLenum>(context_back.g_rgb_blending_eq), static_cast<GLenum>(context_back.g_alpha_blending_eq));
		});

		std::for_each(context_framebuffer.color_buffer_masks.begin(), context_framebuffer.color_buffer_masks.end(),
			[this](std::pair<uint32_t, bvec4> elem) -> void
		{
			elem.second = context_back.g_color_buffer_mask;

			if (isActive())
			{
				bvec4 mask = context_back.g_color_buffer_mask;
				glColorMaski(elem.first, mask.x, mask.y, mask.z, mask.w);
			}
		});
	}
}


void Framebuffer::attachRenderer(const GenericFramebufferRenderer& renderer)
{
	this->renderer = renderer;
}


uint32_t Framebuffer::getLastSupportedColorAttachmentPoint()
{
	GLint max_draw_buffers;
	glGetIntegerv(GL_MAX_DRAW_BUFFERS, &max_draw_buffers);

	return max_draw_buffers - 1;
}


void Framebuffer::setRGBSourceBlendFactor(uint32_t color_buffer_index, ColorBlendFactor rgb_source_bf)
{
	context_framebuffer.rgb_sources[color_buffer_index] = rgb_source_bf;
	if (context_framebuffer.alpha_sources.find(color_buffer_index) == context_framebuffer.alpha_sources.end())
		context_framebuffer.alpha_sources[color_buffer_index] = ColorBlendFactor::ONE;

	if (context_framebuffer.rgb_destinations.find(color_buffer_index) == context_framebuffer.rgb_destinations.end())
		context_framebuffer.rgb_destinations[color_buffer_index] = ColorBlendFactor::ZERO;
	if (context_framebuffer.alpha_destinations.find(color_buffer_index) == context_framebuffer.alpha_destinations.end())
		context_framebuffer.alpha_destinations[color_buffer_index] = ColorBlendFactor::ZERO;
}

ColorBlendFactor Framebuffer::getRGBSourceBlendFactor(uint32_t color_buffer_index) const
{
	auto rgb_source = context_framebuffer.rgb_sources.find(color_buffer_index);

	if (rgb_source != context_framebuffer.rgb_sources.end())
		return (*rgb_source).second;
	else
		return context_back.g_rgb_source;
}

void Framebuffer::setAlphaSourceBlendFactor(uint32_t color_buffer_index, ColorBlendFactor alpha_source_bf)
{
	if (context_framebuffer.rgb_sources.find(color_buffer_index) == context_framebuffer.rgb_sources.end())
		context_framebuffer.rgb_sources[color_buffer_index] = ColorBlendFactor::ONE;
	context_framebuffer.alpha_sources[color_buffer_index] = alpha_source_bf;

	if (context_framebuffer.rgb_destinations.find(color_buffer_index) == context_framebuffer.rgb_destinations.end())
		context_framebuffer.rgb_destinations[color_buffer_index] = ColorBlendFactor::ZERO;
	if (context_framebuffer.alpha_destinations.find(color_buffer_index) == context_framebuffer.alpha_destinations.end())
		context_framebuffer.alpha_destinations[color_buffer_index] = ColorBlendFactor::ZERO;
}

ColorBlendFactor Framebuffer::getAlphaSourceBlendFactor(uint32_t color_buffer_index) const
{
	auto alpha_source = context_framebuffer.alpha_sources.find(color_buffer_index);

	if (alpha_source != context_framebuffer.alpha_sources.end())
		return (*alpha_source).second;
	else
		return context_back.g_alpha_source;
}

void Framebuffer::setSourceBlendFactor(uint32_t color_buffer_index, ColorBlendFactor source_bf)
{
	context_framebuffer.rgb_sources[color_buffer_index] = source_bf;
	context_framebuffer.alpha_sources[color_buffer_index] = source_bf;

	if (context_framebuffer.rgb_destinations.find(color_buffer_index) == context_framebuffer.rgb_destinations.end())
		context_framebuffer.rgb_destinations[color_buffer_index] = ColorBlendFactor::ZERO;
	if (context_framebuffer.alpha_destinations.find(color_buffer_index) == context_framebuffer.alpha_destinations.end())
		context_framebuffer.alpha_destinations[color_buffer_index] = ColorBlendFactor::ZERO;
}

void Framebuffer::setRGBDestinationBlendFactor(uint32_t color_buffer_index, ColorBlendFactor rgb_destination_bf)
{
	if (context_framebuffer.rgb_sources.find(color_buffer_index) == context_framebuffer.rgb_sources.end())
		context_framebuffer.rgb_sources[color_buffer_index] = ColorBlendFactor::ONE;
	if (context_framebuffer.alpha_sources.find(color_buffer_index) == context_framebuffer.alpha_sources.end())
		context_framebuffer.alpha_sources[color_buffer_index] = ColorBlendFactor::ONE;

	context_framebuffer.rgb_destinations[color_buffer_index] = rgb_destination_bf;
	if (context_framebuffer.alpha_destinations.find(color_buffer_index) == context_framebuffer.alpha_destinations.end())
		context_framebuffer.alpha_destinations[color_buffer_index] = ColorBlendFactor::ZERO;
}

ColorBlendFactor Framebuffer::getRGBDestinationBlendFactor(uint32_t color_buffer_index) const
{
	auto rgb_destination = context_framebuffer.rgb_destinations.find(color_buffer_index);

	if (rgb_destination != context_framebuffer.rgb_destinations.end())
		return (*rgb_destination).second;
	else
		return context_back.g_rgb_destination;
}

void Framebuffer::setAlphaDestinationBlendFactor(uint32_t color_buffer_index, ColorBlendFactor alpha_destination_bf)
{
	if (context_framebuffer.rgb_sources.find(color_buffer_index) == context_framebuffer.rgb_sources.end())
		context_framebuffer.rgb_sources[color_buffer_index] = ColorBlendFactor::ONE;
	if (context_framebuffer.alpha_sources.find(color_buffer_index) == context_framebuffer.alpha_sources.end())
		context_framebuffer.alpha_sources[color_buffer_index] = ColorBlendFactor::ONE;

	if (context_framebuffer.rgb_destinations.find(color_buffer_index) == context_framebuffer.rgb_destinations.end())
		context_framebuffer.rgb_destinations[color_buffer_index] = ColorBlendFactor::ZERO;
	context_framebuffer.alpha_destinations[color_buffer_index] = alpha_destination_bf;
}

ColorBlendFactor Framebuffer::getAlphaDestinationBlendFactor(uint32_t color_buffer_index) const
{
	auto alpha_destination = context_framebuffer.alpha_destinations.find(color_buffer_index);

	if (alpha_destination != context_framebuffer.alpha_destinations.end())
		return (*alpha_destination).second;
	else
		return context_back.g_alpha_destination;
}

void Framebuffer::setDestinationBlendFactor(uint32_t color_buffer_index, ColorBlendFactor destination_bf)
{
	if (context_framebuffer.rgb_sources.find(color_buffer_index) == context_framebuffer.rgb_sources.end())
		context_framebuffer.rgb_sources[color_buffer_index] = ColorBlendFactor::ONE;
	if (context_framebuffer.alpha_sources.find(color_buffer_index) == context_framebuffer.alpha_sources.end())
		context_framebuffer.alpha_sources[color_buffer_index] = ColorBlendFactor::ONE;

	context_framebuffer.rgb_destinations[color_buffer_index] = destination_bf;
	context_framebuffer.alpha_destinations[color_buffer_index] = destination_bf;
}

void Framebuffer::setRGBBlendEquation(uint32_t color_buffer_index, ColorBlendEquation rgb_blend_eq)
{
	context_framebuffer.rgb_blending_eqs[color_buffer_index] = rgb_blend_eq;
	if (context_framebuffer.alpha_blending_eqs.find(color_buffer_index) == context_framebuffer.alpha_blending_eqs.end())
		context_framebuffer.alpha_blending_eqs[color_buffer_index] = ColorBlendEquation::ADD;
}

ColorBlendEquation Framebuffer::getRGBBlendEquation(uint32_t color_buffer_index) const
{
	auto rgb_blend_eq = context_framebuffer.rgb_blending_eqs.find(color_buffer_index);

	if (rgb_blend_eq != context_framebuffer.rgb_blending_eqs.end())
		return (*rgb_blend_eq).second;
	else
		return context_back.g_rgb_blending_eq;
}

void Framebuffer::setAlphaBlendEquation(uint32_t color_buffer_index, ColorBlendEquation alpha_blend_eq)
{
	if (context_framebuffer.rgb_blending_eqs.find(color_buffer_index) == context_framebuffer.rgb_blending_eqs.end())
		context_framebuffer.rgb_blending_eqs[color_buffer_index] = ColorBlendEquation::ADD;
	context_framebuffer.alpha_blending_eqs[color_buffer_index] = alpha_blend_eq;
}

ColorBlendEquation Framebuffer::getAlphaBlendEquation(uint32_t color_buffer_index) const
{
	auto alpha_blend_eq = context_framebuffer.alpha_blending_eqs.find(color_buffer_index);

	if (alpha_blend_eq != context_framebuffer.alpha_blending_eqs.end())
		return (*alpha_blend_eq).second;
	else
		return context_back.g_alpha_blending_eq;
}

void Framebuffer::setBlendEquation(uint32_t color_buffer_index, ColorBlendEquation blend_eq)
{
	context_framebuffer.alpha_blending_eqs[color_buffer_index] = blend_eq;
	context_framebuffer.rgb_blending_eqs[color_buffer_index] = blend_eq;
}

void Framebuffer::setColorBufferMask(uint32_t color_buffer_index, bool red, bool green, bool blue, bool alpha)
{
	context_framebuffer.color_buffer_masks[color_buffer_index] = bvec4{ static_cast<GLboolean>(red), static_cast<GLboolean>(green),
		static_cast<GLboolean>(blue), static_cast<GLboolean>(alpha) };
}

void Framebuffer::setColorBufferMask(uint32_t color_buffer_index, bvec4 mask)
{
	context_framebuffer.color_buffer_masks[color_buffer_index] = mask;
}

bvec4 Framebuffer::getColorBufferMask(uint32_t color_buffer_index) const
{
	auto color_buffer_mask = context_framebuffer.color_buffer_masks.find(color_buffer_index);

	if (color_buffer_mask != context_framebuffer.color_buffer_masks.end())
		return (*color_buffer_mask).second;
	else
		return context_back.g_color_buffer_mask;
}


void Framebuffer::makeActive()
{
	AbstractRenderingDevice::makeActive();
	glBindFramebuffer(GL_FRAMEBUFFER, ogl_framebuffer_id);
	applyOpenGLContextSettings();
}


void Framebuffer::makeActiveForReading() const
{
	glBindFramebuffer(GL_READ_FRAMEBUFFER, ogl_framebuffer_id);
}


void Framebuffer::makeActiveForDrawing()
{
	AbstractRenderingDevice::makeActiveForDrawing();
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, ogl_framebuffer_id);
	applyOpenGLContextSettings();
}


void Framebuffer::attachTexture(FramebufferAttachmentPoint attachment_point, const FramebufferAttachmentInfo& attachment_details)
{
	GLint max_draw_buffers;
	glGetIntegerv(GL_MAX_DRAW_BUFFERS, &max_draw_buffers);

	GLint currently_bound_framebuffer;
	glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &currently_bound_framebuffer);

	//Update framebuffer's internal state
	switch (attachment_point)
	{
	case FramebufferAttachmentPoint::STENCIL:
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, ogl_framebuffer_id);
		stencil_attachment = attachment_details;
		stencil_depth_attachment = OwnedFramebufferAttachmentInfo{};
		break;

	case FramebufferAttachmentPoint::DEPTH:
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, ogl_framebuffer_id);
		depth_attachment = attachment_details;
		stencil_depth_attachment = OwnedFramebufferAttachmentInfo{};
		break;

	case FramebufferAttachmentPoint::STENCIL_DEPTH:
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, ogl_framebuffer_id);
		stencil_depth_attachment = attachment_details;
		stencil_attachment = OwnedFramebufferAttachmentInfo{};
		depth_attachment = OwnedFramebufferAttachmentInfo{};
		break;

	case FramebufferAttachmentPoint::COLOR0:
	case FramebufferAttachmentPoint::COLOR1:
	case FramebufferAttachmentPoint::COLOR2:
	case FramebufferAttachmentPoint::COLOR3:
	case FramebufferAttachmentPoint::COLOR4:
	case FramebufferAttachmentPoint::COLOR5:
	case FramebufferAttachmentPoint::COLOR6:
	case FramebufferAttachmentPoint::COLOR7:
	case FramebufferAttachmentPoint::COLOR8:
	case FramebufferAttachmentPoint::COLOR9:
	case FramebufferAttachmentPoint::COLOR10:
	case FramebufferAttachmentPoint::COLOR11:
	case FramebufferAttachmentPoint::COLOR12:
	case FramebufferAttachmentPoint::COLOR13:
	case FramebufferAttachmentPoint::COLOR14:
	case FramebufferAttachmentPoint::COLOR15:
		uint32_t color_attachment_entry =
			static_cast<uint32_t>(static_cast<GLenum>(attachment_point)-static_cast<GLenum>(FramebufferAttachmentPoint::COLOR0));
		if (color_attachment_entry >= static_cast<uint32_t>(max_draw_buffers))
		{
			set_error_state(true);
			std::string err_msg = "Color attachment point " + std::to_string(color_attachment_entry) + " is not supported by the present hardware. Use color attachment points from 0 to " +
				std::to_string(max_draw_buffers - 1);
			set_error_string(err_msg);
			call_error_callback(err_msg);
			return;
		}


		color_attachments[color_attachment_entry] = attachment_details;
		
		//Define draw color buffers
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, ogl_framebuffer_id);
		std::array<GLenum, max_color_attachments> draw_buffer_list;
		for (int i = 0; i < static_cast<int>(color_attachments.size()); ++i)
		{
			if (color_attachments[i])
				draw_buffer_list[i] = GL_COLOR_ATTACHMENT0 + i;
			else
				draw_buffer_list[i] = GL_NONE;
		}
		glDrawBuffers(max_draw_buffers, draw_buffer_list.data());

		break;
	}

	//Attach texture to the framebuffer
	if (attachment_details.texture->isBufferTexture())
		dynamic_cast<const BufferTexture*>(attachment_details.texture)->attachToFBO(attachment_point);
	else
		if (attachment_details.attachment_layer >= 0 && (attachment_details.texture->isArrayTexture() || attachment_details.texture->getNumberOfFaces() > 1))
			dynamic_cast<const ImmutableTexture*>(attachment_details.texture)->attachToFBO(attachment_point, attachment_details.attachment_layer, attachment_details.mipmap_level);
		else
			dynamic_cast<const ImmutableTexture*>(attachment_details.texture)->attachToFBO(attachment_point, attachment_details.mipmap_level);


	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, currently_bound_framebuffer);
}





FramebufferAttachmentInfo Framebuffer::retrieveAttachmentDetails(FramebufferAttachmentPoint attachment_point) const
{
	switch (attachment_point)
	{
	case FramebufferAttachmentPoint::STENCIL:
		return stencil_attachment;

	case FramebufferAttachmentPoint::DEPTH:
		return depth_attachment;

	case FramebufferAttachmentPoint::STENCIL_DEPTH:
		return stencil_depth_attachment;

	case FramebufferAttachmentPoint::COLOR0:
	case FramebufferAttachmentPoint::COLOR1:
	case FramebufferAttachmentPoint::COLOR2:
	case FramebufferAttachmentPoint::COLOR3:
	case FramebufferAttachmentPoint::COLOR4:
	case FramebufferAttachmentPoint::COLOR5:
	case FramebufferAttachmentPoint::COLOR6:
	case FramebufferAttachmentPoint::COLOR7:
	case FramebufferAttachmentPoint::COLOR8:
	case FramebufferAttachmentPoint::COLOR9:
	case FramebufferAttachmentPoint::COLOR10:
	case FramebufferAttachmentPoint::COLOR11:
	case FramebufferAttachmentPoint::COLOR12:
	case FramebufferAttachmentPoint::COLOR13:
	case FramebufferAttachmentPoint::COLOR14:
	case FramebufferAttachmentPoint::COLOR15:
		uint32_t color_attachment_entry =
			static_cast<uint32_t>(static_cast<GLenum>(attachment_point)-static_cast<GLenum>(FramebufferAttachmentPoint::COLOR0));
		return color_attachments[color_attachment_entry];
	}

	return FramebufferAttachmentInfo{};
}


void Framebuffer::detachTexture(FramebufferAttachmentPoint attachment_point)
{
	switch (attachment_point)
	{
	case FramebufferAttachmentPoint::STENCIL:
		stencil_attachment = OwnedFramebufferAttachmentInfo{};
		break;
	case FramebufferAttachmentPoint::DEPTH:
		depth_attachment = OwnedFramebufferAttachmentInfo{};
		break;
	case FramebufferAttachmentPoint::STENCIL_DEPTH:
		stencil_depth_attachment = OwnedFramebufferAttachmentInfo{};
		break;
	case FramebufferAttachmentPoint::COLOR0:
	case FramebufferAttachmentPoint::COLOR1:
	case FramebufferAttachmentPoint::COLOR2:
	case FramebufferAttachmentPoint::COLOR3:
	case FramebufferAttachmentPoint::COLOR4:
	case FramebufferAttachmentPoint::COLOR5:
	case FramebufferAttachmentPoint::COLOR6:
	case FramebufferAttachmentPoint::COLOR7:
	case FramebufferAttachmentPoint::COLOR8:
	case FramebufferAttachmentPoint::COLOR9:
	case FramebufferAttachmentPoint::COLOR10:
	case FramebufferAttachmentPoint::COLOR11:
	case FramebufferAttachmentPoint::COLOR12:
	case FramebufferAttachmentPoint::COLOR13:
	case FramebufferAttachmentPoint::COLOR14:
	case FramebufferAttachmentPoint::COLOR15:
		uint32_t color_attachment_entry =
			static_cast<uint32_t>(static_cast<GLenum>(attachment_point)-static_cast<GLenum>(FramebufferAttachmentPoint::COLOR0));
		color_attachments[color_attachment_entry] = OwnedFramebufferAttachmentInfo{};

		GLint currently_bound_framebuffer;
		glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &currently_bound_framebuffer);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, ogl_framebuffer_id);
		
		//Define draw color buffers
		std::array<GLenum, max_color_attachments> draw_buffer_list;
		for (int i = 0; i < static_cast<int>(color_attachments.size()); ++i)
		{
			if (color_attachments[i])
				draw_buffer_list[i] = GL_COLOR_ATTACHMENT0 + i;
			else
				draw_buffer_list[i] = GL_NONE;
		}

		GLint max_draw_buffers;
		glGetIntegerv(GL_MAX_DRAW_BUFFERS, &max_draw_buffers);
		glDrawBuffers(max_draw_buffers, draw_buffer_list.data());

		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, currently_bound_framebuffer);
		break;
	}
}


void Framebuffer::setPixelReadSource(FramebufferColorAttachmentPoint read_source)
{
	GLint current_framebuffer;
	glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &current_framebuffer);

	glBindFramebuffer(GL_READ_FRAMEBUFFER, ogl_framebuffer_id);
	glReadBuffer(static_cast<GLenum>(read_source));

	glBindFramebuffer(GL_READ_FRAMEBUFFER, current_framebuffer);
}


void Framebuffer::update()
{
	applyOpenGLContextSettings();		//update context settings
	refresh();		//refresh the contents of the framebuffer
}


void Framebuffer::refresh()
{
	if (!isActive()) return;	//if the framebuffer is not active on the calling thread, do nothing

	//Call rendering sequence
	renderer(*this);
}


bool Framebuffer::is_complete(std::string* p_string_description, bool is_read_complete) const
{
	GLenum framebuffer_binding = is_read_complete ? GL_READ_FRAMEBUFFER_BINDING : GL_DRAW_FRAMEBUFFER_BINDING;
	GLenum framebuffer_target = is_read_complete ? GL_READ_FRAMEBUFFER : GL_DRAW_FRAMEBUFFER;

	GLint ogl_current_framebuffer;
	glGetIntegerv(framebuffer_binding, &ogl_current_framebuffer);

	glBindFramebuffer(framebuffer_target, ogl_framebuffer_id);

	GLenum rv = glCheckFramebufferStatus(framebuffer_target);

	glBindFramebuffer(framebuffer_target, ogl_current_framebuffer);

	if (p_string_description)
	{
		switch (rv)
		{
		case GL_FRAMEBUFFER_COMPLETE:
			*p_string_description = "framebuffer is complete for " + (is_read_complete ? std::string{ "reading" } : std::string{ "drawing" });
			break;

		case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
			*p_string_description = "some of framebuffer attachments are not complete";
			break;

		case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
			*p_string_description = "framebuffer has no valid attachments and is not configured for rendering without attachments";
			break;

		case GL_FRAMEBUFFER_UNSUPPORTED:
			*p_string_description = "the combination of attached image formats is not supported by the video driver";
			break;

		case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
			*p_string_description = "incomplete multisampling parameters: probably some textures have fixed locations of the samples and some of them do not";
			break;
		}
	}

	return rv == GL_FRAMEBUFFER_COMPLETE;
}


bool Framebuffer::isReadComplete(std::string* p_description_buf /* = nullptr */) const
{
	return is_complete(p_description_buf, true);
}


bool Framebuffer::isDrawComplete(std::string* p_description_buf /* = nullptr */) const
{
	return is_complete(p_description_buf, false);
}


bool Framebuffer::isScreenBasedDevice() const { return false; }


Framebuffer& Framebuffer::operator=(const Framebuffer& other)
{
	//account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	AbstractRenderingDevice::operator=(other);

	//Copy settings from the source framebuffer
	stencil_attachment = OwnedFramebufferAttachmentInfo{};
	if (other.stencil_attachment)
		attachTexture(FramebufferAttachmentPoint::STENCIL, other.stencil_attachment);

	depth_attachment = OwnedFramebufferAttachmentInfo{};
	if (other.depth_attachment)
		attachTexture(FramebufferAttachmentPoint::DEPTH, other.depth_attachment);

	stencil_depth_attachment = OwnedFramebufferAttachmentInfo{};
	if (other.stencil_depth_attachment)
		attachTexture(FramebufferAttachmentPoint::STENCIL_DEPTH, other.stencil_depth_attachment);

	for (int i = 0; i < static_cast<int>(color_attachments.size()); ++i)
	{
		if (!other.color_attachments[i]) continue;

		color_attachments[i] = OwnedFramebufferAttachmentInfo{};
		FramebufferAttachmentPoint color_attachment_entry =
			static_cast<FramebufferAttachmentPoint>(static_cast<GLenum>(FramebufferAttachmentPoint::COLOR0) + i);

		attachTexture(color_attachment_entry, other.color_attachments[i]);
	}

	context_framebuffer = other.context_framebuffer;
	renderer = other.renderer;

	return *this;
}

Framebuffer& Framebuffer::operator=(Framebuffer&& other)
{
	//account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	AbstractRenderingDevice::operator=(std::move(other));

	//release currently owned OpenGL source responsible for the framebuffer
	if (ogl_framebuffer_id) glDeleteFramebuffers(1, &ogl_framebuffer_id);

	//capture OpenGL framebuffer from the move-assignment source
	ogl_framebuffer_id = other.ogl_framebuffer_id;
	other.ogl_framebuffer_id = 0;

	//Move settings to the destination framebuffer
	stencil_attachment = std::move(other.stencil_attachment);
	depth_attachment = std::move(other.depth_attachment);
	stencil_depth_attachment = std::move(other.stencil_depth_attachment);
	color_attachments = std::move(other.color_attachments);

	context_framebuffer = std::move(other.context_framebuffer);

	renderer = std::move(other.renderer);

	return *this;
}