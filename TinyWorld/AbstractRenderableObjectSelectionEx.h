#ifndef TW__ABSTRACT_RENDERABLE_OBJECT_SELECTION_EX__

#include <list>

#include "AbstractRenderableObject.h"
#include "Framebuffer.h"

namespace tiny_world
{
	//Implements selection buffer. Note that this extension uses COLOR5 framebuffer channel to output selection buffer data
	class AbstractRenderableObjectSelectionEx : virtual public AbstractRenderableObject
	{
	private:
		std::list<ShaderProgramReferenceCode> modified_program_ref_code_list;	//list of reference codes of the shader programs maintained by AbstractRenderableObject, which have been modified by injecting selection extension into their code

	protected:

		//Links functionality of the selection extension to a shader program maintained by the drawable object.
		//The function returns 'true' on success and 'false' on failure. Note that functionality of the selection extension can only be linked to the fragment stage of a shader program.
		bool injectExtension(const ShaderProgramReferenceCode& program_ref_code, std::initializer_list<PipelineStage> program_stages) override;

		//Assigns values to the uniforms used by extension's infrastructure.
		void applyExtension() override;

		//Informs extension about the coordinate transform that converts the world space into the viewer space
		void applyViewerTransform(const AbstractProjectingDevice& projecting_device) override;

		//Allows extension to release the resources it has allocated before the rendering. This function is invoked automatically immediately after the drawing commands have been executed
		//This is especially useful when extension makes use of image units, since ImageUnit object should be released as soon as possible upon completion of the rendering to vacate the 
		//resource it holds
		void releaseExtension() override;


		AbstractRenderableObjectSelectionEx();	//default initialization constructor
		AbstractRenderableObjectSelectionEx(const AbstractRenderableObjectSelectionEx& other);	//copy initialization constructor
		AbstractRenderableObjectSelectionEx(AbstractRenderableObjectSelectionEx&& other);	//move initialization constructor
		AbstractRenderableObjectSelectionEx& operator=(const AbstractRenderableObjectSelectionEx& other);	//copy assignment operator
		AbstractRenderableObjectSelectionEx& operator=(AbstractRenderableObjectSelectionEx&& other);	//move assignment operator

	public:
		virtual ~AbstractRenderableObjectSelectionEx();	//destructor

		//Returns strong identifier of the abstract renderable object, which covers point with window coordinates provided via 'uv2ScreenPoint'.
		//Note that in order for the function to work correctly the target framebuffer referenced by the input parameter 'target_framebuffer' must have
		//the selection buffer texture attached to the 5-th color channel output (i.e. COLOR5). The dimensions of this texture must be equal to that of the viewport
		//used for the rendering in the target framebuffer, and the format of the texture must be either integer, unsigned integer, or floating point
		static unsigned long long getPointSelectionObjectId(const Framebuffer& target_framebuffer, const uvec2& uv2ScreenPoint);
	};
}

#define TW__ABSTRACT_RENDERABLE_OBJECT_SELECTION_EX__
#endif