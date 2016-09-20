//Implements HDR-Bloom extension of AbtractRenderableObject class

#ifndef TW__ABSTRACT_RENDERABLE_OBJECT_HDR_BLOOM_EX__


#include "AbstractRenderableObject.h"

namespace tiny_world
{
	//Implements abstract drawable geometry that can be post-processed by HDR bloom filter
	//Note, that this class does not itself implement HDR-Bloom post processing, but only defines
	//infrastructure for objects that provide output information needed for such filtering.
	//Namely, objects that support HDR-Bloom are requested to have two color outputs at locations 0 and 1,
	//where location 0 is used for usual fragment shading and location 1 is reserved to contain "bloom" regions.
	//The values at location 1 must be computed by calling fragment shader stage function having following calling syntax:
	//								vec4 computeBloomFragment(vec4 v4FragmentColor),
	//where v4FragmentColor is the normal fragment color stored at output location 0.
	//Function computeBloomFragment(...) is automatically embedded into the fragment shader of each object, which is inherited 
	//from this class and properly attached to its infrastructure.
	class AbstractRenderableObjectHDRBloomEx : virtual public AbstractRenderableObject
	{
	private:
		float bloom_min_threshold;	//minimal threshold for extracting the brightest areas from the set of fragments covered by the object. Defaults to 0.8.
		float bloom_max_threshold;	//maximal threshold for extracting the brightest areas from the set of fragments covered by the object. Defaults to 1.2.
		float bloom_intensity;		//intensity factor of the bloom effect. Higher values lead to outputs with "wider" light ranges. Default value is 4.
		int bloom_enabled;			//if equals 1 bloom filtering is enabled. If equals 0, then bloom filtering is disabled. Default value is 1.

		std::list<ShaderProgramReferenceCode> modified_program_ref_code_list;		//list of reference codes of the shader programs that have been modified by injecting HDR-Bloom shaders into them

	protected:
		//Links HDR-Bloom functionality to a shader program maintained by the drawable object.
		//The function returns 'true' on success and 'false' on failure. Note that HDR bloom functionality can only be linked to the fragment shader stage 
		//of a shader program.
		bool injectExtension(const ShaderProgramReferenceCode& program_ref_code, std::initializer_list<PipelineStage> program_stages) override;

		//Assigns values to the uniforms used by HDR-Bloom infrastructure.
		void applyExtension() override;

		//Informs extension about the coordinate transform that converts the world space into the viewer space
		void applyViewerTransform(const AbstractProjectingDevice& projecting_device) override;

		//Allows extension to release the resources it has allocated before the rendering. This function is invoked automatically immediately after the drawing commands have been executed
		//This is especially useful when extension makes use of image units, since ImageUnit object should be released as soon as possible upon completion of the rendering to vacate the 
		//resource it holds
		void releaseExtension() override;

		AbstractRenderableObjectHDRBloomEx();		//Default initialization of an object that supports HDR-Bloom
		AbstractRenderableObjectHDRBloomEx(const AbstractRenderableObjectHDRBloomEx& other);	//Copy construction
		AbstractRenderableObjectHDRBloomEx& operator=(const AbstractRenderableObjectHDRBloomEx& other);	//Copy assignment

	public:
		virtual ~AbstractRenderableObjectHDRBloomEx();	//Destructor

		//The following functions allow to alter settings of the bloom effect

		//Sets minimal threshold of the bloom filter. Assigning smaller values to this property leads to "brighter" appearance of the object
		void setBloomMinimalThreshold(float threshold);

		//Sets maximal threshold of the bloom filter. Assigning smaller values to this property visually reduces amount of light in the scene.
		void setBloomMaximalThreshold(float threshold);

		//Enables or disables output of the bloom regions.
		void useBloom(bool bloom_enable_state);

		//Sets intensity of bloom effect. Higher intensity values lead to output with wider color ranges.
		void setBloomIntensity(float bloom_intensity);


		//The following functions are used to retrieve state of the object

		float getBloomMinimalThreshold() const;		//returns minimal threshold of the bloom filter
		float getBloomMaximalThreshold() const;		//returns maximal threshold of the bloom filter
		bool isBloomInUse() const;	//returns 'true' if output of the bloom regions is enabled
		float getBloomIntensity() const;	//returns intensity of the bloom effect
	};
}


#define TW__ABSTRACT_RENDERABLE_OBJECT_HDR_BLOOM_EX__
#endif