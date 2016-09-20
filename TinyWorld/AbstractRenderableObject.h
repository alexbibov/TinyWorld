#ifndef TW__ABSTRACT_RENDERABLE_OBJECT__

//Disable the warning regarding inheritance via dominance for Microsoft compilers
//Explanation: there is a "strange" behavior of Microsoft compilers, in accordance with which
//compiler applies name resolution via dominance also to pure virtual functions.
//The dominance is used to inject GLSL extensions into drawable objects, so the only workaround for now
//is to suppress the corresponding warning.
#ifdef _MSC_VER
#pragma warning(disable:4250)
#endif

#include <map>
#include <list>
#include <iterator>
#include <initializer_list>
#include <functional>

#include "ShaderProgram.h"
#include "CompleteShaderProgram.h"
#include "SeparateShaderProgram.h"
#include "MatrixTypes.h"
#include "QuaternionTypes.h"
#include "TextureUnitBlock.h"
#include "ImmutableTexture1D.h"
#include "ImmutableTexture2D.h"
#include "ImmutableTexture3D.h"
#include "ImmutableTextureCubeMap.h"
#include "BufferTexture.h"
#include "AbstractProjectingDevice.h"
#include "AbstractRenderingDevice.h"
#include "Entity.h"

namespace tiny_world
{
	//Describes concept of an abstract object, which can be rendered
	class AbstractRenderableObject : public Entity
	{
	protected:
		struct ShaderProgramReferenceCode
		{
			int first, second;
			ShaderProgramReferenceCode();	//default initializer: initializes shader program reference code with default pair (-1, -1) representing invalid reference code
			ShaderProgramReferenceCode(int first, int second);
			operator bool() const;
		};

	private:
		typedef std::map<uint32_t, CompleteShaderProgram> complete_shader_program_list;
		typedef std::map<uint32_t, SeparateShaderProgram> separate_shader_program_list;
		enum class supported_shader_program_type : int { unsupported = -1, complete = 0, separate = 1 };


		const vec3 default_location;	//default location of the object
		vec3 location;	//location of the object in the world space (defaults to the world space's origin)
		vec3 scale_factors;	//contains object scale factors defined in the object space
		const mat3 default_object_transform;	//default transformation of the object
		mat3 object_transform;	//object coordinate transform (does NOT take translation into account!)
		uint32_t rendering_mode;	//currently selected rendering mode of the object. Default value is 0
		uvec2 screen_size;	//size of the render target

		complete_shader_program_list complete_rendering_programs;	//list of complete rendering programs implemented by object
		separate_shader_program_list separate_rendering_programs;	//list of separate rendering programs implemented by object

		//Customizable part of the setScreenSize(...) functions, which must be implemented by the inherited classes
		virtual void applyScreenSize(const uvec2& screen_size) = 0;

		//Performs custom rendering configuration needed to visualize the object properly. This function depends on the nature of drawable object and thus
		//has to be implemented by the most inherited class of the infrastructure
		virtual bool configureRendering(AbstractRenderingDevice& render_target, uint32_t rendering_pass) = 0;

		//Allows to perform custom configuration steps based on the projecting device used to visualize the object
		virtual void configureViewProjectionTransform(const AbstractProjectingDevice& projecting_device) = 0;

		//Performs custom finalization of the rendering. This function depends on what particular finalization is required by certain drawable object.
		//Therefore the implementation must be provided by the most derived object in the inheritance chain
		virtual bool configureRenderingFinalization() = 0;

	protected:
		class ShaderProgramIterator : public std::iterator < std::bidirectional_iterator_tag, ShaderProgram >
		{
		private:
			uint32_t position;
			complete_shader_program_list* p_complete_programs;
			separate_shader_program_list* p_separate_programs;

		public:
			ShaderProgramIterator();
			ShaderProgramIterator(complete_shader_program_list& complete_rendering_programs, separate_shader_program_list& separate_rendering_programs, uint32_t position);

			ShaderProgramIterator& operator++();
			ShaderProgramIterator operator++(int);

			ShaderProgramIterator& operator--();
			ShaderProgramIterator operator--(int);

			ShaderProgram& operator*();
			ShaderProgram* operator->();

			bool operator==(const ShaderProgramIterator& other) const;
			bool operator!=(const ShaderProgramIterator& other) const;
		};

		//Injects extension to the given shader program. This function must be implemented by each inherited class that embodies an
		//extension providing certain additional functionality for the shaders employed by the drawable object. The function must return
		//'true' on successful injection and 'false' on failure
		virtual bool injectExtension(const ShaderProgramReferenceCode& program_ref_code, std::initializer_list<PipelineStage> program_stages) = 0;

		//Applies properties related to an extension to all shader programs, which have this extension injected. This function must be implemented
		//by all inherited classes providing additional functionality for GLSL shaders.
		virtual void applyExtension() = 0;

		//Informs extension about the coordinate transform that converts the world space into the viewer space
		virtual void applyViewerTransform(const AbstractProjectingDevice& projecting_device) = 0;

		//Allows extension to release the resources it has allocated before the rendering. This function is invoked automatically immediately after the drawing commands have been executed
		//This is especially useful when extension makes use of image units, since ImageUnit object should be released as soon as possible upon completion of the rendering to vacate the 
		//resource it holds
		virtual void releaseExtension() = 0;


		//Returns iterator pointing to the first element in the list of rendering programs maintained by the object
		ShaderProgramIterator ShaderProgramListBegin();

		//Returns iterator pointing to position after the last element in the list of rendering programs maintained by the object
		ShaderProgramIterator ShaderProgramListEnd();

		//Creates new complete shader program with given string name and adds it to the list of rendering programs maintained by the object. 
		//The function returns reference code of the newly created program, which allows to access it in future. In case of failure, the function
		//returns an incorrect reference code, which can be identified by its boolean value that is equal to 'false'.
		ShaderProgramReferenceCode createCompleteShaderProgram(const std::string& program_string_name, std::initializer_list<PipelineStage> program_stages);

		//Creates new separate shader program with given string name and adds it to the list of rendering programs maintained by the object. 
		//The function returns reference code of the newly created program, which allows to access it in future. In case of failure, the function
		//returns an incorrect reference code, which can be identified by its boolean value that is equal to 'false'.
		ShaderProgramReferenceCode createSeparateShaderProgram(const std::string& program_string_name, std::initializer_list<PipelineStage> program_stages);

		//Retrieves pointer to a shader program from the list of rendering programs maintained by the object using provided shader program reference code.
		ShaderProgram* retrieveShaderProgram(const ShaderProgramReferenceCode& shader_program_ref_code);

		//Updates shader program referred by the given shader program reference code. Returns 'true' on success
		bool updateShaderProgram(const ShaderProgramReferenceCode& shader_program_ref_code, const ShaderProgram& new_shader_program);


		
		//AbstractRenderableObject can not be created explicitly

		//Default initialization: called only by the intermediate objects in the chain of inheritance that never get instantiated 
		AbstractRenderableObject();

		//Default initialization: constructs object located at the origin of the world space and having default orientation
		explicit AbstractRenderableObject(const std::string& renderable_object_class_string_name);

		//Creates new object at default location and with default orientation using provided string name
		AbstractRenderableObject(const std::string& renderable_object_class_string_name, const std::string& object_string_name);
		
		//Creates object with given location
		AbstractRenderableObject(const std::string& renderable_object_class_string_name, const std::string& object_string_name, const vec3& location);

		//Creates object and rotates it around x-, around y- and finally around z-axis (in this order) in object space by the given angles represented in radians
		AbstractRenderableObject(const std::string& renderable_object_class_string_name, const std::string& object_string_name, const vec3& location, 
			float x_rot_angle, float y_rot_angle, float z_rot_angle);
		

		AbstractRenderableObject(const AbstractRenderableObject& other);	//copy constructor
		AbstractRenderableObject(AbstractRenderableObject&& other);	//move constructor
		AbstractRenderableObject& operator=(const AbstractRenderableObject& other);		//assignment operator
		AbstractRenderableObject& operator=(AbstractRenderableObject&& other);	//move assignment operator

	public:
		virtual ~AbstractRenderableObject();	//Destructor

		vec3 getLocation() const;			//Returns location of the object
		mat4 getObjectTransform() const;	//Returns object coordinate transform, which converts model-space coordinates into world-space coordinates
		
		//Returns object scale transform. The returned matrix transforms coordinates from the raw object modeling space into 
		//the scaled model space. In the raw object modeling space coordinates are allowed to use length scales that are different
		//from those used to describe the world space. Scaling is important as all lighting-related computations are performed
		//in scaled coordinates.
		mat4 getObjectScaleTransform() const; 

		vec3 getObjectScale() const;	//Returns scale factors of the object represented in the model space

		void setLocation(const vec3& new_location);			//Sets new object location
		void rotateX(float angle, RotationFrame frame);		//rotates object around its x-axis. NOTE: rotation is performed in object space
		void rotateY(float angle, RotationFrame frame);		//rotates object around its y-axis. NOTE: rotation is performed in object space
		void rotateZ(float angle, RotationFrame frame);		//rotates object around its z-axis. NOTE: rotation is performed in object space
		void rotate(const vec3& axis, float angle, RotationFrame frame);		//rotates object around an arbitrary axis, where the axis is given in world space coordinates. NOTE: rotation is performed in world space.
		void translate(const vec3& translation);		//translates object in world space by adding translation vector to the current object's location. Here translation vector must be represented in world space coordinates
		void scale(float x_scale_factor, float y_scale_factor, float z_scale_factor);		//scales object in object space
		void scale(const vec3& new_scale_factors);		//applies scaling transform defined by the given vector in object space
		void apply3DTransform(const mat3& _3d_transform_matrix, RotationFrame frame);	//applies 3D transformation to the object. Note that if frame is GLOBAL the transformation is applied in world coordinates. If frame is LOCAL, then it is applied in the object space.
		void applyRotation(const quaternion& q, RotationFrame frame);	//applies rotation represented by quaternion q to the object
		void resetObjectTransform();	//resets object transformation to the value, which was provided on initialization of the object
		void resetObjectRotation();		//resets rotation part of the object transform to the value, which was provided on initialization of the object
		void resetObjectLocation();		//resets location part of the object transform to the value, which was provided on initialization of the object
		uint32_t selectRenderingMode(uint32_t new_rendering_mode);	//changes currently selected rendering mode of the object. Returns previously active rendering mode
		uint32_t getActiveRenderingMode() const;	//returns active rendering mode of the object

		void setScreenSize(const uvec2& screen_size);	//sets size of the render target to which the object is drawn
		void setScreenSize(uint32_t width, uint32_t height);	//sets width and height of the render target to which the object is drawn
		uvec2 getScreenSize() const;	//returns current size of the render target to which the object is drawn
		 
		virtual bool supportsRenderingMode(uint32_t rendering_mode) const = 0;	//Returns 'true' if requested rendering mode is supported by the object. Returns 'false' otherwise
		void applyViewProjectionTransform(const AbstractProjectingDevice& projecting_device);	//Informs the object about the projection device, which is used to view it
		virtual uint32_t getNumberOfRenderingPasses(uint32_t rendering_mode) const = 0;	//Retrieves the number of rendering passes needed to properly draw the object for requested rendering mode
		bool prepareRendering(AbstractRenderingDevice& render_target, uint32_t rendering_pass);	//Prepares object for rendering. Returns 'true' on success and 'false' otherwise. The function is allowed to change the state of the provided render target
		virtual bool render() = 0;	//This function should perform all the tasks related to object rendering (although, that does not mean that anything actually gets drawn in the viewport). Returns 'true' on success, 'false' otherwise
		bool finalizeRendering();	//Performs post-rendering tasks. Returns 'true' on success and 'false' otherwise. If prepareRendering(...) has altered the state of provided render target, it is expected that finalizeRendering() will restore the original state
	};




	//Describes an abstract drawable object that uses texture unit block
	class AbstractRenderableObjectTextured : virtual public AbstractRenderableObject
	{
	protected:
		struct TextureReferenceCode
		{
			//Here "first" contains offset from the beginning of the storage, and "second" contains numeric code of the storage
			int first, second;

			TextureReferenceCode();	//default initializer (initializes texture reference code to (-1, -1))
			TextureReferenceCode(int first, int second);	//initializes the first and the second elements of the pair during object's construction
			operator bool() const;	//returns 'true' if the object represents a valid texture reference code
		};

		class TextureSamplerReferenceCode
		{
		private:
			int first;	//contains offset from the beginning of the storage

		public:
			TextureSamplerReferenceCode();	//default initializer of the sampler reference code (creates invalid reference)
			TextureSamplerReferenceCode(int first);	//initializes sampler reference code using provided value

			operator bool() const;	//returns 'true' of the object represents a valid texture sampler reference code
			int getFirst() const;	//returns offset value wrapped by the reference code
		};


		template<typename T>
		using TextureResource = std::pair < T, TextureReferenceCode > ;

		typedef TextureResource<ImmutableTexture1D> Texture1DResource;
		typedef TextureResource<ImmutableTexture2D> Texture2DResource;
		typedef TextureResource<ImmutableTexture3D> Texture3DResource;
		typedef TextureResource<ImmutableTextureCubeMap> TextureCubeMapResource;
		typedef TextureResource<BufferTexture> TextureBufferResource;


	private:
		template<typename TT> using texture_entry = triplet < uint32_t, TT, TextureSamplerReferenceCode > ;	//describes a single entry in a list of textures attached to the object
		typedef std::vector<texture_entry<ImmutableTexture1D>> _1d_texture_list;	//describes list of immutable 1D textures attached to the object
		typedef std::vector<texture_entry<ImmutableTexture2D>> _2d_texture_list;	//describes list of immutable 2D textures attached to the object
		typedef std::vector<texture_entry<ImmutableTexture3D>> _3d_texture_list;	//describes list of immutable 3D textures attached to the object
		typedef std::vector<texture_entry<ImmutableTextureCubeMap>> cubemap_texture_list;	//describes list of cubemap textures attached to the object
		typedef std::vector<texture_entry<BufferTexture>> buffer_texture_list;	//describes list of buffer textures attached to the object
		typedef std::vector<TextureSampler> texture_sampler_list;	//describes list of texture samplers maintained by the object

		//describes textures currently supported by textured objects of the TinyWorld Graphical Engine
		enum class supported_texture_type : int { unsupported = -1, cubemap = 0, _1d_texture = 1, _2d_texture = 2, _3d_texture = 3, buffer_texture = 4 };

		static TextureUnitBlock *p_texture_unit_block;	//stores global pointer to singleton object representing texture unit block
		static std::string texture_lookup_path;	//default path to look for the textures (used by some objects)

		_1d_texture_list _1d_textures;	//1D-textures attached to the object
		_2d_texture_list _2d_textures;	//2D-textures attached to the object
		_3d_texture_list _3d_textures;	//3D-textures attached to the object
		cubemap_texture_list cubemap_textures;	//cubemap textures attached to the object
		buffer_texture_list buffer_textures; //buffer textures attached to the object 
		texture_sampler_list samplers;	//texture samplers maintained by the object

		TextureSamplerReferenceCode default_sampler_ref_code;	//specifies reference code of the default texture sampler, which is used when no user-defined sampler is provided

		uint32_t texture_unit_offset;	//first texture unit, which can be used by the object
		uint32_t texture_unit_counter;	//stores identifier of the next vacant texture unit

		//Recognizes a texture represented by its abstract base class and returns the corresponding supported texture type
		static supported_texture_type getTextureType(const Texture& texture);


	protected:
		//Attaches a new texture to the object and returns its reference code, which determines where the texture is stored in the object's texture list. 
		//The reference code is needed to update the texture or the associated sampler after it has been attached to the object.
		//Texture sampling parameters are retrieved using the sampler reference code supplied in the second argument of the function.
		//If the sampler reference code is not provided, then the values from the texture are sampled using default sampler object, which has the following configuration:
		//Magnification filter: LINEAR
		//Minification filter: LINEAR_MIPMAP_NEAREST
		//Wrapping mode: CLAMP_TO_EDGE for R-, S- and T- texture coordinates
		//CLAMP_TO_EDGE boundary resolution.
		//If the function did not manage to recognize the texture object, it returns an invalid reference code
		TextureReferenceCode registerTexture(const Texture& texture, TextureSamplerReferenceCode sampler_reference_code = TextureSamplerReferenceCode{});

		//Allows to explicitly declare desired binding unit for the texture
		TextureReferenceCode registerTexture(const Texture& texture, uint32_t unit_binding_block, TextureSamplerReferenceCode sampler_reference_code = TextureSamplerReferenceCode{});

		//Replaces texture and sampler that were previously attached to the object using the reference code of the texture.
		//If sampler object is not provided, the old sampler gets coupled with the new texture. It is allowed for texture reference codes to change the type of the texture
		//they refer to. However, as long as the new reference code was obtained using updateTexture(), it points to the same texture binding unit as the old one.
		//Therefore, usage of the old and the new texture reference codes to bind textures to GLSL sampler objects that have different types will lead to 
		//incorrect behavior of the OpenGL API. 
		//The function returns 'true' on success or 'false' in case of error
		//REMARKS: Note that reference_code is an in/out argument. If the texture being updated differs in type with the replacement texture, 
		//the texture reference code gets modified. Otherwise it remains unchanged.
		bool updateTexture(TextureReferenceCode& texture_reference_code, const Texture& new_texture, TextureSamplerReferenceCode sampler_reference_code = TextureSamplerReferenceCode{});

		//Updates sampler for the given texture reference code
		bool updateSampler(const TextureReferenceCode& texture_reference_code, TextureSamplerReferenceCode sampler_reference_code = TextureSamplerReferenceCode{});

		//Returns binding unit corresponding to the texture determined by the given storage reference code. The function returns -1 if requested reference code is invalid.
		int getBindingUnit(TextureReferenceCode texture_reference_code) const;


		//Creates new texture sampler object, which will be maintained by the infrastructure provided by AbstractRenderableObjectTextured
		TextureSamplerReferenceCode createTextureSampler(const std::string& string_name, SamplerMagnificationFilter magnification_filter = SamplerMagnificationFilter::LINEAR,
			SamplerMinificationFilter minification_filter = SamplerMinificationFilter::LINEAR_MIPMAP_NEAREST, 
			SamplerWrapping wrapping = SamplerWrapping{ SamplerWrappingMode::CLAMP_TO_EDGE, SamplerWrappingMode::CLAMP_TO_EDGE, SamplerWrappingMode::CLAMP_TO_EDGE }, 
			const vec4& v4BorderColor = vec4(0.0f));

		//Returns pointer to the texture sampler object corresponding to the given texture sampler reference code
		TextureSampler* retrieveTextureSampler(TextureSamplerReferenceCode sampler_reference_code);



		//Sets texture unit identifier starting at which it is allowed to bind object textures.
		//This means that the first texture will be bound to texture unit with id "offset", the second — to texture unit with id "offset + 1" and so on.  
		//This function should be called when some texture units should be reserved for the needs of derived objects.
		void setTextureUnitOffset(uint32_t offset);

		void bindTextures() const;	//binds all registered textures to the corresponding texture units using the corresponding sampler objects
		bool bindTexture(TextureReferenceCode reference_code) const;	//binds texture with given reference code in the corresponding texture list of the object


		//Explicit creation of abstract drawable textured objects is not allowed

		AbstractRenderableObjectTextured();		//default initialization of a textured object
		AbstractRenderableObjectTextured(const AbstractRenderableObjectTextured& other);	//copy constructor
		AbstractRenderableObjectTextured(AbstractRenderableObjectTextured&& other);		//move constructor
		AbstractRenderableObjectTextured& operator=(const AbstractRenderableObjectTextured& other);		//copy-assignment operator
		AbstractRenderableObjectTextured& operator=(AbstractRenderableObjectTextured&& other);	//move-assignment operator
		

	public:
		static void defineTextureUnitBlockGlobalPointer(TextureUnitBlock* p_texture_unit_block);
		static TextureUnitBlock* getTextureUnitBlockPointer();

		static void defineTextureLookupPath(const std::string& new_texture_lookup_path);
		static std::string getTextureLookupPath();


		//Helper functions

		uint32_t getNumberOfCubemaps() const;	//returns the number of cubemap textures currently registered on the object

		uint32_t getNumberOf1DTextures() const;	//returns the number of 1D textures currently registered on the object

		uint32_t getNumberOf2DTextures() const;	//returns the number of 2D textures currently registered on the object

		uint32_t getNumberOf3DTextures() const;	//returns the number of 3D textures currently registered on the object

		uint32_t getNumberOfBufferTextures() const;	//returns the number of buffer textures currently registered on the object

		uint32_t getNumberOfTextures() const;	//returns total number of textures currently registered for the object

		virtual ~AbstractRenderableObjectTextured();
	};




	//The following templates are intended to simplify injection of extended functionality into drawable objects


	template<typename... ExtensionClasses>
	class AbstractRenderableObjectExtensionAggregator : virtual public AbstractRenderableObject
	{
	protected:

		bool injectExtension(const ShaderProgramReferenceCode& program_ref_code, std::initializer_list<PipelineStage> program_stages) override
		{
			return true;
		}

		void applyExtension() override
		{

		}

		void applyViewerTransform(const AbstractProjectingDevice& projecting_device) override
		{

		}

		void releaseExtension() override
		{

		}



		AbstractRenderableObjectExtensionAggregator()
		{

		}


		AbstractRenderableObjectExtensionAggregator(const AbstractRenderableObjectExtensionAggregator& other)
		{

		}


		AbstractRenderableObjectExtensionAggregator& operator=(const AbstractRenderableObjectExtensionAggregator& other)
		{
			return *this;
		}
	};


	template<typename Head, typename... ExtensionClasses>
	class AbstractRenderableObjectExtensionAggregator<Head, ExtensionClasses...> : public Head, public AbstractRenderableObjectExtensionAggregator <ExtensionClasses... >
	{
	private:
		template <bool B, typename T = void> using disable_if = std::enable_if<!B, T>;

	protected:
		AbstractRenderableObjectExtensionAggregator() : Head(), AbstractRenderableObjectExtensionAggregator<ExtensionClasses...>()
		{

		}

		AbstractRenderableObjectExtensionAggregator(const AbstractRenderableObjectExtensionAggregator& other) : Head(other), AbstractRenderableObjectExtensionAggregator<ExtensionClasses...>(other)
		{

		}

		AbstractRenderableObjectExtensionAggregator(AbstractRenderableObjectExtensionAggregator&& other) : 
			Head(std::move(other)),
			AbstractRenderableObjectExtensionAggregator<ExtensionClasses...>(std::move(other))
		{

		}

		template<typename Arg0, typename... Args, typename = 
			typename disable_if<sizeof...(Args)==0 && std::is_convertible<Arg0, AbstractRenderableObjectExtensionAggregator>::value, int>::type>
		AbstractRenderableObjectExtensionAggregator(const Arg0& arg0, const Args&... args) : Head(arg0), AbstractRenderableObjectExtensionAggregator<ExtensionClasses...>(args...)
		{

		}


		bool injectExtension(const ShaderProgramReferenceCode& program_ref_code, std::initializer_list<PipelineStage> program_stages) override
		{
			bool rv1 = Head::injectExtension(program_ref_code, program_stages);
			bool rv2 = AbstractRenderableObjectExtensionAggregator<ExtensionClasses...>::injectExtension(program_ref_code, program_stages);
			return rv1 && rv2;
		}

		void applyExtension() override
		{
			Head::applyExtension();
			AbstractRenderableObjectExtensionAggregator<ExtensionClasses...>::applyExtension();
		}

		void applyViewerTransform(const AbstractProjectingDevice& projecting_device) override
		{
			Head::applyViewerTransform(projecting_device);
			AbstractRenderableObjectExtensionAggregator<ExtensionClasses...>::applyViewerTransform(projecting_device);
		}

		void releaseExtension()
		{
			Head::releaseExtension();
			AbstractRenderableObjectExtensionAggregator<ExtensionClasses...>::releaseExtension();
		}
	};



	//The following template implements vertex attribute specification
	template<unsigned int vertex_attribute_id, typename vertex_attribute_format, unsigned int vertex_attribute_size, bool special_vertex_attribute_format=false>
	class VertexAttributeSpecification
	{
	private:
		VertexAttributeSpecification(){};	//Explicit creation of objects of this type is not allowed. The object serves as a namespace container only

	public:
		typedef vertex_attribute_format value_type;

		static void setVertexAttributeBufferLayout(uint32_t relative_offset, uint32_t binding_index, bool normalized = false)
		{
			glVertexAttribFormat(vertex_attribute_id, vertex_attribute_size, ogl_type_traits<vertex_attribute_format, special_vertex_attribute_format>::ogl_data_type_enum, normalized, static_cast<GLuint>(relative_offset));
			glVertexAttribBinding(vertex_attribute_id, binding_index);
		}

		static unsigned int getId() { return vertex_attribute_id; }

		static unsigned int getSize() { return vertex_attribute_size; }

		static unsigned int getCapacity() { return vertex_attribute_size*sizeof(value_type); }

		static decltype(ogl_type_traits<vertex_attribute_format, special_vertex_attribute_format>::ogl_data_type_enum) getDataType()
		{
			return ogl_type_traits<vertex_attribute_format, special_vertex_attribute_format>::ogl_data_type_enum;
		}
	};

	typedef VertexAttributeSpecification<0, ogl_type_mapper<float>::ogl_type, 4> vertex_attribute_position;
	typedef VertexAttributeSpecification<1, ogl_type_mapper<float>::ogl_type, 4> vertex_attribute_color;
	typedef VertexAttributeSpecification<2, ogl_type_mapper<float>::ogl_type, 3> vertex_attribute_normal;
	typedef VertexAttributeSpecification<3, ogl_type_mapper<float>::ogl_type, 2> vertex_attribute_texcoord;
	typedef VertexAttributeSpecification<4, ogl_type_mapper<float>::ogl_type, 3> vertex_attribute_texcoord_3d;

	//Defines number of generic vertex attribute IDs reserved for usage by TinyWorld. In other words, all generic attributes 
	//that have IDs with values greater than or equal to the value of this definition are guaranteed to be not used by the TinyWorld engine
	#define TW_RESERVED_VA_IDs 10
}


#define TW__ABSTRACT_RENDERABLE_OBJECT__
#endif