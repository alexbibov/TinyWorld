#ifndef TW__STATIC_CLOUDS__
#define TW__STATIC_CLOUDS__

#include "AbstractRenderableObject.h"
#include "AbstractRenderableObjectHDRBloomEx.h"
#include "AbstractRenderableObjectSelectionEx.h"
#include "LightingConditions.h"
#include "std140UniformBuffer.h"

#include <random>

namespace tiny_world
{
class StaticClouds final : virtual public AbstractRenderableObject,
    public AbstractRenderableObjectExtensionAggregator<AbstractRenderableObjectHDRBloomEx, AbstractRenderableObjectSelectionEx>
{
private:
    static const std::string rendering_program_name;
    static char vp_source[];
    static char fp_source[];

    GLuint ogl_vertex_attribute_project_id;    //native OpenGL vertex attribute object

    ShaderProgramReferenceCode rendering_program_ref_code;    //reference code of the clouds shading program

    void applyScreenSize(const uvec2& screen_size) override;
    bool configureRendering(AbstractRenderingDevice& render_target, uint32_t rendering_pass) override;
    void configureViewProjectionTransform(const AbstractProjectingDevice& projecting_device) override;
    bool configureRenderingFinalization() override;


    inline void setup_object();	//runs initial procedures required on object's initialization

public:
    StaticClouds();

    //Initializes clouds simulation given the spatial size and the resolution of the simulation domain
    StaticClouds(float cloud_domain_size_x, float cloud_domain_size_y, float cloud_domain_size_z, uint32_t cloud_domain_resolution_x = 512, uint32_t cloud_domain_resolution_y = 512,
        uint32_t cloud_domain_resolution_z = 64);

    //Copy initialization
    StaticClouds(const StaticClouds& other);

    //Move initialization
    StaticClouds(StaticClouds&& other);

    //Copy assignment
    StaticClouds& operator=(const StaticClouds& other);

    //Move assignment
    StaticClouds& operator=(StaticClouds&& other);

    //Destructor
    ~StaticClouds();


    void setDomainDimensions(float cloud_domain_size_x, float cloud_domain_size_y, float cloud_domain_size_z);	//sets spatial dimensions of the cloud domain
    void setDomainDimensions(const vec3& cloud_domain_size);	//sets spatial dimensions of the cloud domain
    vec3 getDomainDimensions() const;	//retrieves spatial dimensions of the cloud domain


   //Standard infrastructure of a drawable object

    bool supportsRenderingMode(uint32_t rendering_mode) const override;
    uint32_t getNumberOfRenderingPasses(uint32_t rendering_mode) const override;
    bool render() override;
};

}

#endif