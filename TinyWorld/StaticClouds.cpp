#include "StaticClouds.h"

using namespace tiny_world;

void StaticClouds::applyScreenSize(const uvec2& screen_size)
{
}

bool StaticClouds::configureRendering(AbstractRenderingDevice& render_target, uint32_t rendering_pass)
{
    if (!render_target.isActive())
        render_target.makeActive();

    //Bind object's data buffer
    glBindVertexArray(ogl_vertex_attribute_project_id);

    COMPLETE_SHADER_PROGRAM_CAST(retrieveShaderProgram(rendering_program_ref_code)).activate();
}

void StaticClouds::configureViewProjectionTransform(const AbstractProjectingDevice& projecting_device)
{
}
