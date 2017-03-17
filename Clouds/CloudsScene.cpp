#include "CloudsScene.h"

#include <iostream>
#include <cstddef>
#include <numeric>


CloudsScene* CloudsScene::p_myself = nullptr;
Screen* CloudsScene::p_screen = nullptr;
const float CloudsScene::pi = 3.1415926535897932384626433832795f;
const vec3 CloudsScene::v3SunLight{ 1.2f, 1.2f, 0.9f };
const vec3 CloudsScene::v3MoonLight{ 0.1f, 0.2f, 0.7f };


CloudsScene* CloudsScene::initializeScene(const std::string& title, uint32_t init_screen_width, uint32_t init_screen_height,
    const std::string& topography_file_name, const std::string& shader_base_catalog, const std::string& texture_base_catalog, uint32_t reflection_map_resolution /* = 1024U */)
{
    if (!p_myself)
    {
        //Initialize screen
        p_screen = new Screen(45, 45, init_screen_width, init_screen_height);
        p_screen->setStringName(title.c_str());
        p_screen->setScreenVideoMode(p_screen->getVideoModes()[0]);
        p_screen->defineViewport(0, 0, static_cast<float>(init_screen_width), static_cast<float>(init_screen_height));
        p_screen->attachRenderer(onScreenRedraw);
        p_screen->registerOnChangeSizeCallback(onScreenSizeChange);
        p_screen->setClearColor(0, 0, 0, 0);

        //Register GLFW event handlers
        glfwSetKeyCallback(*p_screen, onKeyPress);
        glfwSetCursorPosCallback(*p_screen, onMouseMove);

        //Set asset lookup directories
        ShaderProgram::setShaderBaseCatalog(shader_base_catalog);
        AbstractRenderableObjectTextured::defineTextureLookupPath(texture_base_catalog);

        //Create the scene
        p_myself = new CloudsScene(topography_file_name, reflection_map_resolution);
        if (p_myself->error_state)
        {
            delete p_myself;
            p_myself = nullptr;
        }
    }
    return p_myself;
}


void CloudsScene::destroyScene()
{
    if (p_myself)
    {
        delete p_myself;
        delete p_screen;
        p_myself = nullptr;
        p_screen = nullptr;
    }
}


bool CloudsScene::updateScene()
{
    //static uint32_t reflection_update_counter = 0;
    ////Reflection frame buffer must be refreshed 6 times in order to get all six faces of the cube map
    /*if (reflection_update_counter == 0 || reflection_update_counter == 2)
    {
        p_myself->reflection_framebuffer.makeActive();
        p_myself->reflection_framebuffer.refresh();
        p_myself->reflection_framebuffer.refresh();
        p_myself->reflection_framebuffer.refresh();
        p_myself->reflection_framebuffer.refresh();
        p_myself->reflection_framebuffer.refresh();
        p_myself->reflection_framebuffer.refresh();

        p_myself->reflection_map.generateMipmapLevels();

        reflection_update_counter = 1;
    }*/

    p_myself->rendering_composition_framebuffer.makeActive();
    p_myself->rendering_composition_framebuffer.refresh();

    p_screen->makeActive();
    p_screen->refresh();

    //++reflection_update_counter;

    return p_screen->shouldClose();
}


void CloudsScene::onScreenRedraw(Screen& target_screen)
{
    if (p_myself)
    {
        target_screen.clearBuffers(BufferClearTarget::COLOR);
        p_myself->postprocess.pass(p_myself->main_camera, target_screen);
        TwDraw();
    }
}


void CloudsScene::onSceneRedraw(Framebuffer& target_framebuffer)
{
    target_framebuffer.clearBuffers(BufferClearTarget::COLOR_DEPTH);

    p_myself->skydome.selectRenderingMode(TW_RENDERING_MODE_DEFAULT);

    p_myself->skydome.applyViewProjectionTransform(p_myself->main_camera);
    for (uint32_t i = 0; i < p_myself->skydome.getNumberOfRenderingPasses(TW_RENDERING_MODE_DEFAULT); ++i)
    {
        p_myself->skydome.prepareRendering(target_framebuffer, i);
        p_myself->skydome.render();
        p_myself->skydome.finalizeRendering();
    }

    /*float daytime = p_myself->skydome.getDaytime() + 0.0001f;
    daytime -= std::floor(daytime);
    p_myself->skydome.setDaytime(daytime);*/
    p_myself->skydome.setDaytime(p_myself->daytime);

    if (p_myself->skydome.isDay())
    {
        vec2 sun_location = p_myself->skydome.getSunLocation();
        vec3 sun_direction{ -std::cos(sun_location.x)*std::cos(sun_location.y),
            -std::sin(sun_location.x),
            std::cos(sun_location.x)*std::sin(sun_location.y) };
        p_myself->skybody_light.setDirection(sun_direction);

        float fCurrentDaytime = p_myself->skydome.getDaytime();
        vec3 day_light_intencity = fCurrentDaytime <= 0.25f ?
            v3MoonLight + 4 * fCurrentDaytime * (v3SunLight - v3MoonLight) :
            v3SunLight + 4 * (fCurrentDaytime - 0.25f) * (v3MoonLight - v3SunLight);

        p_myself->skybody_light.setColor(day_light_intencity);
    }
    else
    {
        vec2 moon_location = p_myself->skydome.getMoonLocation();
        vec3 moon_direction{ -std::cos(moon_location.x)*std::cos(moon_location.y),
            -std::sin(moon_location.x),
            std::cos(moon_location.x)*std::sin(moon_location.y) };
        p_myself->skybody_light.setDirection(moon_direction);
        p_myself->skybody_light.setColor(v3MoonLight);
    }

    p_myself->tess_terrain.applyViewProjectionTransform(p_myself->main_camera);
    for (uint32_t i = 0; i < p_myself->tess_terrain.getNumberOfRenderingPasses(TW_RENDERING_MODE_DEFAULT); ++i)
    {
        p_myself->tess_terrain.prepareRendering(target_framebuffer, i);
        p_myself->tess_terrain.render();
        p_myself->tess_terrain.finalizeRendering();
    }

    p_myself->clouds.applyViewProjectionTransform(p_myself->main_camera);
    for (uint32_t i = 0; i < p_myself->clouds.getNumberOfRenderingPasses(TW_RENDERING_MODE_DEFAULT); ++i)
    {
        p_myself->clouds.prepareRendering(target_framebuffer, i);
        p_myself->clouds.render();
        p_myself->clouds.finalizeRendering();
    }
}


void CloudsScene::onReflectionMapUpdate(Framebuffer& target_framebuffer)
{
    static int32_t cubemap_pass = 0;
    tiny_world::PerspectiveProjectingDevice aux_camera = p_myself->reflection_camera;
    tiny_world::vec3 v3ReflectionCameraLocation = p_myself->main_camera.getLocation();
    aux_camera.setLocation(v3ReflectionCameraLocation);

    switch (cubemap_pass)
    {
    case 0:
        aux_camera.rotateY(-pi / 2, tiny_world::RotationFrame::LOCAL);
        aux_camera.mirrorXY();
        break;

    case 1:
        aux_camera.rotateY(pi / 2, tiny_world::RotationFrame::LOCAL);
        aux_camera.mirrorXY();
        break;

    case 2:
        aux_camera.rotateX(pi / 2, tiny_world::RotationFrame::LOCAL);
        break;

    case 3:
        aux_camera.rotateX(-pi / 2, tiny_world::RotationFrame::LOCAL);
        break;

    case 4:
        aux_camera.rotateY(pi, tiny_world::RotationFrame::LOCAL);
        aux_camera.mirrorXY();
        break;

    case 5:
        aux_camera.mirrorXY();
        break;
    }
    p_myself->reflection_framebuffer.attachTexture(FramebufferAttachmentPoint::COLOR0,
        FramebufferAttachmentInfo{ 0, cubemap_pass, &p_myself->reflection_map });



    //*****************************************************Do reflection rendering**************************************************
    target_framebuffer.clearBuffers(tiny_world::BufferClearTarget::COLOR_DEPTH);

    p_myself->skydome.selectRenderingMode(TW_RENDERING_MODE_DEFAULT);
    p_myself->skydome.applyViewProjectionTransform(aux_camera);
    for (uint32_t i = 0; i < p_myself->skydome.getNumberOfRenderingPasses(TW_RENDERING_MODE_DEFAULT); ++i)
    {
        p_myself->skydome.prepareRendering(target_framebuffer, i);
        p_myself->skydome.render();
        p_myself->skydome.finalizeRendering();
    }

    float tess_terrain_lod = p_myself->tess_terrain.getLODFactor();
    p_myself->tess_terrain.setLODFactor(200);
    p_myself->tess_terrain.setNormalMapEnableState(false);

    /*for (uint32_t i = 0; i < p_myself->tess_terrain.getNumberOfRenderingPasses(TW_RENDERING_MODE_DEFAULT); ++i)
    {
        p_myself->tess_terrain.applyViewProjectionTransform(aux_camera);
        p_myself->tess_terrain.prepareRendering(target_framebuffer, i);
        p_myself->tess_terrain.render();
        p_myself->tess_terrain.finalizeRendering();
    }*/

    p_myself->tess_terrain.setLODFactor(tess_terrain_lod);
    p_myself->tess_terrain.setNormalMapEnableState(true);

    //****************************************************************************************************************************

    cubemap_pass = (cubemap_pass + 1) % 6;
}


void CloudsScene::onKeyPress(GLFWwindow* p_window, int key, int scancode, int actions, int mods)
{
    float translation_speed = 2.5f;

    if (glfwGetKey(p_window, GLFW_KEY_A) == GLFW_PRESS)
        p_myself->main_camera.translate(tiny_world::vec3{ -translation_speed, 0.0f, 0.0f });
    if (glfwGetKey(p_window, GLFW_KEY_D) == GLFW_PRESS)
        p_myself->main_camera.translate(tiny_world::vec3{ translation_speed, 0.0f, 0.0f });
    if (glfwGetKey(p_window, GLFW_KEY_W) == GLFW_PRESS)
        p_myself->main_camera.translate(tiny_world::vec3{ 0.0f, 0.0f, -translation_speed });
    if (glfwGetKey(p_window, GLFW_KEY_S) == GLFW_PRESS)
        p_myself->main_camera.translate(tiny_world::vec3{ 0.0f, 0.0f, translation_speed });

    TwEventKeyGLFW(key, actions);
}


void CloudsScene::onMouseMove(GLFWwindow* p_window, double xpos, double ypos)
{
    TwEventMousePosGLFW(static_cast<int>(xpos), static_cast<int>(ypos));

    if (!p_myself->suppress_mouse_user_input)
    {
        static double old_xpos, old_ypos;
        static bool _1st_call = true;

        if (_1st_call)
        {
            old_xpos = xpos;
            old_ypos = ypos;
            _1st_call = false;
        }
        else
        {
            if (glfwGetMouseButton(p_window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS)
            {
                int width, height;
                glfwGetWindowSize(p_window, &width, &height);

                double dx = (xpos - old_xpos) / width * 2;
                double dy = (ypos - old_ypos) / height * 2;


                if (abs(dy) > abs(dx))
                    p_myself->main_camera.rotateX(static_cast<float>(dy), RotationFrame::LOCAL);

                if (abs(dy) <= abs(dx))
                    p_myself->main_camera.rotateY(-static_cast<float>(dx), RotationFrame::GLOBAL);

                old_xpos = xpos;
                old_ypos = ypos;
            }
            else
            {
                _1st_call = true;
            }
        }
    }
    p_myself->suppress_mouse_user_input = false;
}


void CloudsScene::onScreenSizeChange(Screen& screen, int width, int height)
{
    //Update screen size
    screen.defineViewport(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height));	//update viewport
    screen.applyOpenGLContextSettings();
    TwWindowSize(width, height);



    //Update textures
    TextureSize tex_size{ static_cast<uint32_t>(width), static_cast<uint32_t>(height), 0 };

    p_myself->color_buffer = ImmutableTexture2D{ p_myself->color_buffer.getStringName() };
    p_myself->color_buffer.allocateStorage(1, 0, tex_size, InternalPixelFormat::SIZED_FLOAT_RGBA32);

    p_myself->bloom_texture = ImmutableTexture2D{ p_myself->bloom_texture.getStringName() };
    p_myself->bloom_texture.allocateStorage(1, 0, tex_size, InternalPixelFormat::SIZED_FLOAT_RGBA32);

    p_myself->depth_buffer = ImmutableTexture2D{ p_myself->depth_buffer.getStringName() };
    p_myself->depth_buffer.allocateStorage(1, 0, tex_size, InternalPixelFormat::SIZED_DEPTH32);

    p_myself->normal_map = ImmutableTexture2D{ p_myself->normal_map.getStringName() };
    p_myself->normal_map.allocateStorage(1, 0, tex_size, InternalPixelFormat::SIZED_FLOAT_RGB32);

    p_myself->ad_map = ImmutableTexture2D{ p_myself->ad_map.getStringName() };
    p_myself->ad_map.allocateStorage(1, 0, tex_size, InternalPixelFormat::SIZED_FLOAT_RGBA32);

    p_myself->linear_depth_buffer = ImmutableTexture2D{ p_myself->linear_depth_buffer.getStringName() };
    p_myself->linear_depth_buffer.allocateStorage(1, 0, tex_size, InternalPixelFormat::SIZED_FLOAT_R32);

    p_myself->selection_buffer = ImmutableTexture2D{ p_myself->selection_buffer.getStringName() };
    p_myself->selection_buffer.allocateStorage(1, 0, tex_size, InternalPixelFormat::SIZED_FLOAT_RG32);

    p_myself->refraction_map = ImmutableTexture2D{ p_myself->refraction_map.getStringName() };
    p_myself->refraction_map.allocateStorage(1, 0, tex_size, InternalPixelFormat::SIZED_FLOAT_RGBA32);



    //Re-initialize the cascade filter
    p_myself->bloom_blur_x.defineInputTexture(p_myself->bloom_texture);
    p_myself->ssao.defineScreenSpaceNormalMap(p_myself->normal_map);
    p_myself->ssao.defineLinearDepthBuffer(p_myself->linear_depth_buffer);
    p_myself->immediate_shader.defineColorMap(p_myself->color_buffer);
    p_myself->immediate_shader.defineADMap(p_myself->ad_map);
    p_myself->atmospheric_fog_filter.defineLinearDepthBuffer(p_myself->linear_depth_buffer);
    p_myself->light_haze_filter.defineLinearDepthBuffer(p_myself->linear_depth_buffer);
    p_myself->postprocess.setCommonOutputResolutionValue(uvec2{ static_cast<uint32_t>(width), static_cast<uint32_t>(height) });
    p_myself->p_myself->postprocess.initialize();



    //Update projection matrix
    float w = static_cast<float>(width) / std::max(width, height);
    float h = static_cast<float>(height) / std::max(width, height);
    p_myself->main_camera.setProjectionVolume(-w / 2, w / 2, -h / 2, h / 2, 1.0f, 1000.0f);



    p_myself->init_rendering_composer();



    //Update screen information for the drawable objects
    p_myself->tess_terrain.setScreenSize(uvec2{ static_cast<uint32_t>(width), static_cast<uint32_t>(height) });
}


void CloudsScene::onMouseClick(GLFWwindow* p_glfw_window, int button, int action, int mods)
{
    TwEventMouseButtonGLFW(button, action);

    if (!p_myself->suppress_mouse_user_input)
    {
        //Detect the object currently selected by user
        double xpos, ypos;
        int wsize_x, wsize_y;
        auto screen_size = p_screen->getScreenSize();
        glfwGetCursorPos(p_glfw_window, &xpos, &ypos);
        glfwGetWindowSize(p_glfw_window, &wsize_x, &wsize_y);
        xpos = xpos / wsize_x * screen_size.first;
        ypos = (wsize_y - ypos) / wsize_y * screen_size.second;

        if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_2)
        {
            //Get coordinates of the point having been selected
            float depth;
            p_myself->rendering_composition_framebuffer.readPixels(RenderingColorBuffer::COLOR3,
                static_cast<int>(xpos), static_cast<int>(ypos), 1, 1, PixelReadLayout::RED, PixelDataType::FLOAT, &depth);

            float left, right, bottom, top, near, far;
            p_myself->main_camera.getProjectionVolume(&left, &right, &bottom, &top, &near, &far);

            vec3 v3SelectedPoint_VS =
                vec3{ static_cast<float>((xpos / screen_size.first - 0.5)*(right - left)), static_cast<float>((ypos / screen_size.second - 0.5)*(top - bottom)), -near };
            v3SelectedPoint_VS *= -depth / near;
            vec4 v4SelectedPoint_WS = p_myself->main_camera.getViewTransform().inverse()*vec4 { v3SelectedPoint_VS.x, v3SelectedPoint_VS.y, v3SelectedPoint_VS.z, 1 };

            //Get id of the object having been selected
            unsigned long long selected_object_id =
                AbstractRenderableObjectSelectionEx::getPointSelectionObjectId(p_myself->rendering_composition_framebuffer,
                    uvec2{ static_cast<uint32_t>(xpos), static_cast<uint32_t>(ypos) });


            if (selected_object_id == p_myself->tess_terrain.getId())
            {
                bool is_topography_updated = false;	//equals 'true' if topography shall be updated

                                                    //Look for the terrain vertex located closest to the selected point
                vec3 v3SelectedPoint_unscaled = vec3{ v4SelectedPoint_WS.x, v4SelectedPoint_WS.y, v4SelectedPoint_WS.z } / p_myself->tess_terrain.getObjectScale();


                uint32_t topography_x_res = p_myself->topography_x_res, topography_y_res = p_myself->topography_y_res;
                std::vector<triplet<float, uint32_t, uint32_t>> terrain_slide_region;    //Set of triplets each containing coordinates of a certain point in the tessellated terrain billet together with the distance from the point, which has been clicked by the user
                for (uint32_t i = 0; i < topography_y_res; ++i)
                {
                    for (uint32_t j = 0; j < topography_x_res; ++j)
                    {
                        vec2 v2TessBilletPoint = vec2{ static_cast<float>(j), static_cast<float>(i) } / vec2{ topography_x_res - 1.0f, topography_y_res - 1.0f };
                        v2TessBilletPoint.x -= 0.5f;
                        v2TessBilletPoint.y = 0.5f - v2TessBilletPoint.y;
                        float dist = (v2TessBilletPoint - vec2{ v3SelectedPoint_unscaled.x, v3SelectedPoint_unscaled.z }).norm();
                        if (dist < 0.02f + p_myself->modification_strength*(0.05f - 0.02f))
                        {
                            switch (p_myself->interaction_mode)
                            {
                            case UserInteractionMode::RAISE_TERRAIN:
                            {
                                p_myself->p_raw_topography_data[i*topography_x_res + j] += std::exp(-dist*dist * (5 + p_myself->modification_strength*5))*(0.1f + p_myself->modification_strength*0.1f);
                                is_topography_updated = true;
                                break;
                            }

                            case UserInteractionMode::LOWER_TERRAIN:
                            {
                                p_myself->p_raw_topography_data[i*topography_x_res + j] -= std::exp(-dist*dist * (5 + p_myself->modification_strength*5))*(0.1f + p_myself->modification_strength*0.1f);
                                is_topography_updated = true;
                                break;
                            }


                            case UserInteractionMode::SLIDE_TERRAIN:
                            {
                                terrain_slide_region.push_back(triplet<float, uint32_t, uint32_t>{ dist, i, j });
                                break;
                            }
                            }
                        }
                    }
                }
                if (p_myself->interaction_mode == SLIDE_TERRAIN)
                {
                    std::sort(terrain_slide_region.begin(), terrain_slide_region.end(),
                        [](const triplet<float, uint32_t, uint32_t>& e1, const triplet<float, uint32_t, uint32_t>& e2)->bool
                    {
                        return e1.first < e2.first;
                    });

                    uint32_t num_points_slide_down = std::max(static_cast<uint32_t>(0.2f * terrain_slide_region.size()), 1U);
                    float hill_height = p_myself->p_raw_topography_data[terrain_slide_region[0].second*topography_x_res + terrain_slide_region[0].third];

                    float mass_loss = 0.0f;
                    for (uint32_t i = 0; i < num_points_slide_down; ++i)
                    {
                        float dist = terrain_slide_region[i].first;
                        float subsidence = std::exp(-dist*dist * (5 + p_myself->modification_strength*(10 - 5)))*(0.1f + p_myself->modification_strength*(0.3f - 0.1f));
                        p_myself->p_raw_topography_data[terrain_slide_region[i].second*topography_x_res + terrain_slide_region[i].third] -= subsidence;
                        mass_loss += subsidence;
                    }

                    float D = std::accumulate(terrain_slide_region.begin() + num_points_slide_down, terrain_slide_region.end(), 0.0f,
                        [&terrain_slide_region, hill_height, topography_x_res](float S, const triplet<float, uint32_t, uint32_t>& e)->float
                    {
                        float e_height = p_myself->p_raw_topography_data[e.second*topography_x_res + e.third];
                        if (e_height < hill_height)
                            return S + e.first + hill_height - e_height;
                        else
                            return S;
                    });

                    for (uint32_t i = num_points_slide_down; i < terrain_slide_region.size(); ++i)
                    {
                        float dist = terrain_slide_region[i].first;
                        uint32_t k = terrain_slide_region[i].second;
                        uint32_t l = terrain_slide_region[i].third;
                        float& point_height = p_myself->p_raw_topography_data[k*topography_x_res + l];
                        if (point_height < hill_height)
                            point_height += terrain_slide_region[i].first*(hill_height - point_height) / D*mass_loss;

                        vec2 v2TessBilletPoint = vec2{ static_cast<float>(k), static_cast<float>(l) } / vec2{ topography_x_res - 1.0f, topography_y_res - 1.0f };
                        vec2 v2Aux = (v2TessBilletPoint - vec2{ v3SelectedPoint_unscaled.x, v3SelectedPoint_unscaled.z });
                        vec3 v3Scale = p_myself->tess_terrain.getObjectScale();
                        v2Aux *= vec2{ v3Scale.x, -v3Scale.z };
                        float fDimensionalDistance = v2Aux.norm();
                        v2Aux /= fDimensionalDistance;
                        float hu = mass_loss * v2Aux.x;
                        float hv = mass_loss * v2Aux.y;
                    }

                    is_topography_updated = true;
                }


                if (is_topography_updated)
                {
                    p_myself->tess_terrain.defineHeightMap(p_myself->p_raw_topography_data, topography_x_res, topography_y_res, false);
                }

            }
        }
    }
    p_myself->suppress_mouse_user_input = false;
}


void CloudsScene::onScroll(GLFWwindow* p_glfw_window, double xoffset, double yoffset)
{
    TwEventMouseWheelGLFW(static_cast<int>(yoffset));
}


void CloudsScene::onCharInput(GLFWwindow* p_glfw_window, unsigned int codepoint)
{
    TwEventCharGLFW(codepoint, 1);
}


void CloudsScene::onParamSet(const void* value, void* client_data)
{
    p_myself->suppress_mouse_user_input = true;
    std::string param_name{ static_cast<const char*>(client_data) };

    if (param_name == "daytime") { p_myself->daytime = *static_cast<const float*>(value); }
    if (param_name == "fog_density")
    {
        p_myself->fog_density = *static_cast<const float*>(value);
        p_myself->lighting.setAtmosphericFogGlobalDensity(p_myself->fog_density);
    }
    if (param_name == "fog_height_fall_off")
    {
        p_myself->fog_height_fall_off = *static_cast<const float*>(value);
        p_myself->lighting.setAtmosphericFogHeightFallOff(p_myself->fog_height_fall_off);
    }
    if (param_name == "modification_strength")
    {
        p_myself->modification_strength = *static_cast<const float*>(value);
    }
    if (param_name == "mie_scattering")
    {
        LightScatteringSettings scattering_settings = p_myself->skydome.getAtmosphericScatteringSettings();
        float new_mie_scattering_coefficient = *static_cast<const float*>(value);
        scattering_settings.mie_coefficient = new_mie_scattering_coefficient;
        p_myself->skydome.setAtmosphereScatteringSettings(scattering_settings);
    }
    if (param_name == "rayleigh_scattering")
    {
        LightScatteringSettings scattering_settings = p_myself->skydome.getAtmosphericScatteringSettings();
        RayleighScatteringCoefficients rayleigh_scattering = *static_cast<const RayleighScatteringCoefficients*>(value);
        scattering_settings.red_wavelength = rayleigh_scattering.wavelength_red;
        scattering_settings.green_wavelength = rayleigh_scattering.wavelength_green;
        scattering_settings.blue_wavelength = rayleigh_scattering.wavelength_blue;
        p_myself->skydome.setAtmosphereScatteringSettings(scattering_settings);
    }
}


void CloudsScene::onParamGet(void* value, void* client_data)
{
    p_myself->suppress_mouse_user_input = true;
    std::string param_name{ static_cast<const char*>(client_data) };

    if (param_name == "daytime") { *static_cast<float*>(value) = p_myself->daytime; }
    if (param_name == "fog_density") { *static_cast<float*>(value) = p_myself->fog_density; }
    if (param_name == "fog_height_fall_off") { *static_cast<float*>(value) = p_myself->fog_height_fall_off; }
    if (param_name == "modification_strength") { *static_cast<float*>(value) = p_myself->modification_strength; }
    if (param_name == "mie_scattering")
    {
        LightScatteringSettings scattering_settings = p_myself->skydome.getAtmosphericScatteringSettings();
        *static_cast<float*>(value) = scattering_settings.mie_coefficient;
    }
    if (param_name == "rayleigh_scattering")
    {
        LightScatteringSettings scattering_settings = p_myself->skydome.getAtmosphericScatteringSettings();
        RayleighScatteringCoefficients rayleigh_scattering{ scattering_settings.red_wavelength, scattering_settings.green_wavelength, scattering_settings.blue_wavelength };
        *static_cast<RayleighScatteringCoefficients*>(value) = rayleigh_scattering;
    }
}


CloudsScene::CloudsScene(const std::string& topography_file_name, uint32_t reflection_map_resolution) :
    error_state{ false },
    main_camera{ "main_scene_camera" },
    reflection_camera{ "global_reflection_camera" },
    rendering_composition_framebuffer{ "renderer_framebuffer" },
    reflection_framebuffer{ "reflection_framebuffer" },
    reflection_map{ "scene_env_map" },
    reflection_map_depth_buffer{ "env_map_depth_buffer" },
    refraction_map{ "scene_refraction_map" },
    normal_map{ "scene_normal_map" },
    ad_map{ "ad_map" },
    linear_depth_buffer{ "scene_linear_depth_texture" },
    color_buffer{ "scene_color_buffer" },
    bloom_texture{ "scene_bloom_texture" },
    depth_buffer{ "scene_depth_buffer" },
    selection_buffer{ "scene_selection_buffer" },
    tess_terrain{ "topography_map", 10.0f, 400U, 400U, 1.0f / 50, 1.0f / 50 },
    ambient_light{ "scene_ambient_light" },
    skybody_light{ "skybody_light" },
    clouds{ vec3{ 100, 100, 100 }, uvec3{ 20, 20, 20 } },
    skydome{ "sky_simulator", 2e4f, 0.05f, 1000U, 128U, 128U },
    p_raw_topography_data{ nullptr },
    modification_strength{ 0.0f },
    suppress_mouse_user_input{ false },
    daytime{ 0.25f },
    fog_density{ 0.05f },
    fog_height_fall_off{ 0.5f }
{
    auto screen_size = p_screen->getScreenSize();

    TwInit(TW_OPENGL_CORE, NULL);
    TwWindowSize(static_cast<int>(screen_size.first), static_cast<int>(screen_size.second));
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);



    //Initialize texture unit object
    p_tex_unit_block = TextureUnitBlock::initialize();
    AbstractRenderableObjectTextured::defineTextureUnitBlockGlobalPointer(p_tex_unit_block);



    //Initialize texture objects
    TextureSize ss_tex_size{ screen_size.first, screen_size.second, 0 };	//size of a screen-space texture
    color_buffer.allocateStorage(1, 0, ss_tex_size, InternalPixelFormat::SIZED_FLOAT_RGBA32);
    bloom_texture.allocateStorage(1, 0, ss_tex_size, InternalPixelFormat::SIZED_FLOAT_RGBA32);
    depth_buffer.allocateStorage(1, 0, ss_tex_size, InternalPixelFormat::SIZED_DEPTH32);
    normal_map.allocateStorage(1, 0, ss_tex_size, InternalPixelFormat::SIZED_FLOAT_RGB32);
    ad_map.allocateStorage(1, 0, ss_tex_size, InternalPixelFormat::SIZED_FLOAT_RGBA32);
    linear_depth_buffer.allocateStorage(1, 0, ss_tex_size, InternalPixelFormat::SIZED_FLOAT_R32);
    selection_buffer.allocateStorage(1, 0, ss_tex_size, InternalPixelFormat::SIZED_FLOAT_RG32);



    //Initialize rendering composer
    init_rendering_composer();



    //Initialize reflection buffer
    init_reflection_buffer(reflection_map_resolution);



    //Setup main scene camera
    float w = static_cast<float>(screen_size.first) / std::max(screen_size.first, screen_size.second);
    float h = static_cast<float>(screen_size.second) / std::max(screen_size.first, screen_size.second);
    main_camera.setProjectionVolume(-w / 2, w / 2, -h / 2, h / 2, 1.0f, 2000.0f);
    main_camera.setLocation(vec3{ 0, 350, 0 });
    main_camera.setTarget(tiny_world::vec3{ 0, -.3f, -1.f });


    //Setup reflection camera
    reflection_camera.setProjectionVolume(-1, 1, -1, 1, 1.0f, 1000.0f);

    //Initialize the SS-filters
    init_ss_filters();

    //Configure topography
    std::vector<float> topography_heightmap;
    std::string topography_parse_error;
    /*if (!TessellatedTerrain::parseHeightMapFromFile(topography_file_name, topography_heightmap,
        topography_x_res, topography_y_res, topography_offset, topography_max_height, topography_parse_error))
    {
        error_state = true;
        std::cout << "ERROR: " << topography_parse_error << std::endl;
        return;
    }


    //Perform topography interpolation
    p_raw_topography_data = new float[topography_x_res*topography_y_res];
    std::copy(topography_heightmap.begin(), topography_heightmap.end(), stdext::make_unchecked_array_iterator(p_raw_topography_data));


    tess_terrain.defineHeightMap(p_raw_topography_data, topography_x_res, topography_y_res, false);
    tess_terrain.setScreenSize(screen_size.first, screen_size.second);
    tess_terrain.scale(1000.f, 100.f, 1000.f);*/

    //TessellatedTerrain tess_terrain1 = tess_terrain;

    //Load terrain textures from file and combine them to an array texture
    KTXTexture sand, grass, slope, rock;
    sand.loadTexture("textures/sand_COLOR.ktx");
    grass.loadTexture("textures/grass_COLOR.ktx");
    slope.loadTexture("textures/slope_COLOR.ktx");
    rock.loadTexture("textures/rock_COLOR.ktx");
    ImmutableTexture2D terrain_texture = TEXTURE_2D(sand.getContainedTexture()).combine(TEXTURE_2D(grass.getContainedTexture()));
    terrain_texture = terrain_texture.combine(TEXTURE_2D(slope.getContainedTexture()).combine(TEXTURE_2D(rock.getContainedTexture())));

    //Install terrain texture to the TessellatedTerrain object
    terrain_texture.setStringName("terain_texture_COLOR");
    tess_terrain.installTexture(terrain_texture);



    //Setup light model textures

    //Bump textures
    KTXTexture sand_bump, grass_bump, slope_bump, rock_bump;
    sand_bump.loadTexture("textures/sand_NRM.ktx");
    grass_bump.loadTexture("textures/grass_NRM.ktx");
    slope_bump.loadTexture("textures/slope_NRM.ktx");
    rock_bump.loadTexture("textures/rock_NRM.ktx");
    ImmutableTexture2D terrain_texture_bump = TEXTURE_2D(sand_bump.getContainedTexture()).combine(TEXTURE_2D(grass_bump.getContainedTexture()));
    terrain_texture_bump = terrain_texture_bump.combine(TEXTURE_2D(slope_bump.getContainedTexture()).combine(TEXTURE_2D(rock_bump.getContainedTexture())));

    terrain_texture_bump.setStringName("terain_texture_NRM");
    tess_terrain.applyNormalMapSourceTexture(terrain_texture_bump);


    //Specular textures
    KTXTexture sand_spec, grass_spec, slope_spec, rock_spec;
    sand_spec.loadTexture("textures/sand_SPEC.ktx");
    grass_spec.loadTexture("textures/grass_SPEC.ktx");
    slope_spec.loadTexture("textures/slope_SPEC.ktx");
    rock_spec.loadTexture("textures/rock_SPEC.ktx");
    ImmutableTexture2D terrain_texture_spec = TEXTURE_2D(sand_spec.getContainedTexture()).combine(TEXTURE_2D(grass_spec.getContainedTexture()));
    terrain_texture_spec = terrain_texture_spec.combine(TEXTURE_2D(slope_spec.getContainedTexture()).combine(TEXTURE_2D(rock_spec.getContainedTexture())));

    terrain_texture_spec.setStringName("terain_texture_SPEC");
    tess_terrain.applySpecularMapSourceTexture(terrain_texture_spec);


    tess_terrain.setBloomMinimalThreshold(0.3f);
    tess_terrain.setBloomMaximalThreshold(1.0f);
    tess_terrain.useBloom(true);
    tess_terrain.setBloomIntensity(0.5f);
    tess_terrain.applyLightingConditions(lighting);
    tess_terrain.setSpecularExponent(2.5f);



    //Configure lighting conditions
    ambient_light.setColor(vec3{ 0.3f, 0.3f, 0.3f });
    lighting.addLight(ambient_light);
    lighting.addLight(skybody_light);



    //Configure the skydome
    skydome.setNominalMoonSize(128);
    skydome.setMoonLocation(vec2{ pi / 12, 0 });
    skydome.setMoonLightIntensity(vec3{ 0.5f });
    skydome.setBloomMinimalThreshold(0.7f);
    skydome.setBloomMaximalThreshold(1.0f);
    skydome.setBloomIntensity(0.9f);
    skydome.useBloom(true);
    skydome.setDaytime(daytime);


    //Configure clouds
    clouds.setParticleSize(10.f);
    clouds.setLightingConditions(lighting);
    clouds.setLocation(vec3{ 0.f, 150.f, 0.f });

    //Initializes the toolbar
    init_toolbar();
}


CloudsScene::~CloudsScene()
{
    TextureUnitBlock::reset();
    if (p_raw_topography_data) delete[] p_raw_topography_data;

    TwTerminate();
}


void CloudsScene::init_ss_filters()
{
    auto screen_size = p_screen->getScreenSize();


    //*****************************************************Configure screen-space filters*****************************************************
    CL0.setOutputColorBitWidth(CascadeFilterLevel::ColorWidth::_32bit);
    CL0.setOutputColorComponents(true, true, true, false);
    CL0.addFilter(&bloom_blur_x);

    CL1.setOutputColorBitWidth(CascadeFilterLevel::ColorWidth::_32bit);
    CL1.setOutputColorComponents(true, false, false, false);
    CL1.addFilter(&ssao);

    CL2.setOutputColorBitWidth(CascadeFilterLevel::ColorWidth::_32bit);
    CL2.setOutputColorComponents(true, true, true, false);
    CL2.addFilter(&bloom_blur_y);
    CL2.addFilter(&ssao_blur_x);

    CL3.setOutputColorBitWidth(CascadeFilterLevel::ColorWidth::_32bit);
    CL3.setOutputColorComponents(true, true, true, false);
    CL3.addFilter(&ssao_blur_y);

    CL4.setOutputColorBitWidth(CascadeFilterLevel::ColorWidth::_32bit);
    CL4.setOutputColorComponents(true, true, true, false);
    CL4.addFilter(&immediate_shader);

    CL5.setOutputColorBitWidth(CascadeFilterLevel::ColorWidth::_32bit);
    CL5.setOutputColorComponents(true, true, true, false);
    CL5.addFilter(&hdr_bloom);

    CL6.setOutputColorBitWidth(CascadeFilterLevel::ColorWidth::_32bit);
    CL6.setOutputColorComponents(true, true, true, true);
    CL6.addFilter(&atmospheric_fog_filter);

    CL7.setOutputColorBitWidth(CascadeFilterLevel::ColorWidth::_32bit);
    CL7.setOutputColorComponents(true, true, true, true);
    CL7.addFilter(&light_haze_filter);

    postprocess.addBaseLevel(CL0);
    postprocess.addLevel(CL1, std::list < CascadeFilter::DataFlow >{});
    postprocess.addLevel(CL2, std::list < CascadeFilter::DataFlow >{ CascadeFilter::DataFlow{ 0, 0, 0 }, CascadeFilter::DataFlow{ 1, 1, 0 } });
    postprocess.addLevel(CL3, std::list < CascadeFilter::DataFlow >{ CascadeFilter::DataFlow{ 0, 2, 1 } });
    postprocess.addLevel(CL4, std::list < CascadeFilter::DataFlow >{ CascadeFilter::DataFlow{ 2, 3, 0 } });
    postprocess.addLevel(CL5, std::list < CascadeFilter::DataFlow >{ CascadeFilter::DataFlow{ 0, 4, 0 }, CascadeFilter::DataFlow{ 1, 2, 0 } });
    postprocess.addLevel(CL6, std::list < CascadeFilter::DataFlow >{ CascadeFilter::DataFlow{ 0, 5, 0 } });
    postprocess.addLevel(CL7, std::list < CascadeFilter::DataFlow >{ CascadeFilter::DataFlow{ 0, 6, 0 } });
    postprocess.useCommonOutputResolution(true);
    postprocess.setCommonOutputResolutionValue(uvec2{ screen_size.first, screen_size.second });


    //Configure bloom blur filters
    bloom_blur_x.defineInputTexture(bloom_texture);
    bloom_blur_x.setKernelMipmap(0);
    bloom_blur_x.setKernelScale(6.0f);
    bloom_blur_x.setKernelSize(5);

    bloom_blur_y.setKernelMipmap(0);
    bloom_blur_y.setKernelScale(6.0f);
    bloom_blur_y.setKernelSize(5);
    bloom_blur_y.setDirection(SSFilter_Blur::BlurDirection::VERTICAL);


    //Configure SSAO
    ssao.setKernelRadius(1.0f);
    ssao.setOcclusionRange(2.0f);
    ssao.setNoiseSize(3);
    ssao.setNumberOfSamples(32);
    ssao.defineScreenSpaceNormalMap(normal_map);
    ssao.defineLinearDepthBuffer(linear_depth_buffer);

    //Configure SSAO-blur filters
    ssao_blur_x.setKernelMipmap(0);
    ssao_blur_x.setKernelScale(1.0f);
    ssao_blur_x.setKernelSize(5);
    ssao_blur_y.setKernelMipmap(0);
    ssao_blur_y.setKernelScale(1.0f);
    ssao_blur_y.setKernelSize(5);


    //Configure immediate shader
    immediate_shader.defineColorMap(color_buffer);
    immediate_shader.defineADMap(ad_map);


    //Configure HDR-Bloom filter
    hdr_bloom.setBloomImpact(0.1f);
    hdr_bloom.setContrast(1.8f);


    //Configure Atmospheric Fog filter
    atmospheric_fog_filter.defineLinearDepthBuffer(linear_depth_buffer);
    lighting.setAtmosphericFogGlobalDensity(fog_density);
    lighting.setAtmosphericFogHeightFallOff(fog_height_fall_off);
    lighting.setAtmosphericFogMiePhaseFunctionParameter(-0.8f);
    lighting.setSkydome(&skydome);
    atmospheric_fog_filter.setLightingConditions(lighting);
    atmospheric_fog_filter.setDistanceCutOff(500.0f);

    //Configure Light Haze filter
    light_haze_filter.defineLinearDepthBuffer(linear_depth_buffer);
    light_haze_filter.setLightingConditions(lighting);


    //Initialize the cascade filter
    postprocess.initialize();

    //****************************************************************************************************************************************
}


void CloudsScene::init_rendering_composer()
{
    auto screen_size = p_screen->getScreenSize();

    //Configure rendering composition framebuffer
    rendering_composition_framebuffer.defineViewport(0, 0, static_cast<float>(screen_size.first), static_cast<float>(screen_size.second));
    rendering_composition_framebuffer.setCullTestEnableState(true);
    rendering_composition_framebuffer.setDepthTestEnableState(true);

    rendering_composition_framebuffer.attachTexture(FramebufferAttachmentPoint::COLOR0, FramebufferAttachmentInfo{ 0, 0, &color_buffer });
    rendering_composition_framebuffer.attachTexture(FramebufferAttachmentPoint::COLOR1, FramebufferAttachmentInfo{ 0, 0, &bloom_texture });
    rendering_composition_framebuffer.attachTexture(FramebufferAttachmentPoint::COLOR2, FramebufferAttachmentInfo{ 0, 0, &normal_map });
    rendering_composition_framebuffer.attachTexture(FramebufferAttachmentPoint::COLOR3, FramebufferAttachmentInfo{ 0, 0, &linear_depth_buffer });
    rendering_composition_framebuffer.attachTexture(FramebufferAttachmentPoint::COLOR4, FramebufferAttachmentInfo{ 0, 0, &ad_map });
    rendering_composition_framebuffer.attachTexture(FramebufferAttachmentPoint::COLOR5, FramebufferAttachmentInfo{ 0, 0, &selection_buffer });
    rendering_composition_framebuffer.attachTexture(FramebufferAttachmentPoint::DEPTH, FramebufferAttachmentInfo{ 0, 0, &depth_buffer });
    rendering_composition_framebuffer.attachRenderer(onSceneRedraw);

    std::string framebuffer_completeness_description;
    if (!rendering_composition_framebuffer.isDrawComplete(&framebuffer_completeness_description))
    {
        std::cout << "ERROR: " + framebuffer_completeness_description << std::endl;
        error_state = true;
        return;
    }
}


void CloudsScene::init_reflection_buffer(uint32_t reflection_buffer_resolution)
{
    auto screen_size = p_screen->getScreenSize();

    //Configure reflection framebuffer
    reflection_framebuffer.defineViewport(0, 0, static_cast<float>(reflection_buffer_resolution), static_cast<float>(reflection_buffer_resolution));
    reflection_framebuffer.setCullTestEnableState(true);
    reflection_framebuffer.setDepthTestEnableState(true);

    reflection_map.allocateStorage(static_cast<uint32_t>(std::log2(reflection_buffer_resolution) + 1), 0,
        TextureSize{ reflection_buffer_resolution, reflection_buffer_resolution, 0 }, InternalPixelFormat::SIZED_FLOAT_RGBA16);
    reflection_map_depth_buffer.allocateStorage(1, 1, TextureSize{ reflection_buffer_resolution, reflection_buffer_resolution, 0 }, InternalPixelFormat::SIZED_DEPTH16);
    refraction_map.allocateStorage(1, 1, TextureSize{ screen_size.first, screen_size.second, 0 }, InternalPixelFormat::SIZED_FLOAT_RGBA32);

    reflection_framebuffer.attachTexture(FramebufferAttachmentPoint::DEPTH, FramebufferAttachmentInfo{ 0, 0, &reflection_map_depth_buffer });
    reflection_framebuffer.attachRenderer(onReflectionMapUpdate);
}


void CloudsScene::init_toolbar()
{
    p_main_bar = TwNewBar("MainBar");
    glfwSetMouseButtonCallback(*p_screen, onMouseClick);
    glfwSetScrollCallback(*p_screen, onScroll);
    glfwSetCharCallback(*p_screen, onCharInput);


    TwDefine(" MainBar label='Settings' color='50 127 220' movable='false' resizable='false' size='400 200' ");


    TwAddVarCB(p_main_bar, "daytime", TW_TYPE_FLOAT, onParamSet, onParamGet, "daytime",
        " label='Time of the day' help='Allows to alter the current time of the day' min=0 max=1 step=0.0001 precision=4 ");

    TwAddVarCB(p_main_bar, "fog_density", TW_TYPE_FLOAT, onParamSet, onParamGet, "fog_density",
        " label='Fog density' help='Allows to alter density of the atmospheric fog' min=0.05 max=1 step=0.0001 precision=4 ");

    TwAddVarCB(p_main_bar, "fog_height_fall_off", TW_TYPE_FLOAT, onParamSet, onParamGet, "fog_height_fall_off",
        " label='Fog height fall off' help='Allows to set how rapidly the atmospheric fog evaporates with respect to the altitude' min=0.01 max=1 step=0.0001 precision=4 ");

    TwAddVarCB(p_main_bar, "rayleigh_scattering",
        TwDefineStruct("rayleigh_scattering_coefficients",
            std::vector<TwStructMember>{
        TwStructMember{ "wavelength_red", TW_TYPE_FLOAT, offsetof(RayleighScatteringCoefficients, wavelength_red), " label='Red' help='Rayleigh coefficient used for the red channel' min=0.0001 max=1.0 step=0.0001 precision=4 " },
            TwStructMember{ "wavelength_green", TW_TYPE_FLOAT, offsetof(RayleighScatteringCoefficients, wavelength_green), " label='Green' help='Rayleigh coefficient used for the green channel' min=0.0001 max=1.0 step=0.0001 precision=4 " },
            TwStructMember{ "wavelength_blue", TW_TYPE_FLOAT, offsetof(RayleighScatteringCoefficients, wavelength_blue), " label='Blue' help='Rayleigh coefficient used for the blue channel' min=0.0001 max=1.0 step=0.0001 precision=4 " } }.data(),
            3, sizeof(RayleighScatteringCoefficients), nullptr, nullptr),
        onParamSet, onParamGet, "rayleigh_scattering",
            " label='Rayleigh scattering coefficients' help='Allows to configure coefficients of Rayleigh scattering' ");

    TwAddVarCB(p_main_bar, "mie_scattering", TW_TYPE_FLOAT, onParamSet, onParamGet, "mie_scattering",
        " label='Mie scattering coefficient' help='Adjusts Mie scattering coefficient' min=0.0001 max=1.0 step=0.0001 precision=4 ");


    TwAddVarRW(p_main_bar, "interaction_mode",
        TwDefineEnum("UserInteractionMode",
            std::vector < TwEnumVal > {TwEnumVal{ UserInteractionMode::RAISE_TERRAIN, "Raise terrain" },
            TwEnumVal{ UserInteractionMode::LOWER_TERRAIN, "Lower terrain" },
            TwEnumVal{ UserInteractionMode::SLIDE_TERRAIN, "Slide terrain" }}.data(), 3),
        &interaction_mode,
        " label='Interaction mode' help='Defines the way the user is able to interact with the virtual environment' ");
    TwAddVarCB(p_main_bar, "modification_strength", TW_TYPE_FLOAT, onParamSet, onParamGet, "modification_strength",
        " label='Strength' help='Determines strength of interactive action' min=0.0 max=1.0 step=0.01 precision=4 ");
}