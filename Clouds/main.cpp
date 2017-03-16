#include "CloudsScene.h"
#include "AppParams.h"

#include <cstdlib>

int main(int argc, char* argv[])
{
    CloudsScene* p_shallow_water_scene = CloudsScene::initializeScene(WINDOW_TITLE, WINDOW_RESOLUTION_X, WINDOW_RESOLUTION_Y, "riverdam.txt", "../tw_shaders/", "../tw_textures/", REFLECTION_RESOLUTION);

    while (!p_shallow_water_scene->updateScene());

    CloudsScene::destroyScene();

    return EXIT_SUCCESS;
}