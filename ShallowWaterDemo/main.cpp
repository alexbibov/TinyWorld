#include "ShallowWaterScene.h"
#include "AppParams.h"

#include <cstdlib>

int main(int argc, char* argv[])
{
	ShallowWaterScene* p_shallow_water_scene = ShallowWaterScene::initializeScene(WINDOW_RESOLUTION_X, WINDOW_RESOLUTION_Y, "riverdam.txt", "../tw_shaders/", "../tw_textures/", REFLECTION_RESOLUTION);

	while (!p_shallow_water_scene->updateScene());

	ShallowWaterScene::destroyScene();

	return EXIT_SUCCESS;
}