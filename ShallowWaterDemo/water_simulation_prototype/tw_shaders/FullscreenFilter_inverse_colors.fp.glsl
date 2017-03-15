#version 430 core

in vec2 tex_coord;

out vec4 fColor;

uniform sampler2D source0;

void main()
{
	fColor = vec4(1.0f) - texture(fimage_sampler, tex_coord);
}