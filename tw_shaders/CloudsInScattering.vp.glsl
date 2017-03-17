#version 430 core

in vec4 v4VertexPosition;

uniform mat4 m4ModelViewTransform;
uniform mat4 m4ProjectionTransform;

void main()
{
    gl_Position = m4ProjectionTransform*m4ModelViewTransform*v4VertexPosition;
}