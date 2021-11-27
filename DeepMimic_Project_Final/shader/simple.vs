#version 330 core

layout (location = 0) in vec3 aPos;

uniform mat4 mvpTransform;

void main() {
    gl_Position = mvpTransform * vec4(aPos, 1.0);
}
