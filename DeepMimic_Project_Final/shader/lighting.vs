#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

out vec3 vPos;
out vec3 vNormal;
out vec2 vTexCoord;

uniform mat4 mvpTransform;
uniform mat4 modelTransform;

void main() {
    gl_Position = mvpTransform * vec4(aPos, 1.0);
    vPos = (modelTransform * vec4(aPos, 1.0)).xyz;
    vNormal = (inverse(transpose(modelTransform)) * vec4(aNormal, 0.0)).xyz;
    vTexCoord = aTexCoord;
}
