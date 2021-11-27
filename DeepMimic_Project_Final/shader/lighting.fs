#version 330 core

in vec3 vPos;
in vec3 vNormal;
in vec2 vTexCoord;

out vec4 fragColor;

uniform vec3 lightPos;
uniform vec3 ambientLight;
uniform vec3 diffuseLight;
uniform vec3 specularLight;
uniform float shininess;
uniform vec3 viewPos;

uniform sampler2D tex;
uniform int useTexture;
uniform vec3 matColor;

void main() {
    vec3 color = vec3(0.0, 0.0, 0.0);
    color += ambientLight;
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(lightPos - vPos);
    float diffuse = max(dot(normal, lightDir), 0.0);
    color += diffuseLight * diffuse;
    vec3 viewDir = normalize(viewPos - vPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float specular = max(dot(viewDir, reflectDir), 0.0);
    specular = pow(specular, shininess);
    color += specularLight * specular;

    if (useTexture == 1) {
        vec4 texColor = texture(tex, vTexCoord);
        fragColor = vec4(color * texColor.rgb, 1.0);
    }
    else {
        fragColor = vec4(color * matColor, 1.0);
    }
}