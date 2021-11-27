from OpenGL.GL import *
from shader import Shader
import numpy as np

class Program:
    @classmethod
    def from_vert_and_frag_code(cls, vert_code, frag_code):
        vert_shader = Shader.from_str(GL_VERTEX_SHADER, vert_code)
        frag_shader = Shader.from_str(GL_FRAGMENT_SHADER, frag_code)
        return cls( [vert_shader, frag_shader] )

    def __init__(self, shaders):
        self.id = 0
        self.link_shader(shaders)

    def __del__(self):
        if self.id != 0:
            glDeleteProgram(self.id)

    def link_shader(self, shaders):
        self.id = glCreateProgram()
        for shader in shaders:
            glAttachShader(self.id, shader.id)
        glLinkProgram(self.id)

        success = glGetProgramiv(self.id, GL_LINK_STATUS)
        if not success:
            log = glGetProgramInfoLog(self.id)
            glDeleteProgram(self.id)
            self.id = 0
            raise Exception("failed to link shader: {}".format(log))

        return self.id

    def use(self):
        glUseProgram(self.id)

    def get_location(self, name):
        return glGetUniformLocation(self.id, name)

    def set_uniform_mat4_np(self, name, mat4):
        glUniformMatrix4fv(self.get_location(name), 1, GL_FALSE, mat4.T)

    def set_texture(self, name, tex, active_id=0):
        glUniform1i(self.get_location(name), active_id)
        glActiveTexture(GL_TEXTURE0 + active_id)
        tex.bind()

    def set_uniform_vec3(self, name, vec3):
        glUniform3f(self.get_location(name), *vec3)

    def set_uniform_vec4(self, name, vec4):
        glUniform4f(self.get_location(name), *vec4)

    def set_uniform_float(self, name, value):
        glUniform1f(self.get_location(name), value)

    def set_uniform_int(self, name, value):
        glUniform1i(self.get_location(name), value)