from OpenGL.GL import *

class Shader:
    @classmethod
    def from_file(cls, type, filename):
        return cls(type, filename=filename)

    @classmethod
    def from_str(cls, type, code):
        return cls(type, code=code)

    def __init__(self, type, code="", filename=""):
        self.id = 0
        self.type = type
        if code != "":
            self.compile_shader(code)
        else:
            self.load_and_compile_shader(filename)

    def __del__(self):
        if self.id != 0:
            glDeleteShader(self.id)

    def load_and_compile_shader(self, filename):
        raise Exception("not implemented")

    def compile_shader(self, code):
        self.id = glCreateShader(self.type)
        glShaderSource(self.id, code)
        glCompileShader(self.id)

        success = glGetShaderiv(self.id, GL_COMPILE_STATUS)
        if not success:
            log = glGetShaderInfoLog(self.id)
            glDeleteShader(self.id)
            self.id = 0
            raise Exception("failed to compile shader: {}".format(log))

        return self.id
