from OpenGL.GL import *
import numpy as np
from PIL import Image

class Texture:
    def __init__(self):
        self.id = 0

    def __del__(self):
        if self.id != 0:
            glDeleteTextures(1, self.id)

    def load_texture(self, filename):
        image = Image.open(filename).transpose(Image.FLIP_TOP_BOTTOM)
        image_data = np.array(image, np.uint8)
        if not image_data.any():
            raise Exception("failed to load image: {}".format(filename))

        self.id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.size[0], image.size[1],
            0, GL_RGBA, GL_UNSIGNED_BYTE, image_data.ctypes.data_as(ctypes.c_void_p))
        glGenerateMipmap(GL_TEXTURE_2D)
        return self.id

    def bind(self):
        glBindTexture(GL_TEXTURE_2D, self.id)
