from OpenGL.GL import *
import numpy as np

class Mesh:
    @classmethod
    def from_vertices_and_indices(cls, vertices, indices, primitive=GL_TRIANGLES):
        mesh = cls()
        mesh.create(vertices, indices, primitive=primitive)
        return mesh

    @classmethod
    def box(cls, size=(1, 1, 1)):
        w = size[0] * 0.5
        h = size[1] * 0.5
        d = size[2] * 0.5

        mesh = cls()
        vertices = np.array([
            -w, -h,  d, 0.0, 0.0, 1.0, 0.0, 0.0,
             w, -h,  d, 0.0, 0.0, 1.0, 1.0, 0.0,
             w,  h,  d, 0.0, 0.0, 1.0, 1.0, 1.0,
            -w,  h,  d, 0.0, 0.0, 1.0, 0.0, 1.0,
             w, -h,  d, 1.0, 0.0, 0.0, 0.0, 0.0,
             w, -h, -d, 1.0, 0.0, 0.0, 1.0, 0.0,
             w,  h, -d, 1.0, 0.0, 0.0, 1.0, 1.0,
             w,  h,  d, 1.0, 0.0, 0.0, 0.0, 1.0,
            -w, -h, -d, 0.0, 0.0, -1.0, 0.0, 0.0,
             w, -h, -d, 0.0, 0.0, -1.0, 1.0, 0.0,
             w,  h, -d, 0.0, 0.0, -1.0, 1.0, 1.0,
            -w,  h, -d, 0.0, 0.0, -1.0, 0.0, 1.0,
            -w, -h,  d, -1.0, 0.0, 0.0, 0.0, 0.0,
            -w, -h, -d, -1.0, 0.0, 0.0, 1.0, 0.0,
            -w,  h, -d, -1.0, 0.0, 0.0, 1.0, 1.0,
            -w,  h,  d, -1.0, 0.0, 0.0, 0.0, 1.0,
            -w,  h, -d, 0.0, 1.0, 0.0, 0.0, 0.0,
             w,  h, -d, 0.0, 1.0, 0.0, 1.0, 0.0,
             w,  h,  d, 0.0, 1.0, 0.0, 1.0, 1.0,
            -w,  h,  d, 0.0, 1.0, 0.0, 0.0, 1.0,
            -w, -h, -d, 0.0, -1.0, 0.0, 0.0, 0.0,
             w, -h, -d, 0.0, -1.0, 0.0, 1.0, 0.0,
             w, -h,  d, 0.0, -1.0, 0.0, 1.0, 1.0,
            -w, -h,  d, 0.0, -1.0, 0.0, 0.0, 1.0,

        ], dtype=np.float32)
        indices = np.array([
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4,
            8, 9, 10, 10, 11, 8,
            12, 13, 14, 14, 15, 12,
            16, 17, 18, 18, 19, 16,
            20, 21, 22, 22, 23, 20
        ], dtype=np.uint32)
        mesh.create(vertices, indices)
        return mesh

    @classmethod
    def plane(cls, size=(1, 1)):
        w = size[0] * 0.5
        h = size[1] * 0.5

        mesh = cls()
        vertices = np.array([
            -w, -h, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0,
             w, -h, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0,
             w,  h, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0,
            -w,  h, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0,
        ], dtype=np.float32)
        indices = np.array([
            0, 1, 2, 2, 3, 0,
        ], dtype=np.uint32)
        mesh.create(vertices, indices)
        return mesh

    @classmethod
    def grid(cls, interval, count):
        size = interval * (count - 1)
        half_size = size * 0.5

        vert_array = []
        for i in range(count):
            t = i * interval - half_size
            vert_array += [t, -half_size, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            vert_array += [t, half_size, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            vert_array += [-half_size, t, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            vert_array += [half_size, t, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

        mesh = cls()
        vertices = np.array(vert_array, dtype=np.float32)
        indices = np.array(list(range(len(vert_array))), dtype=np.uint32)
        mesh.create(vertices, indices, primitive=GL_LINES)
        return mesh

    @classmethod 
    def cylinder(cls, upper_radius, lower_radius, height, segment_count=16):
        index_array = []

        # lower lid, upper lid, side
        vert_array = [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.5, 0.5]
        for i in range(segment_count + 1):
            angle = i / segment_count * np.pi * 2.0
            tx = np.cos(angle)
            ty = np.sin(angle)
            x = tx * lower_radius
            y = ty * lower_radius
            vert_array += [x, y, 0.0, 0.0, 0.0, -1.0, tx * 0.5 + 0.5, ty * 0.5 + 0.5]
        for i in range(segment_count):
            index_array.append(0)
            index_array.append(i + 1)
            index_array.append(i + 2)
        
        offset = int(len(vert_array) / 8)
        vert_array += [0.0, 0.0, height, 0.0, 0.0, 1.0, 0.5, 0.5]
        for i in range(segment_count + 1):
            angle = i / segment_count * np.pi * 2.0
            tx = np.cos(angle)
            ty = np.sin(angle)
            x = tx * upper_radius
            y = ty * upper_radius
            vert_array += [x, y, height, 0.0, 0.0, 1.0, tx * 0.5 + 0.5, ty * 0.5 + 0.5]
        for i in range(segment_count):
            index_array.append(offset)
            index_array.append(offset + i + 2)
            index_array.append(offset + i + 1)

        offset = int(len(vert_array) / 8)
        for i in range(segment_count + 1):
            t = i / segment_count
            angle = t * np.pi * 2.0
            tx = np.cos(angle)
            ty = np.sin(angle)
            lx = tx * lower_radius
            ly = ty * lower_radius
            ux = tx * upper_radius
            uy = ty * upper_radius
            vert_array += [lx, ly, 0.0,    tx, ty, 0.0, t, 0.0]
            vert_array += [ux, uy, height, tx, ty, 0.0, t, 1.0]
        for i in range(segment_count):
            index_array.append(offset + 2*i)
            index_array.append(offset + 2*i + 2)
            index_array.append(offset + 2*i + 1)
            index_array.append(offset + 2*i + 1)
            index_array.append(offset + 2*i + 2)
            index_array.append(offset + 2*i + 3)

        mesh = cls()
        vertices = np.array(vert_array, dtype=np.float32)
        indices = np.array(index_array, dtype=np.uint32)
        mesh.create(vertices, indices)
        return mesh
        

    def __init__(self):
        self.vao = 0
        self.vbo = 0
        self.ibo = 0

    def __del__(self):
        if self.vbo != 0:
            glDeleteBuffers(1, self.vbo)
        if self.ibo != 0:
            glDeleteBuffers(1, self.ibo)
        if self.vao != 0:
            glDeleteVertexArrays(1, self.vao)

    def create(self, vertices, indices, primitive=GL_TRIANGLES): # 3, 3, 2
        self.primitive = primitive
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes,
            vertices.ctypes.data_as(ctypes.c_void_p), GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, None)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24))

        self.ibo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes,
            indices.ctypes.data_as(ctypes.c_void_p), GL_STATIC_DRAW)
        self.index_count = indices.shape[0]

    def bind(self):
        glBindVertexArray(self.vao)

    def draw(self):
        self.bind()
        glDrawElements(self.primitive, self.index_count, GL_UNSIGNED_INT, None)