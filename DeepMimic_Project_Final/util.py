import numpy as np

def deg2rad(degree):
    return degree / 180.0 * np.pi

def rot_mat_x(radian):
    c = np.cos(radian)
    s = np.sin(radian)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0,   c,  -s, 0.0],
        [0.0,   s,   c, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

def rot_mat_y(radian):
    c = np.cos(radian)
    s = np.sin(radian)
    return np.array([
        [  c, 0.0,   s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [ -s, 0.0,   c, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

def rot_mat_z(radian):
    c = np.cos(radian)
    s = np.sin(radian)
    return np.array([
        [  c,  -s, 0.0, 0.0],
        [  s,   c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

def rot_mat(radian, axis):
    c = np.cos(radian)
    s = np.sin(radian)
    invc = 1.0 - c
    x = axis[0]
    y = axis[1]
    z = axis[2]
    return np.array([
        [invc*x*x +   c, invc*x*y - s*z, invc*x*z + s*y, 0.0],
        [invc*x*y + s*z, invc*y*y +   c, invc*y*z - s*x, 0.0],
        [invc*x*z - s*y, invc*y*z + s*x, invc*z*z +   c, 0.0],
        [0.0, 0.0, 0.0, 1.0]])


def translate_mat(pos):
    return np.array([
        [1.0, 0.0, 0.0, pos[0]],
        [0.0, 1.0, 0.0, pos[1]],
        [0.0, 0.0, 1.0, pos[2]],
        [0.0, 0.0, 0.0, 1.0]])

def scale_mat(scale):
    return np.array([
        [scale[0], 0.0, 0.0, 0.0],
        [0.0, scale[1], 0.0, 0.0],
        [0.0, 0.0, scale[2], 0.0],
        [0.0, 0.0, 0.0, 1.0]])

def look_at(eye, target, up):
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    mat = np.concatenate((x, y, z, eye)).reshape((4, 3)).T
    mat = np.pad(mat, ((0, 1), (0, 0)), 'constant', constant_values=0.0)
    mat[3, 3] = 1.0
    mat = np.linalg.inv(mat)
    return mat

def perspective(fov, aspect_ratio, near, far):
    cot = 1.0 / np.tan(fov * 0.5)
    return np.array([
        [cot / aspect_ratio, 0.0, 0.0, 0.0],
        [0.0, cot, 0.0, 0.0],
        [0.0, 0.0, (near + far) / (near - far), 2.0 * near * far / (near - far)],
        [0.0, 0.0, -1.0, 0.0]])
