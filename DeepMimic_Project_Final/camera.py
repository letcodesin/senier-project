import numpy as np
from util import *

class Camera:
    ACTION_NONE = 0
    ACTION_ORBIT = 1
    ACTION_PAN = 2
    ACTION_ZOOM = 3

    def __init__(self):
        self.action = Camera.ACTION_NONE
        self.pitch = 45.0
        self.yaw = 0.0
        self.distance = 3.0
        self.target = (0.0, 0.0, 0.0)
        self.up = np.array([0.0, 1.0, 0.0])

    def compute_look_at(self):
        pos = self.camera_pos()
        l = look_at(pos, self.target, self.up)
        return l

    def camera_pos(self):
        rot_x = rot_mat_x(deg2rad(-self.pitch))
        rot_y = rot_mat_y(deg2rad(self.yaw))
        dir = np.matmul(np.matmul(rot_y, rot_x), np.array([0.0, 0.0, 1.0, 0.0]))[:3]
        pos = self.target + self.distance * dir

        return pos

    def modify_with_mouse_pos(self, dx, dy):
        if self.action == Camera.ACTION_ZOOM:
            self.distance -= dx * 0.05
            if self.distance < 0.1:
                self.distance = 0.1

        if self.action == Camera.ACTION_PAN:
            look_at = self.compute_look_at().T
            x = look_at[:3, 0]
            y = look_at[:3, 1]
            target = self.target + (-dx * x + dy * y) * 0.01
            self.target = (target[0], target[1], target[2])

        if self.action == Camera.ACTION_ORBIT:
            self.pitch += dy
            self.yaw -= dx
            if self.pitch > 89:
                self.pitch = 89
            elif self.pitch < -89:
                self.pitch = -89
