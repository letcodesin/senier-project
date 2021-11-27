from util import *
import numpy as np
from bvhtoolbox import bvh
from mesh import Mesh
import math
import os
from scipy.spatial.transform import Rotation as R

class Mocap:
    @classmethod
    def load(cls, filename, use_gl=True):
        return cls(filename, use_gl=use_gl)

    def __init__(self, filename, use_gl=True):
        self.filename = filename

        self.current_time = 0.0
        self.current_joint_index = 0
        self.joint_scale = 0.01
        self.joint_width = 0.1
        self.data = None

        with open(filename) as f:
            self.data = bvh.Bvh(f.read())
            joints = self.data.get_joints()
            print("#frames:", self.data.nframes)
            print("frame time:", self.data.frame_time)

            if use_gl:
                self.joint_mesh_box = Mesh.box((1.0, 1.0, 1.0))
                self.joint_mesh = Mesh.cylinder(0.3, 0.5, 1.0)

            self.joint_count = len(joints)
            ### #frame * #joint * 4 * 4
            self.local_transform = np.zeros([self.data.nframes, self.joint_count, 4, 4])
            self.world_transform = np.zeros([self.data.nframes, self.joint_count, 4, 4])
            self.com = np.zeros([self.data.nframes, 3])

            cache_filename = filename + ".transform.npy"
            if os.path.isfile(cache_filename):
                with open(cache_filename, 'rb') as f:
                    self.local_transform = np.load(f)
                    self.world_transform = np.load(f)
                    self.com = np.load(f)
            else:
                print("computing local transforms...")
                for i in range(self.data.nframes):
                    if i % 100 == 0:
                        print("compute frame {} / {}".format(i, self.data.nframes))
                    for j in range(self.joint_count):
                        self.local_transform[i, j, :, :] = self.compute_local_transform(joints[j], i)

                print("computing world transforms...")
                for i in range(self.data.nframes):
                    if i % 100 == 0:
                        print("compute frame {} / {}".format(i, self.data.nframes))
                    for j in range(self.joint_count):
                        self.world_transform[i, j, :, :] = self.compute_world_transform(joints[j], i)
                
                print("computing center-of-mass...")
                for i in range(self.data.nframes):
                    if i % 100 == 0:
                        print("compute frame {} / {}".format(i, self.data.nframes))
                    self.com[i, :] = np.mean(self.world_transform[i, :, 0:3, 3].reshape((-1, 3)), axis=0)

                with open(cache_filename, 'wb') as f:
                    np.save(f, self.local_transform)
                    np.save(f, self.world_transform)
                    np.save(f, self.com)

            end_effector_names = ["LeftToe", "RightToe", "LeftHand", "RightHand"]
            self.end_effectors = []
            for joint in self.data.get_joints():
                if joint.name in end_effector_names:
                    self.end_effectors.append(joint)
                    print("end effector:", joint.name)

    def compute_local_transform(self, joint, current_frame):
        parent_joint = self.data.joint_parent(joint.name)
        if parent_joint is None:
            offset = np.array((0, 0, 0))
        else:
            offset = np.array(self.data.joint_offset(joint.name))
        channels = self.data.joint_channels(joint.name)
        channel_values = self.data.frame_joint_channels(current_frame, joint.name, channels)
        joint_mat = translate_mat(offset * self.joint_scale)
        for (channel, value) in zip(channels, channel_values):
            if channel == 'Xposition':
                channel_mat = translate_mat([value * self.joint_scale, 0.0, 0.0])
            elif channel == 'Yposition':
                channel_mat = translate_mat([0.0, value * self.joint_scale, 0.0])
            elif channel == 'Zposition':
                channel_mat = translate_mat([0.0, 0.0, value * self.joint_scale])
            elif channel == 'Xrotation':
                channel_mat = rot_mat_x(deg2rad(value))
            elif channel == 'Yrotation':
                channel_mat = rot_mat_y(deg2rad(value))
            elif channel == 'Zrotation':
                channel_mat = rot_mat_z(deg2rad(value))
            joint_mat = np.matmul(joint_mat, channel_mat)
        
        return joint_mat

    def compute_world_transform(self, joint, current_frame):
        model_mat = np.identity(4)
        curr_joint = joint
        while curr_joint is not None:
            joint_mat = self.get_local_transform(curr_joint, current_frame)
            model_mat = np.matmul(joint_mat, model_mat)
            curr_joint = self.data.joint_parent(curr_joint.name)

        return model_mat

    def get_local_transform(self, joint, current_frame):
        return self.local_transform[current_frame, self.data.get_joint_index(joint.name), :, :]

    def get_world_transform(self, joint, current_frame):
        return self.world_transform[current_frame, self.data.get_joint_index(joint.name), :, :]

    def get_current_frame(self):
        if self.data is None:
            return 0
        current_frame = round(self.current_time / self.data.frame_time)
        if current_frame < 0:
            current_frame = 0
        if current_frame > self.data.nframes - 1:
            current_frame = self.data.nframes - 1
        return current_frame

    def set_current_frame(self, frame):
        if self.data is None:
            return
        self.current_time = frame * self.data.frame_time

    def has_multiple_children(self, joint):
        child_joints = list(joint.filter('JOINT'))
        return len(child_joints) > 1

    def draw(self, program, view_proj_mat, mocap_offset):
        current_frame = self.get_current_frame()
        for joint in self.data.get_joints():
            joint_index = self.data.get_joint_index(joint.name)
            parent_joint = self.data.joint_parent(joint.name)
            if parent_joint is None:
                model_mat = self.get_world_transform(joint, current_frame)
            else:
                model_mat = self.get_world_transform(parent_joint, current_frame)
            offset = self.data.joint_offset(joint.name)
            offset = np.array(offset)
            offset_length = np.linalg.norm(offset)
            offset_dir = offset / offset_length
            offset_length *= self.joint_scale
            cylinder_dir = np.array([0.0, 0.0, 1.0])

            cross = np.cross(cylinder_dir, offset_dir)
            c = np.dot(cylinder_dir, offset_dir)
            rot_axis = cross / np.linalg.norm(cross)

            model_mat = np.matmul(translate_mat(mocap_offset), model_mat)
            if parent_joint is None:
                model_mat = np.matmul(model_mat, scale_mat(np.ones(3) * self.joint_width))
                mvp_mat = np.matmul(view_proj_mat, model_mat)
                program.set_uniform_mat4_np("mvpTransform", mvp_mat)
                program.set_uniform_mat4_np("modelTransform", model_mat)
                program.set_uniform_int("useTexture", 0)
                program.set_uniform_vec3("matColor",
                    (1.0, 1.0, 0.0) if joint_index == self.current_joint_index else (1.0, 0.0, 0.0))
                self.joint_mesh_box.draw()
            else:
                scale = scale_mat([self.joint_width, self.joint_width, offset_length])
                rot = rot_mat(math.acos(c), rot_axis)
                model_mat = np.matmul(np.matmul(model_mat, rot), scale)
                mvp_mat = np.matmul(view_proj_mat, model_mat)

                program.set_uniform_mat4_np("mvpTransform", mvp_mat)
                program.set_uniform_mat4_np("modelTransform", model_mat)
                program.set_uniform_int("useTexture", 0)
                program.set_uniform_vec3("matColor",
                    (1.0, 1.0, 0.0) if joint_index == self.current_joint_index else (1.0, 0.0, 0.0))
                self.joint_mesh.draw()

    def draw_pose(self, program, view_proj_mat, target_pose, mocap_offset):
        local_mat = []
        world_mat = []
        for joint in self.data.get_joints():
            joint_index = self.data.get_joint_index(joint.name)
            parent_joint = self.data.joint_parent(joint.name)

            if parent_joint is None:
                offset = np.array((0, 0, 0))
            else:
                offset = np.array(self.data.joint_offset(joint.name))
            joint_mat = translate_mat(offset * self.joint_scale)

            model_mat = target_pose[joint_index]
            model_mat[:3, 3] = np.zeros(3)
            local_mat.append(np.matmul(joint_mat, model_mat))

        for joint in self.data.get_joints():
            joint_index = self.data.get_joint_index(joint.name)
            parent_joint = self.data.joint_parent(joint.name)
            if parent_joint is None:
                world_mat.append(local_mat[joint_index])
                continue
            parent_joint_index = self.data.get_joint_index(parent_joint.name)
            parent_world_mat = world_mat[parent_joint_index]
            world_mat.append(np.matmul(parent_world_mat, local_mat[joint_index]))
        
        for joint in self.data.get_joints():
            joint_index = self.data.get_joint_index(joint.name)
            parent_joint = self.data.joint_parent(joint.name)
            if parent_joint is None:
                model_mat = world_mat[joint_index]
            else:
                parent_joint_index = self.data.get_joint_index(parent_joint.name)
                model_mat = world_mat[parent_joint_index]
            offset = self.data.joint_offset(joint.name)
            offset = np.array(offset)
            offset_length = np.linalg.norm(offset)
            offset_dir = offset / offset_length
            offset_length *= self.joint_scale
            cylinder_dir = np.array([0.0, 0.0, 1.0])

            cross = np.cross(cylinder_dir, offset_dir)
            c = np.dot(cylinder_dir, offset_dir)
            rot_axis = cross / np.linalg.norm(cross)

            model_mat = np.matmul(translate_mat(mocap_offset), model_mat)
            if parent_joint is None:
                model_mat = np.matmul(model_mat, scale_mat(np.ones(3) * self.joint_width))
                mvp_mat = np.matmul(view_proj_mat, model_mat)
                program.set_uniform_mat4_np("mvpTransform", mvp_mat)
                program.set_uniform_mat4_np("modelTransform", model_mat)
                program.set_uniform_int("useTexture", 0)
                program.set_uniform_vec3("matColor", (0.0, 0.0, 1.0))
                self.joint_mesh_box.draw()
            else:
                scale = scale_mat([self.joint_width, self.joint_width, offset_length])
                rot = rot_mat(math.acos(c), rot_axis)
                model_mat = np.matmul(np.matmul(model_mat, rot), scale)
                mvp_mat = np.matmul(view_proj_mat, model_mat)

                program.set_uniform_mat4_np("mvpTransform", mvp_mat)
                program.set_uniform_mat4_np("modelTransform", model_mat)
                program.set_uniform_int("useTexture", 0)
                program.set_uniform_vec3("matColor", (0.0, 0.0, 1.0))
                self.joint_mesh.draw()