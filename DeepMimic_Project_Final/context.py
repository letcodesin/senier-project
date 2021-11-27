from camera import Camera, perspective
from program import Program
from shader_code import *
from mesh import Mesh
from dart_env import DartEnv
from stable_baselines3 import PPO

import numpy as np
from util import *

import imgui
import glfw
from OpenGL.GL import *

import torch
import os

def load_text(filename):
    with open(filename) as f:
        text = f.read()
        return text
    raise Exception('cannot load file: {}'.format(filename))

class Context:
    @classmethod
    def create(cls, width, height):
        return cls(width, height)

    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.mouse_pos = (0.0, 0.0)
        self.camera = Camera()

        self.program = {
            'lighting': Program.from_vert_and_frag_code(
                load_text('shader/lighting.vs'),
                load_text('shader/lighting.fs')),
            'simple': Program.from_vert_and_frag_code(
                load_text('shader/simple.vs'),
                load_text('shader/simple.fs'))
        }

        self.grid = Mesh.grid(0.5, 51)
        self.plane = Mesh.plane(size=(0.5, 0.5))
        self.box = Mesh.box(size=(0.1, 0.1, 0.1))

        glViewport(0, 0, self.width, self.height)
        glEnable(GL_DEPTH_TEST)

        self.bgcolor = [0.1, 0.2, 0.3, 0.0]
        self.light_pos = (3.0, 3.0, 3.0)
        self.light_ambient = (0.1, 0.1, 0.1)
        self.light_diffuse = (0.8, 0.8, 0.8)
        self.shininess = 32
        self.light_specular = (1.0, 1.0, 1.0)
        self.on_simulating = False

        self.on_animating = False
        self.on_track_camera = False

        self.mocap_offset = (1.0, 0.0, 0.0)
        self.use_mocap_target = True
        self.target_mocap_pose = None
        self.show_target_pose = False

        glClearColor(*self.bgcolor)

        # TODO: mlp policy + height map
        policy_kwargs = {
            "activation_fn": torch.nn.ReLU,
            "net_arch": [{
                "pi": [1024, 512],
                "vf": [1024, 512],
            }],
        }

        # self.env = DartEnv(bvh_filename="bvh/walk1_subject1_short.bvh", skel_filename="data/fullbody1_ball.skel")
        self.env = DartEnv(bvh_filename="bvh/walk1_subject1_short.bvh")
        if os.path.isfile("walk.model.zip"):
            self.model = PPO.load("walk.model.zip", self.env)
            print("load from previous model...")
        else:
            self.model = PPO("MlpPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1)
        # self.model.learn(total_timesteps=10240)
        # self.model.save("walk.model.zip")
        self.obs = self.env.reset()
        self.done = False

    def cursor_pos(self, x, y):
        dx = x - self.mouse_pos[0]
        dy = y - self.mouse_pos[1]
        self.camera.modify_with_mouse_pos(dx, dy)
        self.mouse_pos = (x, y)

    def mouse_button(self, button, action, mods, x, y):
        # print("mouse_button")
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            if mods & glfw.MOD_CONTROL:
                self.camera.action = Camera.ACTION_ZOOM
            elif mods & glfw.MOD_SHIFT:
                self.camera.action = Camera.ACTION_PAN
            elif mods & glfw.MOD_ALT:
                self.camera.action = Camera.ACTION_ORBIT
            else:
                self.camera.action = Camera.ACTION_NONE
            self.mouse_pos = (x, y)
        elif self.camera.action != Camera.ACTION_NONE and button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
            self.camera.action = Camera.ACTION_NONE

    def framebuffer_size(self, width, height):
        self.width = width
        self.height = height
        glViewport(0, 0, self.width, self.height)

    def render_ui(self):
        if imgui.begin("view"):
            imgui.text("this is from imgui")
            if imgui.button("click me"):
                print("clicked")
            edit, self.bgcolor = imgui.color_edit4("background", *self.bgcolor)
            if edit:
                glClearColor(*self.bgcolor)
            edit, self.camera.pitch = imgui.drag_float("camera pitch", self.camera.pitch, 0.5, -89, 89)
            edit, self.camera.yaw = imgui.drag_float("camera yaw", self.camera.yaw, 0.5)
            edit, self.camera.distance = imgui.drag_float("camera distance", self.camera.distance, 0.1)
            edit, self.camera.target = imgui.drag_float3("camera target", *self.camera.target, 0.05)

            imgui.separator()
            edit, self.light_pos = imgui.drag_float3("light pos", *self.light_pos, 0.05)
            edit, self.light_ambient = imgui.color_edit3("light ambient", *self.light_ambient)
            edit, self.light_diffuse = imgui.color_edit3("light diffuse", *self.light_diffuse)
            edit, self.shininess = imgui.drag_float("shininess", self.shininess)
            edit, self.light_specular = imgui.color_edit3("light specular", *self.light_specular)
        imgui.end()

        if self.env.mocap is not None:
            mocap = self.env.mocap
            if imgui.begin("bvh"):
                edit, self.on_track_camera = imgui.checkbox("track camera", self.on_track_camera)            
                current_frame = mocap.get_current_frame()
                imgui.label_text("frame", "{} / {}".format(current_frame, mocap.data.nframes))
                edit, mocap.current_time = imgui.drag_float("time", mocap.current_time, 0.01, 0,
                    mocap.data.nframes * mocap.data.frame_time)
                edit, self.on_animating = imgui.checkbox("animation", self.on_animating)

                edit, self.mocap_offset = imgui.drag_float3("mocap offset",
                    self.mocap_offset[0], self.mocap_offset[1], self.mocap_offset[2], 0.01)

                edit, mocap.current_joint_index = imgui.drag_int("joint index",
                    mocap.current_joint_index, 0.2, 0, len(mocap.data.get_joints()) - 1)
                joint = mocap.data.get_joints()[mocap.current_joint_index]
                imgui.label_text("joint name", joint.name)
                offset = mocap.data.joint_offset(joint.name)
                imgui.drag_float3("joint offset", offset[0], offset[1], offset[2])
                channels = mocap.data.joint_channels(joint.name)
                channel_values = mocap.data.frame_joint_channels(current_frame, joint.name, channels)
                for (channel, value) in zip(channels, channel_values):
                    imgui.drag_float(channel, value)
                
            imgui.end()

        if self.env.dp_world is not None:
            dp_world = self.env.dp_world
            if imgui.begin("simulation"):
                time_step = dp_world.getTimeStep()
                edit, new_time_step = imgui.input_float("time step", time_step)
                if edit:
                    dp_world.setTimeStep(new_time_step)
                    self.env.initialize_controller()

                edit, self.env.kp_param = imgui.input_float("propotion", self.env.kp_param)
                if edit:
                    self.env.initialize_controller()
                edit, self.env.kd_param = imgui.input_float("derivative", self.env.kd_param)
                if edit:
                    self.env.initialize_controller()

                edit, self.on_simulating = imgui.checkbox("simulate", self.on_simulating)
                edit, self.use_mocap_target = imgui.checkbox("use mocap target", self.use_mocap_target)

                edit, self.show_target_pose = imgui.checkbox("show target pose", self.show_target_pose)

                if imgui.button("reset simulation"):
                    self.obs = self.env.reset()
                    self.done = False

                imgui.separator()

                for i in range(dp_world.getNumSkeletons()):
                    dp_skeleton = dp_world.getSkeleton(i)
                    if imgui.button("control skel {:2d}:{}".format(i, dp_skeleton.getName())):
                        self.set_body_skeleton(i)

                imgui.separator()

                if imgui.button("set target pos"):
                    current_frame = self.env.mocap.get_current_frame()
                    self.env.target_pos = self.env.mocap_target_pos[current_frame, :]
                    self.env.target_vel = self.env.mocap_target_vel[current_frame, :]

                if imgui.button("set body to target pos"):
                    self.env.set_body_skeleton_to_target_pos()

            imgui.end()

        if self.on_simulating:
            # if self.use_mocap_target:
            #     # set current frame as target
            #     # then increase current frame
            #     current_frame = self.env.mocap.get_current_frame()
            #     self.env.target_pos = self.env.mocap_target_pos[current_frame, :]
            #     self.env.target_vel = self.env.mocap_target_vel[current_frame, :]
            #     self.env.mocap.set_current_frame(current_frame + 1)

            # if not self.done:
            _, pos, _ = self.env.get_observation()
            action, _state = self.model.predict(self.obs)
            target_pos = self.env.convert_action_to_target_pos(action, pos)
            self.target_mocap_pose = self.env.convert_target_pos_to_mocap_pose(target_pos)
            self.obs, rewards, self.done, info = self.env.step(action)

        if self.on_track_camera:
            current_frame = self.env.mocap.get_current_frame()
            joint = self.env.mocap.data.get_joints()[self.env.mocap.current_joint_index]
            transform = self.env.mocap.get_world_transform(joint, current_frame)
            self.camera.target = (
                transform[3][0] * 0.5 + self.camera.target[0] * 0.5,
                transform[3][1] * 0.5 + self.camera.target[1] * 0.5,
                transform[3][2] * 0.5 + self.camera.target[2] * 0.5)

    def render_scene(self, delta_time):
        aspect_ratio = self.width / self.height
        look_at = self.camera.compute_look_at()
        proj = perspective(np.pi / 3.0, aspect_ratio, 0.01, 50)
        view_proj_mat = np.matmul(proj, look_at)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        program = self.program['simple']
        program.use()
        program.set_uniform_vec4("color", (1.0, 1.0, 1.0, 1.0))
        program.set_uniform_mat4_np("mvpTransform",
            np.matmul(view_proj_mat, rot_mat_x(deg2rad(90.0))))
        self.grid.draw()

        program = self.program['lighting']
        program.use()
        program.set_uniform_vec3("lightPos", self.light_pos)
        program.set_uniform_vec3("ambientLight", self.light_ambient)
        program.set_uniform_vec3("diffuseLight", self.light_diffuse)
        program.set_uniform_vec3("specularLight", self.light_specular)
        program.set_uniform_float("shininess", self.shininess)
        program.set_uniform_vec3("viewPos", self.camera.camera_pos())

        if self.env is not None:
            self.env.render(program, view_proj_mat)

        if self.on_animating:
            self.env.mocap.current_time += delta_time
            if self.env.mocap.current_time > self.env.mocap.data.nframes * self.env.mocap.data.frame_time:
                self.env.mocap.current_time = 0.0

        # draw bvh
        if self.env.mocap is not None:
            self.env.mocap.draw(program, view_proj_mat, self.mocap_offset)

        if self.target_mocap_pose is not None and self.show_target_pose:
            com = self.env.get_root_pos()
            mocap_offset = [-self.mocap_offset[0] + com[0], self.mocap_offset[1] + com[1], self.mocap_offset[2] + com[2]]
            self.env.mocap.draw_pose(program, view_proj_mat, self.target_mocap_pose, mocap_offset)

        if self.env.rel_pos is not None:
            rel_pos = self.env.rel_pos
            joint_count = rel_pos.shape[0] // 3
            root_mat = self.env.get_root_transform()

            model_mat = np.matmul(root_mat, translate_mat([1, 0, 0]))
            program.set_uniform_mat4_np("mvpTransform", np.matmul(view_proj_mat, model_mat))
            program.set_uniform_mat4_np("modelTransform", model_mat)
            program.set_uniform_int("useTexture", 0)
            program.set_uniform_vec3("matColor", (1.0, 0.0, 0.0))
            self.box.draw()

            model_mat = np.matmul(root_mat, translate_mat([0, 1, 0]))
            program.set_uniform_mat4_np("mvpTransform", np.matmul(view_proj_mat, model_mat))
            program.set_uniform_mat4_np("modelTransform", model_mat)
            program.set_uniform_int("useTexture", 0)
            program.set_uniform_vec3("matColor", (0.0, 1.0, 0.0))
            self.box.draw()

            model_mat = np.matmul(root_mat, translate_mat([0, 0, 1]))
            program.set_uniform_mat4_np("mvpTransform", np.matmul(view_proj_mat, model_mat))
            program.set_uniform_mat4_np("modelTransform", model_mat)
            program.set_uniform_int("useTexture", 0)
            program.set_uniform_vec3("matColor", (0.0, 0.0, 1.0))
            self.box.draw()

            for i in range(joint_count):
                model_mat = np.matmul(root_mat, translate_mat(rel_pos[3*i:3*i+3]))
                program.set_uniform_mat4_np("mvpTransform", np.matmul(view_proj_mat, model_mat))
                program.set_uniform_mat4_np("modelTransform", model_mat)
                program.set_uniform_int("useTexture", 0)
                program.set_uniform_vec3("matColor", (1.0, 1.0, 0.0))
                self.box.draw()
