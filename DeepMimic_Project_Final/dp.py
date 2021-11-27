from util import *
import dartpy as dart
from mesh import Mesh
import xml.etree.ElementTree as ET
import numpy as np
import math
import os

class World:
    def __init__(self):
        self.skeletons = []
        self.reference = None

    @classmethod
    def from_dartpy(cls, dp_world):
        world = cls()
        world.reference = dp_world
        for i in range(dp_world.getNumSkeletons()):
            dp_skeleton = dp_world.getSkeleton(i)
            skeleton = Skeleton.from_dartpy(dp_skeleton)
            world.skeletons.append(skeleton)
        return world

    def print(self):
        print("----")
        for skeleton in self.skeletons:
            skeleton.print()
        print("----")

    def draw(self, program, view_proj_mat):
        for skeleton in self.skeletons:
            skeleton.draw(program, view_proj_mat)

class Skeleton:
    def __init__(self):
        self.bodies = []
        self.reference = None

    @classmethod
    def from_dartpy(cls, dp_skeleton):
        skeleton = cls()
        skeleton.reference = dp_skeleton
        for i in range(dp_skeleton.getNumBodyNodes()):
            dp_body = dp_skeleton.getBodyNode(i)
            body = Body.from_dartpy(dp_body)
            skeleton.bodies.append(body)
        return skeleton

    def print(self):
        print("name: {}".format(self.reference.getName()))
        for body in self.bodies:
            body.print()

    def draw(self, program, view_proj_mat):
        for body in self.bodies:
            body.draw(program, view_proj_mat)

class Body:
    def __init__(self):
        self.color = (0.5, 0.5, 0.5)
        self.size = (1.0, 1.0, 1.0)
        self.position = (0.0, 0.0, 0.0)
        self.rotation = (0.0, 0.0, 0.0)
        self.mesh = None
        self.body_reference = None
        self.shape_reference = None

    @classmethod
    def from_dartpy(cls, dp_body):
        body = cls()
        body.body_reference = dp_body
        for shape in dp_body.getShapeNodes():
            if shape.getVisualAspect():
                body.shape_reference = shape
                break

        v = body.shape_reference.getVisualAspect()
        s = body.shape_reference.getShape()
        body.color = v.getRGB()
        body.size = s.getSize()

        transform = dp_body.getTransform()
        body.position = transform.translation()
        body.rotation = transform.rotation()
        body.mesh = Mesh.box(body.size)
        return body

    def draw(self, program, view_proj_mat):
        # draw self.mesh using position / rotation
        model_mat = self.shape_reference.getTransform().matrix()
        mvp_mat = np.matmul(view_proj_mat, model_mat)

        program.set_uniform_mat4_np("mvpTransform", mvp_mat)
        program.set_uniform_mat4_np("modelTransform", model_mat)
        program.set_uniform_int("useTexture", 0)
        program.set_uniform_vec3("matColor", self.color)
        self.mesh.draw()

    def print(self):
        print("size: {}, pos: {}, rot: {}".format(self.size, self.position, self.rotation))
        print("transform: {}".format(self.body_reference.getTransform()))
        t = self.shape_reference.getTransform()
        print("shape transform: {}".format(t))

def convert_bvh_to_skel(mocap, filename, weight=60.0):
    joint_pos = []
    sum_of_joint_length = 0.0
    joint_length = np.zeros((len(mocap.data.get_joints())))
    for joint in mocap.data.get_joints():
        joint_index = mocap.data.get_joint_index(joint.name)
        parent_joint = mocap.data.joint_parent(joint.name)
        offset = mocap.data.joint_offset(joint.name)
        
        current_joint = joint
        joint_depth = 0
        while current_joint is not None:
            joint_depth += 1
            current_joint = mocap.data.joint_parent(current_joint.name)

        if parent_joint is None: 
            offset = np.array([0.0, 0.0, 0.0])
            offset_length = 0.0
        else:
            offset = np.array(offset)
            offset = offset * mocap.joint_scale
            offset_length = np.linalg.norm(offset)

        # print("{}: {}".format(joint.name, joint_depth))
        offset_length = max(mocap.joint_width, offset_length) / (joint_depth + 0.5)

        joint_length[joint_index] = offset_length
        sum_of_joint_length += offset_length

        model_mat = mocap.get_world_transform(joint, 0)
        joint_pos.append((joint_index, model_mat[1][3]))

    # print("joint length:", joint_length)
    # print("sum of joint length:", sum_of_joint_length)
    body_weight = joint_length / sum_of_joint_length * weight

    joint_pos = sorted(joint_pos, key=lambda j: j[1])
    # print("joint index/y pair:", joint_pos)

    foot_joint_indices = [joint_pos[0][0], joint_pos[1][0]]

    root = ET.Element("skel")
    tree = ET.ElementTree(root)
    root.set("version", "1.0")
    world = ET.SubElement(root, "world")
    world.set("name", "world 1")
    physics = ET.SubElement(world, "physics")
    time_step = ET.SubElement(physics, "time_step")
    time_step.text = "0.001"
    gravity = ET.SubElement(physics, "gravity")
    gravity.text = "0 -9.81 0"

    skeleton = ET.SubElement(world, "skeleton")
    skeleton.set("name", "ground")
    mobile = ET.SubElement(skeleton, "mobile")
    mobile.text = "false"
    body = ET.SubElement(skeleton, "body")
    body.set("name", "ground_body")

    visual_shape = ET.SubElement(body, "visualization_shape")
    ET.SubElement(visual_shape, "transformation").text = "0 0 0 0 0 0"
    visual_geometry = ET.SubElement(visual_shape, "geometry")
    visual_box = ET.SubElement(visual_geometry, "box")
    ET.SubElement(visual_box, "size").text = "20.0 0.05 20.0"
    ET.SubElement(visual_shape, "color").text = "0.5 0.2 0.0"

    col_shape = ET.SubElement(body, "collision_shape")
    ET.SubElement(col_shape, "transformation").text = "0 0 0 0 0 0"
    col_geometry = ET.SubElement(col_shape, "geometry")
    col_box = ET.SubElement(col_geometry, "box")
    ET.SubElement(col_box, "size").text = "20.0 0.05 20.0"
    
    joint = ET.SubElement(skeleton, "joint")
    joint.set("name", "ground_joint")
    joint.set("type", "free")
    ET.SubElement(joint, "parent").text = "world"
    ET.SubElement(joint, "child").text = "ground_body"

    skeleton = ET.SubElement(world, "skeleton")
    skeleton.set("name", "fullbody1")
    
    for joint in mocap.data.get_joints():
            
        joint_index = mocap.data.get_joint_index(joint.name)
        parent_joint = mocap.data.joint_parent(joint.name)

        # ### TODO: remove multiple child joint
        # if parent_joint is not None and mocap.has_multiple_children(parent_joint):
        #     continue
            
        body = ET.SubElement(skeleton, "body")
        body.set("name", joint.name)
        offset = mocap.data.joint_offset(joint.name)
        if parent_joint is None: 
            offset = np.array([0.0, 0.0, 0.0])
            offset_length = 0.0
            model_mat = mocap.get_world_transform(joint, 0)
        else:
            offset = np.array(offset)
            offset = offset * mocap.joint_scale
            offset_length = np.linalg.norm(offset)
            model_mat = mocap.get_world_transform(parent_joint, 0)

        offset_str = str(offset[0] * 0.5) +" "+str(offset[1] * 0.5) +" "+str(offset[2] * 0.5)
        pos = model_mat[:3, 3]
        rot = model_mat[:3, :3]
        if joint_index in foot_joint_indices:
            foot_y_axis = model_mat[:3, 1]
            y_axis = np.array([0.0, 1.0, 0.0])

            print("foot y axis:", foot_y_axis)

            cross = np.cross(foot_y_axis, y_axis)
            c = np.dot(y_axis, foot_y_axis)
            rot_axis = cross / np.linalg.norm(cross)

            model_mat = np.matmul(rot_mat(math.acos(c), rot_axis), model_mat)
            rot = model_mat[:3, :3]

        # print(rot.dtype, rot.shape)
        euler_angle = dart.math.matrixToEulerXYZ(rot.astype(np.float64))
        xform_str = "{} {} {} {} {} {}".format(
            pos[0], pos[1], pos[2], euler_angle[0], euler_angle[1], euler_angle[2])
        # print("transformation: " + xform_str)
        ET.SubElement(body, "transformation").text = xform_str

        inertia = ET.SubElement(body, "inertia")
        ET.SubElement(inertia, "mass").text = str(body_weight[joint_index])
        ET.SubElement(inertia, "offset").text = offset_str

        visual_shape = ET.SubElement(body, "visualization_shape")
        ET.SubElement(visual_shape, "transformation").text = offset_str + " 0 0 0"
        visual_geometry = ET.SubElement(visual_shape, "geometry")
        visual_box = ET.SubElement(visual_geometry, "box")

        if joint_index in foot_joint_indices:
            offset_length *= 1.6

        # length = max(offset_length, 0.1) - 0.04
        length = max(offset_length, 0.1)
        width = np.sqrt(0.0005 * body_weight[joint_index] / length)

        # child_joints = mocap.data.joint_direct_children(joint.name)
        # if len(child_joints) == 1 and mocap.data.get_joint_index(child_joints[0].name) in foot_joint_indices:
        #     print("{} is parent of foot".format(joint.name))
        #     length -= 0.1

        box_size = ("{} {} {}").format(length, width, width)

        ET.SubElement(visual_box, "size").text = box_size
        #ET.SubElement(visual_shape, "color").text = "0.0 1.0 0.0"

        col_shape = ET.SubElement(body, "collision_shape")
        ET.SubElement(col_shape, "transformation").text = offset_str + " 0 0 0"
        col_geometry = ET.SubElement(col_shape, "geometry")
        col_box = ET.SubElement(col_geometry, "box")
        ET.SubElement(col_box, "size").text = box_size

        joint_element = ET.SubElement(skeleton, "joint")
        joint_element.set("name", joint.name+"_joint")
        if parent_joint is None: 
            joint_element.set("type", "free")
            ET.SubElement(joint_element, "parent").text = "world"            
        else:
            joint_element.set("type", "ball")
            parent_joint = mocap.data.joint_parent(joint.name)

            # ### TODO: remove multiple child joint
            grand_parent_joint = mocap.data.joint_parent(parent_joint.name)
            parent_set = False
            # if grand_parent_joint is not None and mocap.has_multiple_children(grand_parent_joint):
            #     print("set parent to", grand_parent_joint.name)
            #     ET.SubElement(joint_element, "parent").text = grand_parent_joint.name
            #     parent_set = True

            if not parent_set:
                ET.SubElement(joint_element, "parent").text = parent_joint.name

        ET.SubElement(joint_element, "child").text = joint.name

    tree.write(filename)
