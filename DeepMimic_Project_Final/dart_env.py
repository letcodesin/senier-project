import dartpy as dart
from mocap import Mocap
from dp import World, convert_bvh_to_skel

import numpy as np
from scipy.linalg import cho_factor, cho_solve
import math
import os
import gym
from gym import spaces
from scipy.spatial.transform import Rotation as R

import csv

# TODO: reference state initialization
# TODO: observation space (linear term)

# TODO: early termination (already done)

class DartEnv(gym.Env):
    def __init__(self,
            bvh_filename="",
            character_weight=60.0,
            skel_filename="",
            use_gl=True,
            active_joint_name=["LeftLeg","RightLeg","LeftFoot","RightFoot","LeftToe","RightToe","LeftArm","LeftForeArm","LeftHand","RightArm","RightForeArm","RightHand"],
            #"LeftUpLeg","RightUpLeg"
            #"LeftLeg","RightLeg","LeftFoot","RightFoot","LeftToe","RightToe","LeftArm","LeftForeArm","LeftHand","RightArm","RightForeArm","RightHand"
            inc_frame_for_step=True,#False
            angle_range=0.5,
            sigma=0.0005):

        self.dp_world = None
        self.world = None
        self.body_skeleton = None
        self.initial_pos = None
        self.initial_vel = None
        self.character_weight = character_weight

        self.mocap = Mocap.load(bvh_filename, use_gl=use_gl)

        self.angle_range = angle_range
        self.sigma = sigma
        self.inc_frame_for_step = inc_frame_for_step

        if len(skel_filename) > 0:
            path = os.path.abspath(skel_filename)
            self.load_world(path)

            # find bvh-skel mapping from cache file
            self.joint_map = []
            max_bvh_joint_index = len(self.mocap.data.get_joints()) - 1
            for i in range(self.body_skeleton.getNumJoints()):
                self.joint_map.append(min(i, max_bvh_joint_index))
            print("joint map:", self.joint_map)
            # if not exist, initialize
            # compute target pos from mapping

            self.mocap_target_pos = np.zeros([self.mocap.data.nframes, self.initial_pos[self.body_skeleton_index].shape[0]])
            self.mocap_target_vel = np.zeros([self.mocap.data.nframes, self.initial_vel[self.body_skeleton_index].shape[0]])
            for i in range(self.mocap.data.nframes):
                if i % 100 == 0:
                    print("{} / {}".format(i, self.mocap.data.nframes))
                self.mocap_target_pos[i, :] = self.compute_target_pos_from_joint_map(i)
                # self.mocap_target_vel[i, :] = self.compute_target_vel_from_joint_map(i)

            return

        # automatically skel generation from bvh
        convert_bvh_to_skel(self.mocap, "temp.skel",
            weight=self.character_weight)
        path = os.path.abspath("temp.skel")
        print("path:", path)

        self.load_world(path, use_gl=use_gl)
        self.set_body_skeleton(1)

        self.mocap_target_pos = np.zeros([self.mocap.data.nframes, self.initial_pos[self.body_skeleton_index].shape[0]])
        self.mocap_target_vel = np.zeros([self.mocap.data.nframes, self.initial_vel[self.body_skeleton_index].shape[0]])

        cache_filename = bvh_filename + ".target.npy"
        if os.path.isfile(cache_filename):
            with open(cache_filename, 'rb') as f:
                self.mocap_target_pos = np.load(f)
                self.mocap_target_vel = np.load(f)
        else:    
            print("compute target pos")
            for i in range(self.mocap.data.nframes):
                if i % 100 == 0:
                    print("{} / {}".format(i, self.mocap.data.nframes))
                self.mocap_target_pos[i, :] = self.compute_target_pos(i)
                self.mocap_target_vel[i, :] = self.compute_target_vel(i)
            
            with open(cache_filename, 'wb') as f:
                np.save(f, self.mocap_target_pos)
                np.save(f, self.mocap_target_vel)

        print("creating action/observation space")
        joint_count = (self.dof - 6) // 3
        dof = joint_count * 3 # additional axis-angle rotation for a given state
        low = np.zeros((dof), dtype=np.float32)
        high = np.zeros((dof), dtype=np.float32)
        if active_joint_name is None:
            low.fill(-self.angle_range)
            high.fill(self.angle_range)
        elif type(active_joint_name) is list:
            active_joint_index = []
            for name in active_joint_name:
                for i in range(self.body_skeleton.getNumBodyNodes()):
                    body_node = self.body_skeleton.getBodyNode(i)
                    if body_node.getName() == name:
                        active_joint_index.append(i)

            for joint_index in active_joint_index:
                low[3*joint_index-3:3*joint_index] = -self.angle_range
                high[3*joint_index-3:3*joint_index] = self.angle_range
                

        #for i in range(joint_count):
        #    low[4*i+3] = -1.0
        #    high[4*i+3] = 1.0
        self.action_space = spaces.Box(low, high)

        # pos = quat * joint
        # vel = 3 * joint
        # rel_pos = 3 * joint
        # lin_vel = 3 * joint
        quat_shape = joint_count * 4
        ang_vel_shape = joint_count * 3
        rel_pos_shape = joint_count * 3
        lin_vel_shape = joint_count * 3
        ang_vel_range = quat_shape + ang_vel_shape
        rel_pos_range = ang_vel_range + rel_pos_shape
        lin_vel_range = rel_pos_range + lin_vel_shape
        low = np.zeros(lin_vel_range + 1, dtype=np.float32)
        high = np.zeros(lin_vel_range + 1, dtype=np.float32)
        low[0:quat_shape] = -1.0
        high[0:quat_shape] = 1.0
        low[quat_shape:ang_vel_range] = -20.0
        high[quat_shape:ang_vel_range] = 20.0
        low[ang_vel_range:rel_pos_range] = -1.5
        high[ang_vel_range:rel_pos_range] = 1.5
        low[rel_pos_range:lin_vel_range] = -20.0
        high[rel_pos_range:lin_vel_range] = 20.0
        # phase variable
        low[lin_vel_range] = 0.0
        high[lin_vel_range] = 1.0
        self.observation_space = spaces.Box(low, high)
        print("creating action/observation space done:", self.action_space.shape, self.observation_space.shape)

        self.rel_pos = None

    def load_world(self, path, use_gl=True):
        dp_world = dart.utils.SkelParser.readWorld("file://" + path)
        self.init_world(dp_world, use_gl=use_gl)

    def init_world(self, dp_world, use_gl=True):
        self.dp_world = dp_world
        if use_gl:
            self.world = World.from_dartpy(self.dp_world)
            # self.world.print()

        self.kp_param = 1500.0
        self.kd_param = 15.0

        self.initial_pos = []
        self.initial_vel = []
        self.body_skeleton = None
        for i in range(self.dp_world.getNumSkeletons()):
            dp_skeleton = self.dp_world.getSkeleton(i)
            pos = dp_skeleton.getPositions()
            vel = dp_skeleton.getVelocities()
            self.initial_pos.append(pos)
            self.initial_vel.append(vel)
            print("#bodies of skel[{}]: {}".format(i, dp_skeleton.getNumBodyNodes()))
            print("#DOF of skel[{}]: {}".format(i, self.initial_pos[i].shape[0]))
            print("initial pos:", pos)

    def get_root_transform(self):
        root = self.body_skeleton.getBodyNode("Hips")
        transform = root.getTransform().matrix()
        x = transform[:3, 2]
        y = np.array([0, 1, 0])
        z = np.cross(x, y)
        z /= np.linalg.norm(z)
        y = np.cross(z, x)
        y /= np.linalg.norm(y)
        pos = transform[:3, 3]

        mat = np.concatenate((x, y, z, pos)).reshape((4, 3)).T
        mat = np.pad(mat, ((0, 1), (0, 0)), 'constant', constant_values=0.0)
        mat[3, 3] = 1.0
        return mat

    def get_root_velocity(self):
        root = self.body_skeleton.getBodyNode("Hips")
        return root.getLinearVelocity()

    def get_observation(self):
        pos = self.body_skeleton.getPositions()
        vel = self.body_skeleton.getVelocities()
        joint_count = (self.dof - 6) // 3
        quats = np.zeros((joint_count * 4))
        for i in range(joint_count):
            index = i*3 + 6
            euler_angle = pos[index:index+3]
            quats[4*i:4*i+4] = R.from_euler("XYZ", euler_angle).as_quat()

        root_transform = self.get_root_transform()
        root_velocity = self.get_root_velocity()
        inv_root_transform = np.linalg.inv(root_transform)
        rel_pos = np.zeros((joint_count * 3))
        rel_vel = np.zeros((joint_count * 3))
        for i in range(joint_count):
            body_node = self.body_skeleton.getBodyNode(i + 1)
            rel_transform = np.matmul(inv_root_transform, body_node.getTransform().matrix())
            start_pos = rel_transform[:3, 3]
            com = (np.matmul(inv_root_transform, np.pad(body_node.getCOM(), (0, 1), 'constant', constant_values=1)))[:3]
            rel_pos[3*i:3*i+3] = start_pos + (com - start_pos) * 2

            v = body_node.getLinearVelocity() - root_velocity
            v = (np.matmul(inv_root_transform, np.pad(v, (0, 1), 'constant', constant_values=0)))[:3]
            rel_vel[3*i:3*i+3] = v

        phase = np.array([self.mocap.get_current_frame() / (self.mocap.data.nframes - 1.0)])
        observation = np.concatenate((quats, vel[6:], rel_pos, rel_vel, phase), axis=None).reshape((-1))

        # self.rel_pos = rel_pos

        return observation, pos, vel

    def convert_action_to_target_pos(self, action, pos, use_random=True, use_relative=False):
        joint_count = action.shape[0] // 3
        target_pos = np.zeros((self.dof))
        if use_random:
            sample = np.random.normal(action, self.sigma)
        else:
            sample = action
        for i in range(joint_count):
            index = i*3 + 6
            if use_relative:
                r = R.from_euler("XYZ", pos[index:index+3])
                ar = R.from_rotvec(sample[3*i:3*i+3])
                target_pos[index:index+3] = (r * ar).as_euler("XYZ")
            else:
                ar = R.from_rotvec(sample[3*i:3*i+3])
                target_pos[index:index+3] = ar.as_euler("XYZ")
        return target_pos

    def step(self, action):
        time_step = self.dp_world.getTimeStep()
        step_size = int(self.mocap.data.frame_time / time_step)

        observation, pos, vel = self.get_observation()
        target_pos = self.convert_action_to_target_pos(action, pos)
        for _ in range(step_size):
            observation, pos, vel = self.get_observation()

            if np.sum(np.isnan(pos)) != 0 or np.sum(np.isinf(pos)) != 0:
                print("reward: done")
                self.record_avg_reward()
                return np.zeros_like(observation), 0.0, True, {}

            cor_force = self.body_skeleton.getCoriolisAndGravityForces()
            con_force = self.body_skeleton.getConstraintForces()

            ## PD-control for rotational joints
            ## tau = kp * log(R.T * R_d) + kd * (v_d - v)
            ## naive difference
            # diff_pos = pos - self.target_pos
            ## rotational difference
            # diff_pos = DartEnv.compute_difference_of_rotation(pos, self.target_pos)
            # diff_pos = self.body_skeleton.getPositionDifferences(pos, self.target_pos)

            diff_pos = self.body_skeleton.getPositionDifferences(pos, target_pos)
            p = -np.matmul(self.kp, diff_pos + vel * time_step)
            d = -np.matmul(self.kd, vel)

            m = self.body_skeleton.getMassMatrix() + self.kd * time_step
            # inv_m = np.linalg.inv(self.body_skeleton.getMassMatrix() + self.kd * time_step)
            # qddot = np.matmul(inv_m, -cor_force + p + d + con_force) 
            c, low = cho_factor(m)

            check_nan = -cor_force + p + d + con_force #
            if np.sum(np.isnan(check_nan)) != 0 or np.sum(np.isinf(check_nan)) != 0:
                print("reward: done")
                self.record_avg_reward()
                return np.zeros_like(observation), 0.0, True, {} #

            qddot = cho_solve((c, low), -cor_force + p + d + con_force)

            force = p + d - np.matmul(self.kd, qddot) * time_step
            self.body_skeleton.setForces(force)

            self.dp_world.step()

        observation, pos, vel = self.get_observation()

        if np.sum(np.isnan(pos)) != 0 or np.sum(np.isinf(pos)) != 0:
            print("reward: done")
            self.record_avg_reward()
            return np.zeros_like(observation), 0.0, True, {}

        # diff_pos = self.body_skeleton.getPositionDifferences(pos, action)
        head = self.body_skeleton.getBodyNode("Head")
        com = head.getCOM()
        # reward_pos = DartEnv.compute_reward_imitation_pos(pos, self.target_pos)
        reward_pos = np.exp(-2.0 * np.sum((pos - self.target_pos) ** 2))
        reward_vel = np.exp(-0.1 * np.sum((vel - self.target_vel) ** 2))
        reward_end_effector = 0.0
        if len(self.end_effectors) > 0:
            end_effector_sqr_error = 0.0
            for i in range(len(self.end_effectors)):
                dp_joint = self.end_effectors[i]
                joint = self.mocap.end_effectors[i]
                parent_joint = self.mocap.data.joint_parent(joint.name)
                dp_body = dp_joint.getParentBodyNode()
                body_transform = dp_body.getTransform().matrix()
                joint_transform = dp_joint.getRelativeTransform().matrix()
                world_transform_dp = np.matmul(body_transform, joint_transform)
                world_transform_mocap = self.mocap.get_world_transform(parent_joint, self.mocap.get_current_frame())
                diff = world_transform_dp[:3, 3] - world_transform_mocap[:3, 3]
                end_effector_sqr_error += np.dot(diff, diff)
            reward_end_effector = np.exp(-40 * end_effector_sqr_error)
        
        body_com = self.body_skeleton.getCOM()
        reward_com = np.exp(-10 * np.sum((body_com - self.mocap.com[self.mocap.get_current_frame()]) ** 2))
        reward = reward_pos * 0.65 + reward_vel * 0.1 + reward_end_effector * 0.15 + reward_com * 0.1
        print("reward:", reward_pos, reward_vel, reward_end_effector, reward_com, reward)

        self.rewards.append([reward_pos, reward_vel, reward_end_effector , reward_com, reward])
        self.step_count += 1
        if self.inc_frame_for_step:
            current_frame = self.mocap.get_current_frame()
            self.target_pos = self.mocap_target_pos[current_frame, :]
            self.target_vel = self.mocap_target_vel[current_frame, :]
            self.mocap.set_current_frame(current_frame + 1)

        ##
        done = False
        info = {}
        if com[1] < 1.0 or com[1] > 2.0:
            reward = 0.0
            done = True
            print("reward: done")

        if done: self.record_avg_reward()

        return observation, reward, done, info

    def get_root_pos(self):
        root = self.body_skeleton.getBodyNode("Hips")
        com = root.getCOM()
        return com

    def record_avg_reward(self):
        filename = "reward_record"
        cache_filename = filename + ".csv"
        if os.path.isfile(cache_filename):
            f = open(cache_filename, 'a', encoding='utf-8', newline="")
        else:
            f = open(cache_filename, 'w', encoding='utf-8', newline="")
        wr = csv.writer(f)
        
        avg_reward_pos, avg_reward_vel, avg_reward_end, avg_reward_com, avg_reward = 0.0, 0.0, 0.0, 0.0, 0.0
        for r in self.rewards:
            avg_reward_pos += r[0]
            avg_reward_vel += r[1]
            avg_reward_end += r[2]
            avg_reward_com += r[3]
            avg_reward += r[4]
        avg_reward_pos /= self.step_count
        avg_reward_vel /= self.step_count
        avg_reward_end /= self.step_count
        avg_reward_com /= self.step_count
        avg_reward /= self.step_count
        wr.writerow([self.step_count, avg_reward_pos, avg_reward_vel, avg_reward_end, avg_reward_com, avg_reward])
        f.close()

    def reset(self):
        self.step_count = 0
        self.rewards = []

        self.mocap.set_current_frame(0)
        for i in range(self.dp_world.getNumSkeletons()):
            dp_skeleton = self.dp_world.getSkeleton(i)
            initial_pos = self.initial_pos[i].copy()
            if i == self.body_skeleton_index:
                initial_pos[3] += 0.05
            dp_skeleton.setPositions(initial_pos)
            dp_skeleton.setVelocities(self.initial_vel[i])

        observation, _, _ = self.get_observation()
        return observation       

    def render(self, program, view_proj_mat):
        self.world.draw(program, view_proj_mat)

    def close(self):
        pass

    def initialize_controller(self):
        if self.body_skeleton is None:
            return

        body_initial_pos = self.initial_pos[self.body_skeleton_index]
        body_initial_vel = self.initial_vel[self.body_skeleton_index]
        self.dof = body_initial_pos.shape[0]
        self.kp = np.zeros((self.dof, self.dof))
        self.kd = np.zeros((self.dof, self.dof))
        for i in range(6, self.dof):
            self.kp[i, i] = self.kp_param
            self.kd[i, i] = self.kd_param
        self.target_pos = body_initial_pos
        self.target_vel = body_initial_vel

        self.end_effectors = []
        for end_effector in self.mocap.end_effectors:
            for i in range(self.body_skeleton.getNumJoints()):
                dp_joint = self.body_skeleton.getJoint(i)
                if dp_joint.getName() == end_effector.name + "_joint":
                    self.end_effectors.append(dp_joint)
                    print("found same end effector:", dp_joint.getName())


    def set_body_skeleton(self, index):
        dp_skeleton = self.dp_world.getSkeleton(index)
        self.body_skeleton = dp_skeleton
        self.body_skeleton_index = index
        self.initialize_controller()

    def set_body_skeleton_to_target_pos(self):
        for i in range(self.dp_world.getNumSkeletons()):
            dp_skeleton = self.dp_world.getSkeleton(i)
            if i == self.body_skeleton_index:
                dp_skeleton.setPositions(self.target_pos)
            else:
                dp_skeleton.setPositions(self.initial_pos[i])
            dp_skeleton.setVelocities(self.initial_vel[i])

    def compute_target_pos(self, frame):
        target_pos = np.zeros_like(self.initial_pos[self.body_skeleton_index])

        # compute relative rotation values
        index = 0
        for joint in self.mocap.data.get_joints():
            joint_index = self.mocap.data.get_joint_index(joint.name)
            parent_joint = self.mocap.data.joint_parent(joint.name)

            # # ### TODO: remove multiple child joint
            # if parent_joint is not None and self.mocap.has_multiple_children(parent_joint):
            #     continue

            if parent_joint is None:
                init_model_mat = self.mocap.get_local_transform(joint, 0)
                model_mat = self.mocap.get_local_transform(joint, frame)
                rel_mat = np.matmul(np.linalg.inv(init_model_mat), model_mat)
            else:
                init_model_mat = self.mocap.get_local_transform(parent_joint, 0)
                model_mat = self.mocap.get_local_transform(parent_joint, frame)
                rel_mat = np.matmul(np.linalg.inv(init_model_mat), model_mat)

            euler_angle = dart.math.matrixToEulerXYZ(rel_mat[:3, :3].astype(np.float64))
            pos = np.array([rel_mat[3][0], rel_mat[3][1], rel_mat[3][2]])
            if parent_joint is None: 
                target_pos[0:3] = euler_angle
                target_pos[3:6] = pos
            else:
                target_pos[index * 3 + 3: index * 3 + 6] = euler_angle
            index += 1
        
        return target_pos

    def convert_target_pos_to_mocap_pose(self, target_pos):
        # compute relative rotation values
        index = 0
        joint_pose = [None for _ in self.mocap.data.get_joints()]

        for joint in self.mocap.data.get_joints():
            joint_index = self.mocap.data.get_joint_index(joint.name)
            parent_joint = self.mocap.data.joint_parent(joint.name)

            # # ### TODO: remove multiple child joint
            # if parent_joint is not None and self.mocap.has_multiple_children(parent_joint):
            #     continue

            # if parent_joint is None:
            #     init_model_mat = self.mocap.get_local_transform(joint, 0)
            #     rel_mat = np.identity(4)
            #     rel_mat[:3, :3] = dart.math.eulerXYZToMatrix(target_pos[0:3].astype(np.float64))
            #     model_mat = np.matmul(init_model_mat, rel_mat)
            # else:
            #     init_model_mat = self.mocap.get_local_transform(parent_joint, 0)
            #     rel_mat = np.identity(4)
            #     rel_mat[:3, :3] = dart.math.eulerXYZToMatrix(target_pos[index*3+3:index*3+6].astype(np.float64))
            #     model_mat = np.matmul(init_model_mat, rel_mat)

            init_model_mat = self.mocap.get_local_transform(joint, 0)
            rel_mat = np.identity(4)
            rel_mat[:3, :3] = dart.math.eulerXYZToMatrix(target_pos[index*3+3:index*3+6].astype(np.float64))
            model_mat = np.matmul(init_model_mat, rel_mat)

            joint_pose[joint_index] = model_mat
            index += 1
        
        return joint_pose

    def compute_target_pos_from_joint_map(self, frame):
        target_pos = np.zeros_like(self.initial_pos[self.body_skeleton_index])

        # compute relative rotation values
        index = 0
        for i in range(self.body_skeleton.getNumJoints()):
            dp_joint = self.body_skeleton.getJoint(i)

            joint_index = self.joint_map[i]
            joint = self.mocap.data.get_joints()[joint_index]
            parent_joint = self.mocap.data.joint_parent(joint.name)

            if dp_joint.getNumDofs() == 6:
                init_model_mat = self.mocap.get_local_transform(joint, 0)
                model_mat = self.mocap.get_local_transform(joint, frame)
                rel_mat = np.matmul(np.linalg.inv(init_model_mat), model_mat)
                euler_angle = dart.math.matrixToEulerXYZ(rel_mat[:3, :3].astype(np.float64))
                pos = np.array([rel_mat[3][0], rel_mat[3][1], rel_mat[3][2]])
                target_pos[index:index+3] = pos
                target_pos[index+3:index+6] = euler_angle
                index += 6
            elif dp_joint.getNumDofs() == 3:
                init_model_mat = self.mocap.get_local_transform(parent_joint, 0)
                model_mat = self.mocap.get_local_transform(parent_joint, frame)
                rel_mat = np.matmul(np.linalg.inv(init_model_mat), model_mat)
                euler_angle = dart.math.matrixToEulerXYZ(rel_mat[:3, :3].astype(np.float64))
                target_pos[index:index+3] = euler_angle
                index += 3
            else:
                raise "unsupported joint '{}', dofs: {}".format(dp_joint.getName(), dp_joint.getNumDofs())
        
        return target_pos

    def log_map(np_mat):
        ## rotation matrix to axis-angle representation
        # theta = acos((R_11 + R_22 + R_33 - 1) / 2) 
        # v1 = (R_32 - R_23) / (2 * sin (theta))
        # v2 = (R_13 - R_31) / (2 * sin (theta))
        # v3 = (R_21 - R_12) / (2 * sin (theta)) 
        trace = min(max((np.trace(np_mat) - 1.0) / 2.0, -1.0), 1.0)
        theta = math.acos(trace)
        sin_theta = 2.0 * math.sin(theta)
        if sin_theta == 0.0:
            return np.zeros((3,))
        v1 = (np_mat[2, 1] - np_mat[1, 2])
        v2 = (np_mat[0, 2] - np_mat[2, 0])
        v3 = (np_mat[1, 0] - np_mat[0, 1])
        return np.array([v1, v2, v3]) * theta / sin_theta

    def compute_difference_of_rotation(pos, target_pos):
        result = np.zeros_like(pos)
        result[:3] = pos[:3] - target_pos[:3] # position
        for i in range(1, pos.shape[0] // 3):
            rot = dart.math.eulerXYZToMatrix(pos[3*i:3*(i+1)])
            target_rot = dart.math.eulerXYZToMatrix(target_pos[3*i:3*(i+1)])
            diff_rot = np.matmul(target_rot.T, rot)
            log_diff_rot = DartEnv.log_map(diff_rot)
            result[3*i:3*(i+1)] = log_diff_rot
        return result

    def compute_reward_imitation_pos(pos, target_pos):
        total_diff = 0.0
        for i in range(1, pos.shape[0] // 3):
            r1 = R.from_euler("XYZ", pos[3*i:3*(i+1)])
            r2 = R.from_euler("XYZ", target_pos[3*i:3*(i+1)])
            d = r1 * r2.inv()
            total_diff += d.magnitude()

        return np.exp(-2.0 * total_diff)

    def orthogonalize(rot):
        u, _, vt = np.linalg.svd(rot)
        return np.matmul(u, vt)

    def integrate_pos(pos, vel, time_step):
        result = np.zeros_like(pos)
        result[:3] = pos[:3] + vel[:3] * time_step # position
        for i in range(1, pos.shape[0] // 3):
            rot = dart.math.eulerXYZToMatrix(pos[3*i:3*(i+1)])
            ang_vel = vel[3*i:3*(i+1)]
            A = np.array([
                [0, -ang_vel[2], ang_vel[1]],
                [ang_vel[2], 0, -ang_vel[0]],
                [-ang_vel[1], ang_vel[0], 0],
                ]) * time_step
            rot += np.matmul(A, rot)
            result[3*i:3*(i+1)] = dart.math.matrixToEulerXYZ(rot)
        return result

    ### Currently not use desired velocity term
    # check whether computing angular velocity exactly
    def compute_target_vel(self, frame):
        target_vel = np.zeros_like(self.initial_vel[self.body_skeleton_index])

        if frame >= self.mocap.data.nframes - 1:
            prev_frame = frame - 1
            next_frame = frame
        else:
            prev_frame = frame
            next_frame = frame + 1

        # compute relative rotation values
        index = 0
        for joint in self.mocap.data.get_joints():
            joint_index = self.mocap.data.get_joint_index(joint.name)
            parent_joint = self.mocap.data.joint_parent(joint.name)

            # ### remove multiple child joint
            # if parent_joint is not None:
            #     child_joints = list(parent_joint.filter('JOINT'))
            #     if len(child_joints) > 1:
            #         continue

            if parent_joint is None:
                init_model_mat = self.mocap.get_local_transform(joint, 0)
                model_mat_curr = self.mocap.get_local_transform(joint, prev_frame)
                model_mat_next = self.mocap.get_local_transform(joint, next_frame)
                rel_mat_curr = np.matmul(np.linalg.inv(init_model_mat), model_mat_curr)
                rel_mat_next = np.matmul(np.linalg.inv(init_model_mat), model_mat_next)
                rel_mat = (rel_mat_next - rel_mat_curr) / self.mocap.data.frame_time
                rot_mat = np.matmul(rel_mat, np.linalg.inv(rel_mat_curr))
            else:
                init_model_mat = self.mocap.get_local_transform(parent_joint, 0)
                model_mat_curr = self.mocap.get_local_transform(joint, prev_frame)
                model_mat_next = self.mocap.get_local_transform(joint, next_frame)
                rel_mat_curr = np.matmul(np.linalg.inv(init_model_mat), model_mat_curr)
                rel_mat_next = np.matmul(np.linalg.inv(init_model_mat), model_mat_next)
                rel_mat = (rel_mat_next - rel_mat_curr) / self.mocap.data.frame_time
                rot_mat = np.matmul(rel_mat, np.linalg.inv(rel_mat_curr))

            ang_vel = np.array([
                (-rot_mat[1, 2] + rot_mat[2, 1]),
                (-rot_mat[2, 0] + rot_mat[0, 2]),
                (-rot_mat[0, 1] + rot_mat[1, 0]),
                ]) * 0.5
            lin_vel = np.array([rel_mat[3][0], rel_mat[3][1], rel_mat[3][2]])
            if parent_joint is None: 
                target_vel[0:3] = ang_vel
                target_vel[3:6] = lin_vel
            else:
                target_vel[index * 3 + 3: index * 3 + 6] = ang_vel
            index += 1

        return target_vel