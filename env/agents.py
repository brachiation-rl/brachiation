import math
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)

import gym
import numpy as np
import pybullet

from bottleneck import nanmin, nanmax
from numpy import concatenate
from scipy.linalg.blas import sscal as SCAL
from scipy.spatial.transform import Rotation as R


DEG2RAD = np.pi / 180


class WalkerBase:

    root_link_name = None
    mirrored = False

    def __init__(self, bc):
        self._p = bc

        self.action_dim = len(self.power_coef)
        high = np.ones(self.action_dim, dtype=np.float32)
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # globals + angles (sin, cos) + speeds + contacts
        self.state_dim = 6 + self.action_dim * 3 + len(self.foot_names)
        high = np.inf * np.ones(self.state_dim, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.joint_angles, self.joint_speeds = np.zeros(
            (2, self.action_dim), dtype=np.float32
        )

        K = len(self.foot_names) + 1
        self.body_and_feet_xyz = np.zeros((K, 3), dtype="f4")
        self.body_and_feet_vel = np.zeros((K, 3), dtype="f4")
        self.body_and_feet_quat = np.zeros((K, 4), dtype="f4")

    def apply_action(self, action):
        forces = self.joint_gains * action
        pybullet.setJointTorqueArray(
            bodyUniqueId=self.id,
            jointIndices=self.joint_uindices,
            forces=forces,
            physicsClientId=self._p._client,
        )

    def calc_state(self):

        pybullet.getJointStates2(
            self.id,
            self.joint_ids,
            self.joint_angles,
            self.joint_speeds,
            physicsClientId=self._p._client,
        )
        SCAL(0.1, self.joint_speeds)

        pybullet.getLinkStates2(
            self.id,
            self.root_and_foot_ids,
            outPositions=self.body_and_feet_xyz,
            outOrientations=self.body_and_feet_quat,
            outVelocities=self.body_and_feet_vel,
            computeLinkVelocity=1,
            physicsClientId=self._p._client,
        )
        self.body_xyz = self.body_and_feet_xyz[0]
        self.feet_xyz = self.body_and_feet_xyz[1:]
        self.body_world_vel = self.body_and_feet_vel[0]
        self.body_quat = self.body_and_feet_quat[0]

        # In 2D, pitch is better calculated if y-first
        # self.body_rpy = pybullet.getEulerFromQuaternion(self.body_quat)
        pitch, roll, yaw = R.from_quat(self.body_quat).as_euler("yxz").astype("f4")
        self.body_rpy = (roll, pitch, yaw)

        yaw_cos = math.cos(-yaw)
        yaw_sin = math.sin(-yaw)
        vxg, vyg, vzg = self.body_world_vel
        self.body_vel = (
            yaw_cos * vxg - yaw_sin * vyg,
            yaw_sin * vxg + yaw_cos * vyg,
            vzg,
        )
        vx, vy, vz = self.body_vel

        # bottleneck is faster if data is ndarray, otherwise use built-in min()
        height = self.body_xyz[2] - nanmin(self.feet_xyz[:, 2])

        # Faster than np.clip()
        self.joint_speeds[self.joint_speeds < -5] = -5
        self.joint_speeds[self.joint_speeds > +5] = +5

        state = concatenate(
            (
                [height, vx, vy, vz, roll, pitch],
                np.sin(self.joint_angles),
                np.cos(self.joint_angles),
                self.joint_speeds,
                self.feet_contact,
            )
        )

        return state

    def initialize(self):
        self.load_robot_model()
        self.make_robot_utils()

    def parse_model_file(self, model_path, flags):
        self.object_id = self._p.loadMJCF(model_path, flags=flags)
        self.id = self.object_id[0]

        self.parse_joints_and_links()

        self.feet_contact = np.zeros(len(self.foot_names), dtype=np.float32)
        self.feet_xyz = np.zeros((len(self.foot_names), 3), dtype=np.float32)

        self.base_joint_angles = np.zeros(self.action_dim, dtype=np.float32)
        self.base_joint_speeds = np.zeros(self.action_dim, dtype=np.float32)
        self.base_position = np.array([0, 0, 0], dtype=np.float32)
        self.base_orientation = np.array([0, 0, 0, 1], dtype=np.float32)
        self.base_velocity = np.array([0, 0, 0], dtype=np.float32)
        self.base_angular_velocity = np.array([0, 0, 0], dtype=np.float32)

    def make_robot_utils(self):
        # Useful things for mirroring observations
        assert len(self.foot_names) == 2, "Check mirroring indices"
        self.right_obs_indices = concatenate(
            (
                # joint angle (sin) + 6 accounting for global
                6 + self._right_joint_indices,
                # joint angle (cos)
                6 + self._right_joint_indices + self.action_dim,
                # joint velocity
                6 + self._right_joint_indices + self.action_dim * 2,
                # right foot contact
                [6 + self.action_dim * 3],
            )
        )

        self.left_obs_indices = concatenate(
            (
                6 + self._left_joint_indices,
                6 + self._left_joint_indices + self.action_dim,
                6 + self._left_joint_indices + self.action_dim * 2,
                [6 + self.action_dim * 3 + 1],
            )
        )

        self.neg_obs_indices = concatenate(
            (
                [2, 4],  # vy, roll
                6 + self._negation_joint_indices,
                6 + self._negation_joint_indices + self.action_dim,
                6 + self._negation_joint_indices + self.action_dim * 2,
            )
        )

    def parse_joints_and_links(self):
        self.joint_ids = []
        self.joint_gains = []

        self.root_and_foot_ids = []
        self.foot_ids = []

        bc = self._p
        parts = {}
        for j in range(bc.getNumJoints(self.id)):
            joint_info = bc.getJointInfo(self.id, j)
            joint_name = joint_info[1].decode("utf8")
            part_name = joint_info[12].decode("utf8")
            parts[part_name] = j

            self._p.setJointMotorControl2(
                self.id,
                j,
                controlMode=pybullet.POSITION_CONTROL,
                targetPosition=0,
                targetVelocity=0,
                positionGain=0.1,
                velocityGain=0.1,
                force=0,
            )

            if joint_name[:6] == "ignore":
                continue

            if joint_name[:8] != "jointfix":
                self.joint_ids.append(j)
                self.joint_gains.append(self.power_coef[joint_name])

        assert self.root_link_name is not None, "Must specify root link"

        self.foot_ids = [parts[k] for k in self.foot_names]
        self.root_and_foot_ids = [parts[self.root_link_name], *self.foot_ids]

        limits = [bc.getJointInfo(self.id, pid)[8:10] for pid in self.joint_ids]
        self.joint_limits = np.array(limits, dtype=np.float64)

        self.joint_gains = np.array(self.joint_gains, dtype=np.float32)

        uindices = [bc.getJointInfo(self.id, pid)[4] for pid in self.joint_ids]
        self.joint_uindices = np.fromiter(uindices, dtype=np.int32)

        self._zeros = [0 for _ in self.joint_ids]
        self._gains = [0.1 for _ in self.joint_ids]

    def reset(
        self,
        random_pose=True,
        random_mirror=True,
        pos=None,
        quat=None,
        vel=None,
        ang_vel=None,
    ):
        base_joint_angles = np.copy(self.base_joint_angles)
        base_orientation = np.copy(self.base_orientation)
        if random_mirror and self.np_random.rand() < 0.5:
            self.mirrored = True
            base_joint_angles[self._rl] = base_joint_angles[self._lr]
            base_joint_angles[self._negation_joint_indices] *= -1
            base_orientation[0:3] *= -1
        else:
            self.mirrored = False

        if random_pose:
            # Add small deviations, about 5 degrees
            ds = self.np_random.uniform(low=-0.08, high=0.08, size=self.action_dim)
            base_joint_angles = (base_joint_angles + ds).astype(np.float32)

        self.reset_joint_states(base_joint_angles, self.base_joint_speeds)

        pos = pos if pos is not None else self.base_position
        quat = quat if quat is not None else self.base_orientation
        vel = vel if vel is not None else self.base_velocity
        ang_vel = ang_vel if ang_vel is not None else self.base_angular_velocity

        # Reset root position and velocities
        self._p.resetBasePositionAndOrientation(self.id, pos, quat)
        self._p.resetBaseVelocity(self.id, vel, ang_vel)

        self.feet_contact.fill(0.0)
        self.feet_xyz.fill(0.0)

        robot_state = self.calc_state()
        return robot_state

    def reset_joint_states(self, positions, velocities):
        pybullet.resetJointStates(
            self.id,
            self.joint_ids,
            targetValues=positions,
            targetVelocities=velocities,
            physicsClientId=self._p._client,
        )

        pybullet.setJointMotorControlArray(
            bodyIndex=self.id,
            jointIndices=self.joint_ids,
            controlMode=pybullet.POSITION_CONTROL,
            targetPositions=self._zeros,
            targetVelocities=self._zeros,
            positionGains=self._gains,
            velocityGains=self._gains,
            forces=self._zeros,
            physicsClientId=self._p._client,
        )


class Gibbon3D(WalkerBase):

    root_link_name = "torso"
    foot_names = ["right_hand", "left_hand"]

    power_coef = {
        "abdomen_z": 60,
        "abdomen_y": 60,
        "abdomen_x": 60,
        "right_hip_x": 50,
        "right_hip_z": 50,
        "right_hip_y": 50,
        "right_knee": 30,
        "right_ankle": 10,
        "left_hip_x": 50,
        "left_hip_z": 50,
        "left_hip_y": 50,
        "left_knee": 30,
        "left_ankle": 10,
        "right_shoulder_x": 100,
        "right_shoulder_y": 100,
        "right_elbow_z": 60,
        "right_elbow_y": 100,
        "right_wrist": 80,
        "left_shoulder_x": 100,
        "left_shoulder_y": 100,
        "left_elbow_z": 60,
        "left_elbow_y": 100,
        "left_wrist": 80,
    }

    def __init__(self, bc):
        super().__init__(bc)

    def load_robot_model(self, model_path=None, flags=None):
        if model_path is None:
            model_path = os.path.join(parent_dir, "data", "robots", "gibbon3d.xml")

        if flags is None:
            flags = (
                self._p.MJCF_COLORS_FROM_FILE
                | self._p.URDF_USE_SELF_COLLISION
                | self._p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
                | self._p.URDF_MERGE_FIXED_LINKS
            )

        super().parse_model_file(model_path, flags)
        self.base_position = (0, 0, 0.7)

        # Need this to set pose and mirroring
        # hip_[x,z,y], knee, ankle, shoulder_[x,y], elbow
        self._right_joint_indices = np.array(
            [3, 4, 5, 6, 7, 13, 14, 15, 16, 17], dtype=np.int64
        )
        self._left_joint_indices = np.array(
            [8, 9, 10, 11, 12, 18, 19, 20, 21, 22], dtype=np.int64
        )
        self._negation_joint_indices = np.array([0, 2], dtype=np.int64)  # abdomen_[x,z]
        self._rl = concatenate((self._right_joint_indices, self._left_joint_indices))
        self._lr = concatenate((self._left_joint_indices, self._right_joint_indices))

    def set_base_pose(self, pose=None):
        self.base_joint_angles[:] = 0  # reset
        self.base_orientation = np.array([0, 0, 0, 1])

        if pose == "monkey_start":
            self.base_joint_angles[[14]] = -140 * DEG2RAD  # shoulder y
            self.base_joint_angles[[15]] = 180 * DEG2RAD  # elbow z
            self.base_joint_angles[[17]] = -90 * DEG2RAD  # finger
            self.base_joint_angles[[19]] = -170 * DEG2RAD  # shoulder y
            self.base_joint_angles[[20]] = 180 * DEG2RAD  # elbow z
            self.base_joint_angles[[22]] = -90 * DEG2RAD  # finger
            self.base_joint_angles[[6, 11]] = -90 * DEG2RAD  # ankles
            self.base_joint_angles[[7, 12]] = -90 * DEG2RAD  # knees
        if pose == "stand_start":
            self.base_joint_angles[[5, 6]] = -np.pi / 8  # Right leg
            self.base_joint_angles[[10]] = np.pi / 10  # Left leg
            self.base_joint_angles[[7, 12]] = -np.pi / 2  # Ankles
            self.base_joint_angles[[14]] = -140 * DEG2RAD  # shoulder y
            self.base_joint_angles[[19]] = -170 * DEG2RAD  # shoulder y

    def calc_state(self):
        # reverse for monkey
        state = super().calc_state()
        state[0] = nanmax(self.feet_xyz[:, 2]) - self.body_xyz[2]
        return state


class Gibbon2D(Gibbon3D):

    power_coef = {
        "abdomen_y": 30 * 0.7,
        "right_hip_y": 20 * 0.7,
        "right_knee": 20 * 0.7,
        "right_ankle": 10 * 0.7,
        "left_hip_y": 20 * 0.7,
        "left_knee": 20 * 0.7,
        "left_ankle": 10 * 0.7,
        "right_shoulder_y": 50 * 0.7,
        "right_elbow_y": 40 * 0.7,
        "right_wrist": 1 * 0.7,
        "left_shoulder_y": 50 * 0.7,
        "left_elbow_y": 40 * 0.7,
        "left_wrist": 1 * 0.7,
    }

    def set_base_pose(self, pose=None):
        self.base_joint_angles[:] = 0  # reset
        self.base_orientation = np.array([0, 0, 0, 1])

        if pose == "stand_start":
            self.base_joint_angles[[1, 2]] = -np.pi / 8  # Right leg
            self.base_joint_angles[[4]] = np.pi / 10  # Left leg
            self.base_joint_angles[[3, 6]] = -np.pi / 2  # Ankles
            self.base_joint_angles[[7]] = -140 * DEG2RAD  # shoulder y
            self.base_joint_angles[[10]] = -170 * DEG2RAD  # shoulder y
        elif pose == "hanging":
            orientation = pybullet.getQuaternionFromEuler([0, 90 * DEG2RAD, 0])
            self.base_orientation = np.array(orientation)
            self.base_joint_angles[[2, 5]] = 110 * DEG2RAD  # Knees
            self.base_joint_angles[[3, 6]] = -np.pi / 2  # Ankles
            self.base_joint_angles[7] = -np.pi / 3  # Right elbow
            self.base_joint_angles[8] = -np.pi / 4  # Right elbow
            self.base_joint_angles[10] = -np.pi  # Left shoulder

    def load_robot_model(self, model_path=None, flags=None):
        if model_path is None:
            model_path = os.path.join(parent_dir, "data", "robots", "gibbon2d.xml")

        flags = self._p.MJCF_COLORS_FROM_FILE | self._p.URDF_MERGE_FIXED_LINKS

        super().load_robot_model(model_path, flags)
        self.base_position = (0, 0, 1)

        # Need this to set pose and mirroring
        self._right_joint_indices = np.array([1, 2, 3, 7, 8, 9], dtype=np.int64)
        self._left_joint_indices = np.array([4, 5, 6, 10, 11, 12], dtype=np.int64)
        self._negation_joint_indices = np.array([], dtype=np.int64)
        self._rl = concatenate((self._right_joint_indices, self._left_joint_indices))
        self._lr = concatenate((self._left_joint_indices, self._right_joint_indices))
