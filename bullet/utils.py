import functools
import inspect
import os
import time
import types

current_dir = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import pybullet


class BulletClient(object):
    def __init__(self, connection_mode=None, use_ffmpeg=False, fps=60):

        if connection_mode is None:
            self._client = pybullet.connect(pybullet.SHARED_MEMORY)
            if self._client >= 0:
                return
            else:
                connection_mode = pybullet.DIRECT

        options = (
            "--background_color_red=1.0 "
            "--background_color_green=1.0 "
            "--background_color_blue=1.0 "
            "--width=1280 --height=720 "
        )
        if use_ffmpeg:
            from datetime import datetime

            now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            options += f'--mp4="{now_str}.mp4" --mp4fps={fps} '

        self._client = pybullet.connect(connection_mode, options=options)

    def __del__(self):
        """Clean up connection if not already done."""
        try:
            pybullet.disconnect(physicsClientId=self._client)
        except pybullet.error:
            pass

    def __getattr__(self, name):
        """Inject the client id into Bullet functions."""
        attribute = getattr(pybullet, name)
        if inspect.isbuiltin(attribute):
            if name not in [
                "invertTransform",
                "multiplyTransforms",
                "getMatrixFromQuaternion",
                "getEulerFromQuaternion",
                "computeViewMatrixFromYawPitchRoll",
                "computeProjectionMatrixFOV",
                "getQuaternionFromEuler",
            ]:
                attribute = functools.partial(attribute, physicsClientId=self._client)
        return attribute


class Scene:
    def __init__(self, bullet_client, gravity, timestep, frame_skip):
        self._p = bullet_client
        self.timestep = timestep
        self.frame_skip = frame_skip

        self.dt = self.timestep * self.frame_skip
        self.cpp_world = World(self._p, gravity, timestep, frame_skip)

    def global_step(self):
        self.cpp_world.step()


class World:
    def __init__(self, bullet_client, gravity, timestep, frame_skip):
        self._p = bullet_client
        self.gravity = gravity
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.set_physics_parameters()

    def set_physics_parameters(self):
        self._p.setGravity(0, 0, -self.gravity)
        self._p.setPhysicsEngineParameter(
            fixedTimeStep=self.timestep * self.frame_skip,
            numSubSteps=self.frame_skip,
            numSolverIterations=100,
        )

    def step(self):
        pybullet.stepSimulation(physicsClientId=self._p._client)


class StadiumScene(Scene):

    stadium_halflen = 105 * 0.25
    stadium_halfwidth = 50 * 0.25

    def initialize(self, remove_ground=False):
        current_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(current_dir)

        if not remove_ground:
            filename = os.path.join(
                parent_dir, "data", "objects", "misc", "plane_stadium.sdf"
            )
            self.id = self._p.loadSDF(filename, useMaximalCoordinates=True)[0]
            self._p.changeDynamics(
                self.id,
                -1,
                lateralFriction=0.8,
                restitution=0.1,
            )

    def set_friction(self, lateral_friction):
        self._p.changeDynamics(self.id, -1, lateralFriction=lateral_friction)


class Camera:
    def __init__(self, bc, fps=60, dist=2.5, yaw=0, pitch=0, use_egl=False):

        self._p = bc
        self._cam_dist = dist
        self._cam_yaw = yaw
        self._cam_pitch = pitch
        self._coef = np.array([1.0, 1.0, 0.1])

        self.use_egl = use_egl
        self.tracking = False

        self._fps = fps
        self._target_period = 1 / fps
        self._last_frame_time = time.perf_counter()
        self.env_should_wait = False

    def track(self, pos, smooth_coef=None):

        # self.wait()
        if self.env_should_wait or not self.tracking:
            return

        smooth_coef = self._coef if smooth_coef is None else smooth_coef
        assert (smooth_coef <= 1).all(), "Invalid camera smoothing parameters"

        yaw, pitch, dist, lookat_ = self._p.getDebugVisualizerCamera()[-4:]
        lookat = (1 - smooth_coef) * lookat_ + smooth_coef * pos
        self._cam_target = lookat

        self._p.resetDebugVisualizerCamera(dist, yaw, pitch, lookat)

        # Remember camera for reset
        self._cam_yaw, self._cam_pitch, self._cam_dist = yaw, pitch, dist

    def lookat(self, pos):
        self._cam_target = pos
        self._p.resetDebugVisualizerCamera(
            self._cam_dist, self._cam_yaw, self._cam_pitch, pos
        )

    def dump_rgb_array(self):

        camera_info = self._p.getDebugVisualizerCamera()
        width, height, view, proj = camera_info[0:4]
        aspect = width / height

        r = 2 * aspect
        l = -2 * aspect
        t = 2
        b = -2
        n = -3
        f = 3

        proj = (
            2 / (r - l),
            0.0,
            0.0,
            (r + l) / (l - r),
            0.0,
            2 / (t - b),
            0.0,
            (t + b) / (b - t),
            0.0,
            0.0,
            2 / (n - f),
            (f + n) / (n - f),
            0.0,
            0.0,
            0.0,
            1,
        )

        (_, _, rgb_array, _, _) = self._p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=self._p.ER_BULLET_HARDWARE_OPENGL,
            flags=self._p.ER_NO_SEGMENTATION_MASK,
        )

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def wait(self):
        if self.use_egl:
            return

        time_spent = time.perf_counter() - self._last_frame_time

        self.env_should_wait = True
        if self._target_period < time_spent:
            self._last_frame_time = time.perf_counter()
            self.env_should_wait = False
