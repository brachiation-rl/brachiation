import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)

import numpy as np
import pybullet

DEG2RAD = np.pi / 180


class VSphere:
    def __init__(self, bc, radius=None, pos=None, rgba=None, max=True):
        self._p = bc

        radius = 0.3 if radius is None else radius
        pos = (0, 0, 1) if pos is None else pos
        rgba = (219 / 255, 72 / 255, 72 / 255, 1.0) if rgba is None else rgba

        shape = self._p.createVisualShape(
            self._p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=rgba,
        )

        self.id = self._p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=shape,
            basePosition=pos,
            useMaximalCoordinates=max,
        )

        self._rgba = rgba
        self._pos = pos

    def set_position(self, pos=None):

        pos = self._pos if pos is None else pos
        self._pos = pos
        self._p.resetBasePositionAndOrientation(
            self.id, posObj=pos, ornObj=(0, 0, 0, 1)
        )

    def set_color(self, rgba):
        t_rgba = tuple(rgba)
        if t_rgba != self._rgba:
            self._p.changeVisualShape(self.id, -1, rgbaColor=rgba)
            self._rgba = t_rgba


class VCylinder:
    def __init__(self, bc, radius, length, pos=None, quat=None, rgba=None, max=True):
        self._p = bc

        pos = (0, 0, 0) if pos is None else pos
        quat = (0, 0, 0, 1) if quat is None else quat
        rgba = (55 / 255, 55 / 255, 55 / 255, 1) if rgba is None else rgba

        shape = self._p.createVisualShape(
            self._p.GEOM_CYLINDER,
            radius=radius,
            length=length,
            rgbaColor=rgba,
        )

        self.id = self._p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=shape,
            basePosition=pos,
        )

        self._pos = pos
        self._quat = quat
        self._rgba = rgba

    def set_position(self, pos=None, quat=None):
        pos = self._pos if pos is None else pos
        quat = self._quat if quat is None else quat

        self._pos = pos
        self._quat = quat

        self._p.resetBasePositionAndOrientation(
            self.id, posObj=self._pos, ornObj=self._quat
        )

    def set_color(self, rgba):
        t_rgba = tuple(rgba)
        if t_rgba != self._rgba:
            self._p.changeVisualShape(self.id, -1, rgbaColor=rgba)
            self._rgba = t_rgba


class BaseStep:
    def __init__(self, bc, filename, scale, pos=None, quat=None, options=None):
        self._p = bc

        pos = np.zeros(3) if pos is None else pos
        quat = np.array([0, 0, 0, 1]) if quat is None else quat
        options = {} if options is None else options

        self.id = self._p.loadURDF(
            filename,
            basePosition=pos,
            baseOrientation=quat,
            useFixedBase=False,
            globalScaling=scale,
            **options
        )

        self._pos_offset = np.array(self._p.getBasePositionAndOrientation(self.id)[0])

        for link_id in range(-1, self._p.getNumJoints(self.id)):
            self._p.changeDynamics(
                self.id,
                link_id,
                lateralFriction=1.2,
                restitution=0.1,
            )

        self.base_id = -1
        self.cover_id = 0

    def set_position(self, pos=None, quat=None):
        pos = np.zeros(3) if pos is None else pos
        quat = np.array([0, 0, 0, 1]) if quat is None else quat

        pybullet.resetBasePositionAndOrientation(
            self.id,
            posObj=pos + self._pos_offset,
            ornObj=quat,
            physicsClientId=self._p._client,
        )


class Pillar(BaseStep):
    def __init__(self, bc, radius, pos=None, quat=None, options=None):
        filename = os.path.join(parent_dir, "data", "objects", "steps", "pillar.urdf")
        super().__init__(bc, filename, radius, pos, quat, options)


class Plank(BaseStep):
    def __init__(self, bc, width, pos=None, quat=None, options=None):
        filename = os.path.join(parent_dir, "data", "objects", "steps", "plank.urdf")
        super().__init__(bc, filename, 2 * width, pos, quat, options)


class LargePlank(BaseStep):
    def __init__(self, bc, width, pos=None, quat=None, options=None):
        filename = os.path.join(
            parent_dir, "data", "objects", "steps", "plank_large.urdf"
        )
        super().__init__(bc, filename, 2 * width, pos, quat, options)
