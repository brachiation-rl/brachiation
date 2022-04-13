import contextlib
import ctypes
import multiprocessing
import os
import pickle
from abc import ABC, abstractmethod

import cloudpickle
import gym
from gym.core import Wrapper
import numpy as np


def make_env_fns(env_id, seed, rank, **kwargs):
    def _thunk():
        env = gym.make(env_id, **kwargs)
        env.seed(seed + rank)

        if str(env.__class__.__name__).find("TimeLimit") >= 0:
            env = TimeLimitMask(env)

        env = LiteMonitor(env)
        return env

    return _thunk


def make_env(env_id, **kwargs):
    env = gym.make(env_id, **kwargs)
    return env


def make_vec_envs(env_id, seed, num_processes, **kwargs):
    assert num_processes > 1

    env_fns = [make_env_fns(env_id, seed, i, **kwargs) for i in range(num_processes)]

    envs = ShmemVecEnv(env_fns, context="spawn")
    return envs


class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if self.env._max_episode_steps == self.env._elapsed_steps:
            info["bad_transition"] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class LiteMonitor(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env=env)

    def reset(self, **kwargs):
        self.eprew = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.eprew += rew
        if done:
            info["episode"] = {"r": self.eprew}
        return (obs, rew, done, info)


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """

    closed = False
    viewer = None

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode="human"):
        raise NotImplementedError

    @property
    def unwrapped(self):
        return self


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = pickle.loads(ob)


@contextlib.contextmanager
def clear_mpi_env_vars():
    """
    from mpi4py import MPI will call MPI_Init by default.
    If the child process has MPI environment variables, MPI will think
    that the child process is an MPI process just like the parent and do bad things such as hang.
    This context manager is a hacky way to clear those environment variables temporarily such
    as when we are starting multiprocessing Processes.
    """
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ["OMPI_", "PMI_"]:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)


_NP_TO_CT = {
    np.float64: ctypes.c_double,
    np.float32: ctypes.c_float,
    np.int32: ctypes.c_int32,
    np.int8: ctypes.c_int8,
    np.uint8: ctypes.c_char,
    np.bool: ctypes.c_bool,
}


class ShmemVecEnv(VecEnv):
    """
    Optimized version of SubprocVecEnv that uses shared variables to communicate observations.
    """

    def __init__(self, env_fns, context="spawn"):
        """
        If you don't specify observation_space, we'll have to create a dummy
        environment to get it.
        """
        ctx = multiprocessing.get_context(context)

        dummy = env_fns[0]()
        observation_space, action_space = dummy.observation_space, dummy.action_space
        dummy.close()
        del dummy

        VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        self.obs_shape = observation_space.shape
        self.obs_dtype = observation_space.dtype

        self.obs_bufs = [
            ctx.Array(_NP_TO_CT[self.obs_dtype.type], int(np.prod(self.obs_shape)))
            for _ in env_fns
        ]
        self.obs_np_bufs = [
            np.frombuffer(b.get_obj(), dtype=self.obs_dtype) for b in self.obs_bufs
        ]

        self.parent_pipes = []
        self.procs = []
        with clear_mpi_env_vars():
            for i, (env_fn, obs_buf) in enumerate(zip(env_fns, self.obs_bufs)):
                wrapped_fn = CloudpickleWrapper(env_fn)
                parent_pipe, child_pipe = ctx.Pipe()
                proc = ctx.Process(
                    target=_subproc_worker,
                    args=(
                        i,
                        child_pipe,
                        parent_pipe,
                        wrapped_fn,
                        obs_buf,
                        self.obs_shape,
                        self.obs_dtype,
                    ),
                )
                proc.daemon = True
                self.procs.append(proc)
                self.parent_pipes.append(parent_pipe)
                proc.start()

                # Pinning task to CPUs
                target_cpu = i % multiprocessing.cpu_count()
                os.sched_setaffinity(proc.pid, [target_cpu])

                child_pipe.close()
        self.waiting_step = False
        self.viewer = None

    def reset(self):
        if self.waiting_step:
            print("Called reset() while waiting for the step to complete")
            self.step_wait()
        [pipe.send(("reset", None)) for pipe in self.parent_pipes]
        dummy = [pipe.recv() for pipe in self.parent_pipes]
        return self._decode_obses()

    def step_async(self, actions):
        assert len(actions) == len(self.parent_pipes)
        [pipe.send(("step", act)) for pipe, act in zip(self.parent_pipes, actions)]

    def step_wait(self):
        outs = [pipe.recv() for pipe in self.parent_pipes]
        _, rews, dones, infos = zip(*outs)
        return self._decode_obses(), np.array(rews), np.array(dones), infos

    def set_env_params(self, params_dict):
        for pipe in self.parent_pipes:
            pipe.send(("set_env_params", params_dict))

    def get_env_param(self, param_name, default):
        for pipe in self.parent_pipes:
            pipe.send(("get_env_param", (param_name, default)))
        return [pipe.recv() for pipe in self.parent_pipes]

    def close_extras(self):
        if self.waiting_step:
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(("close", None))
        for pipe in self.parent_pipes:
            pipe.recv()
            pipe.close()
        for proc in self.procs:
            proc.join()

    def _decode_obses(self):
        return np.array(self.obs_np_bufs)


def _subproc_worker(
    ind, pipe, parent_pipe, env_fn_wrapper, obs_buf, obs_shape, obs_dtype
):
    """
    Control a single environment instance using IPC and
    shared memory.
    """
    dst = obs_buf.get_obj()
    dst_np = np.frombuffer(dst, dtype=obs_dtype).reshape(obs_shape)

    env = env_fn_wrapper.x()
    parent_pipe.close()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == "reset":
                np.copyto(dst_np, env.reset())
                pipe.send(None)
            elif cmd == "step":
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset()
                np.copyto(dst_np, obs)
                pipe.send((None, reward, done, info))
            elif cmd == "set_env_params":
                env.set_env_params(data)
            elif cmd == "get_env_param":
                param = env.get_env_param(*data)
                pipe.send(param)
            elif cmd == "close":
                pipe.send(None)
                break
            else:
                raise RuntimeError("Got unrecognized cmd %s" % cmd)
    except KeyboardInterrupt:
        if ind == 0:
            print("ShmemVecEnv worker: got KeyboardInterrupt")
    finally:
        env.close()
