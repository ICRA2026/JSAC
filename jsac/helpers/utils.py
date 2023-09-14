import os

import random
import numpy as np
import matplotlib.pyplot as plt
import collections
from gym.core import Env


class MODE:
    IMG = 'img'
    IMG_PROP = 'img_prop'
    PROP = 'prop'

def make_dir(dir_path):
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError:
        pass
    return dir_path

def set_seed_everywhere(seed, env=None):
    np.random.seed(seed)
    random.seed(seed)

    if env is not None:
        env.seed(seed)

def smoothed_curve(returns, ep_lens, x_tick=5000, window_len=5000):
    """
    Args:
        returns: 1-D numpy array with episodic returs
        ep_lens: 1-D numpy array with episodic returs
        x_tick (int): Bin size
        window_len (int): Length of averaging window
    Returns:
        A numpy array
    """
    rets = []
    x = []
    cum_episode_lengths = np.cumsum(ep_lens)

    if cum_episode_lengths[-1] >= x_tick:
        y = cum_episode_lengths[-1] + 1
        steps_show = np.arange(x_tick, y, x_tick)

        for i in range(len(steps_show)):
            rets_in_window = returns[(cum_episode_lengths > max(0, x_tick * (i + 1) - window_len)) *
                                     (cum_episode_lengths < x_tick * (i + 1))]
            if rets_in_window.any():
                rets.append(np.mean(rets_in_window))
                x.append((i+1) * x_tick)

    return np.array(rets), np.array(x)

def show_learning_curve(fname, rets, ep_lens, xtick, xlimit=None, ylimit=None, save_fig=True):
        plot_rets, plot_x = smoothed_curve(
                np.array(rets), np.array(ep_lens), x_tick=xtick, window_len=xtick)
        
        if len(plot_rets):
            plt.clf()
            if xlimit is not None:
                plt.xlim(xlimit)
        
            if ylimit is not None:
                plt.ylim(ylimit)
                
            plt.plot(plot_x, plot_rets)
            plt.pause(0.001)
            if save_fig:
                plt.savefig(fname)

## SRC: https://github.com/kindredresearch/SenseAct/blob/master/senseact/utils.py

class EnvSpec():
    def __init__(self, env_spec, observation_space, action_space):
        self._observation_space = observation_space
        self._action_space = action_space
        self._unwrapped_spec = env_spec

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

Step = collections.namedtuple("Step", ["observation", "reward", "done", "info"])

class NormalizedEnv(Env):
    def __init__(
            self,
            env,
            scale_reward=1.,
            normalize_obs=False,
            normalize_reward=False,
            obs_alpha=0.001,
            reward_alpha=0.001,
    ):
        self._wrapped_env = env
        self._scale_reward = scale_reward
        self._normalize_obs = normalize_obs
        self._normalize_reward = normalize_reward
        self._obs_alpha = obs_alpha
        self._obs_mean = np.zeros(env.observation_space.shape)
        self._obs_var = np.ones(env.observation_space.shape)
        self._reward_alpha = reward_alpha
        self._reward_mean = 0.
        self._reward_var = 1.
        self._spec = EnvSpec(env.spec, self.observation_space, self.action_space)

    def _update_obs_estimate(self, obs):
        flat_obs = self.wrapped_env.observation_space.flatten(obs)
        self._obs_mean = (1 - self._obs_alpha) * self._obs_mean + self._obs_alpha * flat_obs
        self._obs_var = (1 - self._obs_alpha) * self._obs_var + self._obs_alpha * np.square(flat_obs - self._obs_mean)

    def _update_reward_estimate(self, reward):
        self._reward_mean = (1 - self._reward_alpha) * self._reward_mean + self._reward_alpha * reward
        self._reward_var = (1 - self._reward_alpha) * self._reward_var + \
                           self._reward_alpha * np.square(reward - self._reward_mean)

    def _apply_normalize_obs(self, obs):
        self._update_obs_estimate(obs)
        return (obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)

    def _apply_normalize_reward(self, reward):
        self._update_reward_estimate(reward)
        return reward / (np.sqrt(self._reward_var) + 1e-8)

    def reset(self):
        ret = self._wrapped_env.reset()
        if self._normalize_obs:
            return self._apply_normalize_obs(ret)
        else:
            return ret

    def step(self, action):
        # rescale the action
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if self._normalize_obs:
            next_obs = self._apply_normalize_obs(next_obs)
        if self._normalize_reward:
            reward = self._apply_normalize_reward(reward)
        return Step(next_obs, reward * self._scale_reward, done, info)
        # return next_obs, reward * self._scale_reward, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def start(self):
        return self._wrapped_env.start()

    def close(self):
        super(NormalizedEnv, self).close()
        return self._wrapped_env.close()

    @property
    def action_space(self):
        return self._wrapped_env.action_space

    @property
    def observation_space(self):
        return self._wrapped_env.observation_space

    def render(self, *args, **kwargs):
        try:
            return self._wrapped_env.render(*args, **kwargs)
        except TypeError:
            pass
            # return self._wrapped_env.render()

    def log_diagnostics(self, paths, *args, **kwargs):
        self._wrapped_env.log_diagnostics(paths, *args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        self._wrapped_env.terminate()

    def get_param_values(self):
        return self._wrapped_env.get_param_values()

    def set_param_values(self, params):
        self._wrapped_env.set_param_values(params)

    def __getattr__(self, attr):
        orig_attr = self.wrapped_env.__getattribute__(attr)
        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if result == self.wrapped_env:
                    return self
                return result

            return hooked
        else:
            return orig_attr