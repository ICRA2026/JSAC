import os

import random
import numpy as np
import matplotlib.pyplot as plt

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