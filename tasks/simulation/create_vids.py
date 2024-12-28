import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import jax
import flax
import numpy as np
from jax import random
import jax.numpy as jnp
import cv2
from jsac.envs.dmc_visual_env.dmc_env import DMCVisualEnv
from jsac.algo.agent import sample_actions
from jsac.algo.initializers import init_inference_actor, get_init_data
from jsac.helpers.utils import MODE, WrappedEnv

best_actor_paths = [
    'results_final/results_jsac/results/ball_in_cup_img_sync/seed_0/best_actor_params.pkl',
    'results_final/results_jsac/results/cartpole_swingup_img_sync/seed_0/best_actor_params.pkl',
    'results_final/results_jsac/results/cheetah_img_sync/seed_2/best_actor_params.pkl',
    'results_final/results_jsac/results/finger_spin_img_sync/seed_0/best_actor_params.pkl',
    'results_final/results_jsac/results/walker_walk_img_sync/seed_0/best_actor_params.pkl'
]

frames = [60, 200, 200, 100, 200]

img_env_names = ['ball_in_cup', 'cartpole_swingup', 'cheetah', 'finger_spin', 'walker_walk']

config = {
    'conv': [
        # in_channel, out_channel, kernel_size, stride
        [-1, 32, 5, 2],
        [32, 32, 5, 2],
        [32, 64, 3, 1],
        [64, 64, 3, 1],
    ],
    
    'latent_dim': 64,

    'mlp': [1024, 1024],
}

if __name__ == "__main__":
    idx = 0
    all_images = []
    for actor_path, env_name in zip(best_actor_paths, img_env_names):
        images = []
        env = DMCVisualEnv(env_name, MODE.IMG, 0, 3, 96 * 4, 96 * 4, 1, 2)
        env = WrappedEnv(env)

        image_shape = (96, 96, 9)
        proprioception_shape = env.proprioception_space.shape
        action_shape = env.action_space.shape
        env_action_space = env.action_space

        rng = jax.random.PRNGKey(0)
        rng, actor = init_inference_actor(rng, 
                                          image_shape, 
                                          proprioception_shape, 
                                          config, 
                                          action_shape[-1], 
                                          False,
                                          MODE.IMG, 
                                          jnp.float32)
        
        rng, key1, key2 = random.split(rng, 3)
        params= actor.init(key1, key2, *get_init_data(image_shape, proprioception_shape, MODE.IMG))['params']

        with open(actor_path, 'rb') as f: 
            params = flax.serialization.from_bytes(params, f.read())

        episodes = 1200 // frames[idx]
        for j in range(episodes):
            print(env_name, j)
            state = env.reset()
            for i in range(frames[idx]):
                img = state[:, :, 0:3]
                images.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                state = cv2.resize(state, (0, 0), fx = 0.25, fy = 0.25)
                rng, action = sample_actions(rng, 
                                            actor.apply, 
                                            params, 
                                            state, 
                                            MODE.IMG, 
                                            True)

                action = np.asarray(action).clip(-1, 1)
                state, reward, done, info = env.step(action) 
        
        all_images.append(images)

        idx += 1

        del actor
    
    num_images = len(all_images[0])  
    sample_image = all_images[0][0]
    height, width, channels = sample_image.shape 
    total_width = width * len(all_images) 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('results_final/vid.mp4', fourcc, 30, (total_width, height))
 
    for i in range(num_images): 
        row_images = [all_images[j][i] for j in range(len(all_images))]
        concatenated_image = np.hstack(row_images)
 
        out.write(concatenated_image)
 
    out.release() 

    