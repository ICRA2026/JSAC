import os
import sys
import jax
import flax
import numpy as np
import jax.numpy as jnp
import multiprocessing as mp
 
from jsac.helpers.logger import Logger
from jsac.helpers.utils import WrappedEnv
from jsac.algo.agent import sample_actions
from jsac.algo.initializers import init_inference_actor 
from jsac.envs.dmc_visual_env.dmc_env import DMCVisualEnv
from jsac.envs.mujoco_visual_env.mujoco_visual_env import MujocoVisualEnv


def start_eval_process(args, log_dir, eval_queue, num_eval_episodes):
    eval_process = mp.Process(target=eval, args=(args, 
                                                 log_dir, 
                                                 eval_queue, 
                                                 num_eval_episodes))
    eval_process.start()
    return eval_process
    

def eval(args, log_dir, eval_queue, num_eval_episodes):
    if args['env_type'] == 'MUJOCO':
        env = MujocoVisualEnv(args['env_name'], 
                            args['mode'], 
                            args['seed'] + 42, 
                            args['image_history'], 
                            args['image_width'], 
                            args['image_height'])
    else:
        env = DMCVisualEnv(args['env_name'], 
                           args['mode'], 
                           args['seed'] + 42, 
                           args['image_history'], 
                           args['image_width'], 
                           args['image_height'], 
                           args['num_cameras'], 
                           args['action_repeat'])
    env = WrappedEnv(env)
    
    if args['save_wandb']:
        wandb_project_name = args['name']
        wandb_run_name= 'seed_' + str(args['seed']) + '_eval'
        logger = Logger(log_dir, eval=True, use_wandb=True, 
                        wandb_project_name=wandb_project_name, 
                        wandb_run_name=wandb_run_name) 
    else:
        logger = Logger(log_dir, eval=True) 
    rng = jax.random.PRNGKey(0)
    rng, actor = init_inference_actor(rng, 
                                      args['image_shape'],
                                      args['proprioception_shape'],
                                      args['net_params'],
                                      args['action_shape'][-1],
                                      args['spatial_softmax'],
                                      args['layer_norm'],
                                      args['mode'],
                                      jnp.float32)
    
    best_return = -1e8
    best_actor_params_path = os.path.join(log_dir, 'best_actor_params.pkl') 
    params = None
    while True:
        data = eval_queue.get()
        if isinstance(data, str):
            if data == 'close':
                logger.close()
                sys.exit()
        else:
            params = data
            step = int(eval_queue.get())
        
        epi = 0
        state = env.reset()
        
        sum_ret = 0
        while epi < num_eval_episodes: 
            rng, action = sample_actions(rng, 
                                         actor.apply, 
                                         params, 
                                         state, 
                                         args['mode'], 
                                         True)

            action = np.asarray(action).clip(-1, 1)
            state, reward, done, info = env.step(action)
            
            if done or 'truncated' in info:
                sum_ret += info['return']
                state = env.reset()
                info['tag'] = 'eval'
                info['dump'] = True
                info['eval_step'] = step 
                logger.push(info)
                epi += 1
        
        if sum_ret >= best_return:
            best_return = sum_ret
            if os.path.exists(best_actor_params_path):
                os.remove(best_actor_params_path)
            with open(best_actor_params_path, 'wb') as f: 
                f.write(flax.serialization.to_bytes(params))        
                
        logger.plot()