import warnings
warnings.filterwarnings("ignore")

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'
# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from jsac.helpers.logger import Logger
from jsac.envs.create2_orin_visual_reacher.env import Create2VisualReacherEnv
from jsac.helpers.utils import MODE, make_dir, set_seed_everywhere, WrappedEnv, get_episode_and_steps_from_log
from jsac.algo.agent import SACRADAgent, AsyncSACRADAgent
import time
import argparse
import shutil
import multiprocessing as mp
import numpy as np


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

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--name', default='create2_orin_nano_multi_reacher', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--ob_type', default='MASK_OH', type=str) ## OH, MASK, MASK_OH
    
    parser.add_argument('--image_height', default=60, type=int)
    parser.add_argument('--image_width', default=80, type=int)
    parser.add_argument('--image_history', default=2, type=int)
    parser.add_argument('--mode', default='img_prop', type=str, 
                        help="Modes in ['img', 'img_prop', 'prop']")
    parser.add_argument('--apply_weight_clip', default=True, action='store_true')

    parser.add_argument('--camera_id', default=0, type=int)
    parser.add_argument('--episode_length_time', default=12.0, type=float)
    parser.add_argument('--reward_scale', default=1.0, type=float)
    parser.add_argument('--dt', default=0.06, type=float)
    parser.add_argument('--min_target_size', default=0.40, type=float)
    parser.add_argument('--pause_before_reset', default=0, type=float)
    parser.add_argument('--pause_after_reset', default=0, type=float)

    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=5000, type=int)
    
    # train
    parser.add_argument('--init_steps', default=100000, type=int)
    parser.add_argument('--env_steps', default=100000, type=int)
    parser.add_argument('--task_timeout_mins', default=-1, type=int)
    parser.add_argument('--min_charge', default=1005, type=int)
    parser.add_argument('--batch_size', default=160, type=int)
    parser.add_argument('--sync_mode', default=False, action='store_true')
    parser.add_argument('--global_norm', default=1.0, type=float)
    
    # critic
    parser.add_argument('--critic_lr', default=2e-4, type=float) 
    parser.add_argument('--num_critic_networks', default=5, type=int)
    parser.add_argument('--num_critic_updates', default=1, type=int)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--critic_target_update_freq', default=1, type=int)
    
    
    # actor
    parser.add_argument('--actor_lr', default=2e-4, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)
    parser.add_argument('--actor_sync_freq', default=16, type=int)
    
    # encoder
    parser.add_argument('--spatial_softmax', default=False, action='store_true')
    
    # sac
    parser.add_argument('--temp_lr', default=2e-4, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--discount', default=0.99, type=float)
    
    # misc
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tensorboard', default=False, 
                        action='store_true')
    parser.add_argument('--xtick', default=2500, type=int)
    parser.add_argument('--save_wandb', default=False, action='store_true')

    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_model_freq', default=5000, type=int)
    parser.add_argument('--load_model', default=75000, type=int)

    parser.add_argument('--buffer_save_path', default='./buffer/', type=str)
    parser.add_argument('--buffer_load_path', default='', type=str)

    args = parser.parse_args()
    return args

def get_run_flag():
    with open('run_flags.txt', 'r') as f:
        return int(f.readline())

def main(seed=-1):
    task_start_time = time.time()
    args = parse_args()

    if seed != -1:
        args.seed = seed

    args.start_episode, args.start_step = 0, 0

    RF_CONTINUE = 0
    RF_END_RUN_WO_SAVE = 1
    RF_END_RUN_W_SAVE = 2

    assert args.mode == MODE.IMG_PROP and args.sync_mode == False

    args.name = f'{args.name}_{args.ob_type}'
    args.work_dir += f'/results/{args.name}/seed_{args.seed}'

    if os.path.exists(args.work_dir):
        inp = input('The work directory already exists. ' +
                    'Please select one of the following: \n' +  
                    '  1) Press Enter to resume the run.\n' + 
                    '  2) Press X to remove the previous work' + 
                    ' directory and start a new run.\n' + 
                    '  3) Press any other key to exit.\n')
        if inp == 'X' or inp == 'x':
            shutil.rmtree(args.work_dir)
            print('Previous work dir removed.')
        elif inp == '':
            args.start_episode, args.start_step = get_episode_and_steps_from_log(args.work_dir)
        else:
            exit(0)

    rf = get_run_flag()
    if rf != RF_CONTINUE:
        print('Ending the run as the RF flag is set to END_RUN in', end=' ')
        print('run_flags.txt.\nPlease set the flag to 0 and try again!')
        exit(0)

    make_dir(args.work_dir)

    if args.buffer_save_path:
        if args.buffer_save_path == ".":
            args.buffer_save_path = os.path.join(args.work_dir, 'buffers')
        make_dir(args.buffer_save_path)
    
    if args.buffer_load_path == ".":
        args.buffer_load_path = os.path.join(args.work_dir, 'buffers')

    args.model_dir = os.path.join(args.work_dir, 'checkpoints') 
    if args.save_model:
        make_dir(args.model_dir)
        
    args.net_params = config

    if args.save_wandb:
        wandb_project_name = f'{args.name}'
        wandb_run_name=f'seed_{args.seed}'
        L = Logger(args.work_dir, args.xtick, vars(args), 
                   args.save_tensorboard, args.save_wandb, wandb_project_name, 
                   wandb_run_name, args.start_step > 1)
    else:
        L = Logger(args.work_dir, args.xtick, vars(args), 
                   args.save_tensorboard, args.save_wandb)

    channels = 4
    if args.ob_type == "OH":
        channels = 3
    image_shape = (args.image_height, args.image_width, channels*args.image_history)

    env = Create2VisualReacherEnv(
        episode_length_time=args.episode_length_time, 
        dt=args.dt,
        image_shape=image_shape,
        camera_id=args.camera_id,
        min_target_size=args.min_target_size,
        pause_before_reset=args.pause_before_reset,
        pause_after_reset=args.pause_after_reset,
        dense_reward=True,
        multi_target=True,
        ob_type=args.ob_type)
    
    episode_length_step = int(args.episode_length_time // args.dt)
    env = WrappedEnv(env, 
                     episode_max_steps=episode_length_step,
                     is_min_time=False,
                     start_step = args.start_step,
                     start_episode = args.start_episode)
    
    set_seed_everywhere(seed=args.seed)
    env.start()

    args.image_shape = env.image_space.shape
    args.proprioception_shape = env.observation_space.shape
    args.action_shape = env.action_space.shape
    args.env_action_space = env.action_space

    print('Image shape:', args.image_shape, ',   Proprioception shape:', args.proprioception_shape)

    if args.sync_mode:
        sync_queue = None
        agent = SACRADAgent(vars(args))
    else:
        sync_queue = mp.Queue()
        agent = AsyncSACRADAgent(vars(args), sync_queue)

    task_end_time = -1
    if args.task_timeout_mins > 0:
        task_end_time = task_start_time + (args.task_timeout_mins * 60)
    state = env.reset()
    first_step = True
    update_paused = True
    pause_for_update = True
    count = 0

    while env.total_steps < args.env_steps:
        t1 = time.time()
        action = agent.sample_actions(state)
        t2 = time.time()
        next_state, reward, done, info = env.step(action)
        t3 = time.time()

        done = True if done > 0.5 else False
        mask = 1.0 if not done or 'truncated' in info else 0.0

        agent.add(state, action, reward, next_state, mask, first_step)
        first_step = False
        state = next_state

        if done or 'truncated' in info:
            charge = info['battery_charge']

            info['tag'] = 'train'
            info['elapsed_time'] = time.time() - task_start_time
            info['dump'] = True
            L.push(info)
            state = env.reset() 
            first_step = True

            rf = get_run_flag()
            if (task_end_time > 0 and time.time() > task_end_time) or \
                rf == RF_END_RUN_WO_SAVE or rf == RF_END_RUN_W_SAVE:
                break

            if charge < args.min_charge:
                update_paused = True
                agent.pause_update()

            if update_paused and env.total_steps >= args.init_steps \
                and charge > args.min_charge:
                agent.resume_update()
                update_paused = False
                if pause_for_update:
                    sync_queue.put(1)
                    time.sleep(40)  ## Pause for initial jit compilation of udpate fucntion
                    pause_for_update = False

            count += 1
            if count == 5:
                break

        if sync_queue and not update_paused:
            sync_queue.put(1)

        if not update_paused and env.total_steps >= args.init_steps:
            update_infos = agent.update()
            if update_infos is not None:
                for update_info in update_infos:
                    update_info['action_sample_time'] = (t2 - t1) * 1000
                    update_info['env_time'] = (t3 - t2) * 1000
                    update_info['step'] = env.total_steps
                    update_info['tag'] = 'train'
                    update_info['dump'] = False
                    L.push(update_info)

        if env.total_steps % args.xtick == 0:
            L.plot()

        if args.save_model and env.total_steps % args.save_model_freq == 0 and \
            env.total_steps < args.env_steps:
            agent.checkpoint(env.total_steps)
            
    agent.pause_update()
    env.close()

    if rf != RF_END_RUN_WO_SAVE and args.save_model:
        agent.checkpoint(env.total_steps)
    L.plot()
    L.close()

    if rf == RF_END_RUN_WO_SAVE:
        agent.close(without_save=True)
    else:
        agent.close()
    

    end_time = time.time()
    print(f'\nFinished in {end_time - task_start_time}s')


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
