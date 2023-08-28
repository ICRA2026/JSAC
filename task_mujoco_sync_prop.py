import warnings
warnings.filterwarnings("ignore")

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'
# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from jsac.helpers.utils import MODE, make_dir, set_seed_everywhere
from jsac.helpers.logger import Logger
from jsac.envs.mujoco_visual_env import MujocoVisualEnv
from jsac.algo.agent import SACRADAgent, AsyncSACRADAgent
import time
from tensorboardX import SummaryWriter
import tqdm
import jaxlib
import argparse
import multiprocessing as mp
import shutil


config = {
    'conv': [
        # in_channel, out_channel, kernel_size, stride
        [-1, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 1],
    ],
    
    'latent': 50,

    'mlp': [1024, 1024],
}

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--name', default='reacher_sync_prop', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--mode', default='prop', type=str, 
                        help="Modes in ['img', 'img_prop', 'prop']")
    

    parser.add_argument('--env_name', default='Reacher-v2', type=str)
    # parser.add_argument('--image_height', default=120, type=int)
    # parser.add_argument('--image_width', default=120, type=int)
    # parser.add_argument('--stack_frames', default=3, type=int)
    parser.add_argument('--tqdm', default=True, action='store_true')

    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=25000, type=int)
    
    # train
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--env_steps', default=25000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--sync_mode', default=True, action='store_true')
    # parser.add_argument('--apply_rad', default=True, action='store_true')
    # parser.add_argument('--rad_offset', default=0.01, type=float)
    
    # critic
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--critic_target_update_freq', default=1, type=int)
    
    # actor
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)
    # parser.add_argument('--use_critic_encoder', default=True, 
    #                     action='store_true')
    
    # encoder
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    # parser.add_argument('--spatial_softmax', default=True, action='store_true')
    
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--temp_lr', default=1e-4, type=float)
    
    # misc
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tensorboard', default=False, action='store_true')
    parser.add_argument('--xtick', default=300, type=int)
    parser.add_argument('--save_wandb', default=False, action='store_true')

    parser.add_argument('--save_model', default=True, action='store_true')
    parser.add_argument('--save_model_freq', default=5000, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--start_step', default=1, type=int)

    parser.add_argument('--buffer_save_path', default='./buffer/', type=str)
    parser.add_argument('--buffer_load_path', default='', type=str)

    args = parser.parse_args()
    return args

def main(seed=-1):
    args = parse_args()

    assert args.mode == MODE.PROP
    args.apply_rad = False
    args.spatial_softmax = False
    args.use_critic_encoder = False

    if seed != -1:
        args.seed = seed

    args.work_dir += f'/results/{args.name}/seed_{args.seed}/'

    if os.path.exists(args.work_dir):
        inp = input('The work directory already exists. ' +
                    'Please select one of the following: \n' +  
                    '  1) Press Enter to resume the run.\n' + 
                    '  2) Press X to remove the previous work' + 
                    ' directory and start a new run.\n' + 
                    '  3) Press any other key to exit.\n')
        if inp == 'X' or inp == 'x':
            shutil.rmtree(args.work_dir)
        elif inp == '':
            pass
        else:
            exit(0)

    make_dir(args.work_dir)

    if args.buffer_save_path:
        make_dir(args.buffer_save_path)

    args.model_dir = f'{args.work_dir}/checkpoints/'

    if args.save_wandb:
        wandb_project_name = f'{args.name}'
        wandb_run_name=f'seed_{args.seed}'
        L = Logger(args.work_dir, args.xtick, args.save_tensorboard, 
                   args.save_wandb, wandb_project_name, wandb_run_name, 
                   vars(args), args.start_step > 1)
    else:
        L = Logger(args.work_dir, args.xtick, args.save_tensorboard, 
                   args.save_wandb)

    env = MujocoVisualEnv(env_name=args.env_name, use_img=False, seed=args.seed)

    set_seed_everywhere(seed=args.seed)

    args.image_shape = env.image_space.shape
    args.proprioception_shape = env.proprioception_space.shape
    args.action_shape = env.action_space.shape
    args.net_params = config
    args.env_action_space = env.action_space

    proprioception = env.reset()
    
    if args.sync_mode:
        agent = SACRADAgent(args)
    else:
        agent = AsyncSACRADAgent(args)

    task_start_time = time.time()

    for step in tqdm.tqdm(range(args.start_step, args.env_steps + 1), 
                          smoothing=0.1, disable=not args.tqdm):
        t1 = time.time()
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(proprioception)
        t2 = time.time()
        next_proprioception, reward, done, info = env.step(action)
        t3 = time.time()

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        agent.add(proprioception, action, reward, next_proprioception, mask)

        proprioception = next_proprioception

        if done:
            proprioception = env.reset()
            done = False

            log_data = info['episode']
            log_data['tag'] = 'train'
            log_data['dump'] = True
            log_data['step'] = step

            L.push(log_data)

        if step >= args.init_steps:
            update_infos = agent.update()
            if update_infos is not None:
                for update_info in update_infos:
                    update_info['inference_time'] = (t2 - t1) * 1000
                    update_info['env_step_time'] = (t3 - t2) * 1000
                    update_info['tag'] = 'train'
                    update_info['dump'] = False
                    update_info['step'] = step

                    L.push(update_info)

        if step % args.xtick == 0:
            L.plot()

        if args.save_model and step % args.save_model_freq == 0 and \
            step < args.env_steps:
            agent.checkpoint(step)

    if args.save_model:
        agent.checkpoint(args.env_steps)
    L.plot()
    L.close()

    agent.close()

    end_time = time.time()
    print(f'\nFinished in {end_time - task_start_time}s')


if __name__ == '__main__':
    mp.set_start_method('spawn')

    # for i in range(15):
    #     main(i)

    main()





    

