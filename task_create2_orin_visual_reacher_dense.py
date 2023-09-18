import warnings
warnings.filterwarnings("ignore")

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'
# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from jsac.helpers.utils import MODE, make_dir, set_seed_everywhere
from jsac.helpers.logger import Logger
from jsac.envs.create2_orin_visual_reacher.env import Create2VisualReacherEnv
from jsac.helpers.utils import NormalizedEnv
from jsac.algo.agent import SACRADAgent, AsyncSACRADAgent
import time
from tensorboardX import SummaryWriter
import tqdm
import jaxlib
import argparse
import shutil
import multiprocessing as mp
import numpy as np


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
    parser.add_argument('--name', default='create2_orin_visual_reacher_dense', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--mode', default='img_prop', type=str, 
                        help="Modes in ['img', 'img_prop', 'prop']")
    
    parser.add_argument('--image_height', default=90, type=int)
    parser.add_argument('--image_width', default=160, type=int)
    parser.add_argument('--stack_frames', default=3, type=int)

    parser.add_argument('--camera_id', default=0, type=int)
    parser.add_argument('--episode_length_time', default=10.0, type=float)
    parser.add_argument('--dt', default=0.03, type=float)
    parser.add_argument('--min_target_size', default=0.99, type=float)
    parser.add_argument('--pause_before_reset', default=0, type=float)
    parser.add_argument('--pause_after_reset', default=0, type=float)

    parser.add_argument('--tqdm', default=True, action='store_true')

    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    
    # train
    parser.add_argument('--init_steps', default=3000, type=int)
    parser.add_argument('--env_steps', default=100000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--sync_mode', default=False, action='store_true')
    parser.add_argument('--apply_rad', default=True, action='store_true')
    parser.add_argument('--rad_offset', default=0.01, type=float)
    
    # critic
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--critic_target_update_freq', default=1, type=int)
    
    # actor
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)
    parser.add_argument('--use_critic_encoder', default=True, 
                        action='store_true')
    
    # encoder
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--spatial_softmax', default=True, action='store_true')
    
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--temp_lr', default=1e-4, type=float)
    
    # misc
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tensorboard', default=False, 
                        action='store_true')
    parser.add_argument('--xtick', default=1000, type=int)
    parser.add_argument('--save_wandb', default=False, action='store_true')

    parser.add_argument('--save_model', default=True, action='store_true')
    parser.add_argument('--save_model_freq', default=20000, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--start_step', default=1, type=int)

    parser.add_argument('--buffer_save_path', default='./buffers/', type=str)
    parser.add_argument('--buffer_load_path', default='', type=str)

    args = parser.parse_args()
    return args

def main(seed=-1):
    args = parse_args()

    # curriculum = [
    #     (30000, 0.10),
    #     (45000, 0.15),
    #     (60000, 0.20),
    #     (70000, 0.25),
    #     (75000, 0.30),
    #     (args.env_steps + 1000, 0.30)
    # ]
    # curriculum_idx = 0

    assert args.mode == MODE.IMG_PROP
    assert args.sync_mode == False

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
            print('Previous work dir removed.')
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

    image_shape = (args.image_height, args.image_width, 3*args.stack_frames)

    env = Create2VisualReacherEnv(
        episode_length_time=args.episode_length_time, 
        dt=args.dt,
        image_shape=image_shape,
        camera_id=args.camera_id,
        min_target_size=args.min_target_size,
        pause_before_reset=args.pause_before_reset,
        pause_after_reset=args.pause_after_reset,
        dense_reward=True
        )
    
    env = NormalizedEnv(env, save_images=True, 
                        images_path=args.work_dir+'images/',
                        save_data=True,
                        data_path=args.work_dir+'data.txt')
    
    set_seed_everywhere(seed=args.seed)
    env.start()

    episode_length_step = int(args.episode_length_time / args.dt)
    args.image_shape = env.image_space.shape
    args.proprioception_shape = env.observation_space.shape
    args.action_shape = env.action_space.shape
    args.net_params = config
    args.env_action_space = env.action_space

    # print(args.image_shape, image.shape)
    # print(args.proprioception_shape, proprioception.shape)

    if args.sync_mode:
        agent = SACRADAgent(args)
    else:
        agent = AsyncSACRADAgent(args)

    task_start_time = time.time()

    update_paused = True

    (image, proprioception) = env.reset()

    ret = 0
    episode = 0
    step = 0
    epi_steps = 0

    for step in tqdm.tqdm(range(args.start_step, args.env_steps + 1), 
                          smoothing=0.1, disable=not args.tqdm):
        t1 = time.time()
        # if step < args.init_steps:
        #     action = env.action_space.sample()
        #     action = np.tanh(action)
        # else:
        action = agent.sample_actions((image, proprioception))
        t2 = time.time()
        (next_image, next_proprioception), reward, done, info = env.step(action)
        t3 = time.time()

        ret += reward
        epi_steps += 1
        
        mask = 0
        if epi_steps == episode_length_step:
            done = True
            mask = 1.0

        agent.add((image, proprioception), action, reward, 
                  (next_image, next_proprioception),  mask)

        image = next_image
        proprioception = next_proprioception

        if done:
            info['tag'] = 'train'
            info['dump'] = True

            L.push(info)

            ret = 0
            epi_steps = 0
            episode += 1

            if episode == 4:
                break

            (image, proprioception) = env.reset()
            done = False

            if update_paused and step >= args.init_steps:
                agent.resume_update()
                update_paused = False
                time.sleep(25)

        if not update_paused:
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

    agent.pause_update()
    if args.save_model:
        agent.checkpoint(args.env_steps)
    L.plot()
    L.close()


    agent.close()
    env.close()

    end_time = time.time()
    print(f'\nFinished in {end_time - task_start_time}s')


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()


# sudo chmod a+rw /dev/ttyUSB0


    

