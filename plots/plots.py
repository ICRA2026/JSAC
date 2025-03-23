import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd  
import os
import cv2
import numpy as np
from read_files import read_log_file, read_csv_file, get_total_run_time

print(pd.__version__)


def combine_images(folder_path, output_image_name, gap_size=40):
    image_files = sorted([
        img for img in os.listdir(folder_path) 
        if img.endswith('.png')
    ])
    images = [
        cv2.imread(os.path.join(folder_path, img)) 
        for img in image_files
    ]
    top = images[:3]
    bottom = images[3:]
    
    max_top = max(img.shape[0] for img in top)
    max_bottom = max(img.shape[0] for img in bottom)
    
    def pad(img, max_h):
        if img.shape[0] < max_h:
            pad_height = max_h - img.shape[0]
            padding = 255 * np.ones(
                (pad_height, img.shape[1], 3), 
                dtype=img.dtype
            )
            return np.vstack((img, padding))
        return img
    
    top_padded = [pad(img, max_top) for img in top]
    bottom_padded = [pad(img, max_bottom) for img in bottom]
    
    gap = 255 * np.ones((max_top, gap_size, 3), dtype=images[0].dtype)
    top_row = top_padded[0]
    for img in top_padded[1:]:
        top_row = np.hstack([top_row, gap, img])
    
    gap_bottom = 255 * np.ones((max_bottom, gap_size, 3), dtype=images[0].dtype)
    bottom_row = bottom_padded[0]
    for img in bottom_padded[1:]:
        bottom_row = np.hstack([bottom_row, gap_bottom, img])
    
    pad_width = (top_row.shape[1] - bottom_row.shape[1]) // 2
    if pad_width > 0:
        padding = 255 * np.ones(
            (max_bottom, pad_width, 3), 
            dtype=images[0].dtype
        )
        bottom_row = np.hstack([padding, bottom_row, padding])
    
    vertical_gap = 255 * np.ones((gap_size, top_row.shape[1], 3), dtype=images[0].dtype)
    combined = np.vstack([top_row, vertical_gap, bottom_row])
    
    output_path = os.path.join(folder_path, output_image_name)
    cv2.imwrite(output_path, combined)

def avg_graph(file_paths, colors, labels, file_types, seeds, ylim, title, maxlen, output_path, mlts, x_axis_name='steps', remove_ylabel=False):
    sns.set_theme(rc={
        'figure.figsize': (12, 12),
        'axes.labelsize': 28, 
        'axes.titlesize': 34,     
        'xtick.labelsize': 20, 
        'ytick.labelsize': 24, 
        'legend.fontsize': 30 
    })
    sns.set_style("whitegrid")
    
    num_tasks = len(file_paths)

    for itr in range(num_tasks): 
        comb_data = []
        base_path = file_paths[itr] 
        
        print('\t', labels[itr])

        for seed in seeds:
            eval_steps = None
            try:
                if file_types[itr] == 'log':
                    path = f'{base_path}/seed_{seed}/eval.log'
                    eval_steps, returns = read_log_file(path)
                elif file_types[itr] == 'csv':
                    path = f'{base_path}/seed_{seed}/eval_log.csv'
                    eval_steps, returns = read_csv_file(path)
            except:
                print(f'\tSeed {seed}, error reading file.')
                continue
 
            curr_step = eval_steps[0]
            rets = [] 
            for (i, step) in enumerate(eval_steps):
                if step != curr_step:
                    if len(rets) > 0:
                        comb_data.append([curr_step * mlts[itr], sum(rets)/len(rets), labels[itr]]) 
                        rets = []
                    curr_step = step

                ret = returns[i]  
                rets.append(ret) 

                if curr_step == maxlen:
                    break
                
            if len(rets) > 0:  
                comb_data.append([curr_step * mlts[itr], sum(rets)/len(rets), labels[itr]]) 
            
            if abs(curr_step - maxlen) > 10:
                print(f'\tSeed {seed} did not reach {maxlen}. Reached {curr_step}.')
                
        df = pd.DataFrame(comb_data, columns=[x_axis_name, "return", "Task"]) 
        ax1 = sns.lineplot(x=x_axis_name, y='return', data=df,
                   color=sns.color_palette(colors)[itr], 
                   linewidth=2.0, label=labels[itr]) # , errorbar=None)
    
    ax1.legend(loc='lower right', framealpha=0.9, facecolor='white') 
    if remove_ylabel:
        ax1.set_ylabel("")
        ax1.set_xlabel("")
        ax1.get_legend().remove()

    if ylim:
        plt.ylim(*ylim)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_steps():
    prop_env_names = ['Ant-v4', 'HalfCheetah-v4', 'Hopper-v4', 'Humanoid-v4', 'Walker2d-v4']
    img_env_names = ['ball_in_cup', 'cartpole_swingup', 'cheetah', 'finger_spin', 'walker_walk']
    img_env_titles = ['ball_in_cup_catch', 'cartpole_swingup', 'cheetah_run', 'finger_spin', 'walker_walk']

    jsac_results_base_path = 'results/results_jsac'
    sb3_results_base_path = 'results/results_sb3' 
    redq_results_base_path = 'results/results_reqd'

    ## PLOT PROP
    ylabel = False

    os.makedirs('all_plots/prop', exist_ok=True)

    for env_name in prop_env_names:
        print('Steps - Prop, Env name:', env_name)
        title = env_name 
        jsac_env_res_folder = env_name + '_prop'
        jsac_path = os.path.join(jsac_results_base_path, jsac_env_res_folder)
        sb3_path = os.path.join(sb3_results_base_path, env_name)
        redq_path = os.path.join(redq_results_base_path, env_name)

        path = [jsac_path, sb3_path, redq_path]
        output_path = f'all_plots/prop/{env_name}.png'

        avg_graph(
            file_paths=path,
            colors=sns.color_palette('bright'), 
            labels=['jsac', 'sb3', 'redq'], 
            file_types=['log', 'csv', 'log'], 
            seeds=range(15), 
            ylim=None, 
            title=title + ' - Eval', 
            maxlen=1_000_000,
            output_path=output_path,
            mlts=[1, 1, 1],
            remove_ylabel=ylabel)
        
        ylabel = True
        
    combine_images('all_plots/prop/', 'combined.png')
    
    ## PLOT IMGS
    ylabel = False

    os.makedirs('all_plots/img', exist_ok=True)
    
    for idx, env_name in enumerate(img_env_names):
        print('Steps - Img, Env name:', env_name)

        title = img_env_titles[idx] 
        sync_folder = env_name + '_img_sync'
        async_folder = env_name + '_img_async'
        jsac_sync_path = os.path.join(jsac_results_base_path, sync_folder)
        jsac_async_path = os.path.join(jsac_results_base_path, async_folder)

        path = [jsac_sync_path, jsac_async_path]
        output_path = f'all_plots/img/{env_name}.png'

        maxlen = 500_000
        if env_name in ['ball_in_cup', 'cartpole_swingup']:
            maxlen = 250_000

        avg_graph(
            file_paths=path,
            colors=sns.color_palette('bright'), 
            labels=[f'jsac_sync', 'jsac_async'], 
            file_types=['log', 'log'], 
            seeds=range(15), 
            ylim=None, 
            title=title + ' - eval', 
            maxlen=maxlen,
            output_path=output_path,
            mlts=[1, 1],
            remove_ylabel=ylabel)
        
        ylabel = True
        
    combine_images('all_plots/img/', 'combined.png')


def get_timings(base_path, seeds, file_type):
    times = []
    for seed in seeds:
        if file_type == 'log':
            path = f'{base_path}/seed_{seed}/train.log'
            total_time = get_total_run_time(path, file_type)
        elif file_type == 'csv':
            path = f'{base_path}/seed_{seed}/training_log.csv'
            total_time = get_total_run_time(path, file_type)
        times.append(total_time)

    times = sorted(times)[3:-3]
    return sum(times) / len(times)


def plot_times():
    prop_env_names = ['Ant-v4', 'HalfCheetah-v4', 'Hopper-v4', 'Humanoid-v4', 'Walker2d-v4']
    img_env_names = ['ball_in_cup', 'cartpole_swingup', 'cheetah', 'finger_spin', 'walker_walk']
    img_env_titles = ['ball_in_cup_catch', 'cartpole_swingup', 'cheetah_run', 'finger_spin', 'walker_walk']

    jsac_results_base_path = 'results/results_jsac'
    sb3_results_base_path = 'results/results_sb3' 
    redq_results_base_path = 'results/results_reqd'

    os.makedirs('all_plots/times/prop', exist_ok=True)

    ## PLOT PROP
    ylabel = False

    for env_name in prop_env_names:
        print('Times - Prop, Env name:', env_name)

        title = env_name 
        jsac_env_res_folder = env_name + '_prop'
        jsac_path = os.path.join(jsac_results_base_path, jsac_env_res_folder)
        sb3_path = os.path.join(sb3_results_base_path, env_name)
        redq_path = os.path.join(redq_results_base_path, env_name)

        path = [jsac_path, sb3_path, redq_path]
        output_path = f'all_plots/times/prop/{env_name}.png'

        jsac_time = get_timings(jsac_path, range(15), 'log')
        jsac_mlt = (jsac_time / (60 * 1_000_000))

        sb3_time = get_timings(sb3_path, range(15), 'csv')
        sb3_mlt = (sb3_time / (60 * 1_000_000))

        redq_time = get_timings(redq_path, range(15), 'log')
        redq_mlt = (redq_time / (60 * 1_000_000))

        mlts = [jsac_mlt, sb3_mlt, redq_mlt]

        avg_graph(
            file_paths=path,
            colors=sns.color_palette('bright'), 
            labels=[f'jsac', 'sb3', 'redq'], 
            file_types=['log', 'csv', 'log'], 
            seeds=range(15), 
            ylim=None, 
            title=title + ' - Eval', 
            maxlen=1_000_000,
            output_path=output_path,
            mlts=mlts,
            x_axis_name='mins',
            remove_ylabel=ylabel)
        
        ylabel = True
    
    combine_images('all_plots/times/prop/', 'combined.png')

    os.makedirs('all_plots/times/img', exist_ok=True)

    ## PLOT IMGS
    ylabel = False

    for idx, env_name in enumerate(img_env_names):
        print('Times - Img, Env name:', env_name)

        title = img_env_titles[idx] 
        sync_folder = env_name + '_img_sync'
        async_folder = env_name + '_img_async'
        jsac_sync_path = os.path.join(jsac_results_base_path, sync_folder)
        jsac_async_path = os.path.join(jsac_results_base_path, async_folder)

        path = [jsac_sync_path, jsac_async_path]
        output_path = f'all_plots/times/img/{env_name}.png'

        sync_time = get_timings(jsac_sync_path, range(15), 'log')
        sync_mlt = (sync_time / (60 * 500_000))

        async_time = get_timings(jsac_async_path, range(15), 'log')
        async_mlt = (async_time / (60 * 500_000))

        mlts = [sync_mlt, async_mlt]

        maxlen = 500_000
        if env_name in ['ball_in_cup', 'cartpole_swingup']:
            maxlen = 250_000

        avg_graph(
            file_paths=path,
            colors=sns.color_palette('bright'), 
            labels=[f'jsac_sync', 'jsac_async'], 
            file_types=['log', 'log'], 
            seeds=range(15), 
            ylim=None, 
            title=title + ' - eval', 
            maxlen=maxlen,
            output_path=output_path,
            mlts=mlts,
            x_axis_name='mins')
        
        ylabel = True
        
    combine_images('all_plots/times/img/', 'combined.png')


if __name__ == "__main__": 
    plot_steps()
    plot_times()
