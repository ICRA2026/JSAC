import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd  
import os
import cv2
import csv
import numpy as np
from read_files import read_log_file, read_csv_file, read_tlog_file

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

def avg_graph(file_paths, 
              colors, 
              labels, 
              file_types, 
              seeds, 
              ylim, 
              title, 
              maxlen, 
              output_path, 
              mlts, 
              x_axis_name='steps', 
              remove_ylabel=False,
              dreamer_csv=None,
              tdmpc2_csv=None):
    
    sns.set_theme(rc={
        'figure.figsize': (12, 12),
        'axes.labelsize': 28, 
        'axes.titlesize': 34,     
        'xtick.labelsize': 20, 
        'ytick.labelsize': 24, 
        'legend.fontsize': 30 
    })
    sns.set_style("whitegrid")
    
    num_tasks = len(labels)
    
    for itr in range(num_tasks): 
        label = labels[itr]
        print('\t', label)
        
        if label == 'dreamer-v3' or label == 'tdmpc2':
            csv_fl = dreamer_csv if label == 'dreamer-v3' else tdmpc2_csv
            if len(csv_fl) > 0: 
                print(csv_fl)
                data = []
                with open(csv_fl, 'r') as f:
                    r = csv.DictReader(f)
                    for row in r:
                        if int(row['step']) <= maxlen:
                            data.append([int(row['step']), float(row['reward']), label])
                df = pd.DataFrame(data, columns=[x_axis_name, "return", "Task"]) 
                sns.lineplot(x=x_axis_name, y='return', data=df,
                        color=sns.color_palette(colors)[itr], 
                        linewidth=2.5, label=labels[itr], err_kws={'alpha': 0.1}) #, errorbar=None)
        else:
            comb_data = []
            base_path = file_paths[itr] 
            
            for seed in seeds:
                eval_steps = None
                try:
                    if file_types[itr] == 'log':
                        path = f'{base_path}/seed_{seed}/eval.log'
                        eval_steps, returns = read_log_file(path)
                    if file_types[itr] == 'tlog':
                        path = f'{base_path}/seed_{seed}/train.log'
                        ep_lens, returns = read_tlog_file(path)
                    elif file_types[itr] == 'csv':
                        path = f'{base_path}/seed_{seed}/eval_log.csv'
                        eval_steps, returns = read_csv_file(path)
                except:
                    print(f'\tSeed {seed}, error reading file.')
                    continue
    
                
                if file_types[itr] == 'tlog':
                    sns.set_theme(rc={
                        'figure.figsize': (12, 10),
                        'axes.labelsize': 28, 
                        'axes.titlesize': 34,     
                        'xtick.labelsize': 20, 
                        'ytick.labelsize': 24, 
                        'legend.fontsize': 30 
                    })
                    sns.set_style("whitegrid")

                    steps = 0
                    end_step = 5000
                    rets = [] 
                    for (i, epi_s) in enumerate(ep_lens):
                        if steps + epi_s > end_step:
                            if len(rets) > 0:
                                comb_data.append([end_step, sum(rets)/len(rets), label]) 
                                rets = []
                            end_step += 5000
                        
                        steps += epi_s 
                        ret = returns[i]  
                        rets.append(ret) 
                    
                    if len(rets) > 0:  
                        comb_data.append([end_step, sum(rets)/len(rets), label]) 
                    
                else:
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
                    linewidth=2.5, label=labels[itr], err_kws={'alpha': 0.1}) # , errorbar=None)
    
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

def plot_prop():
    prop_env_names = ['Ant-v5', 'HalfCheetah-v5', 'Hopper-v5', 'Humanoid-v5', 'Walker2d-v5']
    
    jsac_base_path = 'results_jsac_no_ln/prop'
    # jsac_with_ln_base_path = 'results_jsac_ln/prop' 
    sb3_results_base_path = 'results_sb3' 
    redq_results_base_path = 'results_redq'
    
    ## PLOT PROP
    ylabel = False

    os.makedirs('all_plots/prop', exist_ok=True)
    for env_name in prop_env_names:
        print('Steps - Prop, Env name:', env_name)
        title = env_name 
        jsac_env_res_folder = env_name + '_prop_sync'
        jsac_path = os.path.join(jsac_base_path, jsac_env_res_folder)
        # jsac_with_ln_path = os.path.join(jsac_with_ln_base_path, jsac_env_res_folder)
        sb3_path = os.path.join(sb3_results_base_path, env_name)
        redq_path = os.path.join(redq_results_base_path, env_name)

        # paths = [jsac_path, jsac_with_ln_path, sb3_path, redq_path]
        paths = [jsac_path, sb3_path, redq_path]
        output_path = f'all_plots/prop/{env_name}.png'

        avg_graph(
            file_paths=paths,
            colors=sns.color_palette('bright'), 
            labels=['jsac', 'sb3', 'redq'], 
            file_types=['log', 'csv', 'log'], 
            seeds=range(15), 
            ylim=None, 
            title=title + ' - Eval', 
            maxlen=1_000_000,
            output_path=output_path,
            mlts=[1, 1, 1, 1],
            remove_ylabel=ylabel)
        
        ylabel = True
        
    combine_images('all_plots/prop/', 'combined.png')
       
def plot_imgs():
    ylabel = False
    img_env_names = ['walker_walk', 'ball_in_cup', 'cartpole_swingup', 'cheetah', 'finger_spin', 'hopper_hop']
    img_env_titles = ['walker_walk', 'ball_in_cup_catch', 'cartpole_swingup', 'cheetah_run', 'finger_spin', 'hopper_hop']

    jsac_base_path = 'results_jsac_no_ln/img'
    # jsac_with_ln_base_path = 'results_jsac_ln/img'  
      
    dreamer_csvs = ['results_tdmpc2/dreamerv3/walker-walk.csv',
                    '', 
                    'results_tdmpc2/dreamerv3/cartpole-swingup.csv', 
                    'results_tdmpc2/dreamerv3/cheetah-run.csv', 
                    'results_tdmpc2/dreamerv3/finger-spin.csv',
                    'results_tdmpc2/dreamerv3/hopper-hop.csv']
    
    tdmpc2_csvs = ['results_tdmpc2/tdmpc2-pixels/walker-walk.csv',
                   '', 
                   '', 
                   'results_tdmpc2/tdmpc2-pixels/cheetah-run.csv', 
                   'results_tdmpc2/tdmpc2-pixels/finger-spin.csv',
                   '']

    os.makedirs('all_plots/img', exist_ok=True)
    
    for idx, env_name in enumerate(img_env_names):
        print('Steps - Img, Env name:', env_name)

        title = img_env_titles[idx] 
        jsac_env_res_folder  = env_name + '_img_sync' 
        jsac_path = os.path.join(jsac_base_path, jsac_env_res_folder)
        # jsac_with_ln_path = os.path.join(jsac_with_ln_base_path, jsac_env_res_folder)

        paths = [jsac_path] #, jsac_with_ln_path]
        output_path = f'all_plots/img/{env_name}.png'
        if env_name == 'walker_walk':
            output_path = f'all_plots/img/a_{env_name}.png'

        maxlen = 500_000
        if env_name in ['ball_in_cup', 'cartpole_swingup']:
            maxlen = 200_000

        avg_graph(
            file_paths=paths,
            colors=sns.color_palette('bright'), 
            labels=['jsac', 'dreamer-v3', 'tdmpc2'], 
            file_types=['log'], 
            seeds=range(15), 
            ylim=None, 
            title=title + ' - eval', 
            maxlen=maxlen,
            output_path=output_path,
            mlts=[1, 1],
            remove_ylabel=ylabel,
            dreamer_csv=dreamer_csvs[idx],
            tdmpc2_csv=tdmpc2_csvs[idx])
        
        ylabel = True
        
    combine_images('all_plots/img/', 'combined.png')

def plot_orin():
    os.makedirs('all_plots/create2_orin', exist_ok=True)
    avg_graph(
            file_paths=["results_orin"],
            colors=sns.color_palette('bright'), 
            labels=['Create2_Orin'], 
            file_types=['tlog'], 
            seeds=range(1), 
            ylim=None, 
            title='Create2 Orin - Real-world Experiment', 
            maxlen=75000,
            output_path='all_plots/create2_orin/plot.png',
            mlts=[1],
            remove_ylabel=False)

if __name__ == "__main__": 
    plot_prop()
    plot_imgs()
    plot_orin()
