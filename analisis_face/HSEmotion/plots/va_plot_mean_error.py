## ----- Imports -----
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
import os
from pathlib import Path

matplotlib.use('TkAgg')
has_prova = True
has_scale = False
smooth = False

## ----- color -----
def set_color(v_name):
    if  v_name == 'Q1_1':
        return "yellow"
    elif  v_name == 'Q1_2':
        return "gold"
    elif  v_name == 'Q2_1':
        return "lime"
    elif  v_name == 'Q2_2':
        return "limegreen"
    elif  v_name == 'Q3_1':
        return "blue"
    elif  v_name == 'Q3_2':
        return "mediumblue"
    elif  v_name == 'Q4_1':
        return "red"
    elif  v_name == 'Q4_2':
        return "firebrick"
    elif  v_name == 'Q5_1':
        return "orange"
    elif  v_name == 'Q5_2':
        return "darkorange"
    elif  v_name == 'Q6_1':
        return "cyan"
    elif  v_name == 'Q6_2':
        return "darkturquoise"
    elif  v_name == 'Q7_1':
        return "blueviolet"
    elif  v_name == 'Q7_2':
        return "darkviolet"
    elif  v_name == 'Q8_1':
        return "magenta"
    elif  v_name == 'Q8_2':
        return "mediumvioletred"
    elif  v_name == 'Q9_1':
        return "greenyellow"
    elif  v_name == 'Q9_2':
        return "yellowgreen"
    else:
        return 'grey'

## ----- Code -----
if __name__ == "__main__":
    ## path
    folder_path = Path("D:/castphys_60/Data")
    save_path = Path("C:/Users/Alex/Desktop/video_saves")

    ## list elements
    ids = [int(patient.split('_')[1]) for patient in os.listdir(folder_path)]
    list_videos = [v for v in os.listdir(folder_path / f"Patient_{ids[0]}") if 'Q' in v]

    fig, axs = plt.subplots(2)
    plt.subplots_adjust(hspace=0.35)

    for v_name in tqdm(list_videos):
        count = 0
        valance = 0
        arousal = 0

        for id in ids:
            if has_scale:
                ## rescale margin
                max_val = [-2,-2]
                min_val = [2,2]

                for v_name2 in list_videos:
                    video_folder2 = folder_path / f"Patient_{id}" / v_name2
                    try:
                        df = pd.read_csv(video_folder2/"FR_predic_HSE.csv")
                        max_val[0] = max(max_val[0], np.max(df['valence'].values))
                        min_val[0] = min(min_val[0], np.min(df['valence'].values))
                        max_val[1] = max(max_val[1], np.max(df['arousal'].values))
                        min_val[1] = min(min_val[1], np.min(df['arousal'].values))
                    except:
                        continue


            video_folder = folder_path / f"Patient_{id}" / v_name
            try:
                df = pd.read_csv(video_folder/"FR_predic_HSE.csv")
                df['time'] = df.index/60
                ## rescale
                if has_scale:
                    df['valence'] = 4*(df['valence']-min_val[0])/(max_val[0]-min_val[0])-2
                    df['arousal'] = 4*(df['arousal']-min_val[0])/(max_val[0]-min_val[0])-2
                count += 1
                df_annotations = pd.read_csv(video_folder/"annotations.csv",header=None) 
                va_pos = [int(float(v)) for [c,v] in df_annotations.values if c in ['valence','arousal']]
            except:
                continue

            try:
                valance = valance[:len(df)] + (np.abs(df['valence']-va_pos[0]))[:len(valance)]
                arousal = arousal[:len(df)] + (np.abs(df['arousal']-va_pos[1]))[:len(arousal)]
            except:
                #print(id)
                valance = (np.abs(df['valence']-va_pos[0]))
                arousal = (np.abs(df['arousal']-va_pos[1]))

        valance = valance/count
        arousal = arousal/count
        if smooth:
            N = 10
            valance = np.array([np.mean(valance[max(0,i-N+1):i+1]) for i in range(len(valance))])
            arousal = np.array([np.mean(arousal[max(0,i-N+1):i+1]) for i in range(len(arousal))])
        axs[0].plot(df['time'][:len(valance)], valance, color=set_color(v_name), label=v_name)
        axs[1].plot(df['time'][:len(arousal)], arousal, color=set_color(v_name), label=v_name)
        
        
    #fig.colorbar(line,ax=axs[0])
    axs[0].set_title(f"Mitja de l'Error")
    axs[0].set_ylabel("València")
    axs[0].set_xlabel("Temps [s]")
    #axs[0].set_ylim([-2,2])
    axs[0].legend(bbox_to_anchor = (1.1, 0.6))
    axs[1].set_ylabel("Activació")
    axs[1].set_xlabel("Temps [s]")
    #axs[1].set_ylim([-2,2])

    if has_prova:
        plt.show()
    else:
        plt.savefig(save_path / f"va_mean_error.png")

