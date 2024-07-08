## ----- Imports -----
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import os
from pathlib import Path

matplotlib.use('TkAgg')
has_mean = True
has_prova = False

## ----- Code -----
if __name__ == "__main__":
    ## path
    folder_path = Path("D:/castphys_60/Data")
    save_path = Path("C:/Users/Alex/Desktop/video_saves/estimation_s")

    ## list elements
    ids = [int(patient.split('_')[1]) for patient in os.listdir(folder_path)]
    list_videos = [v for v in os.listdir(folder_path / f"Patient_{ids[0]}") if 'Q' in v]

    for v_name in list_videos:
        count = 0
        valance = 0
        arousal = 0

        #proba = 0

        fig, axs = plt.subplots(2)
        plt.subplots_adjust(hspace=0.35)

        for id in ids:
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
                
            #print(max_val,min_val)
            video_folder = folder_path / f"Patient_{id}" / v_name
            try:
                df = pd.read_csv(video_folder/"FR_predic_HSE.csv")
                df['time'] = df.index/60
                ## rescale
                df['valence'] = 4*(df['valence']-min_val[0])/(max_val[0]-min_val[0])-2
                df['arousal'] = 4*(df['arousal']-min_val[0])/(max_val[0]-min_val[0])-2
                count += 1
            except:
                continue
            
            if has_mean:
            #proba += df['valence'].values[1000]
                try:
                    valance = valance[:len(df)] + df['valence'].values[:len(valance)]
                    arousal = arousal[:len(df)] + df['arousal'].values[:len(arousal)]
                except:
                    #print(id)
                    valance = df['valence'].values
                    arousal = df['arousal'].values
            

            axs[0].plot(df['time'], df['valence'], color='cornflowerblue',alpha=0.2)
            axs[1].plot(df['time'], df['arousal'], color='orange',alpha=0.2)

        if has_mean:
            #print(valance[1000])
            #print(proba)
            valance = valance/count
            arousal = arousal/count

            axs[0].plot(df['time'][:len(valance)], valance, color='red')
            axs[1].plot(df['time'][:len(arousal)], arousal, color='red')
        
        
        #fig.colorbar(line,ax=axs[0])
        axs[0].set_title(f"Video: {v_name}")
        axs[0].set_ylabel("València")
        axs[0].set_xlabel("Temps [s]")
        axs[0].set_ylim([-2,2])
        axs[1].set_ylabel("Activació")
        axs[1].set_xlabel("Temps [s]")
        axs[1].set_ylim([-2,2])

        if has_prova:
            plt.show()
            break
        else:
            plt.savefig(save_path / f"va_estimation_{v_name}.png")

