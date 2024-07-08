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
    save_path = Path("C:/Users/Alex/Desktop/video_saves/error")

    ## list elements
    ids = [int(patient.split('_')[1]) for patient in os.listdir(folder_path)]
    list_videos = [v for v in os.listdir(folder_path / f"Patient_{ids[0]}") if 'Q' in v]

    for v_name in list_videos:
        count = 0
        mean = 0

        fig, ax = plt.subplots()

        for id in ids:


            video_folder = folder_path / f"Patient_{id}" / v_name
            try:
                df = pd.read_csv(video_folder/"FR_predic_HSE.csv")
                df['time'] = df.index/60
                count += 1
                df_annotations = pd.read_csv(video_folder/"annotations.csv",header=None) 
                va_pos = [int(float(v)) for [c,v] in df_annotations.values if c in ['valence','arousal']]
            except:
                continue
            
            if has_mean:
            #proba += df['valence'].values[1000]
                valance2 = ((df['valence']-va_pos[0]))**2
                arousal2 = ((df['arousal']-va_pos[1]))**2
                error = np.sqrt(valance2 + arousal2)
                try:
                    mean = mean[:len(df)] + error[:len(mean)]
                except:
                    mean = error
            
            ax.plot(df['time'], error, color='cornflowerblue',alpha=0.2)
            

        if has_mean:
            #print(valance[1000])
            #print(proba)
            mean = mean/count

            ax.plot(df['time'][:len(mean)], mean, color='red')
        
        
        #fig.colorbar(line,ax=axs[0])
        ax.set_title(f"Video: {v_name}")
        ax.set_ylabel("Error")
        ax.set_xlabel("Temps [s]")


        if has_prova:
            plt.show()
            break
        else:
            plt.savefig(save_path / f"error_{v_name}.png")

