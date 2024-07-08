## plot el senyal tonic de tots els participants per a un video

## ----- Imports -----
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
import os
from pathlib import Path

matplotlib.use('TkAgg')
has_prova = False

## ----- Code -----
if __name__ == "__main__":
    ## path
    folder_path = Path("D:/castphys_60/Data")
    save_path = Path("C:/Users/Alex/Desktop/video_saves")

    ## list elements
    ids = [int(patient.split('_')[1]) for patient in os.listdir(folder_path)]
    list_videos = [v for v in os.listdir(folder_path / f"Patient_{ids[0]}") if 'Q' in v]

    for v_name in tqdm(list_videos):
        fig, ax = plt.subplots()

        for id in ids:
            video_folder = folder_path / f"Patient_{id}" / v_name

            df = pd.read_csv(video_folder/"eda_analisis.csv")
            df['time'] = df.index/60

            ## Normalize the signal
            tonic_signal = df["tonic"].values
            var_rate = np.max(tonic_signal)-np.min(tonic_signal)
            #tonic_signal = (tonic_signal-np.min(tonic_signal))/(np.max(tonic_signal)-np.min(tonic_signal))
            #driver_signal = (driver_signal-driver_signal.min())/(driver_signal.max()-driver_signal.min())

            #if var_rate > 1000:
            ax.plot(df['time'], tonic_signal, color="orange")
        
        ax.set_title(f"Video: {v_name}")
        ax.set_ylabel("Tònic")
        ax.set_xlabel("Temps [s]")
        #ax.set_ylim([0,1])

        if has_prova:
            plt.show()
            break
        else:
            plt.savefig(save_path / f"tonic_{v_name}.png")

