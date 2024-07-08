## ----- Imports -----
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator
import os
from pathlib import Path

matplotlib.use('TkAgg')

def clr_select(name):
    try:
        c,i = name.split('_')
        i = (int(i)-1.5)*2
    except:
        return "grey"
    if c == 'Q1':
        return "yellow"
    elif c == 'Q2':
        return "lime"
    elif c == 'Q3':
        return "blue"
    elif c == 'Q4':
        return "red"
    elif c == 'Q5':
        return "orange"
    elif c == 'Q6':
        return "cyan"
    elif c == 'Q7':
        return "blueviolet"
    elif c == 'Q8':
        return "magenta"
    elif c == 'Q9':
        return "greenyellow"
    else:
        return "grey"


has_prova = False

## ----- Code -----
if __name__ == "__main__":
    ## path
    folder_path = Path("D:/castphys_60/Data")
    save_path = Path("C:/Users/Alex/Desktop/video_saves")

    ## list elements
    ids = [int(patient.split('_')[1]) for patient in os.listdir(folder_path)]
    list_videos = [v for v in os.listdir(folder_path / f"Patient_{ids[0]}") if 'Q' in v]

    ## mean global pel video
    valence_all = []
    arousal_all = []
    emotion_all = []
    for v_name in tqdm(list_videos):
        if v_name != 'Q1_2':
            N_emo = 6
        else:
            N_emo = 3

        valence_list = []
        arousal_list = []
        emotion_list = []

        for id in ids:
            ## cargar el video
            video_folder = folder_path / f"Patient_{id}" / v_name
            df = pd.read_csv(video_folder/"FR_predic_HSE.csv")
            df['time'] = df.index/60

            ## segmentar en tres perts i agafar l'ultima (Q1_2 especial) 
            N_segment = int(round(len(df)/N_emo))

            # valance & arousal
            valance = df['valence'].values[N_segment*(N_emo-1):]
            arousal = df['arousal'].values[N_segment*(N_emo-1):]

            # discret emotion
            emotions = list(df['emotion'].values[N_segment*(N_emo-1):])
            emotions_dict = dict(zip(emotions,map(lambda x: emotions.count(x),emotions)))
            emotion = list(emotions_dict.keys())[np.argmax(list(emotions_dict.values()))]

            ## calcular i guardar mitjana del participant
            #valence_list.append(np.min(valance))
            #arousal_list.append(np.min(arousal))
            #emotion_list.append(emotion)
            idx_max = np.argmin(valance)
            valence_list.append(valance[idx_max])
            arousal_list.append(arousal[idx_max])
            emotion_list.append(emotions[idx_max])


        ## calcular i guardar mitjana total del video
        valence_all.append(valence_list)
        arousal_all.append(arousal_list)
        emotion_all.append(emotion_list)

    
    ## plot global de cada video
    emotion_class=['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

    for i, v_name in enumerate(list_videos):
        valence = valence_all[i]
        arousal = arousal_all[i]
        emotion = emotion_all[i]

        
        fig, axs = plt.subplots(1,2)
        fig.set_size_inches(12,5)
        ## plot valance & arousal
        axs[0].plot([2/3]*2, [2,-2], color = 'lightgray',linestyle = "--")
        axs[0].plot([-2/3]*2, [2,-2], color = 'lightgray',linestyle = "--")
        axs[0].plot([2,-2], [2/3]*2, color = 'lightgray',linestyle = "--")
        axs[0].plot([2,-2], [-2/3]*2, color = 'lightgray',linestyle = "--")
        axs[0].scatter(valence, arousal,c = 'grey', marker="+", label='Prediction')
        axs[0].scatter(np.mean(valence), np.mean(arousal),c = clr_select(v_name), label='Mean')
        
        axs[0].set_xlim(-2,2)
        axs[0].set_ylim(-2,2)
        axs[0].xaxis.set_major_locator(FixedLocator([-2,-1, 0, 1, 2]))
        axs[0].yaxis.set_major_locator(FixedLocator([-2,-1, 0, 1, 2]))

        axs[0].set_xlabel("Valence")
        axs[0].set_ylabel("Arousal")
        axs[0].legend()

        ## plot discret emotion
        emotions_dict = dict(zip(emotion,map(lambda x: emotion.count(x),emotion)))
        for e in emotion_class:
            try:
                axs[1].bar(x=e, height=emotions_dict[e])
            except:
                axs[1].bar(x=e, height=0)
        axs[1].tick_params(axis='x', labelrotation=30)
        axs[1].set_ylim(0,60)
        axs[1].set_ylabel("Count")

        if has_prova:
            plt.show()
            break
        else:
            plt.savefig(save_path / 'va_2d_emo' / f"va_2d_argmin_{v_name}.png")

