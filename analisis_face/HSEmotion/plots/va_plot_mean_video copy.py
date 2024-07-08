## ----- Imports -----
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
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


has_prova = True

## ----- Code -----
if __name__ == "__main__":
    ## path
    folder_path = Path("D:/castphys_60/Data")
    save_path = Path("C:/Users/Alex/Desktop/video_saves")

    ## list elements
    ids = [int(patient.split('_')[1]) for patient in os.listdir(folder_path)]
    list_videos = [v for v in os.listdir(folder_path / f"Patient_{ids[0]}") if 'Q' in v]

    ## mean global pel video
    valence_mean = []
    arousal_mean = []
    for v_name in tqdm(list_videos):
        if v_name != 'Q1_2':
            N_emo = 6
        else:
            N_emo = 3

        valence_list = []
        arousal_list = []

        for id in ids:
            ## cargar el video
            video_folder = folder_path / f"Patient_{id}" / v_name
            df = pd.read_csv(video_folder/"FR_predic_HSE.csv")
            df['time'] = df.index/60

            ## segmentar en tres perts i agafar l'ultima (Q1_2 especial) 
            N_segment = int(round(len(df)/N_emo))

            valance = df['valence'].values[N_segment*(N_emo-1):]
            arousal = df['arousal'].values[N_segment*(N_emo-1):]

            ## calcular i guardar mitjana del participant
            #valence_list.append(np.max(valance))
            #arousal_list.append(np.max(arousal))
            idx_max = np.argmax(valance)
            valence_list.append(valance[idx_max])
            arousal_list.append(arousal[idx_max])


        ## calcular i guardar mitjana total del video
        valence_mean.append(np.mean(valence_list))
        arousal_mean.append(np.mean(arousal_list))

    
    ## plot global de les means
    fig, ax = plt.subplots()
    ax.plot([0.5/3]*2, [0.5,-0.5], color = 'lightgray',linestyle = "--")
    ax.plot([-0.5/3]*2, [0.5,-0.5], color = 'lightgray',linestyle = "--")
    ax.plot([0.5,-0.5], [0.5/3]*2, color = 'lightgray',linestyle = "--")
    ax.plot([0.5,-0.5], [-0.5/3]*2, color = 'lightgray',linestyle = "--")
    plt.text(-1.3*0.25, 1.15*0.25, 'Q1', color = 'gray')
    plt.text(-0.05*0.25, 1.15*0.25, 'Q2', color = 'gray')
    plt.text(1.15*0.25, 1.15*0.25, 'Q3', color = 'gray')
    plt.text(-1.3*0.25, -0.05*0.25, 'Q4', color = 'gray')
    plt.text(-0.05*0.25, -0.05*0.25, 'Q5', color = 'gray')
    plt.text(1.15*0.25, -0.05*0.25, 'Q6', color = 'gray')
    plt.text(-1.3*0.25, -1.3*0.25, 'Q7', color = 'gray')
    plt.text(-0.05*0.25, -1.3*0.25, 'Q8', color = 'gray')
    plt.text(1.15*0.25, -1.3*0.25, 'Q9', color = 'gray')
    #legend_add = []
    #zoom1 = ['Q1_1','Q3_1','Q3_2','Q6_2',]
    for i, v_name in enumerate(list_videos):
        #if v_name in zoom1:
            #continue 
        clr = clr_select(v_name)
        ax.scatter(valence_mean[i], arousal_mean[i],c = clr_select(v_name))
        ax.annotate(v_name, (valence_mean[i]+0.003, arousal_mean[i]-0.002), fontsize=8)
        #if clr not in legend_add:
            #ax.scatter(valence_mean[i], arousal_mean[i], c = clr, label=v_name.split('_')[0])
            #legend_add.append(clr)
        #else:
            #ax.scatter(valence_mean[i], arousal_mean[i], c = clr)
    #plt.xlim(-0.5,0.5)
    #plt.ylim(-0.5,0.5)
    #plt.xlim(0.15,0.45)
    #plt.ylim(0.1,0.3)
    #ax.grid(which = "major", color = 'lightgray', linestyle = 'dashed')
    #ax.xaxis.set_major_locator(FixedLocator([-2,-1, 0, 1, 2]))
    #ax.yaxis.set_major_locator(FixedLocator([-2,-1, 0, 1, 2]))
    #plt.title("Valence and Arousal mean")
    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    #ax.legend(loc='upper left')#, bbox_to_anchor=(0.5, -0.05))
    if has_prova:
        plt.show()
    else:
        plt.savefig(save_path / f"va_mean_2d.png")

