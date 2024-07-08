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
has_scale = False
smooth = False

## ----- Code -----
if __name__ == "__main__":
    ## path
    folder_path = Path("D:/castphys_60/Data")
    save_path = Path("C:/Users/Alex/Desktop/video_saves")

    ## list elements
    ids = [int(patient.split('_')[1]) for patient in os.listdir(folder_path)]
    list_videos = [v for v in os.listdir(folder_path / f"Patient_{ids[0]}") if 'Q' in v]


    for v_name in tqdm(list_videos):
        count = 0
        valance = 0
        arousal = 0
        N_emo = 3
        list_emotion = [[]]*N_emo

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
            except:
                continue
            
            ## valence & arousal
            try:
                valance = valance[:len(df)] + df['valence'].values[:len(valance)]
                arousal = arousal[:len(df)] + df['arousal'].values[:len(arousal)]
            except:
                #print(id)
                valance = df['valence'].values
                arousal = df['arousal'].values
            ## discet emotion
            emotion_video = []
            N_segment = int(round(len(df)/N_emo))
            for i in range(N_emo):
                if i == N_emo-1:
                    N_segment = len(df) - i*N_segment

                emotions = list(df['emotion'][N_segment*i:N_segment*(i+1)].values)
                #print(type(emotions))
                emotions_dict = dict(zip(emotions,map(lambda x: emotions.count(x),emotions)))
                #if 'Neutral' in emotions_dict:
                    #emotions_dict['Neutral'] = 0
                    #for e,val in [(e, emotions_dict[e]) for e in emotions_dict if e != 'Neutral']:
                        #if val*2 > emotions_dict['Neutral']:
                            #emotions_dict['Neutral'] = 0
                            #break
                #print(np.argmax(list(emotions_dict.values())))
                emotion = list(emotions_dict.keys())[np.argmax(list(emotions_dict.values()))]
                #print(emotion)
                list_emotion[i].append(emotion)

            
        ## valence & arousal
        valance = valance/count
        arousal = arousal/count
        ## discet emotion
        for i in range(N_emo):
            emotions = list_emotion[i]
            emotions_dict = dict(zip(emotions,map(lambda x: emotions.count(x),emotions)))
            list_emotion[i] = list(emotions_dict.keys())[np.argmax(list(emotions_dict.values()))]
        #print(list_emotion)

        if smooth:
            N = 10
            valance = np.array([np.mean(valance[max(0,i-N+1):i+1]) for i in range(len(valance))])
            arousal = np.array([np.mean(arousal[max(0,i-N+1):i+1]) for i in range(len(arousal))])

        ## plot
        fig, axs = plt.subplots(2)
        plt.subplots_adjust(hspace=0.0)

        ## lineas divisories
        N_segment = int(round(len(valance)/N_emo))
        N_time = int(round(len(valance)/N_emo))/60
        for i in range(1,N_emo):
            axs[0].plot([N_time*i]*2, [-2,2], color = 'lightgray',linestyle = "--")
            axs[1].plot([N_time*i]*2, [-2,2], color = 'lightgray',linestyle = "--")
        axs[0].plot([10]*2, [10]*2, color='cornflowerblue',  label='València')
        axs[0].plot([10]*2, [10]*2, color='orange',  label='Activació')

        ## valence & arousal
        axs[0].plot(df['time'][:len(valance)], valance, color='cornflowerblue')
        axs[1].plot(df['time'][:len(arousal)], arousal, color='orange')
        ## discet emotion
        for i in range(N_emo):
            plt.text(N_time*i+N_time*0.1, np.max(arousal)+0.12, list_emotion[i])
        
        
        #fig.colorbar(line,ax=axs[0])
        axs[0].set_title(f"Video: {v_name}")
        axs[0].set_ylabel("València")
        axs[0].set_xticks([])
        #axs[0].set_xlabel("Temps [s]")
        axs[0].set_ylim([np.min(valance)-0.1,np.max(valance)+0.1])
        axs[0].legend()
        axs[1].set_ylabel("Activació")
        axs[1].set_xlabel("Temps [s]")
        axs[1].set_ylim([np.min(arousal)-0.1,np.max(arousal)+0.1])

        if has_prova:
            plt.show()
            break
        else:
            plt.savefig(save_path / "emotion_mean" / f"va_mean_{v_name}.png")

