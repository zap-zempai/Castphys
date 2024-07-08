## ----- Imports -----
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FixedLocator
from matplotlib.patches import Rectangle
import os

import torch
import timm
from torchvision import transforms

from landmarks.landmarks_extraction import load_images_bgr, Video_preprocessing

matplotlib.use('TkAgg')

## ----- Code -----
def quadrant_pos(quadrant):
    if quadrant == 'Q1':
        return (-2,2/3)
    elif quadrant == 'Q2':
        return (-2/3,2/3)
    elif quadrant == 'Q3':
        return (2/3,2/3)
    elif quadrant == 'Q4':
        return (-2,-2/3)
    elif quadrant == 'Q5':
        return (-2/3,-2/3)
    elif quadrant == 'Q6':
        return (2/3,-2/3)
    elif quadrant == 'Q7':
        return (-2,-2)
    elif quadrant == 'Q8':
        return (-2/3,-2)
    elif quadrant == 'Q9':
        return (2/3,-2)
    else:
        return(2,2)

def load_video(video_path):
    ## load video
    video = cv2.VideoCapture(video_path)
    frames = []
    ret_val, frame = video.read()
    while ret_val:
        #frames.append(frame) # BGR
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # RGB
        frames.append(img)
        ret_val, frame = video.read()
    frames = np.array(frames)
    print(frames.shape)
    return frames

def load_va(patient_folder, v_name, file_va, has_scale=False):
    ## load df
    video_folder = patient_folder + v_name
    try:
        df = pd.read_csv(video_folder+'/'+file_va)
    except:
        print("Error:", video_folder+'/'+file_va, "not exist")
        return

    if has_scale:
        ## rescale margin
        max_val = [-2,-2]
        min_val = [2,2]
        list_videos = [v for v in os.listdir(patient_folder) if 'Q' in v]

        for v_name2 in tqdm(list_videos):
            video_folder2 = patient_folder + v_name2
            try:
                df = pd.read_csv(video_folder2/file_va)
                max_val[0] = max(max_val[0], np.max(df['valence'].values))
                min_val[0] = min(min_val[0], np.min(df['valence'].values))
                max_val[1] = max(max_val[1], np.max(df['arousal'].values))
                min_val[1] = min(min_val[1], np.min(df['arousal'].values))
            except:
                continue
        ## rescale
        df['valence'] = 4*(df['valence']-min_val[0])/(max_val[0]-min_val[0])-2
        df['arousal'] = 4*(df['arousal']-min_val[0])/(max_val[0]-min_val[0])-2
    
    return df['valence'].values, df['arousal'].values, df['emotion'].values

def plot_va(frames, valence, arousal, emotion, anotation_path, save_path, has_video=False):
    ## eliminar plots antics
    if has_video:
        try:
            os.remove(save_path + f"plot_{id}_{v_name}_HSE.mp4")
        except:
            print("Plot not exist")
        for file_name in os.listdir(save_path + "img_plot/"):
            os.remove(save_path + f"img_plot/{file_name}")

    ## autoevaluacio
    df_annotations = pd.read_csv(anotation_path,header=None) 
    v_a_eval = [int(float(v)) for [c,v] in df_annotations.values if c in ['valence','arousal']]
    va_pos = v_a_eval
    if va_pos[0] == 2:
        va_pos[0] -= 0.1
    elif va_pos[0] == -2:
        va_pos[0] += 0.1
    if va_pos[1] == 2:
        va_pos[1] -= 0.1
    elif va_pos[1] == -2:
        va_pos[1] += 0.1

    ## plot
    fig = plt.figure(figsize=(15,7))
    a = gridspec.GridSpec(2,1, height_ratios=[2,1])

    axx_model = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=a[1])
    axx_video = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=a[0])

    plt.ion()
    for i in tqdm(range(frames.shape[0])):
        ## model plot
        ax_model = fig.add_subplot(axx_model[0,0])
        if i < 300:
            ax_model.set_xlim([0,600])
        else:
            ax_model.set_xlim([i-300,i+300])
        ax_model.plot(range(i+1),valence[:i+1],label='Valance')
        ax_model.plot(range(i+1),arousal[:i+1],label='Arousal')
        ax_model.set_ylim([-2,2])
        ax_model.set_xlabel("frame")
        ax_model.set_ylabel("Valence & Arousal")
        ax_model.legend()

        ## video
        ax_video = fig.add_subplot(axx_video[0,0])
        im = frames[i,:,:,:]
        ax_video.imshow(np.uint8(im))
        ax_video.set_xticks([])
        ax_video.set_yticks([])
        ax_video.set_title(f"Emotion: {emotion[i]}")

        ## quadrant Valence & Arousal
        ax_va = fig.add_subplot(axx_video[0,1])
        # estructura
        ax_va.set_xlabel('Valence')
        ax_va.set_ylabel('Arousal')
        ax_va.set_xlim([-2, 2])
        ax_va.set_ylim([-2, 2])
        ax_va.xaxis.set_major_locator(FixedLocator([-2,-1, 0, 1, 2]))
        ax_va.yaxis.set_major_locator(FixedLocator([-2,-1, 0, 1, 2]))
        ax_va.plot([2/3]*2, [2,-2], color = 'lightgray',linestyle = "--")
        ax_va.plot([-2/3]*2, [2,-2], color = 'lightgray',linestyle = "--")
        ax_va.plot([2,-2], [2/3]*2, color = 'lightgray',linestyle = "--")
        ax_va.plot([2,-2], [-2/3]*2, color = 'lightgray',linestyle = "--")
        # Quadrant video
        try:
            quadrant = v_name.split('_')[0]
        except:
            quadrant = '0'
        rect = Rectangle(quadrant_pos(quadrant),4/3,4/3,
                alpha=0.2,
                facecolor='seagreen',
                edgecolor='green')
        ax_va.add_patch(rect)
        # auto evaluacio
        ax_va.scatter(x = va_pos[0], y = va_pos[1], label='auto_eval')
        ax_va.scatter(x = valence[i], y = arousal[i], label='estimation')
        ax_va.legend()

        ## end plot
        if has_video:
            plt.savefig(save_path + f"img_plot/{i}.png")
        else:
            plt.show()
            plt.pause(0.0000001)
        ax_model.cla()
        ax_video.cla()

        plt.clf()
    plt.close(fig)

    ## ----- Create Video -----
    if has_video:
        height, width  = cv2.imread(save_path + "img_plot/0.png").shape[:2]
        video = cv2.VideoWriter(save_path + f"plot_{id}_{v_name}_HSE.mp4",cv2.VideoWriter_fourcc(*'mp4v'),60,(width,height))

        for i in tqdm(range(frames.shape[0])):
            img = cv2.imread(save_path + f"img_plot/{i}.png")
            #img = cv2.resize(img, (width, height))
            video.write(img)
        video.release()

if __name__ == "__main__":
    ## path
    id = 19
    v_name = 'Q1_2'
    patient_folder = f"D:/castphys_60/Data/Patient_{id}/"
    file_va = "FR_predic_HSE.csv"
    has_scale=False
    video_path = patient_folder+f"{v_name}/vid_crop.avi"
    anotation_path = patient_folder+f"{v_name}/annotations.csv"

    ## save video
    has_video = True
    save_path = "C:/Users/Alex/Desktop/video_saves/"

    ## code
    print("Strat")
    video = load_video(video_path)
    print("Video Load")
    valence, arousal, emotion = load_va(patient_folder, v_name, file_va, has_scale=has_scale)
    print("va load",len(valence),len(arousal))
    plot_va(video, valence, arousal, emotion, anotation_path, save_path, has_video=has_video)
    print("End")
