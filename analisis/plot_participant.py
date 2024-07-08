## plot de les dades del participant
## input id i v_name

## ---- Import ----
import numpy as np
import pandas as pd
import cv2
import torch
from matplotlib import gridspec
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib.patches import Rectangle
import warnings
import os

from dataloader.evaluation.utils import load_video, quadrant_pos

warnings.filterwarnings('ignore')
torch.manual_seed(0)
matplotlib.use('TkAgg')

## ---- Code ----

def load_video(video_path):
    ## load video
    video = cv2.VideoCapture(video_path)
    all_frames = []
    ret_val, frame = video.read()
    while ret_val:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # RGB
        all_frames.append(img)
        ret_val, frame = video.read()
    video = np.array(all_frames)
    return video

def main(id, v_name):
    save_path = "C:/Users/Alex/Desktop/video_saves/"
    video_folder = f"D:/castphys_60/Data/Patient_{id}/{v_name}/"
    bio_file = video_folder + '/bio.csv'
    annotation_file = video_folder + '/annotations.csv'
    video_path = video_folder + '/vid_crop.avi'

    # --- Biosignal ---
    df_bio = pd.read_csv(bio_file)

    ppg_signal = df_bio['ppg'].values
    eda_signal = df_bio['eda'].values
    brt_signal = df_bio['breath'].values

    # --- video + name ---
    frames = load_video(video_path)

    # --- valence & arousal ---
    df_annotations = pd.read_csv(annotation_file,header=None)
    v_a_values = [int(float(v)) for [c,v] in df_annotations.values if c in ['valence','arousal']]
    v_a_eval = np.array(v_a_values)


    ## Normalize the signal
    ppg_signal = (ppg_signal-ppg_signal.min())/(ppg_signal.max()-ppg_signal.min())*2-1
    eda_signal = (eda_signal-eda_signal.min())/(eda_signal.max()-eda_signal.min())*2-1
    brt_signal = (brt_signal-brt_signal.min())/(brt_signal.max()-brt_signal.min())*2-1


    if True:
        ## eliminar plots antics
        try:
            os.remove(save_path + f"plot_{id}_{v_name}_basic.mp4")
        except:
            print("Plot not exist")
        for file_name in os.listdir(save_path + "img_plot/"):
            os.remove(save_path + f"img_plot/{file_name}")


        fig = plt.figure(figsize=(15,7))
        a = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
        axx_bio = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=a[1])
        axx_vid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=a[0])

        plt.ion()
        for i in range(frames.shape[0]):
            ax_ppg = fig.add_subplot(axx_bio[0, 0])
            ax_eda = fig.add_subplot(axx_bio[1, 0])
            ax_brt = fig.add_subplot(axx_bio[2, 0])
            if i < 450:
                ax_ppg.set_xlim([0, 600])
                ax_eda.set_xlim([0, 600])
                ax_brt.set_xlim([0, 600])
            else:
                ax_ppg.set_xlim([i - 450, i + 150])
                ax_eda.set_xlim([i - 450, i + 150])
                ax_brt.set_xlim([i - 450, i + 150])

            # PPG
            ax_ppg.plot(range(i+1), ppg_signal[:i+1])#, label='Groundtruth')
            ax_ppg.set_ylabel('BVP')
            #ax_ppg.legend()
            ax_ppg.set_ylim([-1.1, 1.1])
            # EDA
            ax_eda.plot(range(i+1), eda_signal[:i+1])
            ax_eda.set_ylabel('EDA')
            ax_eda.set_ylim([-1.1, 1.1])
            # Breath
            ax_brt.plot(range(i+1), brt_signal[:i+1])
            ax_brt.set_xlabel('frames')
            ax_brt.set_ylabel('BR')
            ax_brt.set_ylim([-1.1, 1.1])

            # Video
            ax_vid = fig.add_subplot(axx_vid[0, 0])
            im = frames[i, :, :, :]
            ax_vid.imshow(np.uint8(im))
            ax_vid.set_xticks([])
            ax_vid.set_yticks([])
            ax_vid.set_title(f'Vídeo: {v_name}')
            # Valance & arousal
            ax_va = fig.add_subplot(axx_vid[1, 0])
            if True:
                # estructura
                ax_va.set_xlabel('València')
                ax_va.set_ylabel('Activació')
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
                va_pos = v_a_eval
                if va_pos[0] == 2:
                    va_pos[0] -= 0.1
                elif va_pos[0] == -2:
                    va_pos[0] += 0.1
                if va_pos[1] == 2:
                    va_pos[1] -= 0.1
                elif va_pos[1] == -2:
                    va_pos[1] += 0.1
                ax_va.scatter(x = va_pos[0], y = va_pos[1], label='auto_eval')
                ax_va.legend()

            plt.show()
            plt.pause(0.0000000000001)
            plt.savefig(save_path + f"img_plot/{i}.png")
            ax_vid.cla()
            ax_va.cla()
            ax_ppg.cla()
            ax_eda.cla()
            ax_brt.cla()
            plt.clf()
        plt.close(fig)

        ## ----- Create Video -----
        height, width  = cv2.imread(save_path + "img_plot/0.png").shape[:2]
        video = cv2.VideoWriter(save_path + f"plot_{id}_{v_name}_basic.mp4",cv2.VideoWriter_fourcc(*'mp4v'),60,(width,height))

        for i in range(frames.shape[0]):
            img = cv2.imread(save_path + f"img_plot/{i}.png")
            video.write(img)
        video.release()


if __name__ == '__main__':
    id = 19
    v_name = 'Q1_2'

    main(id, v_name)