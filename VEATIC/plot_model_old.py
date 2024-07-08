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

def normalize(matrix):
    matrix = matrix/255  # normalized matrix
    return matrix

def load_video(video_path, img_size=-1):
    ## load video
    video = cv2.VideoCapture(video_path)
    frames = []
    ret_val, frame = video.read()
    while ret_val:
        if img_size != -1:
            frame =cv2.resize(frame, (img_size,img_size))
        #frames.append(frame) # BGR
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # RGB
        ret_val, frame = video.read()
    return np.array(frames)

def comput_va(path_model, frames, N=300):
    ## frames
    #frames = normalize(frames)
    frames = torch.from_numpy(frames.astype(np.float32)).permute(0,3,1,2)

    ## init model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    model = torch.load(path_model)
    model=model.to(device)
    model.eval()

    ## comput valence & arousal
    val_output = []
    with torch.no_grad():
        for i in tqdm(range(0,len(frames),N)):
            frame = frames[i:i+N].to(device)
            val_output.extend(model(frame)[:,-2:])    
    val_output = np.array(val_output)
    #print(val_output.shape)

    valence = val_output[:,0].reshape(-1)
    arousal = val_output[:,1].reshape(-1)
    #print(valence.shape, arousal.shape)
    return valence, arousal

def plot_va(frames, valence, arousal, anotation_path, has_video=False):
    ## eliminar plots antics
    if has_video:
        try:
            os.remove(f"E:/imgs/plot_{id}_{v_name}_HSE.mp4")
        except:
            print("Plot not exist")
        for file_name in os.listdir("E:/imgs/img_plot/"):
            os.remove(f"E:/imgs/img_plot/{file_name}")

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
    for i in range(frames.shape[0]):
        ## model plot
        ax_model = fig.add_subplot(axx_model[0,0])
        if i < 300:
            ax_model.set_xlim([0,600])
        else:
            ax_model.set_xlim([i-300,i+300])
        ax_model.plot(range(i+1),valence[:i+1],label='Valance')
        ax_model.plot(range(i+1),arousal[:i+1],label='Arousal')
        #ax_model.set_ylim([-2,2])
        ax_model.set_xlabel("frame")
        ax_model.set_ylabel("Valence & Arousal")
        ax_model.legend()

        ## video
        ax_video = fig.add_subplot(axx_video[0,0])
        im = frames[i,:,:,:]
        ax_video.imshow(np.uint8(im))
        ax_video.set_xticks([])
        ax_video.set_yticks([])
        ax_video.set_title("Video")

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
        plt.show()
        if has_video:
            plt.savefig(f"E:/imgs/img_plot/{i}.png")
        else:
            plt.pause(0.0000001)
        ax_model.cla()
        ax_video.cla()

        plt.clf()
    plt.close(fig)

    ## ----- Create Video -----
    if has_video:
        height, width  = cv2.imread("E:/imgs/img_plot/0.png").shape[:2]
        video = cv2.VideoWriter(f"E:/imgs/plot_{id}_{v_name}_HSE.mp4",cv2.VideoWriter_fourcc(*'mp4v'),60,(width,height))

        for i in tqdm(range(frames.shape[0])):
            img = cv2.imread(f"E:/imgs/img_plot/{i}.png")
            #img = cv2.resize(img, (width, height))
            video.write(img)
        video.release()

if __name__ == "__main__":
    ## model
    models_folder='C:/Users/Xavi/Desktop/HSEmotion/models/'
    model_name = 'enet_b0_8_va_mtl.pt'
    path_model = models_folder+model_name

    ## path
    id = 32
    v_name = 'Q3_2'
    patient_folder = f"E:/castphys_60/Patient_{id}/{v_name}/"
    video_path = patient_folder+"vid_crop.avi"
    anotation_path = patient_folder+"annotations.csv"
    has_video = True

    ## code
    frames = load_video(video_path, img_size=260)
    valence, arousal = comput_va(path_model, frames, N=300)
    #valence = valence*2
    #arousal = arousal*2
    print(len(valence),len(arousal))
    plot_va(frames, valence, arousal, anotation_path, has_video=has_video)
