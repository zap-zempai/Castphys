### ----- Imports ------
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from glob import glob
import os
import cv2

from landmarks.landmarks_extraction import load_images_bgr, Video_preprocessing

### ----- Code ------
### Static Crop
## Gloabal
img_size = (658, 492)
fps = 60

### ----- Function -----
def search_margin_static(center,crop_margin):
    p_up = center-crop_margin
    p_down = center+crop_margin

    # set margin
    if p_up[0] < 0:
        p_down[0] -= p_up[0]
        p_up[0] = 0
    if p_up[1] < 0:
        p_down[1] -= p_up[1]
        p_up[1] = 0 
    if p_down[0] > img_size[0]:
        p_up[0] -= p_down[0] - img_size[0]
        p_down[0] = img_size[0]
    if p_down[1] > img_size[1]:
        p_up[1] -= p_down[1] - img_size[1]
        p_down[1] = img_size[1]

    return p_up,p_down

def create_crop(folder_castphys,crop_size):
    crop_margin = np.array(crop_size)//2
    options = {}
    preprocessing = Video_preprocessing(options)

    ids = [int(patient.split('_')[1]) for patient in glob("Patient_*", root_dir=folder_castphys)]
    for i,id in enumerate(ids):
        print("Patient",id,'[',i+1,'/',len(ids),']')
        Patient_path = folder_castphys/ f"Patient_{id}"
        list_videos = glob("Q*",root_dir=Patient_path)# + glob("E*",root_dir=Patient_path)
        for v_name in tqdm(list_videos):
            img_path = Patient_path / v_name / "imgs"

            ## load video and landmarks
            video = load_images_bgr(img_path, [])
            pred_landmarks = preprocessing.landmarks_extraction(video)

            ## Crop de video
            center = np.min(np.min(pred_landmarks,axis=0),axis=0) + (np.max(np.max(pred_landmarks,axis=0),axis=0)-np.min(np.min(pred_landmarks,axis=0),axis=0))//2
            p_up,p_down = search_margin_static(center,crop_margin)

            video_crop = [frm[p_up[1]:p_down[1],p_up[0]:p_down[0],:] for frm in video]

            ## Save crop
            #destini_folder = Patient_path / v_name / "imgs_crop"
            #destini_folder.mkdir(parents=True,exist_ok=True)
            #n_padding = len(str(len(video_crop)))
            #for name_frame, crop_frame in enumerate(video_crop):
            #    cv2.imwrite(str(destini_folder / f"{name_frame}.png".zfill(n_padding+4)),crop_frame)
            ## Save Video
            video_save = cv2.VideoWriter(str(Patient_path / v_name / f"vid_crop.avi"),
                             cv2.VideoWriter_fourcc(*'FFV1'),fps,crop_size)
            for frame in video_crop:
                video_save.write(frame)
            video_save.release()

def main():
    folder_castphys = Path("D:/TFG_ALEX/castphys_60")
    crop_size = (450, 350)

    create_crop(folder_castphys,crop_size)


if __name__ == "__main__":
    main()
