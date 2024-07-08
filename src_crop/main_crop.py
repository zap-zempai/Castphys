### ----- Imports ------
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from glob import glob
import os

from landmarks.landmarks_extraction import load_images_bgr, Video_preprocessing


### ----- Code ------
def comput_area(folder_patient, save_folder):
    ids = [int(patient.split('_')[1]) for patient in glob("Patient_*", root_dir=folder_patient)]
    crop_create = [p.split('.')[0].split('_')[-1] for p in os.listdir(save_folder) if 'crop_lm_' in p]
    #for id in crop_create:
    #    ids.remove(f"Patient_{id}")
    ids.sort()
    img_size = (492, 658)  ## tamany de la imatge
    crop_size = (329, 246) ## no s'utilitza
    options = {}
    preprocessing = Video_preprocessing(options)

    ## for patient
    for id in tqdm(ids):
        result_rs = []
        result_pu = []
        result_pd = []
        result_dd = []
        Patient_path = folder_patient / f"Patient_{id}"
        list_videos = glob("Q*",root_dir=Patient_path) + glob("E*",root_dir=Patient_path)

        ## for video
        for v in list_videos:
            img_path = Patient_path / v / "imgs"
            ## landmarks
            video = load_images_bgr(img_path, crop_size)
            pred_landmarks = preprocessing.landmarks_extraction(video)
            ## area
            result_rs += [(v,np.max(np.max(pred_landmarks,axis=1)-np.min(pred_landmarks,axis=1),axis=0))]
            result_pu += [(v,np.min(np.min(pred_landmarks,axis=1),axis=0))]
            result_pd += [(v,np.max(np.max(pred_landmarks,axis=1),axis=0))]
            result_dd += [(v,img_size - np.max(np.max(pred_landmarks,axis=1),axis=0))]

        ## dataframe
        df_result = pd.DataFrame({"r_size":dict(result_rs),
                                  "p_up":dict(result_pu),
                                  "p_down":dict(result_pd),
                                  "d_down":dict(result_dd)})
        df_result.to_csv(save_folder / f"crop_lm_{id}.csv")

def main():
    folder_castphys = Path("E:/castphys_60")
    save_folder = Path("E:/crop")
    comput_area(folder_castphys, save_folder)

if __name__ == "__main__":
    main()