### ----- Imports ------
import numpy as np
from pathlib import Path
import pandas as pd
import os
import cv2

def Crop_Test(folder_castphys):
    ids = [int(patient.split('_')[1]) for patient in os.listdir(folder_castphys)]

    for id in ids:
        patient_folder = folder_castphys / f"Patient_{id}"

        list_videos = [v for v in os.listdir(patient_folder) if 'Q' in v]

        for v_name in list_videos:
            try:
                video = cv2.VideoCapture(str(patient_folder / v_name / "vid_crop.avi"))
            except:
                print("Patient",id,v_name+": No Video Crop")
                continue
            frames = []
            ret_val, frame = video.read()
            while ret_val:
                #frames.append(frame) # BGR
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # RGB
                ret_val, frame = video.read()
            
            df_bio = pd.read_csv(folder_castphys / f"Patient_{id}" / v_name / "bio.csv")

            if len(df_bio) == len(frames):
                print("\033[0;32m"+"Patient",id,v_name+": Correct"+'\033[0;m')
            else:
                print("\033[0;31m"+"Patient",id,v_name+": Incorrect"+'\033[0;m')

def main():
    folder_castphys = Path("E:/castphys_60")
    Crop_Test(folder_castphys)

if __name__ == "__main__":
    main()