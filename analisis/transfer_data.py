## ---- Imports ----
from pathlib import Path
import os
import shutil
from tqdm import tqdm

## ---- Flies ----
general_files = ["annotations_full.csv","demographic.csv","meta_*.txt"]
video_files = ["annotations.csv","bio.csv","listImages.csv","vid_crop.avi","eda_analisis.csv","eda_event.csv"]

## ---- Code ----
def copy_data(origin_folder,copy_folder):
    ids = [int(patient.split('_')[1]) for patient in os.listdir(origin_folder)]
    for i,id in enumerate(ids):
        print("Patient", id, '[',i+1,'/',len(ids),']')
        # crear la carpeta
        old_patient_folder = origin_folder / f"Patient_{id}"
        new_patient_folder = copy_folder / f"Patient_{id}"
        new_patient_folder.mkdir(parents=True,exist_ok=True)

        # mou arxius
        for file in general_files:
            if '*' in file:
                file = file.split('*')[0] + str(id) + file.split('*')[1]
            if not (old_patient_folder/file).exists():
                print("\033[0;31m"+"Not found this File:",old_patient_folder/file,"\033[0;m")
                continue
            if (new_patient_folder/file).exists():
                #print("File already exist")
                continue
            shutil.copy(old_patient_folder/file,new_patient_folder/file)

        list_videos = [v for v in os.listdir(old_patient_folder) if 'Q' in v]# + [v for v in os.listdir(old_patient_folder) if 'E' in v]
        for v_name in tqdm(list_videos):
            # crear la carpeta
            old_video_folder = old_patient_folder / v_name
            new_video_folder = new_patient_folder / v_name
            new_video_folder.mkdir(parents=True,exist_ok=True)

            # mou arxius
            for file in video_files:
                if not (old_video_folder/file).exists():
                    print("\033[0;31m"+"Not found this File:",old_video_folder/file,"\033[0;m")
                    continue
                if (new_video_folder/file).exists():
                    #print("File already exist")
                    continue
                shutil.copy(old_video_folder/file,new_video_folder/file)


def main():
    origin_folder = Path("F:/castphys_60")
    copy_folder = Path("D:/castphys_60/Data")

    copy_data(origin_folder,copy_folder)

if __name__ == "__main__":
    main()