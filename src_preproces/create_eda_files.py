## crea els arxiu de input de ledalab per podeer fer proves

## ---- imports ----
import numpy as np
from pathlib import Path
import pandas as pd
from glob import glob
from datetime import datetime
import scipy.io as sio

from biosignalsplux.biosignalsProcessor import open_signals_npy

## ---- funcions ----
def create_df2npy_time(df_path, restar_time = False):
    df = open_signals_npy(df_path)
    df['timestamp'] = df.time.apply(lambda y : y.timestamp())
    if restar_time:
        df['timestamp'] = df['timestamp'] - df['timestamp'][0]
    return df

def open_df_time(df_path, restar_time = False):
    df = pd.read_csv(df_path)
    df.time = df.time.apply(lambda y : datetime.fromisoformat(y))
    df['timestamp'] = df.time.apply(lambda y : y.timestamp())
    if restar_time:
        df['timestamp'] = df['timestamp'] - df['timestamp'][0]
    return df

## ---- code ----
def create_all_eda_files(folder_old,folder_castphys,save_folder):
    # lista de videos
    list_videos = pd.read_csv(save_folder / "video_time.csv")['name'].values

    # crear todas las carpetas
    for v_name in list_videos:
        (save_folder / v_name).mkdir(parents=True,exist_ok=True)

    list_sub = [int(patient.split('_')[1]) for patient in glob("Patient_*", root_dir=folder_old)]
    for num_sub in list_sub:
        # buscar id
        with open(folder_old/ f"Patient_{num_sub}" / f"meta_{num_sub}.txt") as meta:
            id = int(meta.readline().split(',')[1][:-1])
        Patient_path = folder_castphys/ f"Patient_{id}"

        # abrir bio senyals
        df_bio = create_df2npy_time(folder_old/ f"Patient_{num_sub}", restar_time = False)

        # para todos los videos
        for v_name in list_videos:
            # encontrar el inicio del video
            event_list = [0]*len(df_bio)
            start_video = open_df_time(Patient_path / v_name / 'bio.csv', restar_time=False)['timestamp'][0]
            event_list[df_bio[df_bio.timestamp == start_video].index[0]] = 1

            # guardar file
            f = open(save_folder / v_name /f"{id}.txt", "w")
            for d,t,e in zip(df_bio['eda'],df_bio['timestamp'],event_list):
                f.write("{:.3f}".format(t)+f" {d} {e}\n")
            f.close()

def main():
    folder_old = Path("D:/castphys_raw/recordings")
    folder_castphys = Path("E:/castphys_60")
    save_folder = Path("E:/EDA")

    create_all_eda_files(folder_old,folder_castphys,save_folder)

if __name__ == "__main__":
    main()