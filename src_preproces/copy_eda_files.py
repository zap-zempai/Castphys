## copia els arxiu de input de ledalab per podeer fer proves

## ---- imports ----
from pathlib import Path
import pandas as pd
from glob import glob
import shutil

## ---- code ----
def copy_all_eda_files(old_folder,new_folder):
    # lista de videos
    list_videos = pd.read_csv(old_folder / "video_time.csv")['name'].values
    # Mou els arxius que no s'alterar
    shutil.copy(old_folder / "video_time.csv", new_folder / "video_time.csv")

    # crear todas las carpetas
    for v_name in list_videos:
        (new_folder / v_name).mkdir(parents=True,exist_ok=True)

        list_files = glob("*.txt", root_dir=old_folder / v_name)
        for file in list_files:
            shutil.copy(old_folder / v_name / file, new_folder / v_name / file)

    

def main():
    old_folder = Path("E:/EDA")
    new_folder = Path("E:/EDA1")

    copy_all_eda_files(old_folder,new_folder)

if __name__ == "__main__":
    main()