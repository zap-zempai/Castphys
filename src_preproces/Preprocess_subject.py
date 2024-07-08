from pathlib import Path
import pandas as pd
import sys
sys.path.append(".")

from psychopy_extractor.parser import create_all_sub_folder
from dataloader.dataloader import DataLoader

def Conver_Patient(patient_num):
    print(f"Start Patient_{patient_num}")
    # inicia path
    path_init = Path("D:/castphys_raw")
    path_desti = Path("C:/Users/Xavi/Desktop/Dataset_prueba")
    path_info = path_init / "info"
    # Create all sub dir that not exist
    create_all_sub_folder(
        psychopy_folder = path_init / "psychopy_annotations/data",
        destiny_folder =  path_init / "cleaned_data"
    )
    # Create Data (patient_number: int, init_path=Path(), path_light="extra", patint_folder="samples",sub_folder="cleaned_data")
    data = DataLoader(patient_num, path_init, path_info, patint_folder="recordings")
    # Trim Video Processed
    data.process_trim_video()
    # video index
    data.videos_index()
    # Crear fitcheros
    data.get_video_id(path_info)
    # Crear la nova data
    data.convert_data(path_desti)

def main():
    patient_num = 171
    Conver_Patient(patient_num)

if __name__ == "__main__":
    main()
