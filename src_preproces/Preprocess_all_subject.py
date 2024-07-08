### --- Imports ---
from pathlib import Path
import pandas as pd
from glob import glob
import sys
sys.path.append(".")

from psychopy_extractor.parser import create_all_sub_folder
from dataloader.dataloader import DataLoader
from video_trimming.bbox import BBox, serch_bbox

### --- Init Path ---
raw_folder = Path("D:/CASTPHYS")
process_folder = Path("D:/CASTPHYS/Dataset_prueba")
path_info = raw_folder / "info"

### --- Convert 1 Subject ---
def Conver_Patient(patient_num):
    print(f"Start Patient_{patient_num}")
    # Create Data (patient_number: int, init_path=Path(), path_light="extra", patint_folder="samples",sub_folder="cleaned_data")
    data = DataLoader(patient_num, raw_folder, path_info)#, patint_folder="")
    # Trim Video Processed
    data.process_trim_video()
    # video index
    data.videos_index()
    # Crear fitcheros
    data.get_video_id(path_info)
    # Crear la nova data
    data.convert_data(process_folder)

### --- crate id to num ---
def create_id2num(drop_id=[]):
    # inicia path
    path_patients = raw_folder / "raw_data/recordings"
    # ids no processar
    id_no_process = drop_id + [int(x.split('_')[1]) for x in glob("Patient_*", root_dir=process_folder)]
    # buscar numeros
    patients_number = [int(x.split('_')[1]) for x in glob("Patient_*", root_dir=path_patients)]
    id2num = {}
    # trobar id
    for n in patients_number:
        try:
            with open(path_patients / f"Patient_{n}/meta_{n}.txt") as meta:
                id = int(meta.readline().split(',')[1][:-1])
            if id not in id_no_process:
                id2num[id] = n
        except:
            print(f"La carpeta Patient_{n} no conte el archiu meta_{n}.txt")

    return id2num

### --- serch all bbox ---
def serch_all_bbox(list_num):
    # path dels subjectes
    patient_dir = raw_folder# / ""
    # trobar els num que no tenen bbox
    df = pd.read_csv(path_info / "id_bbox.csv")
    for num in list_num:
        if num not in list(df.patient):
            # trobar llum i guardar la pos
            #serch_bbox(id:int, info_path, df_path, mark_path, path_img, look_bbox:bool=False) -> BBox:
            serch_bbox(num, 
                       path_info, 
                       "id_bbox.csv", 
                       "marca.png", 
                       patient_dir / f"Patient_{num}/imgs/0.png")


def main():
    # Create all sub dir that not exist
    create_all_sub_folder(
        psychopy_folder = raw_folder / "raw_data/psychopy_annotations/data",
        destiny_folder =  raw_folder / "cleaned_data"
    )

    # trobar id -> num 
    drop_id = [150,1,2,3,4]
    id2num = create_id2num(drop_id)

    # obtenir totes les bbox
    serch_all_bbox(id2num.values())

    # procesar tots els pacients
    for i,id in enumerate(id2num):
        print(f"Process subject {id} [{i+1}/{len(id2num)}]")
        Conver_Patient(patient_num = id2num[id])