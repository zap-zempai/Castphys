## crea el fitxer annotation_full

from pathlib import Path
import pandas as pd
from glob import glob
from datetime import datetime
from tqdm import tqdm

def create_save_annotation(Patient_path, has_save = False):
    ## lista de videos
    list_videos = glob("Q*",root_dir=Patient_path) + glob("E*",root_dir=Patient_path)
    ## crear a pandas
    df_annotation = pd.DataFrame()
    for v in list_videos:
        ## duracion
        listImages = pd.read_csv(Patient_path / v / "listImages.csv")
        duration = datetime.fromisoformat(listImages.time.values[-1])-datetime.fromisoformat(listImages.time.values[0])
        ## annotation
        df = pd.read_csv(Patient_path / v / "annotations.csv", header=None)
        df = pd.DataFrame(dict([(c,[v]) for [c,v] in df.values]+[("duration",duration.total_seconds())]))
        df_annotation = pd.concat([df_annotation,df])
    ## eliminar columnes
    df_annotation = df_annotation.reset_index(drop=True)
    df_annotation = df_annotation.drop(["video_name","skip_video","expected_quadrant"], axis=1)
    ## save
    if has_save:
        df_annotation.to_csv(Patient_path / "annotations_full.csv", index=False)
    
    return df_annotation

def create_all_annotation(folder_castphys, has_save=False):
    ## path
    patients = glob("Patient_*", root_dir=folder_castphys)
    ## for per todos los subjects
    for patient in tqdm(patients):
        Patient_path = folder_castphys / patient
        if not(has_save) or not((Patient_path/"annotations_full.csv").exists()):
            create_save_annotation(Patient_path, has_save)

def main():
    ## path
    has_save = True
    folder_castphys = Path("E:/castphys_60")
    ## annotation
    create_all_annotation(folder_castphys, has_save)


if __name__ == "__main__":
    main()