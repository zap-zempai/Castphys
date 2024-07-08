
from glob import glob
from pathlib import Path
import numpy as np
import pandas as pd

import sys

sys.path.append(".")

from src1.psychopy_extractor.demographic_record import EHTNICITY_MAPPING, GENDER_MAPPING, DemographicRecord, DemographicRecordSaver
from src1.psychopy_extractor.annotation_record import AnnotationRecord, AnnotationRecordSaver, Quadrant

def parse_demographic_data(df: pd.DataFrame) -> DemographicRecord:
    subject_id = int(df.iloc[0]["participant"])
    age = int(df.iloc[0]["ageVar.routineEndVal"])
    gender = GENDER_MAPPING[df.iloc[1]["genderVar.routineEndVal"]]
    try:
        ethnicity_str = df.iloc[2]["ethnicVar.routineEndVal"]
        ethnicity = EHTNICITY_MAPPING[ethnicity_str]
    except KeyError:
        raise Exception(f"Ethnicity [{ethnicity_str}] not supported")
        
    is_tca = df.iloc[2]["tcaVar.routineEndVal"] == "yes"

    return DemographicRecord(
        subject_id,
        age,
        gender,
        ethnicity,
        is_tca)
    


def parse(filename: Path):
    df = pd.read_csv(filename)
    
    demographic_record = parse_demographic_data(df)

    # Find skip video
    skip_video = any(df["skipButton.numClicks"] > 0)

    if skip_video:
        print(f"The subject {demographic_record.subject_id} has skip at least one video")    
    
    annotations = []
    for i, row in df.iterrows():
        video_name = row["videoName"]
        expected_quadrant = row["quadrant"]

        if not isinstance(video_name, str) and np.isnan(video_name):
            continue
        
        if expected_quadrant == "test":
            continue

        arousal = row["arousalSlider1.response"]
        skip_video = row["skipButton.numClicks"] > 0
        valence = row["valenceSlider1.response"]

        annotation_record = AnnotationRecord(
            video_name,
            skip_video,
            arousal,
            valence,
            getattr(Quadrant, expected_quadrant.upper())
        )

        annotations.append(annotation_record)
    
    return annotations, demographic_record


def process_psychopy_data(psychopy_filename: Path,
                          output_annotations_filename: Path,
                          output_demographic_filename: Path):
    annotations, demographic_record = parse(psychopy_filename)

    AnnotationRecordSaver.save(annotations,
                               output_annotations_filename)

    DemographicRecordSaver.save(demographic_record,
                                output_demographic_filename)

def create_sub_folder(id,
                      psychopy_folder = Path("D:/TFG_ALEX/raw_data/psychopy_annotations/data"),
                      destiny_folder =  Path("D:/TFG_ALEX/cleaned_data")):
    # Busca el .csv amb la id
    contenido = glob(f"{id}_elicitation_*.csv", root_dir=psychopy_folder)
    #print(contenido)
    if len(contenido) == 1:
        # Nom dels archius
        psychopy_filename = psychopy_folder / contenido[0]
        sub_folder = destiny_folder / f"sub_{id}"
        # crea la carpeta
        sub_folder.mkdir(parents=True,exist_ok=True)
        # crea i guarda els fitchers
        process_psychopy_data(psychopy_filename, sub_folder/"annotations.csv", sub_folder/"demographic.csv")
    else:
        raise Exception(f"Error: Len of id {id}: {len(contenido)}/1")
    
def create_all_sub_folder(psychopy_folder = Path("raw_data/psychopy_annotations/data"),
                          destiny_folder =  Path("cleaned_data")):
    psychopy_id = [x.split('_')[0] for x in glob("*.csv", root_dir=psychopy_folder)]
    destiny_id = [x.split('_')[1] for x in glob("sub_*", root_dir=destiny_folder)]
    #print(psychopy_id)
    #print(destiny_id)
    no_cleaned = [x for x in psychopy_id if x not in destiny_id]
    #print(no_cleaned)
    if len(no_cleaned) == 0:
        print("All subject cleaned")
    else:
        for i,id in enumerate(no_cleaned):
            print(f"Cleaning subject {id} [{i+1}/{len(no_cleaned)}]")
            create_sub_folder(id,
                psychopy_folder = psychopy_folder,
                destiny_folder = destiny_folder)

def main():
    create_all_sub_folder(psychopy_folder = Path("D:/TFG_ALEX/raw_data/psychopy_annotations/data"),
                          destiny_folder =  Path("D:/TFG_ALEX/cleaned_data"))

if __name__ == "__main__":
    main()
