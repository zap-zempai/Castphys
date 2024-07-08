from pathlib import Path
import pandas as pd
from datetime import datetime
import glob
from typing import List
import numpy as np
import shutil
from tqdm import tqdm

from biosignalsplux.biosignalsProcessor import open_signals_npy
from video_trimming.preprocess import remove_full_path_from_list_imgs_csv
from video_trimming.trim_video import create_trim_video

### FUNCIONS ------------------------------------------------------------------
def repord (m0,m1,d0,d1):
    # m1 sempre es igual o mallor que m0 (el mateix amb d0 i d1)
    d0 += (d1-m1) if d1>m1 else 0
    d1 += (d0-m0) if d0>m0 else 0
    return (d0,d1)

def rep (m0, m1, diff):
    if m0+m1<diff:
        print("Error: Set index: Need reshape")
    d0 = diff//2
    d1 = diff-d0
    if m0 <= m1:
        d0,d1 = repord(m0,m1,d0,d1)
    else:
        d1,d0 = repord(m1,m0,d0,d1)
    return d0,d1

def open_df_time(df_path, colmuns_name):
    df = pd.read_csv(df_path,header=None,names=colmuns_name)
    df.time = df.time.apply(lambda y : datetime.fromisoformat(y))
    df['timestamp'] = df.time.apply(lambda y : y.timestamp())
    return df

### CLASS ---------------------------------------------------------------------
class Patient_path:
    def __init__(self, patient_number: int, patient_dir: Path) -> None:
        self.patient = patient_dir / f"Patient_{patient_number}"
        self.listImages = self.patient / f"listimages_{patient_number}.csv"
        self.trim_video = self.patient / "trim_video.csv"
        self.listImages_p = self.patient / f"listimages_{patient_number}_processed.csv"
        self.metadata = self.patient / f"meta_{patient_number}.txt"
        self.imgs = self.patient / "imgs"
    
    def get_subject_path(self, sub_number: int, sub_dir: Path):
        self.subject = sub_dir / f"sub_{sub_number}"
        self.annotations = self.subject / "annotations.csv"
        self.demographic = self.subject / "demographic.csv"
    
    def has_patient_correct(self):
        return self.metadata.exists() and self.imgs.exists() and self.listImages.exists()
    def has_subject_correct(self):
        return self.annotations.exists() and self.demographic.exists()
    def has_processed(self):
        return self.trim_video.exists() and self.listImages_p.exists()


class DataLoader:
    def __init__(self, patient_number: int, 
                 init_path=Path(),
                 path_info=Path("info"),
                 patint_folder="samples",
                 sub_folder="cleaned_data") -> None:
        # Obtenir el path del pacient
        patient_dir = init_path / patint_folder
        # guardar el numero del pacient i els path corresponents
        self.patient_number = patient_number
        self.path = Patient_path(patient_number, patient_dir)
        # Proba si la data es correcta
        if not self.path.has_patient_correct():
            raise Exception(f"Error: Patient_{patient_number} is not correct")
        # obtenir el numero del pacient
        with open(self.path.metadata) as meta:
            self.num_patient = int(meta.readline().split(',')[1][:-1])
        # obtenir els path del subjecte
        sub_dir = init_path / sub_folder
        self.path.get_subject_path(self.num_patient, sub_dir)
        # Proba si la data es correcta
        if not self.path.has_subject_correct():
            raise Exception(f"Error: sub_{self.num_patient} is not correct")
        ## Preprocess
        # Open: trim_videos, listimages, biosignal
        if not self.path.has_processed(): # sempre crea els fitxers
            print(f"Patient_{patient_number} is not Processed")
            self.open_listImages()
            # Create listImages_p
            self.listImages_p = remove_full_path_from_list_imgs_csv(self.listImages.copy(),
                                                                    self.path.listImages_p)
            # Create trim_video
            self.trim_video = create_trim_video(self.listImages_p.copy(),
                                                self.path.imgs,
                                                patient_number,
                                                path_info,
                                                self.path.trim_video,
                                                threshold = 200)
            print("End processed")
        else:
            print(f"Patient_{patient_number} is Processed")
            self.open_trim_video()
            self.open_listImages_p()
        self.open_biosignal()


    def open_listImages(self):
        self.listImages = open_df_time(self.path.listImages, ['frame','time'])
    def open_listImages_p(self):
        self.listImages_p = open_df_time(self.path.listImages_p, ['frame','time'])
    def open_trim_video(self):
        self.trim_video = open_df_time(self.path.trim_video, ['frame','time','state'])
    
    def open_biosignal(self):
        self.biosignal = open_signals_npy(self.path.patient)
        # crea la time stamp
        self.biosignal['timestamp'] = self.biosignal.time.apply(lambda y : y.timestamp())
    
    def open_annotations(self):
        self.annotations = pd.read_csv(self.path.annotations)
        #self.annotations.video_name = self.annotations.video_name.apply(lambda y : y.split('/')[-1])

    
    def process_trim_video(self):
        # separa en on/off
        trim_video_on = self.trim_video[self.trim_video.state == "ON"].reset_index(drop=True)
        trim_video_off = self.trim_video[self.trim_video.state == "OFF"].reset_index(drop=True)
        # trim video per canvi de llum
        trim_video = pd.concat([trim_video_on,trim_video_off],axis=1)
        trim_video = trim_video.drop(["state","time"], axis=1)
        # Elimina les 5 primeres columnes per quedar-te nomes els canvis de llum dels videos
        trim_video = trim_video.drop(range(5), axis=0).reset_index(drop=True)
        # divideix en principi i final del video
        trim_video_start = trim_video[trim_video.index%2 == 0].reset_index(drop=True)
        trim_video_end = trim_video[trim_video.index%2 == 1].reset_index(drop=True)
        # trim video per video
        self.trim_video = pd.concat([trim_video_start,trim_video_end],axis=1)
        self.trim_video.columns = ["f_on_s","t_on_s","f_off_s","t_off_s",
                                   "f_on_e","t_on_e","f_off_e","t_off_e"]
    
    def search_biosingnal_index(self, timestamp, up=True) -> int:
        biosignal_new = self.biosignal.timestamp - timestamp
        if up:
            t_up = biosignal_new[biosignal_new>=0].iloc[0]
            biosignal_new = biosignal_new[biosignal_new==t_up]
        else:
            t_down = biosignal_new[biosignal_new<=0].iloc[-1]
            biosignal_new = biosignal_new[biosignal_new==t_down]
        return biosignal_new.index[0]
    
    def video_index(self, video):
        ## Falta comprobar si es va fora dels marges i reescalar

        # diferencia entre nº de biosenyals - nº de frames
        diferencia = (video.bio_e-video.bio_s)-(video.f_on_e-video.f_off_s)
        # + : mes biosignals -> add images
        # 0 : igual -> Perfect 
        # - : mes imatges -> add biosignal
        if diferencia == 0:
            img_index = [int(video.f_off_s),int(video.f_on_e)]
            bio_index = [int(video.bio_s),int(video.bio_e)]
        elif diferencia > 0:
            d0,d1 = rep (video.f_off_s-video.f_on_s,
                         video.f_off_e-video.f_on_e,
                        diferencia)
            img_index = [int(video.f_off_s-d0),int(video.f_on_e+d1)]
            bio_index = [int(video.bio_s),int(video.bio_e)]
        else:
            d0,d1 = rep (video.bio_s-self.search_biosingnal_index(video.t_on_s,True),
                        self.search_biosingnal_index(video.t_off_e,False)-video.bio_e,
                        -diferencia)
            img_index = [int(video.f_off_s),int(video.f_on_e)]
            bio_index = [int(video.bio_s-d0),int(video.bio_e+d1)]
        
        return [img_index,bio_index]
    
    def videos_index(self):
        # Busca els index minims de biosenyal
        self.trim_video['bio_s'] = self.trim_video.t_off_s.apply(lambda y : self.search_biosingnal_index(y,True))
        self.trim_video['bio_e'] = self.trim_video.t_on_e.apply(lambda y : self.search_biosingnal_index(y,False)) 

        # quadra els index de les imatges i les biosenyals
        self.index_videos = [self.video_index(self.trim_video.iloc[i]) for i in self.trim_video.index]
    
    def get_video_id(self, path_info, path_id = "video_elicization_id.csv"):
        self.open_annotations()
        video_id = pd.read_csv(path_info / path_id)
        self.annotations["id"] = self.annotations.video_name.apply(lambda y : video_id.id[video_id.name == y].values[0])

    def convert_data(self, destini_path: Path):
        # comprova que els videos quadrin
        if len(self.index_videos) != len(self.annotations):
            raise Exception(f"Error: different number of videos {len(self.index_videos)}/{len(self.annotations)}")
        print("Start of data transfer")
        # Crea la carpeta desti
        destini_folder = destini_path / f"Patient_{self.num_patient}"
        destini_folder.mkdir(parents=True,exist_ok=True)
        # Mou els arxius que no s'alterar
        shutil.copy(self.path.metadata, destini_folder / f"meta_{self.num_patient}.txt")
        shutil.copy(self.path.demographic, destini_folder / "demographic.csv")
        # Crea cada carpeta de video
        n = len(self.annotations)
        for i,id in enumerate(self.annotations.id):
            video_folder = destini_folder / f"{id}"
            video_folder.mkdir(parents=True,exist_ok=True)
            # save csv
            self.annotations.iloc[i].to_csv(video_folder / "annotations.csv", header=False)
            video_images = self.listImages_p.iloc[self.index_videos[i][0][0]:self.index_videos[i][0][1]].drop(["timestamp"], axis=1)
            video_biosignal = self.biosignal.iloc[self.index_videos[i][1][0]:self.index_videos[i][1][1]].drop(["timestamp"], axis=1)
            video_images.to_csv(video_folder / f"listImages.csv", index=False)
            video_biosignal.to_csv(video_folder / f"bio.csv", index=False)
            # Mou i canvia el nom de les imatges
            imgs_folder = video_folder / "imgs"
            imgs_folder.mkdir(parents=True,exist_ok=True)
            n_padding = len(str(len(video_images)))
            print(f"---> Folder {i+1}/{n} : {id} | Images : {len(video_images)}")
            for frame_new, frame_old in tqdm(enumerate(video_images.frame)):
                #shutil.copy(self.path.imgs / f"{frame_old}.png", imgs_folder / f"{frame_new}.png".zfill(n_padding+4))
                continue