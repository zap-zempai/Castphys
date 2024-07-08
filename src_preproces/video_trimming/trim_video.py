from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from video_trimming.compute_mean import LockImage
from video_trimming.bbox import BBox, create_bbox

### FUNCIONS ------------------------------------------------------------------
def  xor (a: bool, b: bool) -> bool:
    return not (a == b)

def trim_video(df_imgs: pd.DataFrame,
               lockimage: LockImage,
               output_filepath: Path):
    
    # Inicia i calcula variables
    df_imgs["distence"] = df_imgs.timestamp[1:].reset_index(drop=True) - df_imgs.timestamp
    list_light = np.array([-1]*len(df_imgs))
    
    ## For per fer un mostreig a la data i reduir els posibles frames
    # Primer element
    t_acum = df_imgs.distence.iloc[0]
    list_light[0] = lockimage.has_light(0)
    # Elements del mig
    for frame, t in tqdm(zip(df_imgs.frame.iloc[1:-1],df_imgs.distence.iloc[1:-1])):
        t_acum += t
        if t_acum > 0.2: # temps de duracio de la llum
            list_light[frame] = lockimage.has_light(frame)
            t_acum = 0
    # Ultim element
    list_light[-1] = lockimage.has_light(len(list_light)-1)

    # Reduir els posibles frames
    df_imgs["state"] = list_light
    df_imgs_range = df_imgs[df_imgs["state"] >= 0].reset_index(drop=True)

    ## For per determinar els rangs on esta la llum
    list_range = []
    for i in range(len(df_imgs_range)-1):
        if xor(df_imgs_range.state.iloc[i],df_imgs_range.state.iloc[i+1]):
            list_range.append([df_imgs_range.frame.iloc[i],df_imgs_range.frame.iloc[i+1]])
    
    ## For per trobar el frame exacte
    list_frame = []
    for i,[start,end] in tqdm(enumerate(list_range)):
        list_frame.append((lockimage.probe_zone(start, end, df_imgs.state.iloc[start]),df_imgs.state.iloc[end]))

    ## Prosses final
    # Create trim_video
    list_frame = np.array(list_frame)
    trim_video = df_imgs.iloc[list_frame[:,0]].reset_index(drop=True)
    trim_video["state"] = ["ON" if i else "OFF" for i in list_frame[:,1]]
    trim_video = trim_video.drop(["distence"], axis=1)
    # Remove timestamp and Saving new csv
    trim_video.drop(["timestamp"], axis=1).to_csv(output_filepath, header=False, index=False)
    return trim_video

def create_trim_video(df_process: pd.DataFrame,
                      path_imgs: Path,
                      id: int,
                      path_info: Path,
                      path_trim_video: Path,
                      threshold = 200):
    # Init BBox
    bbox = create_bbox(id, 
                       path_info, 
                       "id_bbox.csv", 
                       "marca.png", 
                       path_imgs / "0.png", 
                       look_bbox=True)
    # Init LockImage
    lockimage = LockImage(bbox,threshold,path_imgs)
    # Create & return trim_video
    return trim_video(df_process, lockimage, path_trim_video)
