## guarda els resultats de ledalab en un csv 

## ------ Import ------
from pathlib import Path
from glob import glob
import pandas as pd
import scipy.io as sio
from datetime import datetime
from tqdm import tqdm

from biosignalsplux.biosignalsProcessor import open_signals_npy

## ------ Code ------
def open_df_time(df_path):
    df = pd.read_csv(df_path)
    df.time = df.time.apply(lambda y : datetime.fromisoformat(y))
    df['timestamp'] = df.time.apply(lambda y : y.timestamp())
    return df

def obtain_first_time(df_path):
    df = open_signals_npy(df_path)
    df['timestamp'] = df.time.apply(lambda y : y.timestamp())
    return df['timestamp'].values[0]

def id2num_sub(id , folder):
    list_nums = [int(f.split('_')[1]) for f in glob("Patient_*", root_dir=folder)]
    for num_sub in list_nums:
        with open(folder/ f"Patient_{num_sub}" / f"meta_{num_sub}.txt") as meta:
            if id == int(meta.readline().split(',')[1][:-1]):
                return num_sub
    return -1

def obtain_params_mat(mat_loaded):
    mat_dict = {}

    mat_dict['nSCR']      = mat_loaded['results']['CDA'][0,0][0,0][0][0,0]
    mat_dict['Latency']   = mat_loaded['results']['CDA'][0,0][0,0][1][0,0]
    mat_dict['AmpSum']    = mat_loaded['results']['CDA'][0,0][0,0][2][0,0]
    mat_dict['PhasicMax'] = mat_loaded['results']['CDA'][0,0][0,0][5][0,0]
    mat_dict['Tonic']     = mat_loaded['results']['CDA'][0,0][0,0][6][0,0]

    return mat_dict


def eda_analisis_file(video_folder, mat_file, star_time):
    ## load data
    mat_loaded = sio.loadmat(mat_file)
    df_all = pd.DataFrame({'time':mat_loaded['data']['time'][0,0][0],
                           'tonic':mat_loaded['analysis']['tonicDriver'][0,0][0],
                           'driver':mat_loaded['analysis']['driver'][0,0][0]})
    
    ## select zone
    df_all['timestamp'] = df_all['time'] + star_time
    df_bio = open_df_time(video_folder / 'bio.csv')

    df_video = df_all[df_all.timestamp >= df_bio.timestamp[0]]
    df_video = df_video[df_video.timestamp <= df_bio.timestamp.values[-1]]

    ## save in file
    df_video = df_video.drop(["time","timestamp"], axis=1)
    df_video.to_csv(video_folder / "eda_analisis.csv", index=False)

def eda_event_dict(eda_folder, ids):
    ## valors basic
    dict_df_eda = {}

    list_videos = glob("Q*",root_dir=eda_folder)
    for v_name in list_videos:
        dict_all_mat = {'nSCR':[],'Latency':[],'AmpSum':[],'PhasicMax':[],'Tonic':[]}

        for id in ids:
            mat_loaded = sio.loadmat(eda_folder / v_name / f"{id}_era.mat")
            mat_dict = obtain_params_mat(mat_loaded)
            for e in mat_dict:
                dict_all_mat[e].append(mat_dict[e])

        dict_df_eda[v_name] = dict_all_mat
    dict_df_eda

    ## normalitzat
    max_dict = {'nSCR':[],'Latency':[],'AmpSum':[],'PhasicMax':[],'Tonic':[]}
    min_dict = {'nSCR':[],'Latency':[],'AmpSum':[],'PhasicMax':[],'Tonic':[]}

    v_name_0 = list_videos[0]
    for i,id in enumerate(ids):
        for e in max_dict:
            max_v = dict_df_eda[v_name_0][e][i]
            min_v = dict_df_eda[v_name_0][e][i]

            for v_name in list_videos[1:]:
                max_v = max(dict_df_eda[v_name][e][i],max_v)
                min_v = min(dict_df_eda[v_name][e][i],min_v)
            max_dict[e].append(max_v)
            min_dict[e].append(min_v)

    for i,id in enumerate(ids):
        for e in max_dict:
            for v_name in list_videos:
                dict_df_eda[v_name][e][i] = (dict_df_eda[v_name][e][i]-min_dict[e][i])/(max_dict[e][i]-min_dict[e][i])

    for v_name in list_videos:
        dict_df_eda[v_name] = pd.DataFrame(dict_df_eda[v_name])
    return dict_df_eda

def eda_event_file(dict_df_eda, video_folder, v_name, i):
    dict_event = {}
    for c in dict_df_eda[v_name].columns:
        dict_event[c] = [dict_df_eda[v_name][c][i]]
    df_event = pd.DataFrame(dict_event)
    df_event.to_csv(video_folder / "eda_event.csv", index=False)

def eda_estraction_files(folder_eda,folder_castphys,folder_old):
    ids = [int(f.split('_')[1]) for f in glob("Patient_*", root_dir=folder_castphys)]
    dict_df_eda = eda_event_dict(folder_eda, ids)

    for i,id in tqdm(enumerate(ids)):
        Patient_path = folder_castphys/ f"Patient_{id}"
        list_videos = glob("Q*",root_dir=Patient_path)# + glob("E*",root_dir=Patient_path)

        ## star time
        num_sub = id2num_sub(id, folder_old)
        star_time = obtain_first_time(folder_old/ f"Patient_{num_sub}")
        for v_name in list_videos:
            
            ## eda analisis
            eda_analisis_file(Patient_path / v_name, 
                              folder_eda / v_name / f"{id}.mat", 
                              star_time)
            ## eda event
            eda_event_file(dict_df_eda, Patient_path / v_name, v_name, i)

def main():
    folder_eda = Path("E:/EDA")
    folder_castphys = Path("E:/castphys_60")
    folder_old = Path("D:/castphys_raw/recordings")

    eda_estraction_files(folder_eda,folder_castphys,folder_old)

if __name__ == "__main__":
    main()