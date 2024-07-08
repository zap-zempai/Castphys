from pathlib import Path
import matplotlib
from matplotlib import pyplot as plt
import cv2
import pandas as pd
from datetime import datetime
from glob import glob
import numpy as np

matplotlib.use('TkAgg')

def open_df_time(df_path):
    df = pd.read_csv(df_path)
    df.time = df.time.apply(lambda y : datetime.fromisoformat(y))
    df['timestamp'] = df.time.apply(lambda y : y.timestamp()) 
    df['timestamp'] = df['timestamp'] - df['timestamp'].values[0]
    return df

id = 19
v_name = 'Q9_2'

folder_castphys = Path("E:/castphys_60")
sub_forlder = folder_castphys / f"Patient_{id}" / v_name

if False:
    frame = 927

    img = cv2.cvtColor(cv2.imread(str(sub_forlder / 'imgs' / f"0{frame}.png")), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

    video = cv2.VideoCapture(str(sub_forlder / "vid_crop.avi"))
    ret_val, img = video.read()
    count = 0
    while ret_val:
        if count == frame:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            break
        
        ret_val, img = video.read()
        count +=1

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show() 

if False:
    df = open_df_time(sub_forlder / "bio.csv")

    ## BVP
    fig, ax = plt.subplots()
    fig.set_size_inches(10,4)
    ax.plot(df['timestamp'], df['ppg'])
    ax.set_ylabel("BVP")
    ax.set_xlabel('Temps [s]')
    plt.show()

    ## EDA
    fig, ax = plt.subplots()
    fig.set_size_inches(10,4)
    ax.plot(df['timestamp'], df['eda'])
    ax.set_ylabel("EDA")
    ax.set_xlabel('Temps [s]')
    plt.show()

    ## BR
    fig, ax = plt.subplots()
    fig.set_size_inches(10,4)
    ax.plot(df['timestamp'], df['breath'])
    ax.set_ylabel("BR")
    ax.set_xlabel('Temps [s]')
    plt.show()

if False:
    df = pd.read_csv(sub_forlder / "eda_analisis.csv")
    df_bio = pd.read_csv(sub_forlder / "bio.csv")
    df['eda'] = df_bio['eda'].values
    df['time'] = df.index/60

    fig, axs = plt.subplots(2)
    fig.set_size_inches(8,7)
    plt.subplots_adjust(hspace=0.35)

    ## up
    axs[0].plot(df['time'], df['eda'], label = 'EDA')
    axs[0].plot(df['time'], df['tonic'], label = 'Tonic')
    axs[0].set_title("SC Data")
    axs[0].set_ylabel('[µS]')
    axs[0].set_xlabel('Time [s]')
    axs[0].legend()
    ## down
    axs[1].plot(df['time'], df['driver'], label = 'Phasic', color='limegreen')
    axs[1].plot(df['time'], [0]*len(df), color='red')
    axs[1].set_title("Phasic Driver")
    axs[1].set_ylabel('[µS]')
    axs[1].set_xlabel('Time [s]')
    axs[1].legend()

    plt.show()

if True:
    ## Obtain all folders
    name_file = "demographic.csv"
    subs_folder = glob("Patient_*", root_dir=folder_castphys)
    #print(subs_folder)

    ## Read and Concat all info
    subject_df = pd.read_csv(folder_castphys / subs_folder[0] / name_file)
    for sub in subs_folder[1:]:
        if sub == "Patient_5":
            continue
        #print(sub)
        subject_df2 = pd.read_csv(folder_castphys / sub / name_file)
        subject_df = pd.concat([subject_df, subject_df2], ignore_index = True)

    ## Finals settings
    subject_df = subject_df.sort_values(by = "subject_id")
    subject_df = subject_df.reset_index(drop = True)
    #print(subject_df)

    ## Obtain Histogram
    age = np.arange(subject_df["age"].min(), subject_df["age"].max()+1)
    count_a = [len(subject_df[subject_df["age"] == x]) for x in age]
    print("Mean:",np.mean(subject_df["age"].values),"| std:", np.std(subject_df["age"].values))

    ## Plot Histogram
    fig, ax = plt.subplots()
    ax.bar(age, count_a, align='center')
    plt.xticks(age)
    ax.yaxis.grid(True, which='major')
    plt.ylabel('Número')
    plt.xlabel('Edat')
    plt.title("Histograma d'Edat")
    plt.show()

    ## Obtain grafic
    gender = subject_df["gender"].unique()
    count_g = [len(subject_df[subject_df["gender"] == x]) for x in gender]
    for i,g in enumerate(gender):
        if g=="MALE":
            gender[i]="Home"
        elif g=="FEMALE":
            gender[i]="Dona"
    clr_g = ['orange','dodgerblue','green','grey']

    ## Plot grafic
    fig, ax = plt.subplots()
    ax.bar(gender, count_g, color=clr_g)
    ax.set_yticks([])
    for i,n in enumerate(count_g):
        ax.text( i , n, n, ha='center', va='bottom')
    plt.title("Distribució de Gènere")
    plt.show()

    ## Obtain grafic
    ethnicity = subject_df["ethnicity"].unique()
    count_e = [len(subject_df[subject_df["ethnicity"] == x]) for x in ethnicity]

    ## Plot grafic
    fig, ax = plt.subplots()
    ax.pie(count_e, labels=ethnicity)
    ax.set_title("Ethnicity")
    plt.show()
    print(count_e, ethnicity)