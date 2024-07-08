## ------ Imports ------
import numpy as np
import cv2
import torch
from torchvision import transforms
import timm
from tqdm import tqdm
import os
from pathlib import Path
import pandas as pd

from landmarks.landmarks_extraction import load_images_bgr, Video_preprocessing

## ------ Loader Video ------
IMG_SIZE = 224
train_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ]
)

class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, video_path, transform, preprocessing_options):
        self.preprocessing = Video_preprocessing(preprocessing_options)
        self.transform = transform

        video = cv2.VideoCapture(video_path)
        frames = []
        ret_val, frame = video.read()
        while ret_val:
            #frames.append(frame) # BGR
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # RGB
            ## landmarks
            try:
                pred_landmarks = self.preprocessing.landmarks_extraction(img)
                max_p = np.max(pred_landmarks,axis=0)
                min_p = np.min(pred_landmarks,axis=0)
            except:
                pass
            img_crop = img[min_p[1]:max_p[1],min_p[0]:max_p[0],:]
            if False:
                fig, ax = plt.subplots()
                ax.imshow(np.uint8(img_crop))
                ax.set_xticks([])
                ax.set_yticks([])
                plt.show()
                break
            frames.append(cv2.resize(img_crop, (IMG_SIZE,IMG_SIZE))) 
            ret_val, frame = video.read()
        frames = np.array(frames)
        frames = torch.from_numpy(frames.astype(np.float32)).permute(0,3,1,2)
        #print(frames.shape)
        self.imgs = frames

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        #dealing with the image

        img_data=[]
        for id in range(0,self.imgs.shape[0]):
            #print(type(self.imgs[id,:,:,:].numpy()))
            img_data.append(self.transform(self.imgs[id,:,:,:]))

        img_torch = torch.stack(img_data, dim=0)
        #print(img_torch.shape) 
        
        return img_torch

def load_video(video_path):
    ## load video
    test_dataset = MultiTaskDataset(video_path, transform=train_transforms, preprocessing_options={})
    frames = test_dataset[0]
    #print(frames.shape)
    return frames

## ------ Model ------
class ModelHSE:
    def __init__(self, path_model, N=300) -> None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device:", device)
        model = torch.load(path_model)
        model=model.to(device)
        model.eval() # el model nomes es per evaluar

        self.model = model
        self.device = device
        self.N = N

        ## emotion class
        self.idx2class={0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness', 7: 'Surprise'}
        self.num_classes=len(self.idx2class)

    def comput_va(self, frames):
        ## comput valence & arousal
        val_output = []
        emotion_output = []
        
        with torch.no_grad():
            for i in range(0,len(frames),self.N):
                frame = frames[i:i+self.N].to(self.device)
                model_output = self.model(frame).detach().cpu().numpy()
                val_output.extend(model_output[:,-2:]) # valece & arousal
                emotion_output.extend([self.idx2class[i] for i in np.argmax(model_output[:,:self.num_classes], axis=1)]) # emotion class
        val_output = np.array(val_output)
        #print(val_output.shape)

        valence = val_output[:,0].reshape(-1)
        arousal = val_output[:,1].reshape(-1)
        #print(valence.shape, arousal.shape)
        return valence, arousal, emotion_output

## ------ Main ------
def predict_all_subject(folder_path, model_path):
    ## load model
    model = ModelHSE(model_path, N=300)

    # for all patient
    ids = [int(patient.split('_')[1]) for patient in os.listdir(folder_path)]
    for i,id in enumerate(ids):
        #print("Patient", id, '[',i+1,'/',len(ids),']')
        patient_folder = folder_path / f"Patient_{id}"

        # for all videos
        list_videos = [v for v in os.listdir(patient_folder) if 'Q' in v]
        for v_name in list_videos:
            video_folder = patient_folder / v_name
            video_path = str(video_folder / "vid_crop.avi")

            if (video_folder/"FR_predic_HSE.csv").exists():
                #print("File already exist")
                continue
            else:
                print("Patient", id, "|", v_name, "Fail")
            continue
            
            #try:
            ## load video
            frames = load_video(video_path)

            ## predict video
            valence, arousal, emotion = model.comput_va(frames)
            valence = valence*2
            arousal = arousal*2

            ## Save data
            predict_df = pd.DataFrame({"valence":valence,"arousal":arousal,"emotion":emotion})
            #print(predict_df.head)
            predict_df.to_csv(video_folder / "FR_predic_HSE.csv", index=False)
            #except:
                #print("Patient", id, "|", v_name, "Fail")


## ------ Init ------
if __name__ == "__main__":
    ## model
    models_folder = Path('C:/Users/Alex/Desktop/HSEmotion/models')
    model_name = 'enet_b0_8_va_mtl.pt'
    model_path = models_folder / model_name

    ## Data
    folder_path = Path("D:/castphys_60/Data")

    ## code
    predict_all_subject(folder_path, model_path)