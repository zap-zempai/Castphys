## ----- Imports -----
import numpy as np
import cv2
import torch
from torchvision import transforms
import timm
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt

from landmarks.landmarks_extraction import load_images_bgr, Video_preprocessing

matplotlib.use('TkAgg')

## ----- Code -----
IMG_SIZE = 260
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
            pred_landmarks = self.preprocessing.landmarks_extraction(img)
            max_p = np.max(pred_landmarks,axis=0)
            min_p = np.min(pred_landmarks,axis=0)
            img_crop = img[min_p[1]:max_p[1],min_p[0]:max_p[0],:]
            if False:
                fig, ax = plt.subplots()
                ax.imshow(np.uint8(img_crop))
                ax.set_xticks([])
                ax.set_yticks([])
                plt.show()
            frames.append(cv2.resize(img_crop, (IMG_SIZE,IMG_SIZE))) 
            ret_val, frame = video.read()
        frames = np.array(frames)
        frames = torch.from_numpy(frames.astype(np.float32)).permute(0,3,1,2)
        print(frames.shape)
        self.imgs = frames

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        #dealing with the image

        img_data=[]
        for id in range(0,self.imgs.shape[0]):
            #print(type(self.imgs[id,:,:,:].numpy()))
            img_data.append(self.transform(self.imgs[id,:,:,:]))

        img_torch = torch.stack(img_data, dim=0)
        print(img_torch.shape) 
        
        return img_torch


def main(path_model, video_path):
    ## load video
    test_dataset = MultiTaskDataset(video_path, transform=train_transforms, preprocessing_options={})
    frames = test_dataset[0]
    print(frames.shape)

    ## init model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = torch.load(path_model)
    model=model.to(device)
    model.eval()

    ## comput valence & arousal
    N = 300
    val_output = []
    with torch.no_grad():
        for i in tqdm(range(0,len(frames),N)):
            frame = frames[i:i+N].to(device)
            val_output.extend(model(frame).cpu()[:,-2:])    
    val_output = np.array(val_output)
    print(val_output.shape)

    valence = val_output[:,0].reshape(-1)
    arousal = val_output[:,1].reshape(-1)
    print(valence.shape, arousal.shape)

    ## plot
    fig, ax = plt.subplots()
    ax.plot(range(len(valence)),valence,label='Valance')
    ax.plot(range(len(arousal)),arousal,label='Arousal')
    #ax.set_ylim([-1,1])
    #for i in range(val_output.shape[1]):
    #    ax.plot(range(len(val_output)),val_output[:,i].reshape(-1),label=str(i))
    ax.legend()
    plt.show()
    
if __name__ == "__main__":
    ## video
    id = 32
    v_name = 'Q3_2'
    video_path = f"E:/castphys_60/Patient_{id}/{v_name}/vid_crop.avi"

    ## model
    models_folder='C:/Users/Xavi/Desktop/HSEmotion/models/'
    model_name = 'enet_b0_8_va_mtl.pt'
    path_model = models_folder+model_name

    ## main
    main(path_model, video_path)

