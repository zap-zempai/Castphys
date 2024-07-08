## ----- Imports -----
import numpy as np
import cv2
import torch
from torchvision import transforms
import timm
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt

from my_util.fer_util import nn_output
from my_util.detect_util import draw_results_ssd
from landmarks.landmarks_extraction import load_images_bgr, Video_preprocessing

matplotlib.use('TkAgg')

## ----- Code -----
class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, video_path, preprocessing_options):
        self.preprocessing = Video_preprocessing(preprocessing_options)

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
            frames.append(img_crop) 
            ret_val, frame = video.read()
        frames = np.array(frames)
        frames = torch.from_numpy(frames.astype(np.float32)).permute(0,3,1,2)
        print(frames.shape)
        self.imgs = frames

    def __len__(self): return len(self.imgs.shape[0])

    def __getitem__(self, idx): return self.imgs


class AVCE_model():
    def __init__(self,model_path,weights_path="", N=10):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.N = N

        ## net
        protoPath = model_path+"AVCE_demo/face_detector/deploy.prototxt"
        modelPath = model_path+"AVCE_demo/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        ## encoder, regressor
        encoder, regressor = nn_output()
        if weights_path == "":
            encoder.load_state_dict(torch.load(model_path+'AVCE_demo/weights/Sparse_cont_enc.t7',map_location=self.device), strict=False)
            regressor.load_state_dict(torch.load(model_path+'AVCE_demo/weights/Sparse_cont_reg.t7',map_location=self.device), strict=False)
        else:
            encoder.load_state_dict(torch.load(model_path+"pretrained_weights/"+weights_path+'/encoder.t7',map_location=self.device), strict=False)
            regressor.load_state_dict(torch.load(model_path+"pretrained_weights/"+weights_path+'/regressor.t7',map_location=self.device), strict=False)

        encoder.train(False)
        regressor.train(False)
        self.encoder = encoder
        self.regressor = regressor

    def va_compute(self, image_batch):
        input_img = image_batch[-1]
        img_h, img_w, _ = np.shape(input_img)
        
        blob = cv2.dnn.blobFromImage(cv2.resize(input_img, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detected = self.net.forward()

        faces = np.empty((detected.shape[2], 224, 224, 3))
        cropped_face, fd_signal = draw_results_ssd(detected,input_img,faces,0.1,224,img_w,img_h,0,0,0)
    
        croppted_face_tr = torch.from_numpy(cropped_face.transpose(0,3,1,2)[0]/255.)
        cropped_face_th_norm = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))(croppted_face_tr)

        latent_feature = self.encoder(cropped_face_th_norm.unsqueeze_(0).type(torch.FloatTensor)).to(self.device)
        va_output = self.regressor(latent_feature)
        
        valence = va_output.detach().cpu().numpy()[0][0] + 0.15
        arousal = va_output.detach().cpu().numpy()[0][1] + 0.15

        return valence, arousal
    
    def va_batch(self, frames):
        valence_list = [] 
        arousal_list = [] 
        for i in tqdm(range(len(frames))):
            if i < self.N:
                valence, arousal = self.va_compute(frames[0:i+1])
            else:
                valence, arousal = self.va_compute(frames[i-self.N:i+1])
            valence_list.append(valence)
            arousal_list.append(arousal)

        return np.array(valence_list),np.array(arousal_list)


def main(model_path,weights_path, video_path):
    ## load video
    test_dataset = MultiTaskDataset(video_path, preprocessing_options={})
    frames = test_dataset[0]
    print(frames.shape)

    ## init model
    N=10
    model = AVCE_model(model_path,weights_path,N=N)

    ## comput valence & arousal
    valence, arousal = model.va_batch(frames)
    v_all, a_all = model.va_compute(frames)
    print(len(valence),len(arousal))
    print("general prediction:", v_all, a_all)

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
    weights_path = 'enet_b0_8_va_mtl.pt'

    ## main
    main(models_folder, weights_path, video_path)

