## --- imports ----
import numpy as np
import torch
from torch.utils.data import Dataset
from matplotlib import gridspec
import torch.nn.functional as F
import time
import csv
import cv2
import math
from scipy import signal
from scipy.signal import butter, filtfilt
import os
import h5py
import warnings
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.ticker import FixedLocator
from matplotlib.patches import Rectangle
import pandas as pd

try:
    from evaluation.utils import load_video, quadrant_pos
except:
    from dataloader.evaluation.utils import load_video, quadrant_pos

warnings.filterwarnings('ignore')
torch.manual_seed(0)
matplotlib.use('TkAgg')

## ---- code ----
class rPPG_Dataloader(Dataset):
    def __init__(self, data_path=None, dataset=None, temporal_length=300, temporal_stride=10,
                 img_size=96, interpolate=None, normalize=True, face_crop=True, compression=None, 
                 split_set='Train', cv=None, step=1):

        self.last_one = None
        start_time = time.time()
        self.dataset = dataset
        self.data_path = data_path
        self.face_crop = face_crop
        self.compression = compression
        self.temporal_length = temporal_length
        self.temporal_stride = temporal_stride
        self.split_set = split_set
        self.cv = cv
        self.step = step
        self.img_size = img_size
        self.normalize = normalize
        self.samples = []
        self.videos = []
        self.ppg_gt = []
        self.eda_gt = []
        self.brt_gt = []
        self.idxs = []
        self.v_a = []
        self.interpolate = interpolate

        dataset_path = os.path.join(self.data_path, self.dataset)
        set_list = []

        training_path = dataset_path + '/Protocols/Training.txt'
        val_path = dataset_path + '/Protocols/Validation.txt'
        test_path = dataset_path + '/Protocols/Test.txt'

        if cv is None:
            if split_set == 'Train':
                ftrain = open(training_path, "r")
                for x in ftrain:
                    dirr = x.split('\n')[0]
                    set_list.append(dirr)
            elif split_set == 'Dev':
                fdev = open(val_path, "r")
                for x in fdev:
                    dirr = x.split('\n')[0]
                    set_list.append(dirr)
            elif split_set == 'Test':
                ftest = open(test_path, "r")
                for x in ftest:
                    dirr = x.split('\n')[0]
                    set_list.append(dirr)
        elif cv == 'one':
            #set_list = os.listdir(dataset_path + '/Data/')[0:1]
            set_list = ['Patient_45']
        else:
            fold_name = dataset_path + '/Protocols/' + cv
            fold_list = []
            fcv = open(fold_name, "r")
            for x in fcv:
                dirr = x.split('\n')[0]
                fold_list.append(dirr)
            if split_set == 'Train':
                train_set = os.listdir(dataset_path + '/Data/')
                set_list = list(set(train_set) - set(fold_list))
            elif split_set == 'Test':
                set_list = fold_list

        print('------------------------')
        print('Loading ' + self.dataset + ' dataset ...')

        for user in set_list:
            user_folder = dataset_path + '/Data/' + user
            video_name_list = [v for v in os.listdir(user_folder) if 'Q' in v]
            video_name_list = ['Q1_2','Q9_2']
            for v_name in video_name_list:
                video_folder = user_folder + '/' + v_name
                bio_file = video_folder + '/bio.csv'
                annotation_file = video_folder + '/annotations.csv'

                if self.face_crop:
                    if self.compression is None:
                        video_path = video_folder + '/vid_crop.avi'
                    else:
                        video_path = video_folder + '/vid_crop_'+self.compression+'.avi'

                ppg_data = []
                eda_data = []
                breath_data = []
                time_data = []
                with open(bio_file, 'r') as file:
                    bio_reader = csv.reader(file)
                    next(bio_reader)
                    for i,item in enumerate(bio_reader):
                        if i%self.step != 0:
                            continue
                        ppg_data.append(item[0])
                        eda_data.append(item[1])
                        breath_data.append(item[2])
                        time_data.append(item[3])

                ppg_signal = np.asarray(ppg_data, dtype=np.float32)
                eda_signal = np.asarray(eda_data, dtype=np.float32)
                breath_signal = np.asarray(breath_data, dtype=np.float32)

                ## Normalize the signal
                ppg_signal = (ppg_signal-ppg_signal.min())/(ppg_signal.max()-ppg_signal.min())*2-1
                eda_signal = (eda_signal-eda_signal.min())/(eda_signal.max()-eda_signal.min())*2-1
                breath_signal = (breath_signal-breath_signal.min())/(breath_signal.max()-breath_signal.min())*2-1

                if len(ppg_signal) < self.temporal_length:
                    print('Check data size of subject', user, v_name)
                    continue

                df_annotations = pd.read_csv(annotation_file,header=None)
                v_a_values = [int(float(v)) for [c,v] in df_annotations.values if c in ['valence','arousal']]

                video_data = []
                sub_name = []
                #v_a_data = []
                for frame_idx, img in enumerate(load_video(video_path, type_color='RGB', step=self.step)):
                    if self.interpolate is not None:
                        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=self.interpolate)
                    else:
                        img = cv2.resize(img, (self.img_size, self.img_size))
                    video_data.append(img)
                    sub_name.append(user+':'+v_name)
                    #v_a_data.append(v_a_values)
                video_data = np.array(video_data)
                #v_a_data = np.array(v_a_data)
                v_a_values = np.array(v_a_values)

                print('##############')
                print(user, ':', v_name)
                print('##############')
                video_chunks, bvps_chunks, eda_chunks, brt_chunks, idx_chunks, v_a_chunks = self.chunk(video_data, ppg_signal, eda_signal,
                                                                                            breath_signal, sub_name, v_a_values,
                                                                                            self.temporal_length, self.temporal_stride, 
                                                                                            split=self.split_set)
                if True:
                    print('name:   ', idx_chunks[0][0])
                    print('video:  ', video_chunks.shape)
                    print('ppg:    ', bvps_chunks.shape)
                    print('eda:    ', eda_chunks.shape)
                    print('breath: ', brt_chunks.shape)
                    print('V & A:  ', v_a_chunks.shape)
                self.videos.extend(video_chunks)
                self.ppg_gt.extend(bvps_chunks)
                self.eda_gt.extend(eda_chunks)
                self.brt_gt.extend(brt_chunks)
                self.idxs.extend(idx_chunks)
                self.v_a.extend(v_a_chunks)

        self.samples = [self.videos, self.ppg_gt, self.eda_gt, self.brt_gt, self.idxs, self.v_a]

        print("--- %s seconds ---" % (time.time() - start_time))
        print()

    @staticmethod
    def chunk(frames, bvps, edas, brts, sub_name, v_a_data, temporal_length, temporal_overlap, split):

        if split == 'Test': #or split == 'Dev'
            clip_num = math.ceil((frames.shape[0] - temporal_length) / (temporal_length - temporal_overlap)) + 1
        else:
            clip_num = int((frames.shape[0] - temporal_length) / (temporal_length - temporal_overlap)) + 1
        index = (temporal_length - temporal_overlap)
        frames_clips = [frames[i * index:i * index + temporal_length] for i in range(clip_num)]
        bvps_clips = [bvps[i * index:i * index + temporal_length] for i in range(clip_num)]
        edas_clips = [edas[i * index:i * index + temporal_length] for i in range(clip_num)]
        brts_clips = [brts[i * index:i * index + temporal_length] for i in range(clip_num)]
        name_clips = [sub_name[i * index:i * index + temporal_length] for i in range(clip_num)]
        #v_a_clips = [v_a_data[i * index:i * index + temporal_length] for i in range(clip_num)]
        v_a_clips = [v_a_data for _ in range(clip_num)]

        return np.array(frames_clips), np.array(bvps_clips), np.array(edas_clips), np.array(brts_clips), name_clips, np.array(v_a_clips)

    def __getitem__(self, sample_idx):
        bvp_gt = torch.from_numpy(self.samples[1][sample_idx])
        eda_gt = torch.from_numpy(self.samples[2][sample_idx])
        brt_gt = torch.from_numpy(self.samples[3][sample_idx])
        idx = self.samples[4][sample_idx]
        v_a_gt = torch.from_numpy(self.samples[5][sample_idx])
        video = torch.empty(3, len(self.samples[0][sample_idx]), self.img_size, self.img_size, dtype=torch.float)
        for frame_idx, img in enumerate(self.samples[0][sample_idx]):
            out_type = img.dtype
            out_type_info = np.iinfo(out_type)
            img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
            # Normalize between 0 and 1
            if self.normalize:
                img_range = out_type_info.max - out_type_info.min
                img = (img - out_type_info.min) / img_range
            video[:, frame_idx] = img
        return video, bvp_gt, eda_gt, brt_gt, idx, v_a_gt

    def collate_fn(self, samples):
        pad_bvp = []
        pad_eda = []
        pad_brt = []
        pad_name = []
        pad_v_a = []
        pad_imgs = []

        for batch in samples:
            data = list(batch)
            pad_data = []
            for index, sample in enumerate(data):
                if index == 5:
                    pad_v_a.append(sample.reshape((1,2)))
                    continue
                if torch.is_tensor(sample):
                    if len(sample.size()) > 2:
                        sample_length = sample.size(dim=1)
                    else:
                        sample_length = sample.size(dim=0)
                else:
                    sample_length = len(sample)
                padding = self.temporal_length - sample_length
                if torch.is_tensor(sample) and padding != 0:
                    if len(sample.size()) > 2:
                        pad = (0, 0, 0, 0, 0, padding)
                        pad_data.append(torch.cat([self.last_one[index][:, -(padding+self.temporal_stride):, :, :], sample[:, self.temporal_stride:, :, :]], dim=1))
                    else:
                        pad = (0, padding)
                        pad_data.append(torch.cat([self.last_one[index][-(padding+self.temporal_stride):], sample[self.temporal_stride:]])) #F.pad(sample, pad, value=1)

                elif torch.is_tensor(sample) == False and padding != 0:
                    zero_list = [0] * (padding+self.temporal_stride)
                    names_list = sample[self.temporal_stride:]
                    pad_data.append(zero_list+names_list)
                else:
                    self.last_one = data
                    continue
            if pad_data:
                pad_imgs.append(pad_data[0].unsqueeze(0))
                pad_bvp.append(pad_data[1].unsqueeze(0))
                pad_eda.append(pad_data[2].unsqueeze(0))
                pad_brt.append(pad_data[3].unsqueeze(0))
                pad_name.append(pad_data[4])
                #pad_v_a.append(pad_data[5].unsqueeze(0))
            else:
                pad_imgs.append(data[0].unsqueeze(0))
                pad_bvp.append(data[1].unsqueeze(0))
                pad_eda.append(data[2].unsqueeze(0))
                pad_brt.append(data[3].unsqueeze(0))
                pad_name.append(data[4])
                #pad_v_a.append(data[5].unsqueeze(0))
            self.last_one = data

        pad_samples = [torch.cat(pad_imgs, dim=0), torch.cat(pad_bvp, dim=0), torch.concat(pad_eda, dim=0), torch.concat(pad_brt, dim=0), pad_name, torch.concat(pad_v_a, dim=0)]

        return pad_samples


    def __len__(self):
        return len(self.samples[0])

def main():
    image_size = 96
    temporal_length = 150
    temporal_stride = 0
    split_set = 'Test'
    batch_size = 15
    dataset = 'castphys_60'
    data_path = 'D:/'
    face_crop = True
    count = 0
    cv = 'one'
    step = 2 # no pot ser 0
    data = rPPG_Dataloader(data_path=data_path, dataset=dataset, temporal_length=temporal_length,
                           temporal_stride=temporal_stride, img_size=image_size, face_crop=face_crop, normalize=False,
                           split_set=split_set,cv=cv,step=step)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=False, num_workers=1,
                                             pin_memory=True, collate_fn=None if split_set in ['Train', 'Dev'] else data.collate_fn)

    for i, (img_batch, ppg_batch, eda_batch, brt_batch, name_batch, v_a_batch) in enumerate(dataloader):
        print('First name batch: ', name_batch[0][0])
        print('Shape of the video sample batch: ', img_batch.to(device).shape)
        print('Shape of the ppg sample batch: ', ppg_batch.to(device).shape)
        print('Shape of the eda sample batch: ', eda_batch.to(device).shape)
        print('Shape of the breath sample batch: ', brt_batch.to(device).shape)
        print('Shape of the Valence & Arousal sample batch: ', v_a_batch.to(device).shape)
        print('--------------------------------------')
        print('Count number: ', count)
        count+=1

        # video
        frames = img_batch[:, :, :, :, :].permute(0, 2, 1, 3, 4)
        frames = frames.reshape(-1, frames.size(2), frames.size(3), frames.size(4))
        frames = frames.permute(0, 2, 3, 1).detach().cpu().numpy()
        # ppg
        ppg_data = ppg_batch.reshape(-1)
        ppg_data = ppg_data.detach().cpu().numpy()
        ppg_signal = np.asarray(ppg_data, dtype=np.float32)
        # eda
        eda_data = eda_batch.reshape(-1)
        eda_data = eda_data.detach().cpu().numpy()
        eda_signal = np.asarray(eda_data, dtype=np.float32)
        # breath
        brt_data = brt_batch.reshape(-1)
        brt_data = brt_data.detach().cpu().numpy()
        brt_signal = np.asarray(brt_data, dtype=np.float32)
        # name
        name_list = []
        for names in name_batch:
            name_list.extend(names)
        # valence & arousal
        v_a_data = v_a_batch.detach().cpu().numpy()
        v_a_eval = np.asarray([v_a_data for _ in range(img_batch.size(2))], dtype=np.int32)
        v_a_eval = v_a_eval.reshape(-1, v_a_eval.shape[2])


        if False:
            fig = plt.figure(figsize=(15,7))
            a = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
            axx_bio = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=a[1])
            axx_vid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=a[0])

            plt.ion()
            for i in range(frames.shape[0]):
                ax_ppg = fig.add_subplot(axx_bio[0, 0])
                ax_eda = fig.add_subplot(axx_bio[1, 0])
                ax_brt = fig.add_subplot(axx_bio[2, 0])
                if i < 300:
                    ax_ppg.set_xlim([0, 300*2])
                    ax_eda.set_xlim([0, 300*2])
                    ax_brt.set_xlim([0, 300*2])
                else:
                    ax_ppg.set_xlim([i - 300, i + 300])
                    ax_eda.set_xlim([i - 300, i + 300])
                    ax_brt.set_xlim([i - 300, i + 300])

                # PPG
                ax_ppg.plot(range(i+1), ppg_signal[:i+1])#, label='Groundtruth')
                #ax_ppg.set_xlabel('frames')
                ax_ppg.set_ylabel('BVP Amplitude')
                #ax_ppg.legend()
                ax_ppg.set_ylim([-1.1, 1.1])
                # EDA
                ax_eda.plot(range(i+1), eda_signal[:i+1])
                #ax_eda.set_xlabel('frames')
                ax_eda.set_ylabel('EDA Amplitude')
                ax_eda.set_ylim([-1.1, 1.1])
                # Breath
                ax_brt.plot(range(i+1), brt_signal[:i+1])
                ax_brt.set_xlabel('frames')
                ax_brt.set_ylabel('Breath Amplitude')
                ax_brt.set_ylim([-1.1, 1.1])

                # Video
                ax_vid = fig.add_subplot(axx_vid[0, 0])
                im = frames[i, :, :, :]
                ax_vid.imshow(np.uint8(im))
                ax_vid.set_xticks([])
                ax_vid.set_yticks([])
                ax_vid.set_title('Frames dataloading')
                # Valance & arousal
                ax_va = fig.add_subplot(axx_vid[1, 0])
                if True:
                    # estructura
                    ax_va.set_xlabel('Valence')
                    ax_va.set_ylabel('Arousal')
                    ax_va.set_xlim([-2, 2])
                    ax_va.set_ylim([-2, 2])
                    ax_va.xaxis.set_major_locator(FixedLocator([-2,-1, 0, 1, 2]))
                    ax_va.yaxis.set_major_locator(FixedLocator([-2,-1, 0, 1, 2]))
                    ax_va.plot([2/3]*2, [2,-2], color = 'lightgray',linestyle = "--")
                    ax_va.plot([-2/3]*2, [2,-2], color = 'lightgray',linestyle = "--")
                    ax_va.plot([2,-2], [2/3]*2, color = 'lightgray',linestyle = "--")
                    ax_va.plot([2,-2], [-2/3]*2, color = 'lightgray',linestyle = "--")
                    # Quadrant video
                    try:
                        quadrant = name_list[i].split(':')[1].split('_')[0]
                    except:
                        quadrant = '0'
                    rect = Rectangle(quadrant_pos(quadrant),4/3,4/3,
                            alpha=0.2,
                            facecolor='seagreen',
                            edgecolor='green')
                    ax_va.add_patch(rect)
                    # auto evaluacio
                    va_pos = v_a_eval[i,:]
                    if va_pos[0] == 2:
                        va_pos[0] -= 0.1
                    elif va_pos[0] == -2:
                        va_pos[0] += 0.1
                    if va_pos[1] == 2:
                        va_pos[1] -= 0.1
                    elif va_pos[1] == -2:
                        va_pos[1] += 0.1
                    ax_va.scatter(x = v_a_eval[i,0], y = v_a_eval[i,1], label='auto_eval')
                    ax_va.legend()

                plt.show()
                plt.pause(0.0000000000001)
                ax_vid.cla()
                ax_va.cla()
                ax_ppg.cla()
                ax_eda.cla()
                ax_brt.cla()
                plt.clf()
            plt.close(fig)


if __name__ == '__main__':
    main()


