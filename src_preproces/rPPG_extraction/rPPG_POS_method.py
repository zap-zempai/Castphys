from scipy.signal import filtfilt, hilbert
import mediapipe as mp
import copy
import cv2
#from utils_rppg.landmarks import *
import matplotlib.pyplot as plt
import matplotlib
import datetime
from collections import deque
import numpy as np
import os
from scipy.signal import butter
from pathlib import Path
import pandas as pd
from src1.biosignalsplux.biosignalsProcessor import open_signals_npy


def ppg_preprocessing(data, fs, view=False, name=""):
    m_avg = lambda t, x, w: (np.asarray([t[i] for i in range(w, len(x) - w)]),
                             np.convolve(x, np.ones((2 * w + 1,)) / (2 * w + 1),
                                         mode='valid'))

    time = np.linspace(0, len(data), len(data))
    # moving average
    w_size = int(fs * .5)  # width of moving window
    mt, ms = m_avg(time, data, w_size)  # computation of moving average

    # remove global modulation
    sign = data[w_size: -w_size] - ms

    # compute signal envelope
    analytical_signal = np.abs(hilbert(sign))

    fs = len(sign) / (max(mt) - min(mt))
    w_size = int(fs)
    # moving averate of envelope
    mt_new, mov_avg = m_avg(mt, analytical_signal, w_size)

    # remove envelope
    signal_pure = sign[w_size: -w_size] / mov_avg

    if view:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), sharex=True)
        ax1.plot(time, data, "b-", label="Original")
        ax1.legend(loc='best')
        ax1.set_title("File " + str(name) + " Raw", fontsize=14)  # , fontweight="bold")

        ax2.plot(mt, sign, 'r-', label="Pure signal")
        ax2.plot(mt_new, mov_avg, 'b-', label='Modulation', alpha=.5)
        ax2.legend(loc='best')
        ax2.set_title("Raw -> filtered", fontsize=14)  # , fontweight="bold")

        ax3.plot(mt_new, signal_pure, "g-", label="Demodulated")
        ax3.set_xlim(0, mt[-1])
        ax3.set_title("Raw -> filtered -> demodulated", fontsize=14)  # , fontweight="bold")

        ax3.set_xlabel("Time (sec)", fontsize=14)  # common axis label
        ax3.legend(loc='best')

        plt.show()
        plt.clf()

    return signal_pure


def resample_signal(sig, new_length):
    length = new_length
    resampled_signal = np.interp(np.linspace(0.0, 1.0, length, endpoint=False), np.linspace(0.0, 1.0, len(sig),
                                                                                            endpoint=False), sig)
    return resampled_signal


def check_file_exist(path: str):
    if not os.path.exists(path):
        raise Exception('Can not find path: "{}"'.format(path))

def load_images_bgr(imgs_dir, crop_res, list_top=-1, skip=1):
    img_names = [_ for _ in os.listdir(imgs_dir) if '.png' in _]
    pad = len(str(len(img_names)))
    img_names.sort(key=lambda x: str(x.split('.')[0]).zfill(pad))
    list_top = len(img_names) if  list_top==-1 or list_top>=len(img_names) else list_top
    video = []
    for i in range(0,list_top,skip):
        pic_path = os.path.join(imgs_dir, img_names[i])
        pic = cv2.imread(pic_path)
        pic = cv2.resize(pic, crop_res)
        video.append(pic)
    return video

class Video_preprocessing:
    def __init__(self, options):
        self.options = options
    @staticmethod
    def median_filter(lndmrks, win_len=5):
        windowed_sample = deque()
        sample_count = 0
        temporal_length = win_len
        temporal_stride = 1
        samples_data = []
        for i in range(0, len(lndmrks)):
            windowed_sample.append(lndmrks[i])
            sample_count += 1
            if len(windowed_sample) == temporal_length:
                final_windowed_sample = np.median(np.asarray(list(copy.deepcopy(windowed_sample))), axis=0)
                final_windowed_sample = [[int(v) for v in l] for l in final_windowed_sample]
                for t in range(temporal_stride):
                    windowed_sample.popleft()
                samples_data.append(final_windowed_sample)

        final_landmarks = [*lndmrks[0:win_len // 2], *samples_data[:], *lndmrks[-win_len // 2:]]
        return final_landmarks

    @staticmethod
    def merge_add_mask(img_1, mask):
        assert mask is not None
        mask = mask.astype('uint8')
        mask = mask * 255
        b_channel, g_channel, r_channel = cv2.split(img_1)
        r_channel = cv2.bitwise_and(r_channel, r_channel, mask=mask)
        g_channel = cv2.bitwise_and(g_channel, g_channel, mask=mask)
        b_channel = cv2.bitwise_and(b_channel, b_channel, mask=mask)
        res_img = cv2.merge((b_channel, g_channel, r_channel))
        return res_img

    @staticmethod
    def landmarks_transform(image, landmark_list, region):
        lndmrks = []
        lnd_list = np.asarray(landmark_list)[region]
        for landmark in lnd_list:
            x = landmark.x
            y = landmark.y
            shape = image.shape
            relative_x = int(x * shape[1])
            relative_y = int(y * shape[0])
            lndmrks.append([relative_x, relative_y])
        return np.asarray(lndmrks)

    @staticmethod
    def poly2mask(landmarks, img_shape, val=1, b_val=0):
        if b_val == 0:
            hull_mask = np.zeros(img_shape[0:2] + (1,), dtype=np.float32)
        else:
            hull_mask = np.ones(img_shape[0:2] + (1,), dtype=np.float32)
        cv2.fillPoly(hull_mask, [landmarks], (val,))
        return hull_mask

    def landmarks_extraction(self, video):
        mp_face_mesh = mp.solutions.face_mesh
        lnds_video = []
        prev_face_landmarks = None
        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,
                                   min_tracking_confidence=0.99) as face_mesh:
            for i,frame in enumerate(video):
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                results = face_mesh.process(frame)
                try:
                    face_landmarks = results.multi_face_landmarks[0]
                except:
                    face_landmarks = prev_face_landmarks
                pred_landmarks = face_landmarks.landmark
                pred_landmarks = self.landmarks_transform(frame, pred_landmarks, range(0, 468))
                lnds_video.append(pred_landmarks)
                prev_face_landmarks = face_landmarks
        return lnds_video

    def ROI_extraction(self, video, landmarks_video, flag_plot=False):

        leftEyeUpper = [463, 263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
        rightEyeUpper = [243, 173, 246, 161, 160, 159, 158, 157, 173, 33, 7, 163, 144, 145, 153, 154, 155, 133]
        face_countours = [54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379,
                          365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338, 10, 109, 67, 103]
        lips = [76, 184, 74, 73, 72, 11, 302, 303, 304, 408, 306, 307, 320, 404, 315, 16, 85, 180, 90, 96, 62]
        images = []
        landmarks_video = np.asarray(self.median_filter(landmarks_video, 5))
        for num, frame in enumerate(video):
            landmarks = landmarks_video[num]
            (x, y, w, h) = cv2.boundingRect(landmarks)
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            image = frame[y: y + h, x:x + w]
            landmarks = [[landmarks[_][0] - x, landmarks[_][1] - y] for _ in range(len(landmarks))]
            landmarks = np.array(landmarks, dtype=np.int32)
            face_countour = landmarks[face_countours]
            lips_countour = landmarks[lips]
            leye_countour = landmarks[leftEyeUpper]
            reye_countour = landmarks[rightEyeUpper]
            face_mask = self.poly2mask(face_countour, image.shape, val=1, b_val=0)
            lips_mask = self.poly2mask(lips_countour, image.shape, val=0, b_val=1)
            leye_mask = self.poly2mask(leye_countour, image.shape, val=0, b_val=1)
            reye_mask = self.poly2mask(reye_countour, image.shape, val=0, b_val=1)
            roi_mask = np.logical_and.reduce([face_mask, lips_mask, leye_mask, reye_mask])
            image = self.merge_add_mask(image, roi_mask)
            images.append(cv2.resize(image, (329, 246)))
            if flag_plot:
                image_copy = copy.deepcopy(image)
                for lmk_idx, lmk in enumerate(landmarks):
                    land = lmk.tolist()
                    cv2.circle(image_copy,(land[0],land[1]), 1, (255, 0, 0), -1)
                image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)
                plt.imshow(image_copy)
                plt.pause(1 / 60)
                plt.clf()
        return images


class DefaultStrategy:
    @staticmethod
    def extract_means(frame):
        nan_frame = frame
        nan_frame[nan_frame == 0.0] = np.nan
        r = np.nanmean(nan_frame[:, :, 2])
        g = np.nanmean(nan_frame[:, :, 1])
        b = np.nanmean(nan_frame[:, :, 0])
        return [r, g, b]

    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=979357
    @staticmethod
    def detrend(y):
        from scipy.sparse import identity
        from scipy.sparse import spdiags
        T = len(y)
        l = 10
        I = identity(T)
        data = np.ones(shape=[T - 2, 1]) * np.array([1, -2, 1])
        D2 = spdiags(data.T, np.array([0, 1, 2]), T - 2, T)
        operations = I + (l ** 2) * (D2.T * D2)
        inversion = np.linalg.inv(operations.toarray())
        y_stat = np.matmul((I - np.linalg.inv((I + (l ** 2) * (D2.T * D2)).toarray())), y).A[0]
        return y_stat
    def uses_nir(self):
        return False
    @staticmethod
    def moving_average(values, window_size):
        m_average = np.zeros([len(values)])
        left, right = (0, window_size)
        for i in range(len(values)):
            m_average[i] = np.mean(values[i - left: i + right])
            if (left + 1 < right) or (i + right == len(values)):
                left += 1
                right -= 1
        return m_average

    @staticmethod
    def get_fft(y, frame_rate=30):
        sample_rate = 1.0 / float(frame_rate)
        sample_count = len(y)
        yf = np.fft.fft(y)
        xf = np.linspace(0.0, 1.0 / (2.0 * sample_rate), sample_count // 2)
        return xf, 2.0 / sample_count * np.abs(yf[0: sample_count // 2])

    # https://gitlab.idiap.ch/bob/bob.rppg.base/blob/master/bob/rppg/base/utils.py
    @staticmethod
    def build_bandpass_filter(fs, order, min_freq=0.6, max_freq=4.0):
        from scipy.signal import firwin
        nyq = fs / 2.0
        numtaps = order + 1
        return firwin(numtaps, [min_freq / nyq, max_freq / nyq], pass_zero=False)
    @staticmethod
    def bandpass_filter(data, frame_rate=30, min_freq=0.6, max_freq=4.0, order=64):
        from scipy.signal import filtfilt
        return filtfilt(DefaultStrategy.build_bandpass_filter(fs=float(frame_rate),
                                                              order=order,
                                                              min_freq=min_freq,
                                                              max_freq=max_freq), np.array([1]), data)

# https://pure.tue.nl/ws/portalfiles/portal/31563684/TBME_00467_2016_R1_preprint.pdf
# https://github.com/pavisj/rppg-pos/blob/master/pos_face_seg.py
class Wang(DefaultStrategy):
    @staticmethod
    def extract_rppg(temporal_means, frame_rate=25, window_size=None):
        def get_order(size):
            return (size - 6) // 3
        if window_size is None:
            window_size = int(frame_rate * 1.6)
        signal = np.zeros(len(temporal_means), dtype=np.float32)
        for t in range(len(temporal_means) - window_size):
            Cn = temporal_means[t:t + window_size].T
            if np.any(Cn.mean(axis=0) == 0.0):
                continue
            else:
                Cn = Cn / Cn.mean(axis=0)
            projection_matrix = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float32)
            S = np.matmul(projection_matrix, Cn)
            if np.std(S[1]) == 0.0:
                continue
            else:
                h = S[0] + (np.std(S[0]) / np.std(S[1])) * S[1]
            signal[t:t + window_size] += h - h.mean()

        return DefaultStrategy.moving_average(signal, frame_rate // 4)


def POS_extraction(images, fps):
    temporal_means = None
    for img in images:
        cut_bgr = img
        b, g, r = np.nanmean(cut_bgr, axis=(0, 1))
        features = np.array([[r, g, b]])
        if temporal_means is None:
            temporal_means = features
        else:
            temporal_means = np.append(temporal_means, features, axis=0)

    wang_signal = Wang.extract_rppg(temporal_means.copy(), int(fps))
    [b_pulse, a_pulse] = butter(1, [0.75 / fps * 2, 2.5 / fps * 2], btype='bandpass')
    wang_signal = filtfilt(b_pulse, a_pulse, np.double(wang_signal))
    wang_signal = (wang_signal - min(wang_signal)) * (1 - (-1)) / (
            max(wang_signal) - min(wang_signal)) + (-1)

    return wang_signal

def rPPG_detector(img_path:str,df_bio,fps:int,time:int=-1):
    ## millor si els fps son divisors de fps_t=60
    ## time en segons

    ## Parametres
    #print("Start") ## millor si els fps son divisors de fps_t=60
    crop_size = (329, 246)
    fps_t = 60
    list_crop = -1 if time==-1 else time*fps_t

    ## Video
    #print("video")
    video = load_images_bgr(img_path, crop_size, list_crop, fps_t//fps)
    options = {}
    preprocessing = Video_preprocessing(options)
    pred_landmarks = preprocessing.landmarks_extraction(video)
    images = preprocessing.ROI_extraction(video, pred_landmarks, flag_plot=False)

    ## Bio Signal
    #print("biosignal")
    bvp_signal = df_bio.ppg.values
    if list_crop == -1:
        bvp_signal = bvp_signal[0::fps_t//fps]
    else:
        bvp_signal = bvp_signal[0:list_crop:fps_t//fps]
    #bvp_signal = resample_signal(bio_signal[0][:list_crop], len(images))

    ## rPPG
    #print("rPPG")
    [b_pulse, a_pulse] = butter(1, [0.75 / fps * 2, 2.5 / fps * 2], btype='bandpass')
    bvp_signal = filtfilt(b_pulse, a_pulse, np.double(bvp_signal))
    bvp_signal = (bvp_signal - min(bvp_signal)) * (1 - (-1)) / (
            max(bvp_signal) - min(bvp_signal)) + (-1)
    wang_signal_raw = POS_extraction(images, fps)
    wang_signal = np.pad(wang_signal_raw, (16, 16), 'symmetric')
    wang_signal = ppg_preprocessing(wang_signal, fps)

    return bvp_signal, wang_signal, wang_signal_raw

def plot_results(bvp_signal, wang_signal, wang_signal_raw):
    plt.plot(bvp_signal, label='PPG_signal')
    plt.plot(wang_signal, label='rPPG_signal')
    plt.plot(wang_signal_raw, label='rPPG_signal_POS_raw')
    plt.legend()
    plt.show()


## --------------------------------------------------------
#matplotlib.use("TkAgg")
def main():
    init_path = Path('D:/TFG_ALEX/raw_data/recordings') 
    prosseced = False
    dataset_path= init_path / 'Patient_55' # si esta procesat canviar el path afegin la carpeta del video
    img_path = dataset_path / 'imgs'

    if prosseced:
        df_bio = pd.read_csv(dataset_path / 'bio.csv')
    else:
        df_bio = open_signals_npy(dataset_path)
    bvp_signal, wang_signal, wang_signal_raw = rPPG_detector(img_path,df_bio,
                                                             fps=30,
                                                             time=10)
    plot_results(bvp_signal, wang_signal, wang_signal_raw)
