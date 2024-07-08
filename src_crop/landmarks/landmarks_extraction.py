## ---------------- Imports ----------------
import cv2
import numpy as np
import mediapipe as mp
import os
from pathlib import Path
import copy
from collections import deque
import matplotlib.pyplot as plt
import matplotlib

## ---------------- Functions ----------------
def load_images_bgr(imgs_dir, crop_res, list_top=-1):
    img_names = [_ for _ in os.listdir(imgs_dir) if '.png' in _]
    pad = len(str(len(img_names)))
    img_names.sort(key=lambda x: str(x.split('.')[0]).zfill(pad))
    list_top = len(img_names) if  list_top==-1 or list_top>=len(img_names) else list_top
    video = []
    for i in range(0,list_top):
        pic_path = os.path.join(imgs_dir, img_names[i])
        pic = cv2.imread(pic_path)
        #pic = cv2.resize(pic, crop_res)
        video.append(pic)
    return video

## ---------------- Class ----------------
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




## ---------------- Code ----------------
def main ():
    img_path = Path("E:/castphys_60/Patient_10/Q1_1/imgs")
    crop_size = (329, 246)
    video = load_images_bgr(img_path, crop_size)
    #print(len(video))
    options = {}
    preprocessing = Video_preprocessing(options)
    pred_landmarks = preprocessing.landmarks_extraction(video)
    print(pred_landmarks[0])
    #images = preprocessing.ROI_extraction(video, pred_landmarks, flag_plot=False)

if __name__ == "__main__":
    main()