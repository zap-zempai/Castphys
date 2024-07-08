from matplotlib import pyplot as plt
from scipy import stats, signal
import os
import cv2
import numpy as np


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

"""
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

def _calculate_fft_hr(ppg_signal, fs=30, low_pass=0.75, high_pass=2.5):
    "Calculate heart rate based on PPG using Fast Fourier transform (FFT)."
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr, pxx_ppg, f_ppg
"""


def check_file_exist(path: str):
    if not os.path.exists(path):
        raise Exception('Can not find path: "{}"'.format(path))


def load_video(path: str, type_color: str, step: int=1) -> list:
    check_file_exist(path)
    video = cv2.VideoCapture(path)
    frames = []
    ret_val, frame = video.read()
    while ret_val:
        if type_color == 'BGR':
            frames.append(frame)
        elif type_color == 'RGB':
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        elif type_color == 'YUV':
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2YUV))
        elif type_color == 'LAB':
            frame = (frame - frame.min()) / (frame.max() - frame.min())
            im_lab = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_BGR2LAB)
            frames.append(im_lab)
        for _ in range(step):
            ret_val, frame = video.read()
    video.release()
    return frames

def quadrant_pos(quadrant):
    if quadrant == 'Q1':
        return (-2,2/3)
    elif quadrant == 'Q2':
        return (-2/3,2/3)
    elif quadrant == 'Q3':
        return (2/3,2/3)
    elif quadrant == 'Q4':
        return (-2,-2/3)
    elif quadrant == 'Q5':
        return (-2/3,-2/3)
    elif quadrant == 'Q6':
        return (2/3,-2/3)
    elif quadrant == 'Q7':
        return (-2,-2)
    elif quadrant == 'Q8':
        return (-2/3,-2)
    elif quadrant == 'Q9':
        return (2/3,-2)
    else:
        return(2,2)