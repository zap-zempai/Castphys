## ---- Import ----
import numpy as np
import torch
from matplotlib import gridspec
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib.patches import Rectangle
import warnings

from dataloader.dataloader_old import rPPG_Dataloader
from dataloader.evaluation.utils import quadrant_pos
from VEATIC.model_VEATIC.model import VEATIC_baseline


warnings.filterwarnings('ignore')
torch.manual_seed(0)
matplotlib.use('TkAgg')

## ---- Code ----
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
    # Model
    model = VEATIC_baseline()

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
        v_a_data = v_a_batch.reshape(-1, v_a_batch.size(2))
        v_a_data = v_a_data.detach().cpu().numpy()
        v_a_eval = np.asarray(v_a_data, dtype=np.float32)
        # Model
        print('--------------------------------------')
        v_a_predict = []
        for i in range(img_batch.shape[0]):
            video_segment = img_batch[i:i+1,:,:,:,:].permute(2, 0, 1, 3, 4)
            out = model(video_segment)
            print("Batch", i+1 , out.to(device).shape)
            v_a_predict.append(out)
        v_a_predict = torch.cat(v_a_predict, 0)
        print('Shape of the Valence & Arousal Estimation: ', v_a_predict.to(device).shape)
        print('--------------------------------------')
        v_a_predict = v_a_predict.detach().cpu().numpy()
        v_a_predict = np.asarray(v_a_predict, dtype=np.float32)



        if True:
            fig = plt.figure(figsize=(15,7))
            a = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
            axx_bio = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=a[1])
            axx_vid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=a[0])

            plt.ion()
            for i in range(frames.shape[0]):
                ax_ppg = fig.add_subplot(axx_bio[0, 0])
                ax_eda = fig.add_subplot(axx_bio[1, 0])
                ax_brt = fig.add_subplot(axx_bio[2, 0])
                ax_va = fig.add_subplot(axx_bio[3, 0])
                if i < 450:
                    ax_ppg.set_xlim([0, 600])
                    ax_eda.set_xlim([0, 600])
                    ax_brt.set_xlim([0, 600])
                    ax_va.set_xlim([0, 600])
                else:
                    ax_ppg.set_xlim([i - 450, i + 150])
                    ax_eda.set_xlim([i - 450, i + 150])
                    ax_brt.set_xlim([i - 450, i + 150])
                    ax_va.set_xlim([i - 450, i + 150])

                # PPG
                ax_ppg.plot(range(i+1), ppg_signal[:i+1])#, label='Groundtruth')
                ax_ppg.set_ylabel('BVP Amplitude')
                #ax_ppg.legend()
                ax_ppg.set_ylim([-1.1, 1.1])
                # EDA
                ax_eda.plot(range(i+1), eda_signal[:i+1])
                ax_eda.set_ylabel('EDA Amplitude')
                ax_eda.set_ylim([-1.1, 1.1])
                # Breath
                ax_brt.plot(range(i+1), brt_signal[:i+1])
                #ax_brt.set_xlabel('frames')
                ax_brt.set_ylabel('Breath Amplitude')
                ax_brt.set_ylim([-1.1, 1.1])
                # Valence and Arousal
                ax_va.plot(range(i+1), v_a_predict[:i+1,0], label='Valence')
                ax_va.plot(range(i+1), v_a_predict[:i+1,1], label='Arousal')
                ax_va.set_xlabel('frames')
                ax_va.set_ylabel('Valence & Arousal')
                ax_va.legend()
                ax_va.set_ylim([-2.1, 2.1])

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
                    ax_va.scatter(x = va_pos[0], y = va_pos[1], label='auto_eval')
                    ax_va.scatter(x = v_a_predict[i,0], y = v_a_predict[i,1], label='estimation')
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