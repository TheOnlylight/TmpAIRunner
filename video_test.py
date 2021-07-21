import tensorflow as tf
import numpy as np
import scipy.io
import os
import sys
sys.path.append('../')
from model import Attention_mask, TS_CAN
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter
from utils import detrend
from process_gt_test import generate_pulse_gt, process_back_gt
from scipy.signal import resample
from preprocess import preprocess_raw_video
from pathlib import Path
from scipy import ndimage
from skimage.transform import resize
tf.random.set_seed(100)

def calculate_HR(fs, signal):
    N = 30 * fs
    pulse_fft = np.expand_dims(signal, 0)
    f, pxx = scipy.signal.periodogram(pulse_fft, fs=fs, nfft=4 * N, detrend=False)
    fmask = np.argwhere((f >= 0.75) & (f <= 2.5))  # regular Heart beat are 0.75*60 and 2.5*60
    # fmask = np.argwhere(((f >= 0.75) & (f <= 0.95)) | ((f > 1.05) & (f <= 2.5)))
    frange = np.take(f, fmask)
    HR = np.take(frange, np.argmax(np.take(pxx, fmask), 0))[0] * 60
    return HR, np.take(f, fmask), np.take(pxx, fmask)


#%% Model Inference
##############################################################################
# Parameters
img_rows = 36
img_cols = 36
frame_depth = 10
subj_list = os.listdir('./Raw_Dataset_cleaned')
#subj_list.remove('.DS_Store')
subj_list = sorted(subj_list)
subj_list = ['Sinan']
# subj_list = [subj_list[1]]
# print(subj_list)
# subj_list = [subj_list[9]]
camera_type = ['iPhone','Pixel','Galaxy','Huawei','MI']
all_sessions = ['4','5','6','7','8','9']
model_checkpoint = './ts_can_with_syn.hdf5'
batch_size = frame_depth * 10
model = TS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
model.load_weights(model_checkpoint)
attention_sub_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('attention_mask_2').output)
for name in subj_list:
    for camera in camera_type:
        for session in all_sessions:
            
            
            sample_data_path = './Raw_Dataset_cleaned/Camera Comparison/' + name+'_' + camera + '_' + session + '.MOV'
            print(sample_data_path)
            
            if not os.path.exists(sample_data_path):
                 continue
            result_path = './Results_Cleaned_MAE/Camera Comparison/'
            chunk_path = './Tsinghua-rPPG/ProcessedInputFiles/TsinghuaProcessedChunks36x36/'
            Path(result_path).mkdir(parents=True, exist_ok=True)
            PID = 'P' + str(name.split('_')[0])
            if camera == 'Front':
                real_camera = 'RGB'
            else:
                real_camera = 'IR'
            TASK = '_' + real_camera + '_' + session
#            if not Path(sample_backg_ppg_path).is_file():
#                print('==========Missing Back PPG File=================')
#                print(sample_backg_ppg_path)
#                print('================================================')
#                continue
#
#            if not Path(ground_truth_path).is_file():
#                print('==========Missing GT File=================')
#                print(ground_truth_path)
#                print('==========================================')
#                continue

            ##############################################################################
            # Processing Face PPG
            dXsub = preprocess_raw_video(sample_data_path, dim=36)
            print('Original dXsub Shape: ', dXsub.shape)
            num_processed_frame = dXsub.shape[0]
            # Cut off some extra frames (2412 -> 2400)
            dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
            dXsub_trimed = dXsub[:dXsub_len, :, :, :]
            # Face PPG Prediction
            yptest = model.predict((dXsub_trimed[:, :, :, :3], dXsub_trimed[:, :, :, -3:]), batch_size=batch_size, verbose=0)
            attention_mask = attention_sub_model.predict((dXsub_trimed[:, :, :, :3], dXsub_trimed[:, :, :, -3:]), batch_size=batch_size, verbose=0)
            attention_mask = np.squeeze(attention_mask, -1)
            attention_mask = attention_mask[30:35]
            for img_idx in range(5):
                # attention_mask[img_idx] = attention_mask[img_idx] / np.max(attention_mask[img_idx])
                attention_mask[img_idx] = np.rot90(attention_mask[img_idx], k=1)
            attention_mask = np.reshape(attention_mask, [170, 34])
            attention_mask = ndimage.rotate(attention_mask, 270, reshape=True)
            # np.save(result_path + exp_name + '_pred.npy', yptest)
            # Bandpass filtering Parameters
            fs = 30
            # [b, a] = butter(1, [0.75 / fs * 2, 2 / fs * 2], btype='bandpass')
            [b, a] = butter(1, [0.75/fs*2, 2.5/fs*2], btype='bandpass')
            pred_window = yptest
            pred_window = detrend(np.cumsum(pred_window), 100) # Post
            pred_window = scipy.signal.filtfilt(b, a, np.double(pred_window))
            # ##############################################################################
#             # Processing Back-finger PPG
#             try:
#                 green_back_gt, back_time = process_back_gt(sample_backg_ppg_path)
#             except Exception as e:
#                 print('============Broken File===============')
#                 print(sample_backg_ppg_path)
#                 print('=======================================')
#                 continue
#             green_back_gt = detrend(green_back_gt, 50)
#             print('green_back_gt frame: ', green_back_gt.shape)
            # ##############################################################################
             # Processing Ground-Truth ppg
#            gt_pulse, gt_time = generate_pulse_gt(ground_truth_path)
#            gt_pulse = np.array(gt_pulse)
#            gt_time = np.array(gt_time)
            ##############################################################################
            # Resampling Data to ensure same length from video, finger-ppg, and GT

#            gt_pulse = (gt_pulse-np.min(gt_pulse))
#            gt_pulse = gt_pulse / np.max(gt_pulse)

            pred_window = (pred_window-np.min(pred_window))
            pred_window = pred_window / np.max(pred_window)

            # green_back_gt = (green_back_gt-np.min(green_back_gt))
            # green_back_gt = green_back_gt / np.max(green_back_gt)

            # Reshape and interpolation
            # back_time = np.reshape(back_time, (back_time.shape[0]))
            # green_back_gt = np.reshape(green_back_gt, (green_back_gt.shape[0]))
            # green_back_gt = 1 - green_back_gt
            # Get Interp time stamp for GT and back PPG
            # gap = 33 # 30FPS
            # start_time = int(back_time[0])
            # end_time = start_time + num_processed_frame * gap
            # re_time = range(start_time, end_time, gap)

            #re_gt_pulse = np.interp(re_time, gt_time, gt_pulse)
            # re_green_back_gt = np.interp(re_time, back_time, green_back_gt)

            # np.save(result_path + exp_name + '_back.npy', re_green_back_gt)
            # np.save(result_path + exp_name + '_gt.npy', re_gt_pulse)

            # hr
#            gap = 33
#            gt_time = np.reshape(gt_time, (gt_time.shape[0]))
#            start_time = int(gt_time[0])
#            end_time = start_time + + num_processed_frame * gap
#            re_time = range(start_time, end_time, gap)
#            re_gt_pulse = np.interp(re_time, gt_time, gt_pulse)
#            
#            gt_HR, gt_f, gt_pxx = calculate_HR(fs, re_gt_pulse)
            # back_HR, back_f, back_pxx = calculate_HR(fs, re_green_back_gt)
            pred_HR, pred_f, pred_pxx = calculate_HR(fs, pred_window)

            fig = plt.gcf()
            #fig.set_size_inches(6, 13)
#            fig.suptitle(name + '-' + session, fontsize=15)
#            plt.subplot(711)
#            plt.plot(gt_f, np.reshape(gt_pxx, [-1, 1]))
#            plt.title('{} - GT FFT'.format(session))

            # plt.subplot(712)
            # plt.plot(back_f, np.reshape(back_pxx, [-1, 1]))
            # plt.title('Back FFT')

            plt.subplot(713)
            plt.plot(pred_f, np.reshape(pred_pxx, [-1, 1]))
            plt.title('Pred FFT Pre HR:{}'.format(str(pred_HR)))

            ##############################################################################
            # Visualizing the outputs
            # plt.subplot(714)
            # plt.plot(re_gt_pulse)
            # plt.plot(re_green_back_gt)
            # plt.title('GT vs. fPPG, GT HR: {}, MAE: {}'.format(str(gt_HR), str(abs(gt_HR - back_HR))))
            # plt.legend(['GT', 'fPPG'])

#            plt.subplot(715)
#            plt.plot(re_gt_pulse)
#            plt.plot(pred_window)
#            plt.title('GT vs. rPPG, rPPG HR: {}, MAE: {}'.format(str(pred_HR), str(abs(gt_HR - pred_HR))))
#            plt.legend(['GT','rPPG'])

            # plt.subplot(716)
            # plt.plot(re_green_back_gt)
            # plt.plot(pred_window)
            # plt.title('fPPG vs. rPPG, fPPG HR: {}, MAE: {}'.format(str(back_HR), str(abs(back_HR - pred_HR))))
            # plt.legend(['fPPG','rPPG'])

            # plt.subplot(717)
            # plt.imshow(attention_mask, cmap='viridis', interpolation='nearest', aspect='auto')
            # plt.title('Attention Masks')
            fig.tight_layout()
            plt.savefig(result_path + name+'_' + camera+ '_' + session + '.png', dpi=300)
            plt.close()
            ######################
            # # Final Print
            # print('dXsub: ', dXsub.shape)
            # print('re_green_back_gt', re_green_back_gt.shape)
            # print('re_gt_pulse', re_gt_pulse.shape)
