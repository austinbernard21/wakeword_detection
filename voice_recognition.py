import sounddevice as sd
import python_speech_features
import numpy as np
import pickle
import scipy.signal
import queue

model = pickle.load(open('initial_model.sav', 'rb'))

duration = 2  # seconds
sample_rate = 48000
resample_rate = 8000
# myrecording = sd.rec(int(duration * rate), samplerate=rate, channels=1)
#
# sd.wait()

# mfcc = python_speech_features.base.mfcc(myrecording, samplerate=rate, winstep=0.15, winlen=.2, nfft=2048, winfunc=np.hamming)
# dim_1, dim_2 = mfcc.shape
# reshape_dim = dim_1 * dim_2
# mfcc = mfcc.reshape(-1, reshape_dim)
# # print(reshape_dim)
# print(mfcc.shape)
#
# print(model.predict(mfcc))
# sd.play(myrecording, rate)
#
# print(myrecording.shape)

# q = queue.Queue()

def decimate(signal, old_fs, new_fs):
    # Check to make sure we're downsampling
    if new_fs > old_fs:
        print("Error: target sample rate higher than original")
        return signal, old_fs

    # We can only downsample by an integer factor
    dec_factor = old_fs / new_fs
    # if not dec_factor.is_integer():
    #     print("Error: can only decimate by integer factor")
    #     return signal, old_fs

    # Do decimation
    resampled_signal = scipy.signal.decimate(signal, int(dec_factor))

    return resampled_signal, new_fs


def callback(indata, frames, time, status):
    if status:
        print(status)
    # outdata[:] = indata
    rec = np.squeeze(indata)
    rec, fs = decimate(rec, sample_rate, resample_rate)

    mfcc = python_speech_features.base.mfcc(rec, samplerate=fs, winstep=0.15, winlen=.2, nfft=2048, winfunc=np.hamming)
    dim_1, dim_2 = mfcc.shape
    reshape_dim = dim_1 * dim_2
    mfcc = mfcc.reshape(-1, reshape_dim)
    print(model.predict(mfcc))


with sd.InputStream(samplerate=sample_rate, channels=1, blocksize=int(sample_rate * duration), callback=callback):
    while True:
        pass
    # sd.sleep(int(duration * 1000 * 2))
