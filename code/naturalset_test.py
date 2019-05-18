import os
import math
import torch
import librosa
import progressbar
import numpy as np
import soundfile as sf

from u_net import u_net
from utils.audio_helper import spec_normalization, inv_sample

# set path
AUDIO_DIR = '../dataset/naturalset'
OUTPUT_DIR = '../dataset/naturalset_result'
MODEL_DIR = '../dataset/model/mask_net_param.pkl'

# set params
ORIGINAL_RATE = 44100
SAMPLE_RATE = 11000
FRAG_LENGTH = 66302
WIN_LENGTH = 1022
HOP_LENGTH = 256
INSTRUMENTS = ['accordion', 'acoustic_guitar', 'cello', 'flute', 'saxophone', 'trumpet', 'violin', 'xylophone']

def get_instr_class(audio_name):
    instr = []
    for idx, instrument in enumerate(INSTRUMENTS):
        if instrument in audio_name:
            instr.append(idx)
    return np.array(instr)

# load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mask_net = u_net(16, 8).to(device)
mask_net.load_state_dict(torch.load(MODEL_DIR, map_location=device))
mask_net.eval()

# load data
audios = os.listdir(AUDIO_DIR)
for audio in audios:
    # get path
    audio_path = os.path.join(AUDIO_DIR, audio)
    output_path_1 = os.path.join(OUTPUT_DIR, audio)[:-4] + '_seg1.wav'
    output_path_2 = os.path.join(OUTPUT_DIR, audio)[:-4] + '_seg2.wav'
    
    # get instrument class
    instr = get_instr_class(audio)
    
    # load audio
    wave, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    frag_num = math.ceil(len(wave) / FRAG_LENGTH)
    waves = np.zeros((2, frag_num * FRAG_LENGTH))

    # init progressbar
    print(audio[:-4])
    bar = progressbar.ProgressBar(maxval=frag_num)
    bar.start()

    for cnt in range(frag_num):
        # update progressbar
        bar.update(cnt + 1)

        # split into fragments
        if cnt != frag_num - 1:
            frag = wave[cnt * FRAG_LENGTH:(cnt + 1) * FRAG_LENGTH]
        else:
            frag = wave[cnt * FRAG_LENGTH:]
            frag = np.pad(frag, (0, FRAG_LENGTH - len(frag)), 'constant', constant_values=0)
        
        # predict mask
        spec = librosa.stft(frag, n_fft=WIN_LENGTH, hop_length=HOP_LENGTH, center=False).reshape((1,1,512,256))
        spec_input = torch.from_numpy(spec_normalization(spec)).to(device)
        masks_output = mask_net.forward(spec_input)
        
        # get seperated spectrums
        masks_predicted = masks_output[:, instr]
        masks = inv_sample(np.rint(masks_predicted.detach().cpu().numpy()))
        specs = spec * masks
        
        # back to audio
        waves[0, cnt * FRAG_LENGTH:(cnt + 1) * FRAG_LENGTH] = librosa.istft(specs[0, 0], hop_length=HOP_LENGTH, center=False)
        waves[1, cnt * FRAG_LENGTH:(cnt + 1) * FRAG_LENGTH] = librosa.istft(specs[0, 1], hop_length=HOP_LENGTH, center=False)
    
    # save audio
    sf.write(output_path_1, librosa.resample(waves[0, :len(wave)], SAMPLE_RATE, ORIGINAL_RATE), ORIGINAL_RATE)
    sf.write(output_path_2, librosa.resample(waves[1, :len(wave)], SAMPLE_RATE, ORIGINAL_RATE), ORIGINAL_RATE)

    bar.finish()
