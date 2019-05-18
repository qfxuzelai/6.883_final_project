import os
import math
import librosa
import numpy as np

INSTR_DIR = '../dataset/trainset/audio'
SPEC_DIR = '../dataset/trainset/spectrum'
SAMPLE_RATE = 11000
FRAG_LENGTH = 66302
WIN_LENGTH = 1022
HOP_LENGTH = 256

# mkdir if not exist
if os.path.exists(SPEC_DIR) == False:
    os.mkdir(SPEC_DIR)

instr_class = os.listdir(INSTR_DIR)
for instr in instr_class:
    # mkdir if not exist
    spec_dir = os.path.join(SPEC_DIR, instr)
    if os.path.exists(spec_dir) == False:
        os.mkdir(spec_dir)
    
    audio_dir = os.path.join(INSTR_DIR, instr)
    audios = os.listdir(audio_dir)
    for audio in audios:
        # mkdir if not exist
        num = audio[0:-4]
        dst_dir = os.path.join(spec_dir, num)
        if os.path.exists(dst_dir) == False:
            os.mkdir(dst_dir)

        # load audio
        audio_path = os.path.join(audio_dir, audio)
        wave, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        frag_num = math.floor(len(wave)/FRAG_LENGTH)
        for cnt in range(frag_num):
            # STFT
            frag = wave[cnt*FRAG_LENGTH:(cnt+1)*FRAG_LENGTH]
            spec = librosa.stft(frag, n_fft=WIN_LENGTH, hop_length=HOP_LENGTH, center=False)

            # save spectrum
            spec_path = os.path.join(dst_dir, str(cnt))
            np.save(spec_path, spec)
        
        print('audio %2s in %s processed.' % (num, instr))
