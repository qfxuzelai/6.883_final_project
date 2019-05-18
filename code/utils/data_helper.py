import os
import numpy as np
from random import sample
import torch
import torchvision.transforms as transforms

def get_instr_label(instr):
    if instr == 'accordion':
        return 0
    elif instr == 'acoustic_guitar':
        return 1
    elif instr == 'cello':
        return 2
    elif instr == 'flute':
        return 3
    elif instr == 'saxophone':
        return 4
    elif instr == 'trumpet':
        return 5
    elif instr == 'violin':
        return 6
    elif instr == 'xylophone':
        return 7
    else:
        return None

def load_data(spec_dir):
    # load spectrums
    specs = {}
    instr_class = os.listdir(spec_dir)
    for instr in instr_class:
        specs[instr] = {}
        instr_dir = os.path.join(spec_dir, instr)
        audios = os.listdir(instr_dir)
        for audio in audios:
            specs[instr][audio] = []
            audio_dir = os.path.join(instr_dir, audio)
            frags = os.listdir(audio_dir)
            for frag in frags:
                spec_path = os.path.join(audio_dir, frag)
                spec = np.load(spec_path)
                specs[instr][audio].append(spec)
    return specs

def get_input(specs, bs=4, N=2):
    specs_input = np.zeros((bs, N, 512, 256), dtype=np.complex64)
    labels_input = np.zeros((bs, N), dtype=np.int64)
    for batch in range(bs):
        instr_sampled = sample(specs.keys(), N)
        for cnt in range(N):
            instr = instr_sampled[cnt]
            item = sample(specs[instr].keys(), 1)[0]
            frag = sample(range(len(specs[instr][item])), 1)[0]
            specs_input[batch, cnt, :, :] = specs[instr][item][frag]
            labels_input[batch, cnt] = get_instr_label(instr_sampled[cnt])
    return [specs_input, labels_input]

def img_pretreat(prob_list):
    max_pos = np.argmax(prob_list)
    max_val = np.max(prob_list)
    num = max_pos
    if max_pos == 4:
        if max_val<1 and max_val>0.95 and prob_list[3]<0.9 and prob_list[3]>0.85:
            num = 3
        elif max_val<0.15 and prob_list[6] > 1/8*max_val:
            num = 6
    return num