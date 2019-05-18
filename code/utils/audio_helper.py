import librosa
import numpy as np

def mix_specs(specs):
    spec_mixed = np.sum(specs, axis=1, keepdims=True)
    return spec_mixed

def log_sample(spec):
    index = np.logspace(0, 9, num=256, base=2.0, dtype='int64') - 1
    iline = np.linspace(0, 221, num=222, dtype='int64')
    index[0:222] = iline
    spec = spec[:, :, index, :]
    return spec

def get_mask(specs):
    mask = np.zeros((specs.shape[0], specs.shape[1], 256, 256), dtype=np.float32)
    dominant_component = np.argmax(np.abs(specs), axis=1)
    for cnt in range(specs.shape[1]):
        mask[:, cnt:cnt+1, :, :] = log_sample((dominant_component == cnt)[:, np.newaxis, :, :])
    return mask

def spec_normalization(spec):
    spec = log_sample(spec)
    audio_input = librosa.amplitude_to_db(np.abs(spec))
    return audio_input

def inv_sample(mask):
    index = np.logspace(0, 9, num=256, base=2.0, dtype='int64') - 1
    iline = np.linspace(0, 221, num=222, dtype='int64')
    index[0:222] = iline
    mask_isampled = np.zeros((mask.shape[0], mask.shape[1], 512, 256), dtype=np.float32)
    mask_isampled[:, :, index, :] = mask
    idx_pre = 0
    for idx in range(0, 512):
        if idx in index:
            if idx > idx_pre + 1:
                idx_mid = (idx_pre + idx) // 2
                mask_isampled[:, :, idx_pre+1:idx_mid, :] = mask_isampled[:, :, idx_pre:idx_pre+1, :].repeat(idx_mid-idx_pre-1, axis=2)
                mask_isampled[:, :, idx_mid:idx, :] = mask_isampled[:, :, idx:idx+1, :].repeat(idx-idx_mid, axis=2)
            idx_pre = idx
    return mask_isampled

def spec_to_frag(spec, hop_len=256, win_len=1022):
    frag = librosa.istft(spec, hop_length=hop_len, win_length=win_len, center=False)
    return frag
