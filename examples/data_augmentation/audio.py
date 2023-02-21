import numpy as np
import librosa
import librosa.display
import PIL
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import noisereduce as nr
import torchaudio.transforms as t
import torchaudio.sox_effects as sox

class SpecAugment:
    def __init__(self, labeled_dataset, time_stretch_param=80, freq_mask_param=80, time_mask_param=80): 
        self.dataset = labeled_dataset
        self.nargs = 2
        self.specaug = torch.nn.Sequential(
            t.TimeStretch(time_stretch_param, fixed_rate=True),
            t.FrequencyMasking(freq_mask_param),
            t.TimeMasking(time_mask_param)
        )
        
    def __call__(self, img, ix):
        x, sr = self.dataset.get_audio(ix)
        # x = np.array(x.get_array_of_samples()).astype(np.float32)
        spec = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128)
        spec = torch.Tensor(spec).unsqueeze(0)
        spec = self.specaug(spec).numpy().squeeze()
        return get_mel_spec(S=spec, sr=sr).resize(img.size)

class RandomPass:
    def __init__(self, labeled_dataset, lowpass_p=0.5): 
        self.dataset = labeled_dataset
        self.nargs = 2
        self.lowpass_p = lowpass_p
        
    def __call__(self, img, ix):
        # get audio
        x, sr = self.dataset.get_audio(ix)
        x = torch.Tensor(x).unsqueeze(0)

        # sample low vs. high pass randomly, and a strength
        random_effect = [[
            np.random.choice(['lowpass', 'highpass'], p=[self.lowpass_p, 1-self.lowpass_p]),
            "-1",
            f"{np.random.randint(1, 100) * 10}",
        ]]
        x, sr = sox.apply_effects_tensor(x, sr, random_effect)
        x = x.numpy().squeeze()
        return get_mel_spec(x=x, sr=sr).resize(img.size)

class NoiseReduceAugment:
    """
    Using the noisereduce package, reduce the noise in an audio clip against a reference 'empty' clip from the same location (microphone).
    
    Assumes that the dataset has a empty_indices attribute that is a dict of the form {'train': [list of indices], ...},
        where the indices are the indices of empty (no object) examples in the dataset.
    """
    def __init__(self, labeled_dataset, stationary=True, freq_mask_smooth_hz=500): 
        assert labeled_dataset.dataset_name == "birdcalls"
        assert hasattr(labeled_dataset, 'location_array')
        
        self.dataset = labeled_dataset
        self.nargs = 2

        # prepare a list of empty clips to reduce noise against
        if 'train' in self.dataset.empty_indices:
            self.empty_indices = self.dataset.empty_indices['train']
        else:
            self.empty_indices = np.array([], dtype=int)

        self.classes_to_not_augment = [0] # 0 is the empty class for birdcalls

        self.freq_mask_smooth_hz = freq_mask_smooth_hz
        self.stationary = stationary
    
    def __call__(self, img, ix):
        dataset = self.dataset

        # don't transform empty images (on the labeled set)
        y = dataset.y_array[ix].item()
        if y in self.classes_to_not_augment: return img # IDENTITY CASE

        x, sr = dataset.get_audio(ix)

        # sample the empty file
        (bg_x, _), _ = self.get_empty_from_loc(dataset.location_array[ix])
        if bg_x is None: return img # IDENTITY CASE
                
        # reduce noise
        reduced_noise = nr.reduce_noise(y=x, sr=sr, y_noise=bg_x, freq_mask_smooth_hz=self.freq_mask_smooth_hz, stationary=self.stationary)
        
        # convert to image
        return get_mel_spec(x=reduced_noise, sr=sr).resize(img.size)
   
    def __repr__(self) -> str:
        format_string = super().__repr__()[:-1]
        for k in ['stationary', 'freq_mask_smooth_hz']:
            if getattr(self, k): format_string += f" {k}={getattr(self, k)}"
        format_string += ")"
        return format_string
    
    def get_empty_from_loc(self, loc):
        labeled_mask = (self.dataset.location_array[self.empty_indices] == loc)
        assert len(labeled_mask) == len(self.empty_indices)
        # empty indices are numpy arrays, so make masks numpy
        labeled_mask = labeled_mask.numpy()

        # return no image if mask all false
        if np.all(labeled_mask == 0):
            return None
        
        empty_ix = np.random.choice(self.empty_indices[labeled_mask])
        return self.dataset.get_audio(empty_ix), empty_ix

def get_mel_spec(x=None, sr=None, S=None, show_colorbar=False, normalize=True):
    """Return a PIL image of a mel spectrogram for a waveform"""
    assert (x is not None) ^ (S is not None)
    if S is None:
        if normalize: x = librosa.util.normalize(x)
        S = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128)
    fig, ax = plt.subplots(figsize=(5,5))
    plt.tight_layout()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax, cmap=None)    # convert plot to img with no padding
    if show_colorbar: fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.axis('off')
    ax.set_position((0, 0, 1, 1))
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    fig.canvas.draw()
    img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    plt.close()
    return img
