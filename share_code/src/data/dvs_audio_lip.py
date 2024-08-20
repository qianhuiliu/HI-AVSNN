import os
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
import glob
import json
from torchaudio_augmentations import ComposeMany
from torchaudio_augmentations import Gain
from torchaudio_augmentations import Noise
from torchaudio_augmentations import PolarityInversion
from torchaudio_augmentations import RandomApply
from torchaudio_augmentations import Reverb
from .cvtransforms import *

# code from Tan et al.
def events_to_voxel_grid_pytorch(events, num_bins, width, height, device):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    :param device: device to use to perform computations
    :return voxel_grid: PyTorch event tensor (on the device specified)
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    with torch.no_grad():

        events_torch = torch.from_numpy(events).float()
        events_torch = events_torch.to(device)

        voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32, device=device)
        if events_torch.shape[0] == 0:
            return voxel_grid

        voxel_grid = voxel_grid.flatten()

        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = events_torch[-1, 0]
        first_stamp = events_torch[0, 0]
        deltaT = float(last_stamp - first_stamp)

        if deltaT == 0:
            deltaT = 1.0

        events_torch[:, 0] = (num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
        ts = events_torch[:, 0]
        xs = events_torch[:, 1].long()
        ys = events_torch[:, 2].long()
        pols = events_torch[:, 3].float()
        pols[pols == 0] = -1  # polarity should be +1 / -1

        tis = torch.floor(ts)
        tis_long = tis.long()
        dts = ts - tis
        vals_left = pols * (1.0 - dts.float())
        vals_right = pols * dts.float()

        valid_indices = tis < num_bins
        valid_indices &= tis >= 0
        voxel_grid.index_add_(dim=0,
                              index=xs[valid_indices] + ys[valid_indices]
                                    * width + tis_long[valid_indices] * width * height,
                              source=vals_left[valid_indices])

        valid_indices = (tis + 1) < num_bins
        valid_indices &= tis >= 0

        voxel_grid.index_add_(dim=0,
                              index=xs[valid_indices] + ys[valid_indices] * width
                                    + (tis_long[valid_indices] + 1) * width * height,
                              source=vals_right[valid_indices])

        voxel_grid = voxel_grid.view(num_bins, height, width)

    return voxel_grid

def events_to_voxel_all(events, frame_nums, seq_len, num_bins, width, height, device):
    voxel_len = min(seq_len, frame_nums) * num_bins
    voxel_grid_all = np.zeros((num_bins * seq_len, 1, height, width))
    voxel_grid = events_to_voxel_grid_pytorch(events, voxel_len, width, height, device)
    voxel_grid = voxel_grid.unsqueeze(1).cpu().numpy()
    voxel_grid_all[:voxel_len] = voxel_grid
    return voxel_grid_all


class MultiDataset(Dataset):
    def __init__(self, phase, args):
        self.labels = sorted(os.listdir(os.path.join(args.event_root, phase)))
        self.length = args.seq_len
        self.phase = phase
        self.args = args
        self.noise = ['']
        self.file_list = []
        
        self.file_list.extend(sorted(glob.glob(os.path.join(args.event_root, phase, '*', '*.mp3'))))
        print (len(self.file_list))
        with open('/home/qianhui/lip-reading/share_code/src/data/frame_nums.json', 'r') as f:
            self.frame_nums = json.load(f)
            
        if self.phase == "train": 
            transforms = [
                RandomApply([PolarityInversion()], p=0.8),
                RandomApply([Noise(0.0001, 0.9)], 0.1),
                RandomApply([Gain()], p=0.3),
                RandomApply([Reverb(sample_rate=44100)], p=0.6),
            ]
            self.audio_transf = ComposeMany(transforms, num_augmented_samples=1)
        else:
            self.audio_transf = lambda x: x.unsqueeze(dim=0)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        word = self.file_list[index].split('/')[-2]
        person = self.file_list[index].split('/')[-1][:-4]
        frame_num = self.frame_nums[self.phase][word][int(person)]
        #double channel noisy 0, single channel clean 1
        try:
            audio_input, sr = torchaudio.load(self.file_list[index])
            if audio_input.shape[0]>1:
                part = -1
                assert (sr==48000)
                audio_input = torchaudio.functional.resample(audio_input, orig_freq=sr, new_freq=44100)
                sr = 44100
            else:
                part = 0
                assert (sr==44100)
            audio_input = audio_input[0:1,]
            audio_input = audio_input.type(torch.FloatTensor)
            
            event_file = self.file_list[index].replace("audio", "DVS")
            event_file = event_file.replace(".mp3", ".npy")
            if  '_noise' in event_file:
                event_file = event_file.replace(self.phase+'_noise', self.phase)
            else:
                for p in range(1,len(self.noise)):
                    if self.phase+self.noise[p] in event_file:
                        event_file = event_file.replace(self.phase+self.noise[p], self.phase)
                        part = part + p
            events_input = np.load(event_file)
        except:
            print(self.file_list[index])

        ## Visual ##
        events_input = events_input[np.where((events_input['x'] >= 16) & (events_input['x'] < 112) & (events_input['y'] >= 16) & (events_input['y'] < 112))]
        events_input['x'] -= 16
        events_input['y'] -= 16
        events_input['x'] = events_input['x']//2
        events_input['y'] = events_input['y']//2
        t, x, y, p = events_input['t'], events_input['x'], events_input['y'], events_input['p']
        events_input = np.stack([t, x, y, p], axis=-1)
        
        event_voxel_low = events_to_voxel_all(events_input, frame_num, self.length, self.args.num_bins[0],48, 48, device='cpu') # (30*num_bins[0], 96, 96)

        if self.phase == 'train':
            event_voxel_low = RandomCrop(event_voxel_low, (44, 44))
            event_voxel_low = HorizontalFlip(event_voxel_low)
        else:
            event_voxel_low  = CenterCrop(event_voxel_low, (44, 44)) 
        events = torch.FloatTensor(event_voxel_low)
       
        # ## Audio ##
        length = int(1.2 * sr)
        if length < len(audio_input[0]):
            audio_input = audio_input[0:1,:length]
        elif length > len(audio_input[0]):
            audio_input = torch.nn.functional.pad(audio_input, (0, length - len(audio_input[0])), "constant")    
        audio_input = audio_input / torch.max(torch.abs(audio_input))
           
        audio_input = self.audio_transf(audio_input).squeeze(dim=0)   
        spec = torchaudio.compliance.kaldi.fbank(audio_input, frame_length = 120, frame_shift = 40, num_mel_bins=40, sample_frequency=sr)
        
        num_v_frames = event_voxel_low.shape[0]
        num_a_frames = spec.shape[0]
        assert (num_v_frames==num_a_frames)

        return spec, events, self.labels.index(word), part
