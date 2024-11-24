
import torch
import random

def frequency_masking(data, num_masks=1, mask_width=10):
    for _ in range(num_masks):
        mask_start = random.randint(0, data.size(-1) - mask_width)
        data[:, mask_start:mask_start + mask_width] = 0
    return data

class TimeSeriesTransform:
    def __init__(self, jitter=0.01, crop_size=100, freq_mask=True, num_masks=2):
        self.jitter = jitter
        self.crop_size = crop_size
        self.freq_mask = freq_mask
        self.num_masks = num_masks

    def __call__(self, sample):
        jittered = sample + torch.randn_like(sample) * self.jitter
        if len(jittered) > self.crop_size:
            start = torch.randint(0, len(jittered) - self.crop_size, (1,)).item()
            jittered = jittered[start:start + self.crop_size]
        else:
            jittered = torch.nn.functional.pad(jittered, (0, self.crop_size - len(jittered)))
        if self.freq_mask:
            jittered = frequency_masking(jittered.unsqueeze(0), num_masks=self.num_masks).squeeze(0)
        return jittered
    