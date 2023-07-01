import os
import random
from glob import glob

import numpy as np
import torch
import torch.utils.data
import torchaudio
import torchaudio.transforms as T

"""Multi speaker version"""


class SoundStormDataset(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, ds_dir, all_in_mem: bool = False):
        self.audiopaths = glob(os.path.join(ds_dir, "**/*.wav"), recursive=True)
        print(len(self.audiopaths))

        random.shuffle(self.audiopaths)

        self.all_in_mem = all_in_mem
        if self.all_in_mem:
            self.cache = [self.get_audio(p[0]) for p in self.audiopaths]

    def get_audio(self, filename):
        c = torch.load(filename.replace(".wav", "_hubert.pt")).unsqueeze(0)
        codes = torch.load(filename.replace(".wav", ".code.pt"))
        # print(codes.shape, c.shape)

        assert abs(c.shape[-1] - codes.shape[-1]) < 3, (
            c.shape,
            codes.shape,
            filename,
        )

        lmin = min(c.shape[-1], codes.shape[-1])
        c = c[:, :lmin]
        codes = codes[:, :, :lmin]
        return c.detach(), codes.detach()

    def random_slice(self, c, codes, audio):
        # if spec.shape[1] < 30:
        #     print("skip too short audio:", filename)
        #     return None
        if codes.shape[1] > 800:
            start = random.randint(0, codes.shape[1] - 800)
            end = start + 790
            codes, c = (
                codes[:, start:end],
                c[:, start:end],
            )
            audio = audio[:, start * self.hop_length : end * self.hop_length]

        return c, codes, audio

    def __getitem__(self, index):
        if self.all_in_mem:
            return self.cache[index]
        else:
            return self.get_audio(self.audiopaths[index])
        # print(1)

    def __len__(self):
        return len(self.audiopaths)


class TextAudioCollate:
    def __call__(self, batch):
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].shape[-1] for x in batch]), dim=0, descending=True
        )

        max_c_len = max([x[0].shape[-1] for x in batch])
        min_c_len = min([x[0].shape[-1] for x in batch])

        c_padded = torch.LongTensor(len(batch), min_c_len)
        codes_padded = torch.LongTensor(len(batch), batch[0][1].shape[1], min_c_len)

        c_padded.zero_()
        codes_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            len_raw = row[0].shape[-1]

            if len_raw > min_c_len:
                start = random.randint(0, len_raw - min_c_len)
            else:
                start = 0

            c = row[0]
            c_padded[i, :] = c[:, start : start + min_c_len]

            codes = row[1]
            codes_padded[i, :, :] = codes[:, :, start : start + min_c_len]

        codes_padded = codes_padded.transpose(1, 2)

        return c_padded, codes_padded
