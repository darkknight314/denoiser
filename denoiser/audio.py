# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import json
from pathlib import Path
import math
import os
import sys
import soundfile as sf
from torch.nn import functional as F


LENGTH = 10000
def find_audio_files(path, exts=[".raw"], progress=True):
    meta = []
    for root, folders, files in os.walk(path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                meta.append((str(file.resolve()), LENGTH))
    meta.sort()
    return meta


class Audioset:
    def __init__(self, files=None, length=None, stride=None,
                 pad=True, with_path=False, sample_rate=None,
                 channels=None, convert=False):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.length = length
        self.stride = stride or length
        self.with_path = with_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.convert = convert
        for file, file_length in self.files:
            if length is None:
                examples = 1
            elif file_length < length:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length) / self.stride) + 1)
            else:
                examples = (file_length - self.length) // self.stride + 1
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for (file, _), examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            num_frames = 0
            offset = 0
            if self.length is not None:
                offset = self.stride * index
                num_frames = self.length
            # torchaudio.set_audio_backend('soundfile')
            # if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
                # out, sr = torchaudio.load(str(file),
                #                           frame_offset=offset,
                #                           num_frames=num_frames or -1)
            # else:
            #     out, sr = torchaudio.load(str(file), offset=offset, num_frames=num_frames)
            # target_sr = self.sample_rate or sr
            # target_channels = self.channels or out.shape[0]
            # if self.convert:
            #     out = convert_audio(out, sr, target_sr, target_channels)
            # else:
            #     if sr != target_sr:
            #         raise RuntimeError(f"Expected {file} to have sample rate of "
            #                            f"{target_sr}, but got {sr}")
            #     if out.shape[0] != target_channels:
            #         raise RuntimeError(f"Expected {file} to have sample rate of "
            #                            f"{target_channels}, but got {sr}")
            out = torch.from_numpy(sf.read(str(file), samplerate=5000, channels=1, subtype="FLOAT")[0])
            if num_frames:
                out = F.pad(out, (0, num_frames - out.shape[-1]))
            if self.with_path:
                return out, file
            else:
                return out


if __name__ == "__main__":
    meta = []
    for path in sys.argv[1:]:
        meta += find_audio_files(path)
    json.dump(meta, sys.stdout, indent=4)
