import sys 
import os
 
import string
import argparse
import os
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
 
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts


class CTCLabelConverterForBaiduWarpctc(object):
    """ Convert between text-label and text-index for baidu warpctc """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

import os
import sys
import re
import six
import math
import lmdb
import torch

from natsort import natsorted
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
# torch 2.9+ no longer exposes torch._utils._accumulate; provide a fallback.
try:  # pragma: no cover
    from torch._utils import _accumulate  # type: ignore
except ImportError:  # pragma: no cover
    from itertools import accumulate as _it_accumulate

    def _accumulate(iterable):
        for x in _it_accumulate(iterable):
            yield x
import torchvision.transforms as transforms


class Batch_Balanced_Dataset(object):

    def __init__(self, opt):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate, pin_memory=True)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = next(data_loader_iter)
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = next(self.dataloader_iter_list[i])
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts


def hierarchical_dataset(root, opt, select_data='/'):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    print(dataset_log)
    dataset_log += '\n'
    for dirpath, dirnames, filenames in os.walk(root+'/'):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                dataset = LmdbDataset(dirpath, opt)
                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log


class LmdbDataset(Dataset):

    def __init__(self, root, opt):

        self.root = root
        self.opt = opt
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

            if self.opt.data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                use --data_filtering_off and only evaluate on alphabets and digits.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192

                And if you want to evaluate them with the model trained with --sensitive option,
                use --sensitive and --data_filtering_off,
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')

                    if len(label) > self.opt.batch_max_length:
                        # print(f'The length of the label is longer than max_length: length
                        # {len(label)}, {label} in dataset {self.root}')
                        continue

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    out_of_char = f'[^{self.opt.character}]'
                    if re.search(out_of_char, label.lower()):
                        continue

                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.opt.rgb:
                    img = Image.open(buf).convert('RGB')  # for color image
                else:
                    img = Image.open(buf).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'

            if not self.opt.sensitive:
                label = label.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{self.opt.character}]'
            label = re.sub(out_of_char, '', label)

        return (img, label)


class RawDataset(Dataset):

    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            if self.opt.rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
            else:
                img = Image.open(self.image_path_list[index]).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import sys
sys.path.append('../')

import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output
 
class TPS_SpatialTransformerNetwork(nn.Module):
    """ Rectification Network of RARE, namely TPS based STN """

    def __init__(self, F, I_size, I_r_size, I_channel_num=1):
        """ Based on RARE TPS
        input:
            batch_I: Batch Input Image [batch_size x I_channel_num x I_height x I_width]
            I_size : (height, width) of the input image I
            I_r_size : (height, width) of the rectified image I_r
            I_channel_num : the number of channels of the input image I
        output:
            batch_I_r: rectified image [batch_size x I_channel_num x I_r_height x I_r_width]
        """
        super(TPS_SpatialTransformerNetwork, self).__init__()
        self.F = F
        self.I_size = I_size
        self.I_r_size = I_r_size  # = (I_r_height, I_r_width)
        self.I_channel_num = I_channel_num
        self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def forward(self, batch_I):
        batch_C_prime = self.LocalizationNetwork(batch_I)  # batch_size x K x 2
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)  # batch_size x n (= I_r_width x I_r_height) x 2
        build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2])
        
        if torch.__version__ > "1.2.0":
            batch_I_r = F.grid_sample(batch_I, build_P_prime_reshape, padding_mode='border', align_corners=True)
        else:
            batch_I_r = F.grid_sample(batch_I, build_P_prime_reshape, padding_mode='border')

        return batch_I_r

class Attention(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)

    def _char_to_onehot(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, batch_H, text, is_train=True, batch_max_length=25):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.

        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device))

        if is_train:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
            probs = self.generator(output_hiddens)

        else:
            targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes


class VGG_FeatureExtractor(nn.Module):
    """ FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf) """

    def __init__(self, input_channel, output_channel=512):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64x16x50
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 128x8x25
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),  # 256x8x25
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 256x4x25
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),  # 512x4x25
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 512x2x25
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True))  # 512x1x24

    def forward(self, input):
        return self.ConvNet(input)


class RCNN_FeatureExtractor(nn.Module):
    """ FeatureExtractor of GRCNN (https://papers.nips.cc/paper/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf) """

    def __init__(self, input_channel, output_channel=512):
        super(RCNN_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64 x 16 x 50
            GRCL(self.output_channel[0], self.output_channel[0], num_iteration=5, kernel_size=3, pad=1),
            nn.MaxPool2d(2, 2),  # 64 x 8 x 25
            GRCL(self.output_channel[0], self.output_channel[1], num_iteration=5, kernel_size=3, pad=1),
            nn.MaxPool2d(2, (2, 1), (0, 1)),  # 128 x 4 x 26
            GRCL(self.output_channel[1], self.output_channel[2], num_iteration=5, kernel_size=3, pad=1),
            nn.MaxPool2d(2, (2, 1), (0, 1)),  # 256 x 2 x 27
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 2, 1, 0, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True))  # 512 x 1 x 26

    def forward(self, input):
        return self.ConvNet(input)


class ResNet_FeatureExtractor(nn.Module):
    """ FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf) """

    def __init__(self, input_channel, output_channel=512):
        super(ResNet_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])

    def forward(self, input):
        return self.ConvNet(input)


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        # else:
        #     print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_image(bbox, image):
    if np.all(bbox) > 0:
        try:
            word = crop(bbox, image)
            color_coverted = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('Result', color_coverted)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return Image.fromarray(color_coverted).convert('L') 
        except Exception as e:
            print(e)
            return None

import pandas as pd 

def interpret_labels(opt, input_image):

    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    model.load_state_dict(torch.load(opt.saved_model, map_location=device, weights_only=False))

    # predict
    model.eval()
    with torch.no_grad():
        # prep image for input into model
        transform = ResizeNormalize((opt.imgW, opt.imgH))
        image_tensors = [transform(input_image)]
        image_tensors = torch.cat(image_tensors, 0)

        # only dealing with 1 at a time, so adjust dimensions
        batch_size = 1
        image_tensors = image_tensors.unsqueeze(dim=0)
        image = image_tensors.to(device)

        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            # preds_index = preds_index.view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)

        else:
            preds = model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        for pred, pred_max_prob in zip(preds_str, preds_max_prob): # only loops once 
            if 'Attn' in opt.Prediction:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
            weighted = pred_max_prob.apply_(lambda x: (x*0.1))
            weight_sum = weighted.cumsum(dim=0)[-1].item()
            confidence_score = weight_sum / (len(pred_max_prob) * 0.1)
            # print(f'{pred:25s}\t {confidence_score:0.4f}')
            return pred, confidence_score

def prep_read_labels(transformation, feature_extraction, sequence_modeling, prediction, saved_model):
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args(["--Transformation", transformation, "--FeatureExtraction", feature_extraction,
                             "--SequenceModeling", sequence_modeling, "--Prediction", prediction,
                             "--saved_model", saved_model])

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    return opt

def generate_crop_patch(bbox, image):
    """
    Generate a cropped image patch based on the bounding box.

    Args:
        bbox (list[int]): Bounding box coordinates [x1, y1, x2, y2, x3, y3, x4, y4].
        image (numpy.ndarray): The input image.

    Returns:
        PIL.Image or None: Cropped image patch or None if the cropping fails.
    """
    # Convert the bounding box to integer coordinates
    bbox = np.array(bbox, dtype=int).reshape(4, 2)

    if np.any(bbox < 0):  # Check for invalid coordinates
        print(f"Coordinates for bbox : {bbox} are incorrrect")
        return None

    try:
        # Calculate the bounding rectangle
        x_min = np.min(bbox[:, 0])
        y_min = np.min(bbox[:, 1])
        x_max = np.max(bbox[:, 0])
        y_max = np.max(bbox[:, 1])

        # Crop the image
        cropped_img = image[y_min:y_max, x_min:x_max]
        if cropped_img.size == 0:  # Handle invalid crops
            print(f"Cropped bounding box: {bbox} results in no image")
            return None
        # Convert to grayscale and prepare for model input
        cropped_img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        return Image.fromarray(cropped_img_gray).convert('L')  # Convert to PIL image
    except Exception as e:
        print(f"Error cropping bbox {bbox}: {e}")
        return None
 
import re
from fuzzywuzzy import fuzz
import re

def normalize_numeric_text(text):
    """
    Normalize text by replacing common letter and symbol substitutions with numbers.
    
    Args:
        text (str): The input text to normalize.

    Returns:
        str: The normalized text.
    """
    substitutions = {
        'l': '1',  # Lowercase 'l' to '1'
        'L': '1',  # Uppercase 'L' to '1'
        'I': '1',  # Uppercase 'I' to '1'
        'i': '1',  # Lowercase 'i' to '1'
        '|': '1',  # Vertical bar to '1'
        '!': '1',  # Exclamation mark to '1'
        '/': '1',  # Slash to '1'
        '\\': '1', # Backslash to '1'
        'o': '0',  # Lowercase 'o' to '0'
        'O': '0',  # Uppercase 'O' to '0'
        'Q': '0',  # Uppercase 'Q' to '0'
        'D': '0',  # Uppercase 'D' to '0'
        'G': '6',  # Uppercase 'G' to '6'
        'S': '5',  # Uppercase 'S' to '5'
        '$': '5',  # Dollar sign to '5'
        'B': '8',  # Uppercase 'B' to '8'
        'g': '9',  # Lowercase 'g' to '9'
        'q': '9',  # Lowercase 'q' to '9'
        'Z': '2',  # Uppercase 'Z' to '2'
        'z': '2',  # Lowercase 'z' to '2'
        ' ': '',   # Remove spaces
        '-': '',   # Remove dashes
        '_': '',   # Remove underscores
        '.': '',   # Remove periods
        ',': '',   # Remove commas
        '~': '',   # Remove tilde
    }

    return ''.join(substitutions.get(char, char) for char in text)

def validate_bounding_boxes(
    bboxes,
    results,
    valid_room_labels,
    confidence_threshold=0.85,
    fuzzy_threshold=75,
    extra_valid_words=None,
):
    """
    Validate bounding boxes based on confidence scores and predicted text with proper checks.

    Upgrades:
    - `extra_valid_words` lets you whitelist non-room semantic tokens (e.g., 'stairs', 'elev').
      These will be accepted without fuzzy matching against `valid_room_labels`.

    Args:
        bboxes (list[list[int]]): List of bounding box coordinates.
        results (list[dict]): List of results with 'bbox', 'text', and 'confidence' fields.
        valid_room_labels (list[str]): List of valid room labels.
        confidence_threshold (float): Minimum confidence score to accept a bounding box.
        fuzzy_threshold (int): Minimum fuzzy matching score to accept a text as valid.
        extra_valid_words (list[str] or None): Additional words to accept verbatim (e.g., transitions).

    Returns:
        list[list[int]]: Valid bounding boxes (same order as valid_results).
        list[dict]: Valid results (subset of `results`, aligned with returned bboxes).
    """
    from fuzzywuzzy import fuzz  # local import to avoid issues if not used elsewhere

    if extra_valid_words is None:
        extra_valid_words = []
    # Normalize whitelist to lowercase for robust checks
    extra_valid_words_norm = {w.strip().lower() for w in extra_valid_words if isinstance(w, str)}

    valid_bboxes = []
    valid_results = []
    print(f"Verifying {len(results)} labels... ")

    for i, (bbox, result) in enumerate(zip(bboxes, results), start=1):
        raw_text = result['text']
        confidence = result['confidence']

        if confidence < confidence_threshold:
            print(f"Failed (Low Confidence): {i}/{len(results)}: bbox: {bbox} "
                  f"with text: '{raw_text}' has confidence: {confidence:.2f} < {confidence_threshold}")
            continue

        # Case 1: Numeric text (valid as-is)
        if raw_text.isdigit():
            valid_bboxes.append(bbox)
            valid_results.append(result)
            continue

        # Case 1.5: Explicitly whitelisted words (e.g., stairs/elev)
        if raw_text.lower() in extra_valid_words_norm:
            valid_bboxes.append(bbox)
            valid_results.append(result)
            continue

        # Case 2: Single-character strings (normalize and validate)
        if len(raw_text) == 1:
            normalized_text = normalize_numeric_text(raw_text)
            if normalized_text.isdigit():
                result['text'] = normalized_text
                valid_bboxes.append(bbox)
                valid_results.append(result)
            else:
                print(f"Failed (Invalid Single Character): {i}/{len(results)}: "
                      f"Single character '{raw_text}' is invalid.")
            continue

        # Case 3: Mixed alphanumeric text
        if any(char.isdigit() for char in raw_text) and any(char.isalpha() for char in raw_text):
            normalized_text = normalize_numeric_text(raw_text)
            if normalized_text.isdigit():
                result['text'] = normalized_text
                valid_bboxes.append(bbox)
                valid_results.append(result)
            else:
                print(f"Failed (Invalid Mixed): {i}/{len(results)}: Mixed text "
                      f"'{raw_text}' (normalized: '{normalized_text}') did not become valid.")
            continue

        # Case 4: Purely alphabetical text (fuzzy matching against room labels OR exact in whitelist)
        if raw_text.isalpha():
            raw_lower = raw_text.lower()
            if raw_lower in extra_valid_words_norm:
                valid_bboxes.append(bbox)
                valid_results.append(result)
                continue

            is_valid = False
            for label in valid_room_labels:
                match_score = fuzz.ratio(raw_lower, label.lower())
                if match_score >= fuzzy_threshold:
                    is_valid = True
                    break

            if is_valid:
                valid_bboxes.append(bbox)
                valid_results.append(result)
            else:
                print(f"Failed (Invalid Word): {i}/{len(results)}: Word '{raw_text}' "
                      f"did not match any valid room labels (fuzzy score < {fuzzy_threshold}).")
            continue

        # Default: Invalid
        print(f"Failed (Invalid Category): {i}/{len(results)}: '{raw_text}' with bbox: {bbox} "
              f"does not fit any valid category.")

    return valid_bboxes, valid_results

def interpret_bboxes(image_path, bbox_text_file, results_dir):
    """
    Interpret text within bounding boxes, validate results (rooms + transitions), and save the valid ones.

    Upgrades:
    - Accept 'stairs' and 'elev' (plus common variants) as valid **transition** nodes.
    - Return a new list `transition_bboxes` (in addition to room/hallway/outside).
      NOTE: The *text* of each transition bbox remains in the saved results file for downstream
      naming like 'stairs_X' or 'elevator_X'.

    Returns:
        tuple: (room_bboxes, hallway_bboxes, outside_bboxes, transition_bboxes, result_file_path)
    """
    import os

    image_name_no_ext = os.path.splitext(os.path.basename(image_path))[0]

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    interpreter_detect_dir = os.path.join(results_dir, "interpreter_detect")
    os.makedirs(interpreter_detect_dir, exist_ok=True)

    interpret_img_dir = os.path.join(interpreter_detect_dir, f"{image_name_no_ext}")
    os.makedirs(interpret_img_dir, exist_ok=True)
    result_file_path = os.path.join(interpret_img_dir, "room_labels.txt")

    # Load bounding boxes from the text file
    # Format can be: "x1,y1,x2,y2,x3,y3,x4,y4" (old) or "x1,y1,x2,y2,x3,y3,x4,y4 | text" (new)
    try:
        bboxes = []
        with open(bbox_text_file, "r") as bbox_file:
            for line in bbox_file:
                line = line.strip()
                if not line:
                    continue
                # Split by " | " to separate coordinates from text (if present)
                if " | " in line:
                    coord_part = line.split(" | ")[0]
                else:
                    coord_part = line
                # Parse coordinates
                bbox = list(map(int, coord_part.split(",")))
                bboxes.append(bbox)
    except Exception as e:
        raise ValueError(f"Error reading bounding boxes from {bbox_text_file}: {e}")

    interp_model = prep_read_labels(
        "None", "VGG", "BiLSTM", "CTC",
        "/data1/yansari/cad2map/Tesseract++/Model_weights/None-VGG-BiLSTM-CTC.pth"
    )

    # OCR each bbox
    results = []
    failed_labels = []
    for i, bbox in enumerate(bboxes):
        cropped_patch = generate_crop_patch(bbox, image)

        if cropped_patch is None:
            failed_labels.append(bbox)
            print(f"Failed Cropbox: Bbox: {bbox}")
            continue

        pred, score = interpret_labels(interp_model, cropped_patch)
        results.append({'bbox': bbox, 'text': pred, 'confidence': score})

    # Valid label sets
    valid_room_labels = [
        'kitchen', 'bedroom', 'living room', 'bathroom', 'dining room', 'family room',
        'guest room', 'study', 'office', 'den', 'lounge', 'playroom', 'media room', 'hall', 'NA'
    ]
    # NEW: transitions whitelist (accepted verbatim in validation)
    valid_transition_words = [
        'stairs', 'stair', 'staircase',
        'elev', 'elevator', 'lift'
    ]

    # Validate (now keeps transitions too)
    bboxes, results = validate_bounding_boxes(
        bboxes,
        results,
        valid_room_labels,
        confidence_threshold=0.85,
        fuzzy_threshold=75,
        extra_valid_words=valid_transition_words
    )

    # Separate by semantic class
    room_bboxes = []
    hallway_bboxes = []
    outside_bboxes = []
    transition_bboxes = []  # NEW

    # Normalize helper for transitions to standard keys (optional, kept as-is for now)
    transition_aliases = {
        'stairs': 'stairs', 'stair': 'stairs', 'staircase': 'stairs',
        'elev': 'elevator', 'elevator': 'elevator', 'lift': 'elevator'
    }

    for result in results:
        bbox = result['bbox']
        text = result['text'].lower()

        if text == 'hall':
            hallway_bboxes.append(bbox)
        elif text == 'na':
            outside_bboxes.append(bbox)
        elif text in transition_aliases:
            transition_bboxes.append(bbox)  # we keep just bboxes here; the label remains in results file
        else:
            room_bboxes.append(bbox)

    # Overwrite bbox file with valid bboxes AND inferred text
    # Create a mapping from bbox tuple to text for easy lookup
    bbox_to_text = {tuple(result['bbox']): result['text'] for result in results}
    
    with open(bbox_text_file, "w") as bbox_file:
        for bbox in bboxes:
            bbox_tuple = tuple(bbox)
            inferred_text = bbox_to_text.get(bbox_tuple, "")
            # Format: x1,y1,x2,y2,x3,y3,x4,y4 | inferred_text
            bbox_file.write(",".join(map(str, bbox)) + f" | {inferred_text}\n")

    # Save detailed results (includes transition words for later node naming)
    with open(result_file_path, "w") as result_file:
        for i, result in enumerate(results):
            bbox = result['bbox']
            text = result['text']
            confidence = result['confidence']
            result_file.write(f"BBox {i}: {bbox}, Text: {text}, Confidence: {confidence:.4f}\n")

    print(f"{len(results)} Valid bounding boxes interpreted, verified and saved to {result_file_path}")
    # NOTE: Return now includes `transition_bboxes` (new 4th element)
    return room_bboxes, hallway_bboxes, outside_bboxes, transition_bboxes, result_file_path

def parse_transition_labels(results_txt_path):
    """
    Parse lines like:
      BBox i: [x1, y1, x2, y2, ...], Text: <word>, Confidence: 0.9876
    Return: { (bbox_tuple): 'stairs'|'elevator' }
    """
    alias_map = {
        "stairs": "stairs", "stair": "stairs", "staircase": "stairs",
        "elev": "elevator", "elevator": "elevator", "lift": "elevator",
    }
    out = {}
    if not os.path.exists(results_txt_path):
        return out

    # pattern to capture the bbox list and the text token
    pat = re.compile(r"^BBox\s+\d+:\s*\[([^\]]+)\]\s*,\s*Text:\s*([^\s,]+)", re.IGNORECASE)

    with open(results_txt_path, "r") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            bbox_str = m.group(1)
            text = m.group(2).strip().lower()
            norm = alias_map.get(text)
            if norm is None:
                continue  # ignore non-transition words here

            # parse the bbox numbers into a tuple so it matches list->tuple comparisons
            try:
                nums = [int(x.strip()) for x in bbox_str.split(",")]
                out[tuple(nums)] = norm
            except Exception:
                continue
    return out