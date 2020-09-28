# Copyright (c) Yiwen Shao
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from collections import OrderedDict
import json

from io import BytesIO
import librosa
from subprocess import run, PIPE
import torchaudio

import torch
import simplefst
import kaldi_io
from tqdm import tqdm
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from pychain import ChainGraph, ChainGraphBatch


def parse_rxfile(file):
    # separate offset from filename
    if re.search(':[0-9]+$', file):
        (file, offset) = file.rsplit(':', 1)
    return file, int(offset)


def _collate_fn_train(batch):
    # sort the batch by its feature length in a descending order
    batch = sorted(
        batch, key=lambda sample: sample[1], reverse=True)
    max_seqlength = batch[0][1]
    feat_dim = batch[0][0].size(1)
    minibatch_size = len(batch)
    feats = torch.zeros(minibatch_size, max_seqlength, feat_dim)
    feat_lengths = torch.zeros(minibatch_size, dtype=torch.int)
    utt_ids = []
    graph_list = []
    max_num_transitions = 0
    max_num_states = 0
    for i in range(minibatch_size):
        feat, length, utt_id, graph = batch[i]
        feats[i, :length, :].copy_(feat)
        feat_lengths[i] = length
        utt_ids.append(utt_id)
        graph_list.append(graph)
        if graph.num_transitions > max_num_transitions:
            max_num_transitions = graph.num_transitions
        if graph.num_states > max_num_states:
            max_num_states = graph.num_states
    num_graphs = ChainGraphBatch(
        graph_list, max_num_transitions=max_num_transitions, max_num_states=max_num_states)
    return feats, feat_lengths, utt_ids, num_graphs


def _collate_fn_test(batch):
    # sort the batch by its feature length in a descending order
    batch = sorted(
        batch, key=lambda sample: sample[1], reverse=True)
    max_seqlength = batch[0][1]
    feat_dim = batch[0][0].size(1)
    minibatch_size = len(batch)
    feats = torch.zeros(minibatch_size, max_seqlength, feat_dim)
    feat_lengths = torch.zeros(minibatch_size, dtype=torch.int)
    utt_ids = []
    for i in range(minibatch_size):
        feat, length, utt_id = batch[i]
        feats[i, :length, :].copy_(feat)
        feat_lengths[i] = length
        utt_ids.append(utt_id)
    return feats, feat_lengths, utt_ids


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for ChainDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        if self.dataset.train:
            self.collate_fn = _collate_fn_train
        else:
            self.collate_fn = _collate_fn_test


class BucketSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        codes from deepspeech.pytorch 
        https://github.com/SeanNaren/deepspeech.pytorch/blob/master/data/data_loader.py
        """
        super(BucketSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size]
                     for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)


class ChainDataset(data.Dataset):

    def __init__(self, data_json_path, train=True, cache_graph=True, sort=True,
                 on_the_fly=False, feat='mfcc', normalize=False, params=None):
        super(ChainDataset, self).__init__()
        self.train = train
        self.cache_graph = cache_graph
        self.sort = sort
        self.on_the_fly = on_the_fly
        self.normalize = normalize
        if self.on_the_fly:
            print("Doing on-the-fly feature extraction")
        self.feat = feat
        if params:
            self.params = params
        elif self.on_the_fly and self.feat == 'mfcc':
            self.params = {
                "use_energy": False,
                "num_mel_bins": 40,
                "num_ceps": 40,
                "low_freq": 20,
                "high_freq": -400
            }
        self.samples = []  # list of dicts

        with open(data_json_path, 'rb') as f:
            loaded_json = json.load(f, object_pairs_hook=OrderedDict)

        print("Initializing dataset...")
        for utt_id, val in tqdm(loaded_json.items()):
            sample = {}
            sample['utt_id'] = utt_id
            sample['wav'] = val['wav']
            sample['text'] = val['text']
            sample['duration'] = float(val['duration'])
            if not self.on_the_fly:
                sample['feat'] = val['feat']
                sample['length'] = int(val['length'])

            if self.train:  # only training data has fst (graph)
                fst_rxf = val['numerator_fst']
                if self.cache_graph:  # cache all fsts at once
                    filename, offset = parse_rxfile(fst_rxf)
                    fst = simplefst.StdVectorFst.read_ark(filename, offset)
                    graph = ChainGraph(fst, log_domain=True)
                    if graph.is_empty:
                        continue
                    sample['graph'] = graph
                else:
                    sample['graph'] = fst_rxf

            self.samples.append(sample)

        if self.sort:
            # sort the samples by their feature length
            self.samples = sorted(
                self.samples, key=lambda sample: sample['duration'])

    def __getitem__(self, index):
        sample = self.samples[index]
        utt_id = sample['utt_id']
        if not self.on_the_fly:
            feat_ark = sample['feat']
            feat = torch.from_numpy(kaldi_io.read_mat(feat_ark))
            feat_length = sample['length']
        else:
            wav_file = sample['wav']
            if len(wav_file.split()) > 1:
                # wav_file is in command format
                source = BytesIO(run(wav_file, shell=True, stdout=PIPE).stdout)
                wav, sampling_rate = librosa.load(
                    source,
                    sr=None,  # 'None' uses the native sampling rate
                    mono=False,  # Retain multi-channel if it's there
                )
                wav = torch.from_numpy(wav).unsqueeze(0)
            else:
                wav, sampling_rate = torchaudio.load(wav_file)

            if self.feat == 'mfcc':
                feat = torchaudio.compliance.kaldi.mfcc(wav, sample_frequency=sampling_rate,
                                                        **self.params)
                feat_length = feat.size(0)
            elif self.feat == 'raw':
                feat = wav.transpose(0, 1)
                feat_length = feat.size(0)
                if self.normalize:
                    feat = feat / feat.abs().max()  # normalize to [-1, 1]

        if self.train:
            if self.cache_graph:
                graph = sample['graph']
            else:
                fst_rxf = sample['graph']
                filename, offset = parse_rxfile(fst_rxf)
                fst = simplefst.StdVectorFst.read_ark(filename, offset)
                graph = ChainGraph(fst)
            return feat, feat_length, utt_id, graph
        else:
            utt_id = sample['utt_id']
            return feat, feat_length, utt_id

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    json_path = 'data/test_monophone.json'
    # trainset = ChainDataset(json_path, on_the_fly=True)
    # trainloader = AudioDataLoader(
    #     trainset, batch_size=1, shuffle=False)

    # feat, feat_lengths, graphs = next(iter(trainloader))
    # print(feat)
    # print(feat.size())
    # cmvn_ark = "/export/b06/yshao/pychain_ex2/examples/wsj/data/train_si284/cmvn.ark"
    # cmvn_ark = torch.from_numpy(kaldi_io.read_mat(cmvn_ark))
    # print(cmvn_ark.size())
    # print(cmvn_ark)
    trainset_raw = ChainDataset(
        json_path, on_the_fly=True, feat='raw', train=False, normalize=True)
    trainloader = AudioDataLoader(
        trainset_raw, batch_size=2, shuffle=False)

    feat, feat_lengths, utt_id = next(iter(trainloader))
    # params = {
    #     "use_energy": False,
    #     "num_mel_bins": 40,
    #     "num_ceps": 40,
    #     "low_freq": 20,
    #     "high_freq": -400
    # }
    # sampling_rate = 16000
    # print(feat_lengths[0])
    # feat = torchaudio.compliance.kaldi.mfcc(feat[0, :feat_lengths[0]], sample_frequency=sampling_rate,
    #                                         **params)
    print(feat.abs().max())
