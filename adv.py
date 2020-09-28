#!/usr/bin/env python3
# Copyright (c) Yiwen Shao

# Apache 2.0

import argparse
import os

import torch
import torch.nn.parallel

from dataset import ChainDataset, AudioDataLoader
from models import get_model

import kaldi_io
from pychain.loss import ChainLoss
from pychain.graph import ChainGraph
import simplefst

import torchaudio

parser = argparse.ArgumentParser(description='PyChain Attack and Defense')
# Datasets
parser.add_argument('--test', type=str, required=True,
                    help='test set json file')
parser.add_argument('--den-fst', type=str, required=True,
                    help='denominator fst path')

# Model
parser.add_argument('--exp', default='exp/tdnn',
                    type=str, metavar='PATH', required=True,
                    help='dir to load model and save output')
parser.add_argument('--model', default='model_best.pth.tar', type=str,
                    help='model checkpoint')
parser.add_argument('--results', default='posteriors.ark', type=str,
                    help='results filename')
parser.add_argument('--bsz', default=128, type=int,
                    help='test batchsize')
# Feature Extraction
parser.add_argument('--on-the-fly', default=False, type=bool,
                    help='on the fly feature extraction')
parser.add_argument('--normalize', default=False, type=bool,
                    help='waveform normalization on utterance level')
# Adversarial Attack
parser.add_argument('--attack', default='FGSM', type=str,
                    help='attack type')
parser.add_argument('--eps', default='1e-2', type=float,
                    help='epsilon for FGSM/PGD')
parser.add_argument('--alpha', default='1e-2', type=float,
                    help='alpha for PGD')
parser.add_argument('--iters', default=10, type=int,
                    help='iterations for PGD')
# Adversarial Defense
parser.add_argument('--defense', default='random', type=str,
                    help='defense type')
parser.add_argument('--sigma', default=0.1, type=float,
                    help='random smoothing sigma')
# Perturbed Audio Saver
parser.add_argument('--save-audio-dir', default='None', type=str,
                    help='dir to save perturbed audio')

args = parser.parse_args()
# state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
use_cuda = torch.cuda.is_available()


def main():
    # Data
    testset = ChainDataset(
        args.test, on_the_fly=args.on_the_fly, feat='raw', normalize=args.normalize)
    testloader = AudioDataLoader(testset, batch_size=args.bsz)

    # loss
    den_fst = simplefst.StdVectorFst.read(args.den_fst)
    den_graph = ChainGraph(den_fst)
    criterion = ChainLoss(den_graph)

    # Model
    checkpoint_path = os.path.join(args.exp, args.model)
    with open(checkpoint_path, 'rb') as f:
        state = torch.load(f)
        model_args = state['args']
        print("==> creating model '{}'".format(model_args.arch))
        model = get_model(model_args.feat_dim, model_args.num_targets,
                          model_args.layers, model_args.hidden_dims,
                          model_args.arch, kernel_sizes=model_args.kernel_sizes,
                          dilations=model_args.dilations,
                          strides=model_args.strides,
                          bidirectional=model_args.bidirectional)
        print(model)

        if use_cuda:
            model = torch.nn.DataParallel(model).cuda()
            print("model is on gpu")

        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        model.load_state_dict(state['state_dict'])

    output_file = os.path.join(args.exp, args.results)
    test(testloader, model, criterion, output_file, use_cuda, args)


def test(testloader, model, criterion, output_file, use_cuda, args):
    # switch to test mode
    model.eval()
    with open(output_file, 'wb') as f:
        for i, (inputs, input_lengths, utt_ids, graphs) in enumerate(testloader):
            if args.attack == 'FGSM':
                perturbed_inputs = fgsm_attack(
                    inputs, input_lengths, graphs, model, criterion, args.eps)
            elif args.attack == 'PGD':
                perturbed_inputs = pgd_attack(
                    inputs, input_lengths, graphs, model, criterion, args.eps, args.alpha, args.iters)
            else:
                perturbed_inputs = inputs  # no perturbation

            if args.defense == 'random':
                perturbed_inputs = perturbed_inputs + \
                    args.sigma * torch.randn_like(perturbed_inputs)

            perturbed_lprobs, output_lengths = model(
                perturbed_inputs, input_lengths)
            for j in range(inputs.size(0)):
                output_length = output_lengths[j]
                utt_id = utt_ids[j]
                kaldi_io.write_mat(
                    f, (perturbed_lprobs[j, :output_length, :]).cpu().detach().numpy(), key=utt_id)
                # save perturbed data into audio
                if args.save_audio_dir is not None:
                    filepath = os.path.join(
                        args.exp, args.save_audio_dir, utt_id + '.wav')
                    input_length = input_lengths[j]
                    torchaudio.save(
                        filepath, perturbed_inputs[j, :input_length, :].transpose(0, 1), sample_rate=16000)


def fgsm_attack(data, data_lengths, graphs, model, criterion, eps=0.01):
    data.requires_grad = True
    output, output_lengths = model(data, data_lengths)
    loss = criterion(output, output_lengths, graphs)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    sign_data_grad = data_grad.sign()
    perturbed_data = data + eps * sign_data_grad
    perturbed_data = perturbed_data.detach()
    return perturbed_data


def pgd_attack(data, data_lengths, graphs, model, criterion, eps=0.01, alpha=0.01, iters=10):

    original_data = data.clone()
    for i in range(iters):
        data.requires_grad = True
        output, output_lengths = model(data, data_lengths)
        loss = criterion(output, output_lengths, graphs)
        model.zero_grad()
        loss.backward()

        perturbed_data = data + alpha * data.grad.sign()
        perturbed_data = torch.min(perturbed_data, original_data + eps)
        perturbed_data = torch.max(perturbed_data, original_data - eps)
        data = perturbed_data.detach()

    print((data - original_data).abs().max())
    return data


if __name__ == '__main__':
    main()
