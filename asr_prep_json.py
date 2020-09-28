#!/usr/bin/env python3
# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import OrderedDict
import json
import logging
import sys


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)


def read_file(ordered_dict, key, dtype, *paths):
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                utt_id, val = line.strip().split(None, 1)
                if val[-1] == '|':
                    val = val[:-2]
                if utt_id in ordered_dict:
                    assert key not in ordered_dict[utt_id], \
                        "Duplicate utterance id " + utt_id + " in " + key
                    ordered_dict[utt_id].update({key: dtype(val)})
                else:
                    ordered_dict[utt_id] = {key: val}
    return ordered_dict


def main():
    parser = argparse.ArgumentParser(
        description="Wrap all related files of a dataset into a single json file"
    )
    # fmt: off
    parser.add_argument("--wav-files", nargs="+", required=True,
                        help="path(s) to scp raw waveform file(s)")
    parser.add_argument("--dur-files", nargs="+", required=True,
                        help="path(s) to utt2dur file(s)")
    parser.add_argument("--feat-files", nargs="+", default=None,
                        help="path(s) to scp feature file(s)")
    parser.add_argument("--num-frames-files", nargs="+", default=None,
                        help="path(s) to utt2num_frames file(s)")
    parser.add_argument("--text-files", nargs="+", default=None,
                        help="path(s) to text file(s)")
    parser.add_argument("--numerator-fst-files", nargs="+", default=None,
                        help="path(s) to numerator fst file(s)")
    parser.add_argument("--output", required=True, type=argparse.FileType("w"),
                        help="path to save json output")
    args = parser.parse_args()
    # fmt: on

    print(args)
    obj = OrderedDict()
    obj = read_file(obj, "wav", str, *(args.wav_files))
    obj = read_file(obj, "duration", float, *(args.dur_files))
    if args.feat_files is not None:
        obj = read_file(obj, "feat", str, *(args.feat_files))
    if args.text_files is not None:
        obj = read_file(obj, "text", str, *(args.text_files))
    if args.numerator_fst_files is not None:
        obj = read_file(obj, "numerator_fst", str, *(args.numerator_fst_files))
    if args.num_frames_files is not None:
        obj = read_file(obj, "length", int,
                        *(args.num_frames_files))

    json.dump(obj, args.output, indent=4)


if __name__ == "__main__":
    main()
