#!/bin/bash
# Copyright (c) Yiwen Shao

# Apache 2.0

. ./cmd.sh
set -euo pipefail

# Change this location to somewhere where you want to put the data.
data=./corpus/
if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
  data=/export/corpora5
fi

data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11

# data related
rootdir=data
dumpdir=data/dump   # directory to dump full features
langdir=data/lang   # directory for language models
graphdir=data/graph # directory for chain graphs (FSTs)


# Data splits
train_set=train_460
valid_set=dev_clean
test_set=test_clean

# feature configuration
do_delta=false

# Model options
unit=phone # phone/char
type=mono # mono/bi

affix=
stage=0
. ./path.sh
. ./utils/parse_options.sh

dir=exp/tdnn_${type}${unit}${affix:+_$affix}
lang=$langdir/lang_${type}${unit}_e2e
graph=$graphdir/${type}${unit}


mkdir -p $data

if [ $stage -le -2 ]; then
  for part in dev-clean test-clean train-clean-100 train-clean-360; do
    local/download_and_untar.sh $data $data_url $part
  done
fi

if [ $stage -le -1 ]; then
  local/download_lm.sh $lm_url data/local/lm
fi


if [ $stage -le 0 ]; then
  echo "Stage 0: Data Preparation"
  # format the data as Kaldi data directories
  for part in dev-clean test-clean train-clean-100 train-clean-360; do
    # use underscore-separated names in data directories.
    local/data_prep.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
  done
  utils/combine_data.sh data/${train_set} data/train_clean_100 data/train_clean_360
fi

if [ $stage -le 1 ]; then
  echo "Stage 1: Feature Extraction"
  ./prepare_feat.sh --train_set $train_set \
		    --valid_set $valid_set \
		    --test_set $test_set \
		    --dumpdir $dumpdir \
		    --rootdir $rootdir
fi

if [ $stage -le 2 ]; then
  echo "Stage 2: Dictionary and LM Preparation"

  local/prepare_dict.sh --stage 3 --nj 30 --cmd "$train_cmd" \
    data/local/lm data/local/lm data/local/dict_nosp

  utils/prepare_lang.sh data/local/dict_nosp \
    "<UNK>" data/local/lang_tmp_nosp $langdir/lang_nosp

  local/format_lms.sh --src-dir $langdir/lang_nosp data/local/lm
  # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
  utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
    $langdir/lang_nosp $langdir/lang_nosp_test_tglarge
fi

if [ $stage -le 3 ]; then
  echo "Stage 3: Graph Preparation"
  ./prepare_graph.sh --train_set $train_set \
		     --valid_set $valid_set \
		     --test_set $test_set \
		     --rootdir $rootdir \
		     --graphdir $graphdir \
		     --langdir $langdir \
		     --type $type \
		     --unit $unit
fi

if [ ${stage} -le 4 ]; then
  echo "Stage 4: Dump Json Files"
  for dataset in $train_set $valid_set $test_set; do
    wav=$rootdir/$dataset/wav.scp
    dur=$rootdir/$dataset/utt2dur
    feat=$dumpdir/$dataset/feats.scp
    fst=$graph/$dataset/num.scp
    text=$rootdir/$dataset/text
    utt2num_frames=$rootdir/$dataset/utt2num_frames
    asr_prep_json.py --wav-files $wav \
		     --dur-files $dur \
		     --feat-files $feat \
		     --numerator-fst-files $fst \
		     --text-files $text \
		     --num-frames-files $utt2num_frames \
		     --output data/${dataset}_${type}${unit}.json
  done
  
fi

num_targets=$(tree-info $graph/tree | grep num-pdfs | awk '{print $2}')

if [ ${stage} -le 5 ]; then
  echo "Stage 5: Model Training"
  opts=""
  mkdir -p $dir/logs
  log_file=$dir/logs/train.log
  $cuda_cmd $log_file train.py \
    --train data/${train_set}_${type}${unit}.json \
    --valid data/${valid_set}_${type}${unit}.json \
    --den-fst $graph/normalization.fst \
    --epochs 40 \
    --dropout 0.2 \
    --wd 0.01 \
    --optimizer adam \
    --lr 0.001 \
    --scheduler plateau \
    --gamma 0.5 \
    --hidden-dims 384 384 384 384 384 \
    --curriculum 1 \
    --num-targets $num_targets \
    --seed 1 \
    --exp $dir 2>&1 | tee $log_file
fi

attack=FGSM
defense=random
sigma=0.001
eps=0
iters=10
#for eps in 0.0001 0.001 0.01; do
#test_adv=${test_set}_${attack}_eps${eps}_${defense}_${sigma}
# for eps in 0.0001; do
# test_adv=${test_set}_${attack}_eps${eps}
if [ ${stage} -le 6 ]; then
  echo "Stage 6: Dumping Posteriors for Test Data"
  path=$dir/$checkpoint
  log_file=$dir/logs/dump_$test_adv.log
  result_file=${test_adv}/posteriors.ark
  audio_dir=${test_adv}/audio
  mkdir -p $dir/$audio_dir
  $cuda_cmd $log_file adv.py \
	    --test data/test_${type}${unit}.json \
	    --den-fst $graph/normalization.fst \
	    --exp $dir \
	    --bsz 128 \
	    --attack $attack \
	    --defense $defense \
	    --sigma $sigma \
	    --eps $eps \
	    --iters $iters \
	    --on-the-fly true \
	    --model model_best.pth.tar \
	    --results $result_file \
	    --save-audio-dir $audio_dir
fi

if [ ${stage} -le 7 ]; then
  echo "Stage 7: Trigram LM Decoding"
  decode_dir=$dir/decode/$test_set/tgsmall
  mkdir -p $decode_dir
  latgen-faster-mapped --acoustic-scale=1.0 --beam=15 --lattice-beam=8 \
		       --word-symbol-table="$graph/graph_tgsmall/words.txt" \
		       $graph/0.trans_mdl $graph/graph_tgsmall/HCLG.fst \
		       ark:$dir/$test_set/posteriors.ark \
		       "ark:|lattice-scale --acoustic-scale=10.0 ark:- ark:- | gzip -c >$decode_dir/lat.1.gz" \
		       2>&1 | tee $dir/logs/decode_$test_set.log
fi


if [  $stage -le 8 ]; then
  echo "Stage 8: Forthgram LM rescoring"
  oldlang=$langdir/lang_nosp_test_tgsmall
  newlang=$langdir/lang_nosp_test_tglarge
  oldlm=$oldlang/G.fst
  newlm=$newlang/G.carpa
  oldlmcommand="fstproject --project_output=true $oldlm |"
  olddir=$dir/decode/$test_set/tgsmall
  newdir=$dir/decode/$test_set/tglarge
  mkdir -p $newdir
  $train_cmd $dir/logs/rescorelm_$test_set.log \
	     lattice-lmrescore --lm-scale=-1.0 \
	     "ark:gunzip -c ${olddir}/lat.1.gz|" "$oldlmcommand" ark:- \| \
	     lattice-lmrescore-const-arpa --lm-scale=1.0 \
	     ark:- "$newlm" "ark,t:|gzip -c>$newdir/lat.1.gz"
fi

if [ ${stage} -le 9 ]; then
  echo "Stage 9: Computing WER"
  for lmtype in tglarge; do
    local/score_kaldi_wer.sh $rootdir/$test_set $graph/graph_tgsmall $dir/decode/$test_set/$lmtype
    echo "Best WER for $test_set with $lmtype:"
    cat $dir/decode/$test_set/$lmtype/scoring_kaldi/best_wer
  done
fi
