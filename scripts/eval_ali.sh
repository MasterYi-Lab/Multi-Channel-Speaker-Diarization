#!/bin/bash
#SBATCH -J tranBSub
#SBATCH --output=inference.txt
#SBATCH -p p-A800
#SBATCH -A t00120220002
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1


# source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh 
# conda config --add envs_dirs /mntnfs/lee_data1/lijunjie/anaconda3/envs
# conda activate py1.11
# nvidia-smi

gpu=0
exp_name=rerun_tsvad_mc_backup_ami_ali
rs_len=4000
segment_shift=1
gen_subset=Eval

code_path=/home/zxy/speaker_diarization/TS-VAD # need change
data_path=/mntcephfs/lab_data/zxy/Alimeeting # need change

pt_path=the_path_of_your_checkpoint # need change


./parse_options.sh

export CUDA_VISIBLE_DEVICES=$gpu

ts_vad_path=${code_path}/ts_vad

speech_encoder_path=${code_path}/pretrain/ecapa-tdnn.model # Speaker encoder path
spk_path=${data_path}/SpeakerEmbedding

results_path=the_path_of_your_results # need change

python3 ${ts_vad_path}/generate.py ${data_path} \
  --user-dir ${ts_vad_path} \
  --results-path ${results_path} \
  --path ${pt_path} \
  --task ts_vad_task \
  --spk-path ${spk_path} \
  --rs-len ${rs_len} \
  --segment-shift ${segment_shift} \
  --gen-subset ${gen_subset} \
  --batch-size 32 \
  --shuffle-spk-embed-level 3 \
  --inference
