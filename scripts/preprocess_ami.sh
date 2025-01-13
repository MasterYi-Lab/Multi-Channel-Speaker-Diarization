data_path=/mntcephfs/lab_data/zxy/Alimeeting

# echo " Process dataset: Train/Eval dataset, get json files"
# python prepare_data.py \
#     --data_path ${data_path} \
#     --type Eval \

# python prepare_data.py \
#     --data_path ${data_path} \
#     --type Train \


# -m debugpy --listen 0.0.0.0:5678 --wait-for-client
data_path=/mntcephfs/lab_data/zxy/amicorpus
rttm_path=${data_path}/AMI-diarization-setup/only_words/rttms/train
orig_audio_path=${data_path}/data
target_audio_path=${data_path}/target_audio
mode_emb_file=/home/zxy/speaker_diarization/TS-VAD-MC-backup-master_cw_junyi/pretrain/ecapa-tdnn.model
target_embedding_path=${data_path}/target_embedding
echo " Process AMI dataset: Train/Eval dataset, get json files"
python  prepare_data_ami.py \
    --root_path ${data_path} \
    --rttm_path ${rttm_path} \
    --orig_audio_path ${orig_audio_path} \
    --target_audio_path ${target_audio_path}\
    --source ${mode_emb_file} \
    --target_embedding_path ${target_embedding_path}
