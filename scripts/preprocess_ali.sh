data_path=/mntcephfs/lab_data/zxy/Alimeeting

echo " Process dataset: Train/Eval dataset, get json files"
python prepare_data.py \
    --data_path ${data_path} \
    --type Eval \

python prepare_data.py \
    --data_path ${data_path} \
    --type Train \


