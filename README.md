This repository provides the training and testing procedures for our algorithm. For training, you can use the Alimeeting dataset alone or a combination of the Alimeeting and AMI datasets. The testing is then conducted on the NTU dataset. If you only wish to test on the NTU dataset, you can directly use our pre-trained model. Please note that you can place the dataset in any location of your choice; simply modify the corresponding path in the shell script as needed. For the locations that require modifications, I have already added annotations.
# Dataset
### 1.  Download Alimeeting dataset(https://openslr.org/119/)
```
mkdir Alimeeting
cd Alimeeting
wget https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Train_Ali_far.tar.gz
wget https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz
wget https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Test_Ali.tar.gz
tar -xzvf Train_Ali_far.tar.gz
tar -xzvf Eval_Ali.tar.gz
tar -xzvf Test_Ali.tar.gz
```
- Download SpeakerEmbedding.zip from https://drive.google.com/file/d/1tNRnF9ouPbPX9jxAh1HkuNBOY9Yx6Pj9/view?usp=sharing and put it in root folder
- Then make the dataset looks like 
 ```
alimeeting
├── Train_Ali
│   ├── Train_Ali_far 
│     ├── audio_dir
├── Eval_Ali
│   ├── Eval_Ali_far 
│     ├── audio_dir
├── spk_embed
│   ├── SpeakerEmbedding 
│     ├── ...
```

- run scripts/preprocess_ali.sh in scripts directory and then the dataset looks like
```
alimeeting 
├── Train_Ali
│   ├── Train_Ali_far 
│     ├── audio_dir
│     ├── target_audio
│     ├── textgrid_dir
│     ├── Train.json
├── Eval_Ali
│   ├── Eval_Ali_far 
│     ├── audio_dir
│     ├── target_audio
│     ├── textgrid_dir
│     ├── Eval.json
├── spk_embed
│   ├── SpeakerEmbedding 
│     ├── ...
```



### 2. Download AMI dataset
download ami dataset from
link: https://pan.baidu.com/s/1YzrQy6-7HHOknAlnBVGUHg?pwd=d8pj
you alse need download https://github.com/pyannote/AMI-diarization-setup
- to make the  dataset looks like 
```
amicorpus
├── AMI-diarization-setup
├── Train_Ami
├── Eval_Ami
├── Test_Ami
├── data
│   ├── ES2002a
│   ├──  ES2002b
│   ├──  ...
```
- run scripts/preprocess_ami.sh in scripts directory to get target_audio and target_embedding and json file for Train_Ami Eval_Ami and Test_Ami

### 3. Download NTU dataset
you can download the NTU dataset from https://pan.baidu.com/s/1gRzDwW0OVBkvIIzhoB4vsg?pwd=cmb5
follow the pipeline of https://github.com/adnan-azmat/TSVAD_pytorch/tree/u/adnan/e2epipeline to process NTU dataset
```
data/
├── wav/
│   ├── file1.wav
│   ├── file2.wav
│   ├── file3.wav
│   └── ...
├── rttm/
│   ├── file1.rttm
│   ├── file2.rttm
│   ├── file3.rttm
│   └── ...
├── target_audio/
│   ├── file1
│   │   ├── 1.wav
│   │   ├── 2.wav
│   │   ├── 3.wav
│   │   ├── 4.wav
│   │   └── all.wav
│   ├── file2
│   │   ├── 1.wav
│   │   ├── 2.wav
│   │   ├── 3.wav
│   │   ├── 4.wav
│   │   └── all.wav
│   └── ...
├── target_embedding/
│   ├── file1
│   │   ├── 1.pt
│   │   ├── 2.pt
│   │   ├── 3.pt
│   │   └── 4.pt
│   ├── file2
│   │   ├── 1.pt
│   │   ├── 2.pt
│   │   ├── 3.pt
│   │   └── 4.pt
│   └── ...
└── ts_infer.json
```

### 4.Ali_ami dataset
copy AMI dataset to Alimeeting dataset for target_audio ， SpeakerEmbedding and json file

## Train
for train, you need to change the content of scripts/run.sh
```
bash scripts/run.sh
```
## Eval
for eval alimeeting dataset, you need to change the content of scripts/eval_ali.sh
```
bash scripts/eval_ali.sh
```
for eval NTU dataset, you need to change the content of scripts/eval_ntu.sh
```
bash scripts/eval_ali.sh
```


Please note that due to the different file structure formats of various datasets, you need to make corresponding modifications to the code when testing on NTU to ensure it matches the input data and gt_rttm.

## Model zoo
The weight files obtained from training using only Alimeeting are https://drive.google.com/file/d/1QqF4I4rYPgOhCrvtFQtw8YSl3VSvbDXn/view?usp=sharing

The weight files obtained from training using  Alimeeting and AMI are https://drive.google.com/file/d/1o7LYaAOwty-tg0OmGF3nqdqTlVQiCwle/view?usp=sharing