from collections import defaultdict

import torch, os, tqdm, numpy, soundfile, time, argparse, glob, wave,copy,json
from speaker_encoder import ECAPA_TDNN

def remove_overlap(aa, bb):
    # Sort the intervals in both lists based on their start time
    a = aa.copy()
    b = bb.copy()
    a.sort()
    b.sort()

    # Initialize the new list of intervals
    result = []

    # Initialize variables to keep track of the current interval in list a and the remaining intervals in list b
    i = 0
    j = 0

    # Iterate through the intervals in list a
    while i < len(a):
        # If there are no more intervals in list b or the current interval in list a does not overlap with the current interval in list b, add it to the result and move on to the next interval in list a
        if j == len(b) or a[i][1] <= b[j][0]:
            result.append(a[i])
            i += 1
        # If the current interval in list a completely overlaps with the current interval in list b, skip it and move on to the next interval in list a
        elif a[i][0] >= b[j][0] and a[i][1] <= b[j][1]:
            i += 1
        # If the current interval in list a partially overlaps with the current interval in list b, add the non-overlapping part to the result and move on to the next interval in list a
        elif a[i][0] < b[j][1] and a[i][1] > b[j][0]:
            if a[i][0] < b[j][0]:
                result.append([a[i][0], b[j][0]])
            a[i][0] = b[j][1]
        # If the current interval in list a starts after the current interval in list b, move on to the next interval in list b
        elif a[i][0] >= b[j][1]:
            j += 1

    # Return the new list of intervals
    return result


def init_speaker_encoder(source):
	speaker_encoder = ECAPA_TDNN(C=1024).cuda()
	speaker_encoder.eval()
	loadedState = torch.load(source, map_location="cuda")
	selfState = speaker_encoder.state_dict()
	for name, param in loadedState.items():
		if name in selfState:
			selfState[name].copy_(param)
	for param in speaker_encoder.parameters():
		param.requires_grad = False 
	return speaker_encoder

def extract_embeddings(batch, model):	
	batch = torch.stack(batch)    
	with torch.no_grad():
		embeddings = model.forward(batch.cuda())
	return embeddings
def get_args():
    #     python modules/extract_target_speech.py \
    #         --rttm_path exp/predict/res_rttm \
    #         --orig_audio_path ${audio_dir} \
    #         --target_audio_path ${target_audio_path}
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--root_path', required=True,
                        help='the path for the root')
    parser.add_argument('--rttm_path', required=True,
                        help='the path for the rttm_files')
    parser.add_argument('--orig_audio_path', required=True,
                        help='the path for the orig audio')
    parser.add_argument('--target_audio_path', required=True,
                        help='the part for the output audio')
    parser.add_argument('--target_embedding_path', required=True,
                        help='the part for the output audio')
    parser.add_argument('--source', help='the part for the speaker encoder')
    parser.add_argument('--length_embedding', type=float, default=6, help='length of embeddings, seconds')
    parser.add_argument('--step_embedding', type=float, default=1, help='step of embeddings, seconds')
    parser.add_argument('--batch_size', type=int, default=96, help='step of embeddings, seconds')
    args = parser.parse_args()

    return args

def main():
    args = get_args()

    outs = open(os.path.join(args.root_path,'Train.json'), "w")

    # stage1 get target_audio from original audio and rttm file
    for rttm_path in os.listdir(args.rttm_path):
        lines = open(os.path.join(args.rttm_path,rttm_path)).read().splitlines()
        room_set = set()
        spkr_id = set()
        for line in (lines):
            data = line.split()
            room_set.add(data[1])
            spkr_id.add(data[-3])

        spkr_id_list = list(spkr_id)
        string_to_number = {string: idx + 1 for idx, string in enumerate(spkr_id_list)}
        for room_id in tqdm.tqdm(room_set):
            intervals = defaultdict(list)
            new_intervals = defaultdict(list)
            for line in (lines): 
                data = line.split()
                if data[1] == room_id:
                    stime = float(data[3])
                    etime = float(data[3]) + float(data[4])
                    spkr = string_to_number[data[-3]]
                    intervals[spkr].append([stime, etime])

            # Remove the overlapped speeech    
            for key in intervals:
                new_interval = intervals[key]
                for o_key in intervals:
                    if o_key != key:                
                        new_interval = remove_overlap(copy.deepcopy(new_interval), copy.deepcopy(intervals[o_key]))
                new_intervals[key] = new_interval

            wav_file = glob.glob(os.path.join(args.orig_audio_path, room_id) + '/audio/' + '*.wav')[0]
            orig_audio, fs = soundfile.read(wav_file)
            orig_audio = orig_audio[:,0] if orig_audio.ndim == 2 else orig_audio
            length = len(orig_audio)

            # # Cut and save the clean speech part
            id_full = wav_file.split('/')[-1][:-4]
            for key in new_intervals:
                output_dir = os.path.join(args.target_audio_path, id_full)
                os.makedirs(output_dir, exist_ok = True)
                output_wav = os.path.join(output_dir, str(key) + '.wav')
                new_audio = []    
                for interval in new_intervals[key]:
                    s, e = interval
                    s *= 16000
                    e *= 16000
                    new_audio.extend(orig_audio[int(s):int(e)])

                soundfile.write(output_wav, new_audio, 16000)
            output_wav = os.path.join(output_dir, 'all.wav')
            soundfile.write(output_wav, orig_audio, 16000)
		    # Save the labels
            for key in intervals:
                labels = [0] * int(length / 16000 * 25) # 40ms, one label
                for interval in intervals[key]:
                    s, e = interval
                    for i in range(int(s * 25), min(int(e * 25) + 1, len(labels))):
                        # diarization label, 0: inactive, 1: active
                        labels[i] = 1

                res = {'filename':id_full, 'speaker_key':key, 'speaker_id':spkr_id_list[key-1], 'labels':labels}
                json.dump(res, outs)
                outs.write('\n')
               
    # stage2 get target_embedding from target_audio
    files = sorted(glob.glob(args.target_audio_path + "/*/*.wav"))
    model = init_speaker_encoder(args.source)
    for file in tqdm.tqdm(files):
        if 'all' not in file:
            batch = []
            embeddings = []
            wav_length = wave.open(file, 'rb').getnframes() # entire length for target speech
            for start in range(0, wav_length - int(args.length_embedding * 16000), int(args.step_embedding * 16000)):
                stop = start + int(args.length_embedding * 16000)
                target_speech, _ = soundfile.read(file, start = start, stop = stop)
                target_speech = torch.FloatTensor(numpy.array(target_speech))
                batch.append(target_speech)
                if len(batch) == args.batch_size:                
                    embeddings.extend(extract_embeddings(batch, model))
                    batch = []
            if len(batch) != 0:
                embeddings.extend(extract_embeddings(batch, model))             
            embeddings = torch.stack(embeddings)
            output_file = args.target_embedding_path + '/' + file.split('/')[-2] + '/' + file.split('/')[-1].replace('.wav', '.pt')
            os.makedirs(os.path.dirname(output_file), exist_ok = True)
            torch.save(embeddings, output_file)

if __name__ == '__main__':
    main()