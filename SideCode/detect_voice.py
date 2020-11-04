import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def moving_average(a, n=3):
    if n % 2 == 0:
        raise ValueError("Odd size kernel expected")
    
    a = np.concatenate((np.zeros(n//2),a,np.zeros(n//2)))
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

start = 0
end = -1

# start and end are in seconds
# returns start_list and end_list in frame index
def retrieve_blanks_from_signal(y,start,end,sr):
    meaned = moving_average(np.abs(y[0][int(start*sr):int(end*sr)]),1001)
    
    in_active_region = False
    start_list = []
    end_list = []
    
    nb_time_points = meaned.shape[0]
    current = 0
    
    while current < nb_time_points - int(1*sr):
        if in_active_region:
            if meaned[current] < 0.004:
                look_next = int(1*sr)
                next_segment = meaned[current:current+look_next]
                if len(next_segment[next_segment < 0.005]) > 0.3*look_next:
                    if ((current - start_list[-1])/16000) > 1:
                        end_list.append(current)
                        in_active_region = False
                    else:
                        #print("passage à",current/sr)
                        start_list.pop(-1)
                        in_active_region = False
        
        else:
            if meaned[current] > 0.003:
                look_next = int(0.1*sr)
                next_segment = meaned[current:current+look_next]
                if len(next_segment[next_segment > 0.004]) > 0.3*look_next:
                    start_list.append(current)
                    in_active_region = True
                
        current = current + int(0.01*sr)
    
    if len(start_list) == len(end_list) + 1:
        end_list.append(nb_time_points - 1)
    
    if len(start_list) != len(end_list):
        raise ValueError("End and start list of different length.")
    
    return start_list,end_list

def add_instrumental_token(phoneme,start_list,end_list):
    phoneme_out = torch.zeros(phoneme.shape)
    phoneme_out[:,62] = 1.0
    for i in range(len(start_list)):
        duration = (end_list[i] - start_list[i])/sr
        start_frame = int((start_list[i]/sr) / 0.016)
        end_frame = start_frame+int((duration - 0.032)/0.016)
        phoneme_out[start_frame:end_frame] = phoneme[start_frame:end_frame]
    
    return phoneme_out

if __name__ == '__main__':
    exp_name = '10-26_trueVocals_instrumentalTokenMoved'
    
    # we use the already moved version
    phoneme_path = ('/tsi/doctorants/fmarty/Posteriograms/'
                    + '10-19_trueVocals_ctc_moved')
    files = sorted(os.listdir(phoneme_path))
    outPath = '/tsi/doctorants/fmarty/Posteriograms/' + exp_name
    audio_path = '/tsi/doctorants/fmarty/Posteriograms/MUSDB18_trueVocals/'
   
    try: 
        os.mkdir(outPath)
    except:
        print("Directory already exist.")
    
    for index,file in enumerate(files):
        print(index,file)
        if file.endswith('.pt'):
            
            audio_file_name = file.replace('.pt','.wav')
            audio_file_name = audio_file_name.replace('test_','test_vocals_')
            audio_file_name = audio_file_name.replace('train_','train_vocals_')
            
            posteriogram = torch.load(phoneme_path+'/'+file)
            audio, sr = librosa.load(audio_path + audio_file_name, sr=None,mono=False)
            
            start = 0
            end = -1
            start_list,end_list = retrieve_blanks_from_signal(audio,start,end,sr)
            
            phoneme_out = add_instrumental_token(posteriogram,start_list,end_list)
            
            torch.save(phoneme_out,outPath+'/'+file)