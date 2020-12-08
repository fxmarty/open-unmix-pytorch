import os
import numpy as np
import torch

# phoneme [nb_phoneme_frames, nb_phonemes]
# return [nb_phoneme_frames,nb_phonemes-1], with blank token removed and put in the other places
# we assume that 0 is NOT the max in the first frame
def moveCTCTokenIntoMatrix(phoneme):
    nb_phoneme_frames, nb_phonemes = phoneme.shape
    output = np.copy(phoneme)[:,:-1]
    current_ind_max = np.argmax(phoneme[0])
    for i in range(nb_phoneme_frames):
        ind_max = np.argmax(phoneme[i]) 

        if ind_max == 64:
            output[i][current_ind_max] += phoneme[i][64]
        else:
            output[i] = output[i]/output[i].sum()
            current_ind_max = ind_max
    return output

if __name__ == '__main__':
    exp_name = '11-10_polyphonic_ctc_moved'
    path = '/tsi/doctorants/fmarty/Posteriograms/MUSDB18_polyphonic'
    files = sorted(os.listdir(path))
    outPath = '/tsi/doctorants/fmarty/Posteriograms/' + exp_name
   
    try: 
        os.mkdir(outPath)
    except:
        print("Directory already exist.")
    
    for index,file in enumerate(files):
        print(index)
        if file.endswith('.npy'):
            raw_posteriogram = np.load(path+'/'+file)
            phoneme_ctc_moved = moveCTCTokenIntoMatrix(raw_posteriogram)
            phoneme_ctc_moved = torch.from_numpy(phoneme_ctc_moved)
            
            file_name = file.replace('_vocals', '')
            file_name = file_name.replace('_posteriorgram','')
            file_name = file_name.replace('.npy','.pt')
            torch.save(phoneme_ctc_moved,outPath+'/'+file_name)