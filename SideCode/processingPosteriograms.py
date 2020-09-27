import os
import numpy as np
import torch

# phoneme [nb_phoneme_frames, nb_phonemes]
# return [nb_phoneme_frames,nb_phonemes-1], with blank token removed and put in the other places
# we assume that 0 is NOT the max in the first frame
def moveBlankTokenIntoMatrix(phoneme):
    nb_phoneme_frames, nb_phonemes = phoneme.shape
    output = np.copy(phoneme)[:,:-1]
    current_ind_max = np.argmax(phoneme,axis=1)
    for i in range(nb_phoneme_frames):
        ind_max = np.argmax(phoneme[i]) 

        if ind_max == 64:
            output[i][current_ind_max] += phoneme[i][64]
        else:
            output[i] = output[i]/output[i].sum()
            current_ind_max = ind_max
    return output

def selectHighestPhoneme(phoneme):
    return np.argmax(phoneme,axis=1)

if __name__ == '__main__':
    exp_name = '09-26_trueVocals_unique'
    path = '/tsi/doctorants/fmarty/Posteriograms/MUSDB18_trueVocals_posteriogram'
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
            phoneme_moved = moveBlankTokenIntoMatrix(raw_posteriogram)
            phoneme_moved = selectHighestPhoneme(phoneme_moved)
            
            file_name = file.replace('_vocals', '')
            file_name = file_name.replace('_posteriorgram','')
            np.save(outPath+'/'+file_name,phoneme_moved)