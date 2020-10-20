import os
import numpy as np
import torch

# phoneme [nb_phoneme_frames, nb_phonemes]

if __name__ == '__main__':
    exp_name = '10-12_fullPosteriogramsNoProcessing'
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
            
            output = torch.from_numpy(raw_posteriogram)
            file_name = file.replace('_vocals', '')
            file_name = file_name.replace('_posteriorgram','')
            file_name = file_name.replace('.npy','.pt')
            torch.save(output,outPath+'/'+file_name)