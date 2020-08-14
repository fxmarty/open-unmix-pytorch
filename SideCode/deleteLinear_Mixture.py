import librosa
import soundfile as sf

import os
root_dir = '../../MUSDB18_8192wav/'

i = 0
tot = 150
for dir_, _, files in os.walk(root_dir):
    for file_name in files:
        rel_dir = os.path.relpath(dir_, root_dir)
        rel_file = os.path.join(rel_dir, file_name)
        if rel_file.endswith('.wav'):
            #i = i + 1
            #print("Processing",rel_file,"... (",100 * i/tot,"% done)")
            os.remove(root_dir + rel_file)
            
        """
        
        if rel_file.endswith('.wav'):
            print("Processing",rel_file,"... (",i/tot,"% done)")
            y, sr = librosa.load(root_dir + rel_file, sr=8192,mono=False)
            sf.write('/tsi/doctorants/fmarty/MUSDB18_8192wav/'+rel_file, y.T, sr)
            i = i + 1
        """