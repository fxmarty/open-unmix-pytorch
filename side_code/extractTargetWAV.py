from shutil import copyfile
import os
root_dir = '../../executedJobs/08-28_trainUMXreproduceStereo/outDir/'

i = 0
tot = 150
for dir_, _, files in os.walk(root_dir):
    for file_name in files:
        rel_dir = os.path.relpath(dir_, root_dir)
        rel_file = os.path.join(rel_dir, file_name)
        if rel_file.endswith('vocals.wav'):
            #print("Processing",rel_file.split("/")[-3],"... (",100 * i/tot,"% done)")
            outputName = rel_file.split("/")[-3]+"_vocals_" + rel_file.split("/")[-2] + ".wav"
            #print(outputName)
            #y, sr = librosa.load(root_dir + rel_file, sr=16000,mono=False)
            copyfile(root_dir + rel_file, '/tsi/doctorants/fmarty/MUSDB18_estimatedVocals/'+outputName)
            #sf.write('/tsi/doctorants/fmarty/MUSDB18_16000wav/'+rel_file, y.T, sr)
            i = i + 1