import os
import numpy as np
import torch

<<<<<<< HEAD
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


=======
>>>>>>> 48c5b9c843525a1076251baa9f22ea1896458596
if __name__ == '__main__':
    exp_name = '10-16_grouped_trueVocals'
    path = '/tsi/doctorants/fmarty/Posteriograms/MUSDB18_trueVocals_posteriogram'
    files = sorted(os.listdir(path))
    outPath = '/tsi/doctorants/fmarty/Posteriograms/' + exp_name
   
    try: 
        os.mkdir(outPath)
    except:
        print("Directory already exist.")
    
    phoneme_list = ['ɚ', 's', 'ɛ', 'ð', 'j', 'œ', 'k', 'ɔ', 'ɲ', 'ø', 't', 'ʑ', 'd', 'ᵻ', 'ʃ', 'ʋ', 'ɜ', 'p', 'θ', 'ɾ', 'ɡ', 'n', 'ʏ', 'ɨ', 'ç', 'ɒ', 'ə', 'x', 'ʝ', 'ɟ', 'ŋ', 'ʎ', 'i', 'ɵ', 'y', 'ʁ', 'l', 'ʒ', 'ɕ', 'f', 'a', 'ɑ', 'β', 'ɬ', 'ʌ', 'z', 'o', 'r', 'v', 'ɐ', 'ɹ', 'ɣ', 'ɪ', 'w', 'ʔ', 'e', 'ʊ', 'u', 'b', 'h', 'æ', 'm']
    phoneme_list.append("special token 1")
    phoneme_list.append("special token 2")
    phoneme_list.append("special token 3")
    
    dict_phoneme = dict((el,0) for el in phoneme_list)
    dict_phoneme['ɚ'] = 'vowel'
    dict_phoneme['s'] = 'fricative'
    dict_phoneme['ɛ'] = 'vowel'
    dict_phoneme['ð'] = 'fricative'
    dict_phoneme['j'] = 'approximant'
    dict_phoneme['œ'] = 'vowel'
    dict_phoneme['k'] = 'plosive'
    dict_phoneme['ɔ'] = 'vowel'
    dict_phoneme['ɲ'] = 'nasal'
    dict_phoneme['ø'] = 'vowel'
    dict_phoneme['t'] = 'plosive'
    dict_phoneme['ʑ'] = 'fricative' # special one
    dict_phoneme['d'] = 'plosive'
    dict_phoneme['ᵻ'] = 'vowel'
    dict_phoneme['ʃ'] = 'fricative'
    dict_phoneme['ʋ'] = 'approximant'
    dict_phoneme['ɜ'] = 'vowel'
    dict_phoneme['p'] = 'plosive'
    dict_phoneme['θ'] = 'fricative'
    dict_phoneme['ɾ'] = 'plosive' # not sure about this
    dict_phoneme['ɡ'] = 'plosive'
    dict_phoneme['n'] = 'nasal'
    dict_phoneme['ʏ'] = 'vowel'
    dict_phoneme['ɨ'] = 'vowel'
    dict_phoneme['ç'] = 'fricative'
    dict_phoneme['ɒ'] = 'vowel'
    dict_phoneme['ə'] = 'vowel'
    dict_phoneme['x'] = 'fricative'
    dict_phoneme['ʝ'] = 'fricative'
    dict_phoneme['ɟ'] = 'plosive' # very similar to last
    dict_phoneme['ŋ'] = 'nasal'
    dict_phoneme['ʎ'] = 'approximant'
    dict_phoneme['i'] = 'vowel'
    dict_phoneme['ɵ'] = 'vowel'
    dict_phoneme['y'] = 'vowel'
    dict_phoneme['ʁ'] = 'fricative'
    dict_phoneme['l'] = 'approximant'
    dict_phoneme['ʒ'] = 'fricative'
    dict_phoneme['ɕ'] = 'fricative' # special
    dict_phoneme['f'] = 'fricative'
    dict_phoneme['a'] = 'vowel'
    dict_phoneme['ɑ'] = 'vowel'
    dict_phoneme['β'] = 'fricative'
    dict_phoneme['ɬ'] = 'fricative'
    dict_phoneme['ʌ'] = 'vowel'
    dict_phoneme['z'] = 'fricative'
    dict_phoneme['o'] = 'vowel'
    dict_phoneme['r'] = 'fricative' # not sure about this
    dict_phoneme['v'] = 'fricative'
    dict_phoneme['ɐ'] = 'vowel'
    dict_phoneme['ɹ'] = 'approximant'
    dict_phoneme['ɣ'] = 'fricative' # ressemble autre
    dict_phoneme['ɪ'] = 'vowel'
    dict_phoneme['w'] = 'approximant' # special
    dict_phoneme['ʔ'] = 'plosive'
    dict_phoneme['e'] = 'vowel'
    dict_phoneme['ʊ'] = 'vowel'
    dict_phoneme['u'] = 'vowel'
    dict_phoneme['b'] = 'plosive'
    dict_phoneme['h'] = 'fricative'
    dict_phoneme['æ'] = 'vowel'
    dict_phoneme['m'] = 'nasal'
    dict_phoneme['special token 1'] = 'special'
    dict_phoneme['special token 2'] = 'special'
    dict_phoneme['special token 3'] = 'special'

<<<<<<< HEAD
    uniqueValues = list(set(dict_phoneme.values()))
    
=======
    uniqueValues = set(dict_phoneme.values())

>>>>>>> 48c5b9c843525a1076251baa9f22ea1896458596
    for index,file in enumerate(files):
        print(index)
        if file.endswith('.npy'):
            raw_posteriogram = np.load(path+'/'+file)
            
            nb_time_frames, nb_phonemes = raw_posteriogram.shape
            
<<<<<<< HEAD
            output = np.zeros((nb_time_frames,len(uniqueValues)))
            
            for k,class_name in enumerate(uniqueValues):
                class_phonemes_indexes = [i for i, x in enumerate(list(dict_phoneme.values())) if x == class_name]
=======
            output = np.zeros(nb_time_frames,len(uniqueValues))
            
            for k,class_name in enumerate(uniqueValues):
                class_phonemes_indexes = [i for i, x in enumerate(list(dict_phoneme.values()) if x == class_name]
>>>>>>> 48c5b9c843525a1076251baa9f22ea1896458596
                output[:,k] = np.sum(raw_posteriogram[:,class_phonemes_indexes],axis=1)

            file_name = file.replace('_vocals', '')
            file_name = file_name.replace('_posteriorgram','')
            file_name = file_name.replace('.npy','.pt')
            torch.save(torch.from_numpy(output),outPath+'/'+file_name)