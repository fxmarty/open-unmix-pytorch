import torch
import tf_transforms
import musdb
import math

# stride and hop in seconds
# Assumption : phoneme window = 2 * phoneme hop
def timeTransform(phoneme,nb_fft_frames,phoneme_hop,fft_window,fft_hop):
    nb_samples, nb_phoneme_frames, nb_phonemes = phoneme.size()
    
    weight_matrix = torch.zeros(nb_fft_frames,nb_phoneme_frames).to(phoneme.device)
    
    alpha = phoneme_hop/fft_window
    
    for i in range(nb_fft_frames):
        start_time = i * fft_hop
        end_time = start_time + fft_window
        
        # The "1 +" comes from the definition of phonemeStartFrame
        phonemeStartFrame = math.floor(start_time/phoneme_hop) + 1
        phonemeEndFrame = math.floor(end_time/phoneme_hop) + 1
                
        beta_left = (phonemeStartFrame + 1)*phoneme_hop - start_time
        beta_left = beta_left / fft_window
                
        beta_right = end_time - phonemeEndFrame*phoneme_hop
        beta_right = beta_right / fft_window
        
        #nb_alpha = math.floor(((1 - beta_left) * fft_window) / phoneme_hop)
        
        weight_matrix[i][phonemeStartFrame - 1] = beta_left/2
        weight_matrix[i][phonemeStartFrame] = beta_left/2 + alpha/2
        weight_matrix[i][phonemeStartFrame+1:phonemeEndFrame-1] = alpha
        weight_matrix[i][phonemeEndFrame-1] = beta_right/2 + alpha/2
        weight_matrix[i][phonemeEndFrame] = beta_right/2
        
        #print(i,nb_fft_frames,phonemeEndFrame)
        #print(nb_phoneme_frames)
        #print(torch.sum(weight_matrix[i]))
    #output shape [batch_size, nb_channels,nb_fft_frames, nb_phonemes]
    
    return torch.matmul(weight_matrix,phoneme)

if __name__ == '__main__':
    mus = musdb.DB(
        root='/tsi/doctorants/fmarty/MUSDB18wav',
        is_wav=True,
        split='test',
        subsets='test',
        download=False
    )
    
    print(mus.tracks[11].name)
    
    track = torch.from_numpy(mus.tracks[11].audio)
    track = track.T
    track = torch.unsqueeze(track,0)
    print(track.shape)
    print("track length:",track.shape[-1]/mus.tracks[11].rate,"s")
    
    stft_transform = tf_transforms.STFT()
    spec_transform = tf_transforms.Spectrogram()
    
    spec_track = spec_transform(stft_transform(track))
    
    
    print(spec_track.shape)
    
    post_modified = torch.load('/tsi/doctorants/fmarty/Posteriograms/09-15_trueVocals/test_'+mus.tracks[11].name+'_posteriorgram.pt')

    
    print("Post modified:",post_modified.shape)
    
    res = timeTransform(post_modified,spec_track.shape[0],0.016,4096/44100,1024/44100)
    
    print("reshaped posteriogram:",res.shape)
    
    
    a = torch.rand(1,1,4096+1024)
    a_spec = spec_transform(stft_transform(a))
    
    print("---")
    print(a.shape)
    print(a_spec.shape)
    
    """
    for i,track in enumerate(mus.tracks):
        print(i,track.name)
    """