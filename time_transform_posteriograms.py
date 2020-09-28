import torch
import tf_transforms
import musdb
import math

# stride and hop are in seconds
# Assumption : phoneme window = 2 * phoneme hop
def open_unmix(phoneme,nb_fft_frames,phoneme_hop,fft_window,fft_hop,center=False):
    nb_samples, nb_phoneme_frames, nb_phonemes = phoneme.size()
    
    weight_matrix = torch.zeros(nb_fft_frames,nb_phoneme_frames).to(phoneme.device)
    
    for i in range(nb_fft_frames):
        alpha = phoneme_hop/fft_window
        
        # center is an argument of torch.stft that pad the input signal if center=True
        if center == False: # at training time
            start_time = i * fft_hop
            end_time = start_time + fft_window
        
        elif center == True: # at test time
            start_time = i * fft_hop - fft_window/2
            end_time = start_time + fft_window
            
            # case where the start point of the window is in the padding (at begin)
            if start_time < 0:
                start_time = 0
            
            # case where the end point of the window is in the padding (at end)
            if end_time > ((nb_fft_frames - math.ceil(fft_window/fft_hop) - 1) 
                                * fft_hop + fft_window):
                end_time = ((nb_fft_frames - math.ceil(fft_window/fft_hop) - 1) 
                            * fft_hop + fft_window)
        
        # variables defined as the limit numbers of the phoneme frames that overlap 
        # with the current fft window
        phonemeStartFrame = math.floor(start_time/phoneme_hop)
        phonemeEndFrame = math.ceil(end_time/phoneme_hop)
        
        # total number of overlapping phoneme windows with the current fft window
        total_nb_frames = phonemeEndFrame - phonemeStartFrame + 1
        
        # as in the computation of the stft, we will apply a hann window on the
        # phoneme windows overlapping with the current fft window
        # This window has a 0 at the end and the beginning
        window = torch.hann_window(total_nb_frames,periodic=False)

        # normalize to 1 over the contribution of each phoneme window
        sum = torch.sum(window)
        weights = window/sum
        
        weight_matrix[i][phonemeStartFrame:phonemeEndFrame + 1] = weights
        
        """debugging stuff
        if i < 10 or i > nb_fft_frames - 10:
            print("weight_matrix shape",weight_matrix.shape)
            print("NB TOTAL FRAME", total_nb_frames)
            print("start_time",start_time)
            print("end_time",end_time)
            print("phonemeStartFrame",phonemeStartFrame)
            print("i",i)
            print(window)
            print("End frame",phonemeEndFrame)
            print(torch.sum(weight_matrix[i]))
            print(weight_matrix[i][:15])
            print("---")
        """

    #output shape [batch_size, nb_fft_frames, nb_phonemes]
    return torch.matmul(weight_matrix,phoneme)

# stride and hop are in seconds
# Assumption : phoneme window = 2 * phoneme hop
def open_unmix_single(phoneme,nb_fft_frames,phoneme_hop,fft_window,
                    fft_hop,center=False):
    nb_samples, nb_phoneme_frames = phoneme.size()
    
    output = torch.zeros(nb_samples,nb_fft_frames,dtype=torch.long).to(phoneme.device)
    
    # we reserve the 0 for padding in the embedding
    phoneme = phoneme + 1
        
    frame_list = []
    
    for i in range(nb_fft_frames):
        alpha = phoneme_hop/fft_window
        
        # center is an argument of torch.stft that pad the input signal if center=True
        if center == False: # at training time
            start_time = i * fft_hop
            end_time = start_time + fft_window
        
        elif center == True: # at test time
            start_time = i * fft_hop - fft_window/2
            end_time = start_time + fft_window
            
            # case where the start point of the window is in the padding (at begin)
            if start_time < 0:
                start_time = 0
            
            # case where the end point of the window is in the padding (at end)
            if end_time > ((nb_fft_frames - math.ceil(fft_window/fft_hop) - 1) 
                                * fft_hop + fft_window):
                end_time = ((nb_fft_frames - math.ceil(fft_window/fft_hop) - 1) 
                            * fft_hop + fft_window)
        
        # we keep the value from the closest frame to the middle time in its middle
        middle_time = (start_time + end_time)/2
        phonemeCorrespondingFrame = math.ceil(middle_time/phoneme_hop - 0.5)
        
        # in case we are in the very beginning
        if phonemeCorrespondingFrame == 0:
            phonemeCorrespondingFrame = 1
        
        frame_list.append(phonemeCorrespondingFrame)
    
    output[:,] = phoneme[:,frame_list]
    
    #output shape [batch_size, nb_fft_frames]
    return output

# The phoneme window is assumed to be much larger than the encoder window in Conv-TasNet
def conv_tasnet_single(phoneme,nb_frames,padding_size,
                phoneme_hop,encoder_kernel,encoder_stride,sp_rate):
    
    nb_samples, nb_phoneme_frames = phoneme.size()
    
    pad_left = padding_size//2
    pad_right = padding_size - pad_left
    
    output = torch.zeros(nb_samples,nb_frames,dtype=torch.long).to(phoneme.device)
    
    frame_list = []
    
    # we reserve the 0 for padding in the embedding
    phoneme = phoneme + 1
    
    # we leave zeros in the padding regions
    for i in range(pad_left,nb_frames - pad_right + 1):
        correspondingTime = encoder_kernel/2 + i * encoder_stride - pad_left/sp_rate
        
        # Beware that the 'phoneme' also inputs the preceding frame as only zeros,
        # corresponding to indice -1 normally.
        # - 0.5 to take the closest phoneme window in its middle
        phonemeCorrespondingFrame = math.ceil(correspondingTime/phoneme_hop - 0.5)
        
        # in case we are in the very beginning
        if phonemeCorrespondingFrame == 0:
            phonemeCorrespondingFrame = 1
        
        frame_list.append(phonemeCorrespondingFrame)
    
    output[:,pad_left:nb_frames - pad_right + 1] = phoneme[:,frame_list]
    
    # out [batch_size, nb_frames]
    return output

# stride and hop are in seconds
# Assumption : phoneme window = 2 * phoneme hop
def conv_tasnet(phoneme,nb_frames,padding_size,
                phoneme_hop,encoder_kernel,encoder_stride,sp_rate):
    nb_samples, nb_phoneme_frames, nb_phonemes = phoneme.size()
    
    pad_left = padding_size//2
    pad_right = padding_size - pad_left

    output = torch.zeros(nb_samples,nb_frames,nb_phonemes).to(phoneme.device)
    
    frame_list = []

    # we leave zeros in the padding regions
    for i in range(pad_left,nb_frames - pad_right + 1):
        correspondingTime = encoder_kernel/2 + i * encoder_stride - pad_left/sp_rate
        
        # Beware that the 'phoneme' also inputs the preceding frame as only zeros,
        # corresponding to indice -1 normally (but here 0).
        # - 0.5 to take the closest phoneme window in its middle
        phonemeCorrespondingFrame = math.ceil(correspondingTime/phoneme_hop - 0.5)
        
        frame_list.append(phonemeCorrespondingFrame)
        
    output[:,pad_left:nb_frames - pad_right + 1,:] = phoneme[:,frame_list,:]
    
    # out [batch_size, nb_frames, nb_phonemes]
    return output

if __name__ == '__main__':
    import numpy as np
    
    mus = musdb.DB(
        root='/tsi/doctorants/fmarty/Datasets/MUSDB18wav',
        is_wav=True,
        split='test',
        subsets='test',
        download=False
    )
    
    track_number = 35
    
    print(mus.tracks[track_number].name)
    
    track_db = mus.tracks[track_number]
    track_db.chunk_start = 0
    track_db.chunk_duration = min(track_db.duration,600)
    
    track = torch.from_numpy(track_db.audio)
    
    track = track.T
    track = torch.unsqueeze(track,0)
    print(track.shape)
    print("track length:",track.shape[-1]/mus.tracks[track_number].rate,"s")
    
    center=True
    
    stft_transform = tf_transforms.STFT(center=center)
    spec_transform = tf_transforms.Spectrogram()
    
    spec_track = spec_transform(stft_transform(track))
    
    print(spec_track.shape)
    
    post_modified = np.load('/tsi/doctorants/fmarty/Posteriograms/09-16_trueVocals/test_'+mus.tracks[track_number].name+'.npy')
    post_modified = torch.from_numpy(post_modified)
    
    #print("Post modified:",post_modified.shape)
    
    # Concatenated in front in all cases, to mimick the absence of a frame at the beginning
    # of the track
    # The concatenation at the end is just for convenience
    post_modified = torch.cat((torch.zeros(1,64),post_modified,torch.zeros(5,64)),dim=0)
    
    
    post_modified = torch.unsqueeze(post_modified,0) # add batch dimension
    
    res = timeTransform(post_modified,spec_track.shape[0],
                        0.016,4096/44100,1024/44100,center=center)
    
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