import numpy as np
import librosa
import numpy as np
from scipy.fftpack import dct


def mfcc_phase(y,sr,n_mfcc,hop_length=512,n_mels=40):
    # STFT compleja
    D = librosa.stft(
        y=y,
        n_fft=min(2048, len(y)),
        hop_length=hop_length
    )

    # fase
    phase = np.angle(D)

    # unwrap para evitar saltos artificiales
    phase = np.unwrap(phase, axis=1)

    # mel filterbank
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=min(2048, len(y)),
        n_mels=n_mels
    )

        # MAX en vez de suma ponderada
    # mel_phase = np.array([
    #     phase[np.argmax(mel_basis[m]), :]   # bin de frecuencia con mayor peso en ese filtro
    #     for m in range(n_mels)
    # ])
    dominant_bins = np.argmax(mel_basis, axis=1) 
    mel_phase = phase[dominant_bins, :]

    # aplicar filtros mel sobre fase
    #mel_phase = mel_basis @ phase

    # DCT igual que MFCC
    phase_ceps = dct(
        mel_phase,
        type=2,
        axis=0,
        norm='ortho'
    )

    return phase_ceps[:n_mfcc]

def preprocess(X, sr, n_frames=16, n_mfcc=13, phase=False, binary=True):

    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X[None, :]

    X_out = []

    for signal in X:

        if(phase):
            mfcc = mfcc_phase(signal, sr, n_mfcc, hop_length=512, n_mels=40)
            #mfcc = mfcc_phase(signal, sr, n_mfcc, hop_length=max(1, len(signal) // n_frames), n_mels=40)
        else:
            mfcc = librosa.feature.mfcc(y=signal,sr=sr,n_mfcc=n_mfcc)

        T = mfcc.shape[1]

        # 2. dividir en 16 bloques temporales
        idx = np.linspace(0, T, n_frames + 1, dtype=int)

        values = []

        for i in range(n_frames):

            block = mfcc[:, idx[i]:idx[i+1]]  # (13, block_size)

            # 3. el blocke a 1 scalar
            if(phase): #phase
                scalar = np.angle(np.mean(np.exp(1j * block))) 
            else:
                scalar = np.mean(np.abs(block))  
            values.append(scalar)
        
        X_out.append(values)

    X_out = np.array(X_out)  # shape (N, 16)

    # normalización global
    mn = X_out.min()
    mx = X_out.max()

    if(mx - mn < 1e-12):
        return np.zeros_like(X_out, dtype=int)
    
    # if(not(binary)):
    #     for i in range(len(X_out)):
    #         mn, mx = X_out[i].min(), X_out[i].max()
    #         if mx - mn > 1e-12:
    #             X_out[i] = (X_out[i] - mn) / (mx - mn)
    # else:
    X_out = (X_out - mn) / (mx - mn)

    if(binary):
        X_out = np.floor(X_out * 15).astype(int)
    

    

    return X_out
