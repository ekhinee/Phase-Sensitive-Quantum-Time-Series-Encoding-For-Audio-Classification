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

    # aplicar filtros mel sobre fase
    mel_phase = mel_basis @ phase

    # DCT igual que MFCC
    phase_ceps = dct(
        mel_phase,
        type=2,
        axis=0,
        norm='ortho'
    )

    return phase_ceps[:n_mfcc]

def preprocess(X, sr, n_frames=16, n_mfcc=13, phase=False):

    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X[None, :]

    X_out = []

    for signal in X:

        if(phase):
            mfcc = mfcc_phase(signal, sr, n_mfcc, hop_length=max(1, len(signal) // n_frames), n_mels=40)
        else:
            mfcc = librosa.feature.mfcc(y=signal,sr=sr,n_mfcc=n_mfcc)

        T = mfcc.shape[1]

        # 2. dividir en 16 bloques temporales
        idx = np.linspace(0, T, n_frames + 1, dtype=int)

        values = []

        for i in range(n_frames):

            block = mfcc[:, idx[i]:idx[i+1]]  # (13, block_size)

            # 3. el blocke a 1 scalar
            scalar = np.mean(np.abs(block))  
            values.append(scalar)
        
        X_out.append(values)

    X_out = np.array(X_out)  # shape (N, 16)

    # normalización global
    mn = X_out.min()
    mx = X_out.max()

    if(mx - mn < 1e-12):
        return np.zeros_like(X_out, dtype=int)
    
    X_out = (X_out - mn) / (mx - mn)

    X_out = np.floor(X_out * 15).astype(int)
    

    

    return X_out
