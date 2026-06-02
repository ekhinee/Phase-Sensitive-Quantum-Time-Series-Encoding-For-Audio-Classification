from pathlib import Path
import librosa
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import soundfile as sf

def load_audios_gtzan(first_path, second_path, n,sr=22050, duration=30.0):

    X1 = []
    X2 = []

    m = 0

    for file in Path(first_path).glob("*.wav"):
        audio, _ = librosa.load(file, sr=sr, duration=duration)
        X1.append(audio)
        sf.write(f"data_gtzan/classical/classical_{m}.wav",audio,sr)
        m = m + 1
        if(m==n):
            break

    m = 0

    for file in Path(second_path).glob("*.wav"):
        audio, _ = librosa.load(file, sr=sr, duration=duration)
        X2.append(audio)
        sf.write(f"data_gtzan/classical/classical_{m}.wav",audio,sr)
        m = m + 1
        if(m==n):
            break

    X1 = np.array(X1)
    X2 = np.array(X2)

    y1 = np.zeros(len(X1), dtype=int)
    y2 = np.ones(len(X2), dtype=int)

    return X1, y1, X2, y2


def banatu_folds(X,y):
    idx_all = np.arange(len(y))
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    folds = []
    for idx_trainval, idx_test in skf.split(idx_all, y):
        idx_train, idx_val = train_test_split(
            idx_trainval, test_size=0.20, random_state=42, stratify=y[idx_trainval]
        )
        folds.append({
            "train": idx_train,
            "val":   idx_val,
            "test":  idx_test
        })

    np.save("X.npy", X)
    np.save("y.npy", y)
    np.save("folds.npy", folds, allow_pickle=True)
