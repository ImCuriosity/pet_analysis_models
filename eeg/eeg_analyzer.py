# eeg_analyzer.py

import sys, io
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.signal import welch
from scipy.integrate import trapezoid

# -----------------[ 경로 설정 ]-----------------
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "eeg_model"
MODEL_SRC = MODEL_DIR / "PatchedBrainTransformer" / "src"
STATE_PATH = MODEL_DIR / "pbt_dogready_state.pth"

# -----------------[ 상수 정의 ]-----------------
SEQ_LEN, N_CLASSES = 113, 5
FS = 256
CLASS_NAMES = ["Enjoyed","Funny","Relaxed","Sad","Scary"] 
POSITIVE_SET = {"Enjoyed","Funny","Relaxed"}


# -----------------[ 유틸 함수들 ]-----------------
def resample_to_len(x_row, target_len):
    src = np.linspace(0, 1, len(x_row), dtype=np.float32)
    dst = np.linspace(0, 1, target_len, dtype=np.float32)
    return np.interp(dst, src, x_row).astype(np.float32)

def load_sheet_numeric_from_bytes(file_bytes):
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, header=None)
    if isinstance(df, dict):
        df = next(iter(df.values()))
    
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = df.values.astype(np.float32)
    
    if X.shape[1] != SEQ_LEN:
        X = np.vstack([resample_to_len(r, SEQ_LEN) for r in X])
        
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-6)
    return X

def band_power_welch(x, fs, fmin, fmax):
    nperseg = min(1024, len(x))
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    m = (f >= fmin) & (f <= fmax)
    return float(trapezoid(Pxx[m], f[m]))

def rel_bands(x, fs):
    total = band_power_welch(x, fs, 0.5, 45.0) + 1e-12
    return {
        "rel_alpha": band_power_welch(x, fs, 8, 13) / total,
        "rel_theta": band_power_welch(x, fs, 4, 8) / total,
        "rel_beta": band_power_welch(x, fs, 13, 30) / total,
        "rel_gamma": band_power_welch(x, fs, 30, 45) / total,
    }

def active_percent_from_rel(rel):
    a, t, b, g = rel["rel_alpha"], rel["rel_theta"], rel["rel_beta"], rel["rel_gamma"]
    return (b + g) / (a + t + b + g + 1e-9) * 100.0

def positive_percent_from_probs(prob):
    name2idx = {c:i for i,c in enumerate(CLASS_NAMES)}
    idxs = [name2idx[c] for c in CLASS_NAMES if c in POSITIVE_SET]
    return prob.mean(0)[idxs].mean() * 100.0


# -----------------[ 분석기 클래스 ]-----------------
class EEGAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

    def _load_model(self):
        if str(MODEL_SRC) not in sys.path:
             sys.path.insert(0, str(MODEL_SRC)) 
        
        try:
             from model import PBT
        except ImportError:
             raise ImportError(f"PBT model not found. Check if '{MODEL_SRC}/model.py' exists.")
        
        # dim_feedforward 출력 숨기기
        _stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        model = PBT(
            d_input=1, n_classes=N_CLASSES, num_embeddings=SEQ_LEN, num_tokens_per_channel=1,
            d_model=128, n_blocks=4, num_heads=4, dropout=0.2, device=self.device
        ).to(self.device)
        sys.stdout = _stdout 
        
        if not STATE_PATH.exists():
            raise FileNotFoundError(f"Model state file not found: {STATE_PATH}")
        
        state = torch.load(str(STATE_PATH), map_location=self.device)
        model.load_state_dict(state, strict=False)
        model.eval()
        
        print("Model loaded successfully.")
        return model

    def _probs_from_model(self, X2d: np.ndarray) -> np.ndarray:
        xb = torch.tensor(X2d, dtype=torch.float32, device=self.device).unsqueeze(-1)
        pos = torch.arange(SEQ_LEN, device=self.device).unsqueeze(0).repeat(xb.size(0), 1)
        
        with torch.no_grad():
            out = self.model(xb, pos)
            logits = out[1] if isinstance(out, tuple) else out
            prob = torch.softmax(logits, dim=1).cpu().numpy()
        return prob

# eeg_analyzer.py 파일 내 analyze_eeg_data 함수 수정

    def analyze_eeg_data(self, file_bytes: bytes) -> dict:
        """EEG 데이터를 받아 분석 결과를 반환합니다."""
        X = load_sheet_numeric_from_bytes(file_bytes)
        prob = self._probs_from_model(X)
        
        pos_pct = positive_percent_from_probs(prob)
        act_vals = [active_percent_from_rel(rel_bands(x, FS)) for x in X]
        act_pct = float(np.mean(act_vals))
        
        # ★★★ 이 부분을 수정합니다: float()으로 감싸서 표준 float으로 변환 ★★★
        return {
            "positive_percent": round(float(pos_pct), 1),
            "active_percent": round(float(act_pct), 1),
            "result_text": f"Positive {pos_pct:.1f}% | Active {act_pct:.1f}%"
        }