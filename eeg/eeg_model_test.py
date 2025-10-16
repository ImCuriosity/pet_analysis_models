# ==============================================================
# eeg_model_test.py — 시트가 1개인 경우 자동 분석
# 출력: "Positive xx.x% | Active xx.x%"
# ==============================================================

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
STATE_PATH = MODEL_DIR / "pbt_dogready_state.pth"   # human EEG로 학습한 PBT 모델을 fine-tune한 뒤, 그 가중치(state_dict)만 저장한 파일
EXCEL_PATH = MODEL_DIR / "dog_sample01.xlsx"   # 🔹 샘플 파일 (필요 시 dog_sample02.xlsx 로 변경)

# -----------------[ 모델 로드 ]-----------------
sys.path.insert(0, str(MODEL_SRC))  # import 경로 추가
from model import PBT

SEQ_LEN, N_CLASSES = 113, 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dim_feedforward 출력 숨기기
_stdout = sys.stdout
sys.stdout = io.StringIO()
model = PBT(
    d_input=1,
    n_classes=N_CLASSES,
    num_embeddings=SEQ_LEN,
    num_tokens_per_channel=1,
    d_model=128,
    n_blocks=4,
    num_heads=4,
    dropout=0.2,
    device=device
).to(device)
sys.stdout = _stdout

# 가중치 로드
state = torch.load(str(STATE_PATH), map_location=device)
model.load_state_dict(state, strict=False)
model.eval()

# -----------------[ 유틸 함수들 ]-----------------
def resample_to_len(x_row, target_len):
    src = np.linspace(0, 1, len(x_row), dtype=np.float32)
    dst = np.linspace(0, 1, target_len, dtype=np.float32)
    return np.interp(dst, src, x_row).astype(np.float32)

def load_sheet_numeric(path, sheet=None):
    # sheet_name=None → 여러 시트면 dict 반환하므로, 첫 시트만 자동 선택
    data = pd.read_excel(str(path), sheet_name=sheet, header=None)
    if isinstance(data, dict):       # 여러 시트가 dict로 반환된 경우
        df = next(iter(data.values()))  # 첫 번째 시트만 선택
    else:
        df = data

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = df.values.astype(np.float32)
    if X.shape[1] != SEQ_LEN:
        X = np.vstack([resample_to_len(r, SEQ_LEN) for r in X])
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-6)
    return X

def probs_from_model(model, X2d):
    xb = torch.tensor(X2d, dtype=torch.float32, device=device).unsqueeze(-1)
    pos = torch.arange(SEQ_LEN, device=device).unsqueeze(0).repeat(xb.size(0), 1)
    with torch.no_grad():
        out = model(xb, pos)
        logits = out[1] if isinstance(out, tuple) else out
        prob = torch.softmax(logits, dim=1).cpu().numpy()
    return prob

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
        "rel_beta":  band_power_welch(x, fs, 13, 30) / total,
        "rel_gamma": band_power_welch(x, fs, 30, 45) / total,
    }

def active_percent_from_rel(rel):
    a, t, b, g = rel["rel_alpha"], rel["rel_theta"], rel["rel_beta"], rel["rel_gamma"]
    return float((b + g) / (a + t + b + g + 1e-9) * 100.0)

CLASS_NAMES  = ["Enjoyed","Funny","Relaxed","Sad","Scary"]
POSITIVE_SET = {"Enjoyed","Funny","Relaxed"}

def positive_percent_from_probs(prob):
    name2idx = {c:i for i,c in enumerate(CLASS_NAMES)}
    idxs = [name2idx[c] for c in CLASS_NAMES if c in POSITIVE_SET]
    return float(prob.mean(0)[idxs].mean() * 100.0)

# -----------------[ 실행: 시트 자동 읽기 ]-----------------
FS = 256
X = load_sheet_numeric(EXCEL_PATH)  # 자동 첫 시트 읽기
prob = probs_from_model(model, X)

pos_pct = positive_percent_from_probs(prob)
act_vals = [active_percent_from_rel(rel_bands(x, FS)) for x in X]
act_pct = float(np.mean(act_vals))

# -----------------[ 출력: 한 줄만 ]-----------------
if __name__ == "__main__":
    FS = 256
    X = load_sheet_numeric(EXCEL_PATH)  # 자동 첫 시트 읽기
    prob = probs_from_model(model, X)

    pos_pct = positive_percent_from_probs(prob)
    act_vals = [active_percent_from_rel(rel_bands(x, FS)) for x in X]
    act_pct = float(np.mean(act_vals))

    print(f"Positive {pos_pct:.1f}% | Active {act_pct:.1f}%")


