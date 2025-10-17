import os, json, uuid, shutil, sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models
from torch.serialization import add_safe_globals

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware


# ===================== 0. 설정 =====================
# 환경변수/기본값
MODEL_PATH   = os.getenv("MODEL_PATH", "vgg19_bilstm_binary.pt")
CLASSES_STR  = os.getenv("CLASSES", "Negative_Score, Positive_Score")
DOGGIE_ROOT  = os.getenv("DOGGIE_ROOT", "Doggie-smile")
OUT_ROOT     = Path(os.getenv("OUT_ROOT", "outputs/api"))

SSD_PROTO    = str(Path(DOGGIE_ROOT) / "MobileNetSSD_deploy.prototxt")
SSD_WEIGHTS  = str(Path(DOGGIE_ROOT) / "MobileNetSSD_deploy.caffemodel")
HEAD_DAT     = str(Path(DOGGIE_ROOT) / "dogHeadDetector.dat")  # 사용 안함(옵션)

Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)

# ===================== 1. 모델 클래스 (훈련 때와 동일 이름/구조) =====================
class VGG19_BiLSTM_Binary(nn.Module):
    def __init__(self, hidden=256, num_classes=2):
        super().__init__()
        m = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(m.features.children()))
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden,
                            bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden*2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):  # x: (B,T,C,H,W)
        B,T,C,H,W = x.shape
        x = x.view(B*T, C, H, W)
        feat = self.backbone(x)
        feat = self.gap(feat).view(B*T, 512)
        feat = feat.view(B, T, 512)
        out,_ = self.lstm(feat)
        out = out[:, -1, :]
        logits = self.head(out)
        return logits




def load_pt_model_safely(model_path: str, device: str = "cuda") -> nn.Module:
    """
    - Colab에서 전체모델(torch.save(model, ...))로 저장된 .pt
    - 또는 state_dict(torch.save({'state_dict': model.state_dict()}, ...)) 모두 지원
    - torch 2.6이 아닌 버전에서도 동작하도록 __main__ 리맵핑과 safe globals만 사용
    """
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    # Colab에서 __main__.VGG19_BiLSTM_Binary 로 저장된 경우를 대비:
    # 현재 모듈을 __main__으로도 바라보게 함 (간단/효과적)
    sys.modules['__main__'] = sys.modules[__name__]

    # 전체모델 로딩 시, 해당 클래스 허용
    add_safe_globals([VGG19_BiLSTM_Binary])

    # 1) 먼저 전체 모델로 시도
    try:
        m = torch.load(model_path, map_location=dev, weights_only=False)
        if isinstance(m, nn.Module):
            m.eval()
            return m.to(dev)
    except Exception as e_full:
        # print(f"[loader] full-model load failed: {e_full}")
        pass

    # 2) state_dict로 시도
    try:
        obj = torch.load(model_path, map_location="cpu", weights_only=False)
        sd = None
        if isinstance(obj, dict):
            # 흔한 키들 우선 탐색
            for k in ("state_dict", "model_state", "model", "net", "weights"):
                if k in obj and isinstance(obj[k], dict):
                    sd = obj[k]
                    break
            # 키가 없이 곧바로 state_dict일 수 있음
            if sd is None and all(isinstance(k, str) for k in obj.keys()):
                sd = obj

        if sd is not None:
            model = VGG19_BiLSTM_Binary(num_classes=2)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            # 필요하면 missing/unexpected 로깅
            model.eval()
            return model.to(dev)
    except Exception as e_sd:
        # print(f"[loader] state_dict load failed: {e_sd}")
        pass

    raise RuntimeError(
        "모델 로드 실패: 저장 포맷(전체모델/스테이트딕트) 미스매치 또는 손상.\n"
        "Colab에서 다음 중 하나로 다시 저장하세요:\n"
        " - 전체 모델:  torch.save(model, 'vgg19_bilstm_binary.pt')\n"
        " - state_dict: torch.save({'state_dict': model.state_dict()}, 'vgg19_bilstm_binary.pt')"
    )




# ===================== 2. 유틸/디텍터 =====================
def clamp_box(x1,y1,x2,y2,w,h):
    x1=max(0,min(x1,w-1)); y1=max(0,min(y1,h-1))
    x2=max(0,min(x2,w-1)); y2=max(0,min(y2,h-1))
    return x1,y1,x2,y2

def square_with_margin(x1,y1,x2,y2,margin,w,h):
    cx,cy=(x1+x2)/2,(y1+y2)/2
    side=max(x2-x1,y2-y1); side=int(round(side*(1+margin)))
    x1n,y1n=int(round(cx-side/2)),int(round(cy-side/2))
    x2n,y2n=x1n+side,y1n+side
    return clamp_box(x1n,y1n,x2n,y2n,w,h)

def center_box(h,w,scale=0.5):
    bw,bh=int(w*scale),int(h*scale)
    x=(w-bw)//2; y=(h-bh)//2
    return x,y,x+bw,y+bh

def nms_boxes(boxes, iou_thr=0.45):
    if not boxes: return []
    arr=np.asarray(boxes,float)
    x1,y1,x2,y2,s=arr[:,0],arr[:,1],arr[:,2],arr[:,3],arr[:,4]
    areas=(x2-x1)*(y2-y1); order=s.argsort()[::-1]; keep=[]
    while order.size:
        i=order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]])
        yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]])
        yy2=np.minimum(y2[i],y2[order[1:]])
        w=np.maximum(0.0,xx2-xx1); h=np.maximum(0.0,yy2-yy1)
        inter=w*h
        iou=inter/(areas[i]+areas[order[1:]]-inter+1e-9)
        order=order[1:][iou<iou_thr]
    return [tuple(map(float,arr[i])) for i in keep]

class MobileNetSSDDog:
    def __init__(self, proto, weights, conf_thr=0.3, dog_class_id=12):
        if not Path(proto).exists() or not Path(weights).exists():
            raise FileNotFoundError("Doggie-smile 가중치가 없습니다. SSD_PROTO/SSD_WEIGHTS 경로 확인.")
        self.net=cv2.dnn.readNetFromCaffe(proto,weights)
        self.conf_thr=conf_thr; self.dog_class_id=dog_class_id

    def __call__(self,img):
        H,W=img.shape[:2]
        blob=cv2.dnn.blobFromImage(cv2.resize(img,(300,300)),0.007843,(300,300),(127.5,127.5,127.5))
        self.net.setInput(blob); out=self.net.forward()
        boxes=[]
        for i in range(out.shape[2]):
            cid=int(out[0,0,i,1]); conf=float(out[0,0,i,2])
            if conf<self.conf_thr or cid!=self.dog_class_id: continue
            x1=int(out[0,0,i,3]*W); y1=int(out[0,0,i,4]*H)
            x2=int(out[0,0,i,5]*W); y2=int(out[0,0,i,6]*H)
            x1,y1,x2,y2=clamp_box(x1,y1,x2,y2,W,H)
            boxes.append((x1,y1,x2,y2,conf))
        return nms_boxes(boxes)

def choose_largest(boxes):
    if not boxes: return None
    return max(boxes,key=lambda b:(b[2]-b[0])*(b[3]-b[1]))

def detect_face_bbox(img,dog_det):
    H,W=img.shape[:2]; dog_boxes=dog_det(img)
    if not dog_boxes: return None
    bx1,by1,bx2,by2,_=choose_largest(dog_boxes)
    bx1,by1,bx2,by2=clamp_box(int(bx1),int(by1),int(bx2),int(by2),W,H)
    return int(bx1),int(by1),int(bx2-bx1),int(by2-by1)

def process_video(video_path,out_dir,fps=10,out_size=224,margin=0.15):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    dog_det=MobileNetSSDDog(SSD_PROTO,SSD_WEIGHTS,conf_thr=0.3)
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    every_n=max(1,int(orig_fps//max(1,fps)))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pbar=tqdm(total=total,desc="frames",leave=False)
    stem=Path(video_path).stem; subdir=out_dir/stem; subdir.mkdir(parents=True,exist_ok=True)
    idx=saved=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if idx%every_n!=0:
            idx+=1; pbar.update(1); continue
        H,W=frame.shape[:2]
        det=detect_face_bbox(frame,dog_det)
        if det is None:
            cx1,cy1,cx2,cy2=center_box(H,W,0.5)
            x1,y1,x2,y2=square_with_margin(cx1,cy1,cx2,cy2,margin,W,H)
        else:
            x,y,w,h=det
            x1,y1,x2,y2=square_with_margin(x,y,x+w,y+h,margin,W,H)
        crop=cv2.resize(frame[y1:y2,x1:x2],(out_size,out_size))
        cv2.imwrite(str(subdir/f"{stem}_{saved:06d}.jpg"),crop)
        saved+=1; idx+=1; pbar.update(1)
    cap.release(); pbar.close()
    return subdir

def list_crops(crop_dir: Path)->List[Path]:
    return sorted([p for p in crop_dir.glob("*.jpg")])

def make_sequences(paths: List[Path], T=16, stride=8)->List[List[Path]]:
    seqs=[]; n=len(paths)
    for i in range(0,max(0,n-T+1),stride):
        seqs.append(paths[i:i+T])
    if not seqs and n>0:
        seqs=[paths+[paths[-1]]*(T-n)]
    return seqs

def load_seq_imgs(seqp: List[Path], size=224, norm="imagenet")->np.ndarray:
    imgs=[]
    for p in seqp:
        bgr=cv2.imread(str(p))
        if bgr is None:
            imgs.append(np.zeros((3,size,size),np.float32)); continue
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        rgb=cv2.resize(rgb,(size,size))
        arr=rgb.astype(np.float32)/255.0
        mean=np.array([0.485,0.456,0.406],np.float32)
        std =np.array([0.229,0.224,0.225],np.float32)
        arr=(arr-mean)/std
        arr=np.transpose(arr,(2,0,1))
        imgs.append(arr)
    return np.stack(imgs,axis=0)  # (T,C,H,W)

# ===================== 3. Torch 예측 =====================
def predict_torch(model_path:str, seq_batches, device:str="cuda"):
    dev=torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    add_safe_globals([VGG19_BiLSTM_Binary])
    model=torch.load(model_path,map_location=dev,weights_only=False)
    if hasattr(model,"to"): model=model.to(dev)
    if hasattr(model,"eval"): model.eval()

    preds=[]
    with torch.no_grad():
        for seqs in tqdm(seq_batches,desc="Torch infer",leave=False):
            X=np.stack(seqs,axis=0)      # (B,T,C,H,W)
            X=torch.from_numpy(X).to(dev)
            y=model(X)
            if isinstance(y,(tuple,list)): y=y[0]
            y=torch.softmax(y,dim=1)
            preds.extend(y.detach().cpu().numpy())
    return preds

def predict_torch_with_model(model: nn.Module, seq_batches, device="cuda"):
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model.eval()
    preds=[]
    with torch.no_grad():
        for seqs in tqdm(seq_batches, desc="Torch infer"):
            X = np.stack(seqs, axis=0)   # (B,T,C,H,W)
            X = torch.from_numpy(X).to(dev)
            y = model(X)
            y = torch.softmax(y, dim=1)
            preds.extend(y.detach().cpu().numpy())
    return preds


# ===================== 4. FastAPI =====================
app = FastAPI(title="Dog Emotion PT Inference API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

@app.get("/health")
def health():
    ok = Path(MODEL_PATH).exists() and Path(SSD_PROTO).exists() and Path(SSD_WEIGHTS).exists()
    return {"status": "ok" if ok else "missing_files",
            "model_exists": Path(MODEL_PATH).exists(),
            "ssd_prototxt_exists": Path(SSD_PROTO).exists(),
            "ssd_caffemodel_exists": Path(SSD_WEIGHTS).exists()}

@app.post("/predict")
async def predict(
    video: UploadFile = File(..., description="MP4/MOV 등 비디오 파일"),
    fps: int = Form(10),
    size: int = Form(224),
    T: int = Form(16),
    stride: int = Form(8),
    margin: float = Form(0.15),
    device: str = Form("cuda"),
    classes: str = Form(CLASSES_STR),
):
    # 세션 작업 폴더
    sid = uuid.uuid4().hex[:8]
    work = OUT_ROOT / sid
    crops_dir = work / "crops"
    out_csv = work / "preds.csv"
    annot_video = work / "annot.mp4"
    work.mkdir(parents=True, exist_ok=True)

    # 업로드 저장
    ext = Path(video.filename).suffix.lower()
    if ext not in [".mp4", ".mov", ".avi", ".mkv", ".wmv"]:
        raise HTTPException(status_code=400, detail="지원하지 않는 확장자입니다.")
    video_path = work / f"input{ext}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    try:
        # 1) 비디오 → 크롭 이미지
        crop_dir = process_video(
            video_path=str(video_path),
            out_dir=str(crops_dir),
            fps=fps, out_size=size, margin=margin
        )

        # 2) 시퀀스 생성/배칭
        frames = list_crops(crop_dir)
        if not frames:
            raise HTTPException(status_code=422, detail="크롭 프레임이 생성되지 않았습니다.")
        seq_paths = make_sequences(frames, T=T, stride=stride)

        batches=[]; cur=[]
        for sp in seq_paths:
            arr = load_seq_imgs(sp, size=size, norm="imagenet")
            cur.append(arr)
            if len(cur)>=4:  # batch=4 고정
                batches.append(cur); cur=[]
        if cur: batches.append(cur)

        # 3) 예측
        preds = predict_torch_with_model(_MODEL, batches, device=device)

        # 4) 결과 저장/요약
        classes_list = [c.strip() for c in classes.split(",") if c.strip()]
        rows=[]
        for i,(seq,prob) in enumerate(zip(seq_paths, preds)):
            files = [p.name for p in seq]
            row = {"seq_index": i, "first_frame": files[0], "last_frame": files[-1]}
            prob = np.array(prob).ravel()
            for j,cname in enumerate(classes_list):
                row[cname] = float(prob[j]) if j < len(prob) else float("nan")
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)

        agg = df[classes_list].mean().to_frame(name="mean_prob")
        agg_path = out_csv.with_suffix(".summary.csv")
        agg.to_csv(agg_path)

        # Positive만 별도 JSON
        positive_json = None
        if "Positive_Score" in agg.index:
            positive_json = {"Positive_Score_Mean": float(agg.loc["Positive_Score","mean_prob"])}
            with open(out_csv.with_suffix(".positive_score.json"), "w") as f:
                json.dump(positive_json, f, indent=4)

        return JSONResponse({
            "session_id": sid,
            "outputs": {
                "preds_csv": str(out_csv),
                "summary_csv": str(agg_path),
                "positive_json": str(out_csv.with_suffix(".positive_score.json")) if positive_json else None,
                "crops_dir": str(crop_dir),
                # annot_video 생성을 원하면 아래에서 추가 구현 가능
            },
            "summary": {
                "classes": classes_list,
                "mean_prob": {k: float(v) for k,v in agg["mean_prob"].to_dict().items()},
                "positive_only": positive_json
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    finally:
        try:
            await video.close()
        except Exception:
            pass

# @app.get("/", include_in_schema=False)
# def root():
#     # 문서로 리다이렉트
#     return RedirectResponse(url="/docs")

@app.on_event("startup")
def _startup():
    global _MODEL, _DEVICE
    _DEVICE = os.getenv("DEVICE", "cuda")
    mp = os.getenv("MODEL_PATH", "vgg19_bilstm_binary.pt")
    if not Path(mp).exists() or Path(mp).stat().st_size == 0:
        raise RuntimeError(f"MODEL_PATH not found or empty: {mp}")
    _MODEL = load_pt_model_safely(mp, device=_DEVICE)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def homepage():
    # 아주 간단한 업로드 폼 + JS
    return """
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Dog Emotion Inference</title>
<style>
  body{font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;margin:40px;max-width:880px}
  h1{margin:0 0 8px}
  .card{border:1px solid #ddd;border-radius:12px;padding:20px;margin:18px 0;box-shadow:0 2px 10px rgba(0,0,0,.04)}
  .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  input[type=file]{padding:8px}
  button{padding:10px 16px;border:0;border-radius:10px;background:#111;color:#fff;cursor:pointer}
  button[disabled]{opacity:.5;cursor:not-allowed}
  .muted{color:#666}
  .mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
  .result{font-size:18px}
  progress{width:200px}
  .kv{margin:4px 0}
  .ok{color:#0a7a2e}
  .warn{color:#b15a00}
  .err{color:#b00020}
</style>
</head>
<body>
  <h1>Dog Emotion Inference (Torch)</h1>
  <p class="muted">MP4 파일을 업로드하면 <b>Positive_Score</b> 계산</p>

  <div class="card">
    <form id="uploadForm">
      <div class="row">
        <input id="video" name="video" type="file" accept=".mp4,.mov,.avi,.mkv,.wmv" required />
        <button id="btn" type="submit">분석 시작</button>
        <span id="status" class="muted"></span>
      </div>

      <!-- 고급 옵션 (원하면 수정해서 사용) -->
      <details style="margin-top:10px">
        <summary>고급 옵션</summary>
        <div class="row" style="margin-top:8px">
          <label>fps <input type="number" id="fps" value="10" min="1" style="width:80px"></label>
          <label>size <input type="number" id="size" value="224" min="32" step="32" style="width:80px"></label>
          <label>T <input type="number" id="T" value="16" min="1" style="width:80px"></label>
          <label>stride <input type="number" id="stride" value="8" min="1" style="width:80px"></label>
          <label>margin <input type="number" id="margin" value="0.15" step="0.01" min="0" max="1" style="width:90px"></label>
          <label>device
            <select id="device">
              <option value="cuda">cuda</option>
              <option value="cpu">cpu</option>
            </select>
          </label>
        </div>
      </details>
    </form>
  </div>

  <div id="progressWrap" class="card" style="display:none">
    <div class="row">
      <progress id="prog" max="100" value="30"></progress>
      <span id="progText" class="muted">분석 중...</span>
    </div>
  </div>

  <div id="result" class="card" style="display:none">
    <h3>결과</h3>
    <div id="pos" class="result"></div>
    <div id="kv" class="mono"></div>
    <h4>산출물 경로</h4>
    <div id="paths" class="mono"></div>
  </div>

<script>
const $ = (id)=>document.getElementById(id);

$("uploadForm").addEventListener("submit", async (e)=>{
  e.preventDefault();
  const file = $("video").files[0];
  if(!file){ alert("MP4 파일을 선택해주세요."); return; }

  const fd = new FormData();
  fd.append("video", file);
  fd.append("fps", $("fps").value);
  fd.append("size", $("size").value);
  fd.append("T", $("T").value);
  fd.append("stride", $("stride").value);
  fd.append("margin", $("margin").value);
  fd.append("device", $("device").value);
  // 클래스 이름은 서버 기본값을 사용(환경변수 MODEL_PATH/CLASSES 로 제어 가능)
  // fd.append("classes", "Negative_Score, Positive_Score");

  $("btn").disabled = true;
  $("status").textContent = "업로드/분석 중…";
  $("progressWrap").style.display = "";
  $("prog").value = 30;
  $("progText").textContent = "비디오 전처리 중…";

  try{
    const res = await fetch("/predict", { method:"POST", body: fd });
    $("prog").value = 70; $("progText").textContent = "추론/집계 중…";

    if(!res.ok){
      const txt = await res.text();
      throw new Error(`서버 오류 (${res.status}) ${txt}`);
    }
    const data = await res.json();

    // Positive_Score
    let positive = null;
    if(data?.summary?.positive_only?.Positive_Score_Mean !== undefined){
      positive = data.summary.positive_only.Positive_Score_Mean;
    }else if(data?.summary?.mean_prob?.Positive_Score !== undefined){
      positive = data.summary.mean_prob.Positive_Score;
    }

    $("result").style.display = "";
    if(positive !== null){
      $("pos").innerHTML = `<b>Positive_Score (mean)</b>: <span class="ok">${(+positive).toFixed(4)}</span>`;
    }else{
      $("pos").innerHTML = `<span class="warn">Positive_Score 값을 찾지 못했습니다. (클래스 이름 확인)</span>`;
    }

    // 클래스별 평균
    const mp = data?.summary?.mean_prob || {};
    const kvLines = Object.keys(mp).map(k=>`${k.padEnd(16,' ')} : ${(+mp[k]).toFixed(4)}`);
    $("kv").innerText = kvLines.join("\\n");

    // 산출물 경로 (로컬 경로이므로 클릭해도 안 열릴 수 있어요)
    const out = data?.outputs || {};
    const pathLines = Object.keys(out).map(k=>`${k.padEnd(16,' ')} : ${out[k]}`);
    $("paths").innerText = pathLines.join("\\n");

    $("prog").value = 100; $("progText").textContent = "완료!";
    $("status").textContent = "완료";
  }catch(err){
    console.error(err);
    $("result").style.display = "";
    $("pos").innerHTML = `<span class="err">에러: ${err.message}</span>`;
    $("kv").innerText = "";
    $("paths").innerText = "";
    $("status").textContent = "실패";
  }finally{
    $("btn").disabled = false;
    setTimeout(()=>{ $("progressWrap").style.display = "none"; }, 800);
  }
});
</script>
</body>
</html>
    """



# 원하면 favicon도 간단히 무시/처리
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return JSONResponse(status_code=204, content=None)
