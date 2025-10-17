# 실행 예시: python image_model_pt.py --video "Clip 4.mp4" --model "vgg19_bilstm_binary.pt" --framework torch

import os, cv2, re, math, json, numpy as np, pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Sequence
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models
from torch.serialization import add_safe_globals

# ===================== 1. 환경 설정 =====================
MODEL_PATH = "vgg19_bilstm_binary.pt"   # Colab에서 저장한 .pt 파일
VIDEO_PATH = "Clip 4.mp4"                # 추론할 영상
CLASSES = "Negative_Score, Positive_Score"

# Detector 경로 (Doggie-smile 폴더 포함 필수)
SSD_PROTO   = "Doggie-smile/MobileNetSSD_deploy.prototxt"
SSD_WEIGHTS = "Doggie-smile/MobileNetSSD_deploy.caffemodel"
HEAD_DAT    = "Doggie-smile/dogHeadDetector.dat"

OUT_CSV     = "outputs/preds.csv"
ANNOT_VIDEO = "outputs/annot.mp4"
os.makedirs("outputs", exist_ok=True)

# ===================== 2. 모델 클래스 정의 =====================
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

# ===================== 3. 유틸 함수 =====================
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

# ===================== 4. 탐지기 =====================
class MobileNetSSDDog:
    def __init__(self, proto, weights, conf_thr=0.3, dog_class_id=12):
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

# ===================== 5. 비디오 크롭 =====================
def process_video(video_path,out_dir="outputs/infer_crops",fps=10,out_size=224,margin=0.15):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    dog_det=MobileNetSSDDog(SSD_PROTO,SSD_WEIGHTS,conf_thr=0.3)
    cap=cv2.VideoCapture(video_path)
    every_n=max(1,int(cap.get(cv2.CAP_PROP_FPS)//fps))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar=tqdm(total=total,desc="frames")
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
    print(f"Saved {saved} crops → {subdir}")
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
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        rgb=cv2.resize(rgb,(size,size))
        arr=rgb.astype(np.float32)/255.0
        mean=np.array([0.485,0.456,0.406],np.float32)
        std =np.array([0.229,0.224,0.225],np.float32)
        arr=(arr-mean)/std
        arr=np.transpose(arr,(2,0,1))
        imgs.append(arr)
    return np.stack(imgs,axis=0)  # (T,C,H,W)

# ===================== 6. PyTorch 추론 =====================
def predict_torch(model_path:str, seq_batches, batch_size=8, device="cuda"):
    dev=torch.device(device if torch.cuda.is_available() else "cpu")
    add_safe_globals([VGG19_BiLSTM_Binary])
    model=torch.load(model_path,map_location=dev,weights_only=False)
    model.eval()

    preds=[]
    with torch.no_grad():
        for seqs in tqdm(seq_batches,desc="Torch infer"):
            X=np.stack(seqs,axis=0)
            X=torch.from_numpy(X).to(dev)
            y=model(X)
            y=torch.softmax(y,dim=1)
            preds.extend(y.detach().cpu().numpy())
    return preds

# ===================== 7. 메인 =====================
def run_video_infer():
    crop_dir=process_video(VIDEO_PATH)
    frames=list_crops(crop_dir)
    seqs=make_sequences(frames)
    batches=[]
    cur=[]
    for sp in seqs:
        arr=load_seq_imgs(sp)
        cur.append(arr)
        if len(cur)>=4:
            batches.append(cur); cur=[]
    if cur: batches.append(cur)

    preds=predict_torch(MODEL_PATH,batches)
    classes=[c.strip() for c in CLASSES.split(",")]
    rows=[]
    for i,(seq,prob) in enumerate(zip(seqs,preds)):
        row={"seq_index":i,"first":seq[0].name,"last":seq[-1].name}
        prob=np.array(prob).ravel()
        for j,c in enumerate(classes):
            row[c]=float(prob[j]) if j<len(prob) else np.nan
        rows.append(row)
    df=pd.DataFrame(rows)
    df.to_csv(OUT_CSV,index=False)
    print(f"[OUTPUT] Saved → {OUT_CSV}")

    agg=df[classes].mean().to_frame(name="mean_prob")
    agg_path=OUT_CSV.replace(".csv",".summary.csv")
    agg.to_csv(agg_path)
    print(f"[OUTPUT] Summary → {agg_path}")

    try:
        pos=float(agg.loc["Positive_Score","mean_prob"])
        json_out={"Positive_Score_Mean":pos}
        json_path=OUT_CSV.replace(".csv",".positive_score.json")
        with open(json_path,"w") as f: json.dump(json_out,f,indent=4)
        print(f"[OUTPUT] JSON → {json_path}")
    except Exception as e:
        print("[WARN] Positive_Score not found:",e)

if __name__=="__main__":
    run_video_infer()
