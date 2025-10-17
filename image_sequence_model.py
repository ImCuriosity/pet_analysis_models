# 실행 예시: python image_sequence_model.py --video "Clip 4.mp4" --model "vgg19_bilstm_binary_resaved.keras" --framework keras

import os, math, json, re, argparse, sys
from pathlib import Path
from typing import List, Tuple, Optional, Sequence
import urllib.request
import cv2, numpy as np, pandas as pd
from tqdm import tqdm

# dlib optional
try:
    import dlib  # type: ignore
except Exception:
    dlib = None

# ----------------------- 설정 -----------------------
DOGGIE_ROOT = Path("Doggie-smile")                 # 가중치 저장 기본 폴더
SSD_PROTO   = DOGGIE_ROOT / "MobileNetSSD_deploy.prototxt"
SSD_WEIGHTS = DOGGIE_ROOT / "MobileNetSSD_deploy.caffemodel"
HEAD_DAT    = DOGGIE_ROOT / "dogHeadDetector.dat"  # dlib 미설치면 자동 비활성

# Doggie-smile repo
URLS = {
    "prototxt": "https://raw.githubusercontent.com/tureckova/Doggie-smile/master/MobileNetSSD_deploy.prototxt",
    "caffemodel": "https://github.com/tureckova/Doggie-smile/raw/master/MobileNetSSD_deploy.caffemodel",
    "head_dat": "https://github.com/tureckova/Doggie-smile/raw/master/dogHeadDetector.dat",
}



def _compat_custom_objects():
    import keras
    from keras.layers import Reshape

    class PatchedReshape(Reshape):
        def __init__(self, target_shape, **kwargs):
            # int → (int,), list → tuple 로 보정
            if isinstance(target_shape, int):
                target_shape = (target_shape,)
            elif isinstance(target_shape, list):
                target_shape = tuple(target_shape)
            super().__init__(target_shape=target_shape, **kwargs)

    return {"Reshape": PatchedReshape}


def load_model_compat(model_path: str):
    import keras
    return keras.models.load_model(
        model_path,
        compile=False,
        custom_objects=_compat_custom_objects(),
    )



def _download_if_missing(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    print(f"[downloading] {dst.name} ...")
    try:
        urllib.request.urlretrieve(url, dst.as_posix())
    except Exception as e:
        print(f"[warn] download failed: {dst.name} ({e})")

def ensure_weights():
    _download_if_missing(URLS["prototxt"], SSD_PROTO)
    _download_if_missing(URLS["caffemodel"], SSD_WEIGHTS)
    _download_if_missing(URLS["head_dat"], HEAD_DAT)

# ----------------------- 유틸 -----------------------
PATTERN = re.compile(
    r"^(?P<label>[A-Za-z]+)_(?P<dog>\d+)_(?P<frame>\d+)_(?P<aux>\d+)\.(?P<ext>jpg|jpeg|png|bmp|webp)$",
    re.IGNORECASE
)
def parse_name(fname: str):
    m = PATTERN.match(fname)
    if not m: return None
    g = m.groupdict()
    return {"label": g["label"], "dog_id": g["dog"], "frame_idx": int(g["frame"]),
            "aux": g["aux"], "ext": g["ext"]}

def clamp_box(x1:int,y1:int,x2:int,y2:int,w:int,h:int)->Tuple[int,int,int,int]:
    x1 = max(0, min(x1, w-1)); y1 = max(0, min(y1, h-1))
    x2 = max(0, min(x2, w-1)); y2 = max(0, min(y2, h-1))
    if x2 <= x1: x2 = min(w-1, x1+1)
    if y2 <= y1: y2 = min(h-1, y1+1)
    return x1,y1,x2,y2

def square_with_margin(x1:int,y1:int,x2:int,y2:int,margin:float,w:int,h:int)->Tuple[int,int,int,int]:
    cx, cy = (x1+x2)/2, (y1+y2)/2
    side = max(x2-x1, y2-y1)
    side = int(round(side*(1+margin)))
    x1n, y1n = int(round(cx-side/2)), int(round(cy-side/2))
    x2n, y2n = x1n+side, y1n+side
    return clamp_box(x1n,y1n,x2n,y2n,w,h)

def center_box(h:int,w:int,scale:float=0.5)->Tuple[int,int,int,int]:
    bw, bh = int(w*scale), int(h*scale)
    x = (w-bw)//2; y = (h-bh)//2
    return x, y, x+bw, y+bh

def nms_boxes(boxes: Sequence[Tuple[float,float,float,float,float]], iou_thr: float = 0.45):
    if not boxes: return []
    arr = np.asarray(boxes, dtype=float)
    x1,y1,x2,y2,s = arr[:,0],arr[:,1],arr[:,2],arr[:,3],arr[:,4]
    areas = (x2-x1)*(y2-y1)
    order = s.argsort()[::-1]
    keep=[]
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1); h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        iou = inter/(areas[i]+areas[order[1:]]-inter+1e-9)
        order = order[1:][iou < iou_thr]
    return [tuple(map(float, arr[i])) for i in keep]

# ----------------------- 디텍터 -----------------------
class MobileNetSSDDog:
    """Caffe MobileNet-SSD dog detector (VOC dog class_id=12)."""
    def __init__(self, proto:str, weights:str, conf_thr:float=0.3, dog_class_id:int=12):
        if not Path(proto).exists() or not Path(weights).exists():
            raise FileNotFoundError("MobileNet-SSD 가중치/프로토텍스트가 없습니다.")
        self.net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
        self.conf_thr = conf_thr
        self.dog_class_id = dog_class_id
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except Exception:
            pass

    def __call__(self, img_bgr)->List[Tuple[int,int,int,int,float]]:
        H,W = img_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img_bgr,(300,300)),
                                     scalefactor=0.007843, size=(300,300),
                                     mean=(127.5,127.5,127.5), swapRB=False, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        boxes=[]
        for i in range(out.shape[2]):
            cls_id = int(out[0,0,i,1]); conf = float(out[0,0,i,2])
            if conf < self.conf_thr or cls_id != self.dog_class_id: continue
            x1 = int(out[0,0,i,3]*W); y1 = int(out[0,0,i,4]*H)
            x2 = int(out[0,0,i,5]*W); y2 = int(out[0,0,i,6]*H)
            x1,y1,x2,y2 = clamp_box(x1,y1,x2,y2,W,H)
            boxes.append((x1,y1,x2,y2,conf))
        return nms_boxes(boxes, 0.45)

class DlibDogHead:
    """dlib MMOD (dogHeadDetector.dat)."""
    def __init__(self, detector_dat_path:str):
        if dlib is None:
            raise RuntimeError("dlib 미설치: head 검출 비활성.")
        if not Path(detector_dat_path).exists():
            raise FileNotFoundError("dogHeadDetector.dat이 없습니다.")
        try:
            self.net = dlib.cnn_face_detection_model_v1(detector_dat_path)  # type: ignore
            self.is_cnn = True
        except Exception:
            self.net = dlib.deserialize(detector_dat_path)  # type: ignore
            self.is_cnn = False

    def __call__(self, img_bgr, roi:Optional[Tuple[int,int,int,int]]=None):
        if roi is not None:
            x1,y1,x2,y2 = roi
            x1,y1 = max(0,x1), max(0,y1)
            sub = img_bgr[y1:y2, x1:x2]; offset=(x1,y1)
        else:
            sub = img_bgr; offset=(0,0)
        sub_rgb = cv2.cvtColor(sub, cv2.COLOR_BGR2RGB)
        dets = self.net(sub_rgb)
        out=[]
        for d in dets:
            r = d.rect if hasattr(d,"rect") else d
            score = float(getattr(d, "confidence", getattr(d, "detection_confidence", 1.0)))
            x1h,y1h = int(r.left()), int(r.top())
            x2h,y2h = int(r.right())+1, int(r.bottom())+1
            out.append((x1h+offset[0], y1h+offset[1], x2h+offset[0], y2h+offset[1], score))
        return out

def choose_best_head_for_dog(dog_box, heads):
    x1,y1,x2,y2,_ = dog_box
    best=None; best_area=-1
    for (hx1,hy1,hx2,hy2,hs) in heads:
        ix1=max(x1,hx1); iy1=max(y1,hy1)
        ix2=min(x2,hx2); iy2=min(y2,hy2)
        iw=max(0,ix2-ix1); ih=max(0,iy2-iy1)
        inter=iw*ih
        if inter<=0: continue
        harea=(hx2-hx1)*(hy2-hy1)
        if harea>best_area:
            best=(hx1,hy1,hx2,hy2,hs); best_area=harea
    return best

def choose_largest(boxes):
    if not boxes: return None
    return max(boxes, key=lambda b:(b[2]-b[0])*(b[3]-b[1]))

def detect_face_bbox(img_bgr, dog_det: MobileNetSSDDog, head_det: Optional[DlibDogHead]=None):
    H,W = img_bgr.shape[:2]
    dog_boxes = dog_det(img_bgr)
    if not dog_boxes: return None
    best_heads=[]
    if head_det is not None:
        for db in dog_boxes:
            roi = tuple(map(int, db[:4]))
            heads = head_det(img_bgr, roi=roi)
            best = choose_best_head_for_dog(db, heads)
            if best: best_heads.append(best)
    if best_heads:
        fx1,fy1,fx2,fy2,_ = choose_largest(best_heads)
        return int(fx1),int(fy1),int(fx2-fx1),int(fy2-fy1)
    bx1,by1,bx2,by2,_ = choose_largest(dog_boxes)
    bx1,by1,bx2,by2 = clamp_box(int(bx1),int(by1),int(bx2),int(by2), W,H)
    return int(bx1),int(by1),int(bx2-bx1),int(by2-by1)

# ----------------------- 비디오 처리 -----------------------
def process_video(video_path:str, out_dir:str="outputs/crops", sample_fps:int=10, out_size:int=224,
                  margin:float=0.15, ssd_conf:float=0.3,
                  ssd_proto:str=str(SSD_PROTO),
                  ssd_weights:str=str(SSD_WEIGHTS),
                  dog_class_id:int=12, head_dat:Optional[str]=str(HEAD_DAT),
                  center_fallback_scale:float=0.5):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dog_det = MobileNetSSDDog(ssd_proto, ssd_weights, conf_thr=ssd_conf, dog_class_id=dog_class_id)
    head_det = None
    if head_dat and Path(head_dat).exists() and dlib is not None:
        try: head_det = DlibDogHead(head_dat)
        except Exception as e: print("[warn] head detector disabled:", e)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video_path}")
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    every_n = max(1, int(math.floor(orig_fps/float(sample_fps)))) if sample_fps else 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT)>0 else None
    pbar = tqdm(total=total, desc="frames")

    idx, saved = 0, 0
    stem = Path(video_path).stem
    subdir = out_dir / stem
    subdir.mkdir(parents=True, exist_ok=True)

    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx % every_n != 0:
            idx += 1; 
            if total: pbar.update(1)
            continue
        H,W = frame.shape[:2]
        det = detect_face_bbox(frame, dog_det, head_det)
        if det is None:
            cx1,cy1,cx2,cy2 = center_box(H,W,scale=center_fallback_scale)
            x1,y1,x2,y2 = square_with_margin(cx1,cy1,cx2,cy2, margin, W,H)
        else:
            x,y,w,h = det
            x1,y1,x2,y2 = square_with_margin(x,y,x+w,y+h, margin, W,H)
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop,(out_size,out_size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(subdir / f"{stem}_{saved:06d}.jpg"), crop)
        saved += 1
        idx += 1; 
        if total: pbar.update(1)
    cap.release(); pbar.close()
    print(f"Saved {saved} crops → {subdir} (margin={int(margin*100)}%, size={out_size})")
    return subdir

# ----------------------- 시퀀스 & 로더 -----------------------
def list_crops(crop_dir: Path)->List[Path]:
    files = sorted([p for p in crop_dir.glob("*.jpg")])
    if not files:
        exts={".jpg",".jpeg",".png",".bmp",".webp"}
        files = sorted([p for p in crop_dir.iterdir() if p.suffix.lower() in exts])
    return files

def make_sequences(paths: List[Path], T:int=16, stride:int=8)->List[List[Path]]:
    seqs=[]
    n=len(paths)
    for i in range(0, max(0, n-T+1), stride):
        seqs.append(paths[i:i+T])
    if not seqs and n>0:
        seqs=[paths + [paths[-1]]*(T-n)]
    return seqs

def load_seq_imgs(seqp: List[Path], size:int, norm:str, framework:str)->np.ndarray:
    imgs=[]
    for p in seqp:
        bgr = cv2.imread(str(p))
        if bgr is None:
            img = np.zeros((size,size,3), np.float32) if framework=="keras" else np.zeros((3,size,size), np.float32)
            imgs.append(img); continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2]!=(size,size):
            rgb = cv2.resize(rgb,(size,size), interpolation=cv2.INTER_AREA)
        arr = rgb.astype(np.float32)
        if norm=="tf":
            arr /= 255.0
        elif norm=="imagenet":
            arr = arr/255.0
            mean = np.array([0.485,0.456,0.406], np.float32)
            std  = np.array([0.229,0.224,0.225], np.float32)
            arr = (arr-mean)/std
        if framework=="keras":
            imgs.append(arr)              # (HWC)
        else:
            arr = np.transpose(arr,(2,0,1))  # CHW
            imgs.append(arr)
    return np.stack(imgs, axis=0)  # (T,HWC) or (T,CHW)

# ----------------------- 추론 -----------------------
def predict_keras(model_path:str, seq_batches, batch_size:int=8):
    # keras 우선 로더 + Reshape 패치
    try:
        model = load_model_compat(model_path)
    except Exception:
        # 혹시 keras가 실패하면 tf.keras로 시도
        import tensorflow as tf
        model = tf.keras.models.load_model_compat(model_path, compile=False)

    preds=[]
    for seqs in tqdm(seq_batches, total=len(seq_batches), desc="Keras infer"):
        X = np.stack(seqs, axis=0)  # (B,T,H,W,C)
        y = model.predict(X, batch_size=min(batch_size,len(X)), verbose=0)
        preds.extend([y[i] for i in range(y.shape[0])])
    return preds


def predict_torch(model_path:str, seq_batches, batch_size:int=8, device:str="cuda"):
    import torch  # ← 여기서 임포트
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model = torch.load(model_path, map_location=dev)
    if hasattr(model, "eval"): model.eval()
    preds=[]
    with torch.no_grad():
        for seqs in tqdm(seq_batches, total=len(seq_batches), desc="Torch infer"):
            X = np.stack(seqs, axis=0)
            X = torch.from_numpy(X).to(dev)
            y = model(X)
            if isinstance(y,(tuple,list)): y=y[0]
            preds.extend(y.detach().cpu().numpy())
    return preds

# ----------------------- 파이프라인 -----------------------
def run_video_infer(
    video_path:str,
    model_path:str,
    framework:str="keras",
    classes:str="neutral,happy,fear,angry",
    # detection
    ssd_proto:str=str(SSD_PROTO),
    ssd_weights:str=str(SSD_WEIGHTS),
    head_dat:str=str(HEAD_DAT),
    dog_class_id:int=12, ssd_conf:float=0.3, margin:float=0.15,
    fps:int=10, size:int=224,
    # sequences
    T:int=16, stride:int=8, norm:str="tf",
    # batching/device
    batch:int=8, device:str="cuda",
    # outputs
    out_csv:str="outputs/preds.csv", annot_video:str="",
):
    ensure_weights()

    out_root = Path("outputs"); out_root.mkdir(parents=True, exist_ok=True)
    crop_dir = process_video(
        video_path=video_path, out_dir="outputs/infer_crops", sample_fps=fps, out_size=size,
        margin=margin, ssd_conf=ssd_conf, ssd_proto=ssd_proto, ssd_weights=ssd_weights,
        dog_class_id=dog_class_id, head_dat=head_dat, center_fallback_scale=0.5
    )

    frames = list_crops(crop_dir)
    if not frames: raise SystemExit("No crops produced. Check weights/paths or class id.")
    seq_paths = make_sequences(frames, T=T, stride=stride)

    batches=[]; cur=[]
    for sp in seq_paths:
        arr = load_seq_imgs(sp, size=size, norm=norm, framework=framework)
        cur.append(arr)
        if len(cur)>=batch:
            batches.append(cur); cur=[]
    if cur: batches.append(cur)

    if framework.lower()=="keras":
        preds = predict_keras(model_path, batches, batch_size=batch)
    else:
        preds = predict_torch(model_path, batches, batch_size=batch, device=device)

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
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved predictions → {out_csv} (sequences={len(df)})")

    agg = df[classes_list].mean().to_frame(name="mean_prob")
    agg_path = str(Path(out_csv).with_suffix(".summary.csv"))
    agg.to_csv(agg_path)
    print(f"Saved video-level summary → {agg_path}")

    if annot_video:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps0 = cap.get(cv2.CAP_PROP_FPS) or 30
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(annot_video, fourcc, fps0, (W,H))

            # 프레임에 간단히 현재 마지막 예측을 표시
            seq_len = T
            frame_idx=0; seq_ptr=0; current=None
            while True:
                ok, frame = cap.read()
                if not ok: break
                if seq_ptr < len(preds) and (frame_idx % max(1,int(round(fps0))) == 0):
                    current = preds[seq_ptr]; seq_ptr += 1
                if current is not None:
                    probs = np.array(current).ravel()
                    y0 = 28
                    for i, cname in enumerate(classes_list):
                        p = float(probs[i]) if i < len(probs) else 0.0
                        cv2.putText(frame, f"{cname}: {p:.2f}", (10, y0+i*24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10,220,10), 2, cv2.LINE_AA)
                out.write(frame); frame_idx+=1
            out.release(); cap.release()
            print(f"Annotated video → {annot_video}")
        else:
            print("[annot] cannot open src video; skipped.")

    return df

# ----------------------- CLI -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="입력 동영상 경로")
    parser.add_argument("--model", required=True, help="학습 모델 경로(.keras or .pt)")
    parser.add_argument("--framework", default="keras", choices=["keras","torch"])
    parser.add_argument("--classes", default="Negative_Score, Positive_Score",
                        help="모델 출력 순서대로 클래스명(콤마 구분)")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_csv", default="outputs/preds.csv")
    parser.add_argument("--annot", default="outputs/annot.mp4")
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--ssd_conf", type=float, default=0.3)
    args = parser.parse_args()

    run_video_infer(
        video_path=args.video,
        model_path=args.model,
        framework=args.framework,
        classes=args.classes,
        fps=args.fps, size=args.size,
        T=args.T, stride=args.stride,
        batch=args.batch, device=args.device,
        out_csv=args.out_csv, annot_video=args.annot,
        margin=args.margin, ssd_conf=args.ssd_conf
    )

if __name__ == "__main__":
    main()
