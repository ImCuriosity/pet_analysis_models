# main.py

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request # Request, HTMLResponse 추가 확인
from fastapi.responses import JSONResponse, HTMLResponse 
from fastapi.staticfiles import StaticFiles # 추가 확인
from fastapi.templating import Jinja2Templates # 추가 확인
from eeg_analyzer import EEGAnalyzer 
import io
import os

# FastAPI 애플리케이션 생성
app = FastAPI(title="EEG Analysis Service")

# ----------------- 정적 파일 및 템플릿 설정 -----------------
# 이 부분이 있는지, 그리고 'static' 폴더 이름이 맞는지 확인하세요.
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

# ----------------- 모델 로딩 설정 (서버 시작 시 1회) -----------------
ANALYZER: EEGAnalyzer = None

@app.on_event("startup")
async def startup_event():
    # ... (모델 로딩 함수 내용은 이전 답변 참고)
    global ANALYZER
    try:
        ANALYZER = EEGAnalyzer()
    except Exception as e:
        print(f"Error during model loading: {e}")

def get_analyzer() -> EEGAnalyzer:
    # ... (get_analyzer 함수 내용은 이전 답변 참고)
    if ANALYZER is None:
        raise HTTPException(
            status_code=503, 
            detail="Service not available: EEG Analyzer not initialized (Check terminal for model loading errors)."
        )
    return ANALYZER

# ----------------- 라우팅 정의 -----------------

# 1. 기본 경로: HTML 페이지 로드 (★★★★★ 이 부분이 가장 중요합니다 ★★★★★)
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # index.html 파일을 템플릿으로 반환합니다.
    return templates.TemplateResponse("index.html", {"request": request})

# 2. 분석 API 엔드포인트: 파일 업로드 및 분석
@app.post("/analyze/")
async def analyze_eeg(
    file: UploadFile = File(..., description="Excel file (.xlsx) containing EEG data in the first sheet."),
    analyzer: EEGAnalyzer = Depends(get_analyzer)
):
    # ... (analyze_eeg 함수 내용은 이전 답변 참고)
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Only Excel files (.xlsx or .xls) are accepted."
        )

    try:
        file_bytes = await file.read()
        analysis_result = analyzer.analyze_eeg_data(file_bytes)
        
        return JSONResponse(content={
            "filename": file.filename,
            "positive_percent": analysis_result["positive_percent"],
            "active_percent": analysis_result["active_percent"],
            "message": analysis_result["result_text"]
        })
        
    except Exception as e:
        print(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {e}")