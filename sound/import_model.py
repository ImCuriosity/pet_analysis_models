#%%
# import
import torch
import torch.nn as nn  # nn 모듈 임포트 추가
from transformers import Wav2Vec2Processor, Wav2Vec2Config, Wav2Vec2Model # 필요한 모델 임포트
import os
from pydub import AudioSegment

#%%
# ClassificationHead의 정의 (이전에 사용했던 코드)
class ClassificationHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)
    def forward(self, hidden_states):
        hidden_states=self.dropout(hidden_states)
        hidden_states=self.dense(hidden_states)
        hidden_states=torch.tanh(hidden_states)
        hidden_states=self.dropout(hidden_states)
        hidden_states=self.out_proj(hidden_states)
        return hidden_states

# Wav2Vec2ForMultiLabelClassification의 정의 (오류 수정 및 변수 정의 포함)
# 참고: num_activeness_classes와 num_positive_classes가 config에 저장되어 있어야 안전합니다.
num_activeness_classes = 2 # 저장 시 사용했던 값으로 수정 필요
num_positive_classes = 2 # 저장 시 사용했던 값으로 수정 필요

class Wav2Vec2ForMultiLabelClassification(nn.Module):
    # 💡 오류 수정: __init 대신 __init__ 사용
    def __init__(self, config): 
        super().__init__()
        self.wav2vec2=Wav2Vec2Model(config)
        self.activeness_head=ClassificationHead(config, num_activeness_classes)
        self.positive_head=ClassificationHead(config,num_positive_classes)
        # forward 함수에서 사용하기 위해 클래스 변수로 저장
        self.num_activeness_classes = num_activeness_classes
        self.num_positive_classes = num_positive_classes

    def forward(self, input_values, attention_mask=None, 
                        activeness_labels=None, positive_labels=None):
        outputs=self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states=outputs[0]
        pooled_output=hidden_states.mean(dim=1)
        activeness_logits=self.activeness_head(pooled_output)
        positive_logits=self.positive_head(pooled_output)
        loss=None
        if activeness_labels is not None and positive_labels is not None:
            loss_fct=nn.CrossEntropyLoss()
            # 클래스 변수를 사용하여 로스 계산
            activeness_loss=loss_fct(activeness_logits.view(-1, self.num_activeness_classes),
                                     activeness_labels.view(-1))
            positive_loss=loss_fct(positive_logits.view(-1, self.num_positive_classes), 
                                    positive_labels.view(-1))
            loss=activeness_loss + positive_loss 
        return{
            'loss': loss,
            'activeness_logits': activeness_logits,
            'positive_logits': positive_logits
        }
    
#%%
# 저장된 모델 불러오기
saved_model_path = r'C:\workspace\final_proj\dog_sound\save_path'

try:
    # 모델 설정 로드
    config = Wav2Vec2Config.from_pretrained(saved_model_path)
    # 저장된 모델 가중치를 사용하여 모델 로드
    loaded_model = Wav2Vec2ForMultiLabelClassification(config)
    # 모델 가중치 파일 로드
    model_weights_path = os.path.join(saved_model_path, 'pytorch_model.bin')
    # 오류 해결 시도
    state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))
    loaded_model.load_state_dict(state_dict, strict=False)
    # 프로세서 로드
    loaded_processor = Wav2Vec2Processor.from_pretrained(saved_model_path)
    # 모델을 사용 가능한 장치(GPU or CPU)로 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded_model.to(device)
    loaded_model.eval()
    print(f"\n모델과 프로세서를 성공적으로 불러왔습니다. 장치: {device}")
except FileNotFoundError:
    print(f"오류: 지정된 경로에서 모델 파일을 찾을 수 없습니다. 경로를 확인해주세요: {saved_model_path}")
except Exception as e:
    print(f"모델을 불러오는 중 오류가 발생했습니다: {e}")

#%%
# 예: C:\Users\YourUser\Downloads\ffmpeg-6.0-full_build\bin\
FFMPEG_BIN_DIR = r'C:\workspace\final_proj\dog_sound\ffmpeg-8.0\bin' 

# FFmpeg 경로를 os.environ에 추가하여 pydub가 찾을 수 있게 함
# (단, 이 방식은 pydub/AudioSegment를 임포트하기 전에 수행되어야 합니다.)
os.environ["PATH"] += os.pathsep + FFMPEG_BIN_DIR
# os.environ['FFMPEG_PATH'] = os.path.join(FFMPEG_BIN_DIR, 'ffmpeg.exe') 
# os.environ['FFPROBE_PATH'] = os.path.join(FFMPEG_BIN_DIR, 'ffprobe.exe')

# ***************************************************************
# 2. .mp3 to .wav 변환 함수 (기존 코드)
# ***************************************************************
def mp3_to_wav(mp3_path, wav_path):
    """
    Converts an MP3 file to a WAV file.
    """
    try:
        # mp3 파일 여부 확인
        if not os.path.exists(mp3_path):
            raise FileNotFoundError(f'mp3 파일 찾을 수 없음: {mp3_path}')
        
        # mp3파일 로드 (pydub가 FFmpeg을 호출함)
        audio = AudioSegment.from_mp3(mp3_path)
        
        # wav 파일로 추출
        audio.export(wav_path, format='wav')
        
        print(f"성공적으로 변환: '{mp3_path}' to '{wav_path}'" )
        
    except FileNotFoundError as e:
        print(f'에러: {e}')
    except Exception as e:
        # FFmpeg 경로 설정이 잘못되었을 때 이 에러가 발생할 수 있습니다.
        print(f'변환 중 에러 (FFmpeg 경로 문제일 수 있음): {e}')


# ***************************************************************
# 3. 사용 예시
# ***************************************************************
# Raw string (r)을 사용하여 백슬래시 문제를 방지합니다.
# input_mp3_path = r'C:\workspace\final_proj\dog_sound\puppy_0004.mp3' 
# output_wav_path = r'C:\workspace\final_proj\dog_sound\puppy_0004_1.wav'

# 함수 호출
# if FFMPEG_BIN_DIR:
#     mp3_to_wav(input_mp3_path, output_wav_path)
# else:
#     print("🚨 FFMPEG_BIN_DIR 변수를 사용자의 FFmpeg 경로로 먼저 수정해야 합니다.")

#%%
# .mp3 to .wav 변환
# def mp3_to_wav(mp3_path, wav_path):
#     """
#     Converts an MP3 file to a WAV file.

#     Args:
#     mp3_path(str): The path to the input mp3 file.
#     wav_path(str): The path to save the output wav file.
#     """
#     try:
#         # mp3 파일 여부 확인
#         if not os.path.exists(mp3_path):
#             raise FileNotFoundError(f'mp3 파일 찾을 수 없음: {mp3_path}')
#         # mp3파일 로드
#         audio = AudioSegment.from_mp3(mp3_path)
#         # wav 파일로 추출
#         audio.export(wav_path, format='wav')
#         print(f"성공적으로 변환 '{mp3_path} to {wav_path}" )
#     except FileNotFoundError as e:
#         print(f'에러: {e}')
#     except Exception as e:
#         print(f'변환 중 에러: {e}')

# 사용 예시
# input_mp3_path = '/content/drive/MyDrive/ITWILL(기말플젝)/정현/your_audio.mp3' # Replace with your actual MP3 path
# output_wav_path = '/content/drive/MyDrive/ITWILL(기말플젝)/정현/your_audio.wav' # Replace with your desired output WAV path

# Call the function with your paths
# mp3_to_wav(input_mp3_path, output_wav_path)

# %%
# 필요한 라이브러리 임포트 (이전에 임포트되지 않은 것만)
import torch
import torchaudio
import numpy as np
import os

# 예측할 새로운 MP3 파일 경로 설정 (여기를 수정하세요!)
input_mp3_path = r'C:\workspace\final_proj\dog_sound\puppy_0004.mp3' # 실제 MP3 파일 경로로 변경해주세요.
# 변환된 WAV 파일을 저장할 경로 설정
output_wav_path = r'C:\workspace\final_proj\dog_sound\puppy_0004.wav' # 원하는 출력 WAV 파일 경로로 변경해주세요.

# 이전에 정의한 mp3_to_wav 함수가 있는지 확인
if 'mp3_to_wav' not in globals():
    print("mp3_to_wav 함수가 정의되지 않았습니다. 이전 단계를 실행하여 함수를 정의해주세요.")
elif 'loaded_model' not in locals() or 'loaded_processor' not in locals():
    print("불러온 모델(loaded_model) 또는 프로세서(loaded_processor)가 로드되지 않았습니다. 이전 단계를 실행하여 모델과 프로세서를 로드해주세요.")
else:
    try:
        # 1. MP3 파일을 WAV 파일로 변환
        print(f"MP3 파일을 WAV로 변환합니다: {input_mp3_path} -> {output_wav_path}")
        mp3_to_wav(input_mp3_path, output_wav_path)

        # 변환된 WAV 파일이 존재하는지 확인
        if not os.path.exists(output_wav_path):
            print(f"오류: WAV 파일 변환에 실패했거나 변환된 파일을 찾을 수 없습니다: {output_wav_path}")
        else:
            # 2. 변환된 WAV 파일 로드 및 전처리
            print(f"\n변환된 WAV 파일로 예측을 수행합니다: {output_wav_path}")
            speech_array, sampling_rate = torchaudio.load(output_wav_path)

            # 모델의 샘플링 속도 (16000 Hz)와 다를 경우 리샘플링
            target_sampling_rate = 16000
            if sampling_rate != target_sampling_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=target_sampling_rate)
                speech_array = resampler(speech_array)
                sampling_rate = target_sampling_rate

            # Wav2Vec2 입력 형식에 맞게 변환
            input_values = loaded_processor(speech_array.squeeze(0).numpy(), sampling_rate=sampling_rate, return_tensors="pt").input_values

            # 3. 불러온 모델로 예측
            loaded_model.eval()
            with torch.no_grad():
                # 💡 오류 수정: loaded_model.device 대신 정의된 device 변수 사용
                input_values = input_values.to(device)
                outputs = loaded_model(input_values)

            # 로짓 가져오기
            activeness_logits = outputs["activeness_logits"]
            positive_logits = outputs["positive_logits"]

            # 4. 결과 해석
            activeness_probs = torch.softmax(activeness_logits, dim=-1)
            positive_probs = torch.softmax(positive_logits, dim=-1)

            activeness_pred_id = torch.argmax(activeness_probs, dim=-1).item()
            positive_pred_id = torch.argmax(positive_probs, dim=-1).item()

            # 레이블 ID를 실제 레이블 문자열로 변환
            # activeness_label_map과 positive_label_map은 이전 단계에서 정의되었어야 합니다.
            if 'activeness_label_map' not in locals() or 'positive_label_map' not in locals():
                 print("경고: activeness_label_map 또는 positive_label_map이 정의되지 않았습니다. 레이블 문자열 변환이 올바르지 않을 수 있습니다.")
                 # 임시로 기본 맵 사용 (이전 코드 셀에서 사용된 값)
                 activeness_label_map = {0: 'passive', 1: 'active'}
                 positive_label_map = {0: 'negative', 1: 'positive'}

            # 💡 오류 수정: label_map의 value에서 key를 찾는 대신 key에서 value를 찾도록 수정
            # 또한, activeness_pred_id와 positive_pred_id는 이미 0 또는 1이므로, 이를 직접 label_map의 key로 사용합니다.
            activeness_pred_label = activeness_label_map.get(activeness_pred_id, f"Unknown ID: {activeness_pred_id}")
            positive_pred_label = positive_label_map.get(positive_pred_id, f"Unknown ID: {positive_pred_id}")


            print(f"예측된 Activeness: {activeness_pred_label} (확률: {activeness_probs[0][activeness_pred_id].item():.4f})")
            print(f"예측된 Positive: {positive_pred_label} (확률: {positive_probs[0][positive_pred_id].item():.4f})")

    except FileNotFoundError as e:
        print(f"오류: 파일 관련 오류 발생 - {e}")
    except Exception as e:
        print(f"오디오 변환 또는 예측 중 오류가 발생했습니다: {e}")

