# 커스터마이징
## 1. 다른 모델 사용
적용 모델 : `facebook/bart-base`
<br>

디바이스  : `device='cuda'`
<br>

적용 이유 :
- 짧은 불만/문의 문장 분류에 성능이 안정적임
- 토큰 타입 의존이 없어서 전처리가 단순함
- `4-bit`/`동적 양자화` 모두 호환이 좋아 배포 최적화가 쉬움

## 2. 하이퍼파라미터 조정
```py
# 모델/토크나이저
config = ModelConfig(
    pretrained_model_name="facebook/bart-base",
    num_labels=3,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    max_length=64
)

# 학습 파라미터
num_epochs   = 10
learning_rate= 2e-5
batch_size   = 8
device       = "cuda"
```
조정 이유 : 
1. 과적합을 피하면서 수렴을 위해 `lr=2e-5`, `epochs=10`로 설정
2. GPU 메모리와 안정적 업데이트를 고려해 `batch_size=8`
3. LoRA는 `r=16` / `α=32` 로 가볍게 하되 표현력은 충분히 확보
4. 입력이 한두 문장 위주라 `max_length=64`로 불필요한 패딩을 줄여 속도 ↑

## 3. 모델 성능
![alt text](/text_classification-main/f1_score.png)
- 테스트 정확도 0.8863 
- F1(weighted) 0.8869 
- F1(macro) 0.8862
- F1(micro) 0.8863
- 세 클래스가 고르게 맞춰져 매크로/마이크로 F1이 비슷하게 나온 점이 안정적

## 4. 추론 결과
![alt text](/text_classification-main/inference.png)
- 예시 문장에 대해 billing / delivery / product가 명확히 구분가능
  - “You charged my old card again.” → billing (신뢰도 ~0.996)
  - “Tracking says delivered, but nothing arrived.” → delivery (신뢰도 ~0.996)
  - “Battery drains from 100% to 10% quickly.” → product (신뢰도 ~0.905)

- 라벨 경계가 뚜렷하고 확신도가 매우 높음

## 5. 최적화 결과
### 동적 양자화
![alt text](/text_classification-main/quantized_dynamic.png)
- 크기 -45.9% (828.6MB → 448.0MB)
- 처리량 +113.6% (≈8.7 → ≈18.7 infer/s)
- 평균 확률 차이 ~0.164: CPU 배포용으로 속도 이득이 크고, 정확도 영향은 허용 가능한 수준

### 4비트 양자화
![alt text](/text_classification-main/quantized_4bit.png)
- 크기 -66.5% (828.6MB → 277.7MB)
- 처리량 +227.6% (≈8.36 → ≈27.38 infer/s)
- 평균 확률 차이 ~0.101: GPU에서 속도/용량 이득이 가장 큼
- 구현 메모: LoRA 가중치를 `merge_and_unload()`로 병합한 뒤`models/text_classifier_merged` `NF4 + double-quant` 설정으로 로드하여 안정화

### 양자화 모델 활용 추론 시
`inference.py`의 `model_path`를

동적 양자화: `models/quantized_dynamic`
<br>

4비트 양자화: `models/quantized_4bit`

---

## 시작하기

### 1. 환경 설정

먼저 프로젝트 환경을 설정합니다:
1. `cd 커맨드 사용 시, 본인이 압축을 푼 디렉토리로 이동해야 합니다.`

2. conda 가상환경을 만듭니다.

3. `pip install -r requirements.txt`와 `python setup.py` 스크립트로 기본 환경을 설정합니다.
    - `pip install -r requirements.txt`: 라이브러리 설치
    - `python setup.py`: 폴더 생성 및 환경 셋팅
```bash
cd text_classifiaction-main
conda create -n text_classification python=3.11.8 -y
conda activate text_classification
python setup.py
pip install -r requirements.txt
```

이 스크립트는 다음을 수행합니다:
- Python 버전 확인
- 필요한 패키지 설치
- 디렉토리 생성
- 샘플 데이터 생성

**실행 후 생성되는 파일들:**
- `data/` 폴더와 샘플 파일들 (예시용)
- `models/`, `checkpoints/`, `logs/` 폴더들

### 2. 모델 훈련

```bash
python train.py
```

훈련 과정에서 다음을 확인할 수 있습니다:
- 모델 생성 및 설정
- 데이터 로딩 및 전처리
- 훈련 진행 상황
- 성능 평가 결과

### 3. 모델 추론

훈련된 모델을 사용하여 새로운 텍스트를 분류합니다:

```bash
python inference.py
```

추론 과정에서 다음을 경험할 수 있습니다:
- 훈련된 모델 로딩
- 텍스트 전처리
- 예측 실행
- 결과 해석

### 4. 모델 최적화 (선택사항)

훈련된 모델을 양자화하여 크기를 줄이고 속도를 향상시킵니다:

```bash
python quantization.py
```

