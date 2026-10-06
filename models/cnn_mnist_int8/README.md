# cnn_mnist_int8

28×28 흑백 손글씨 숫자(0~9)를 분류하는 소형 CNN이다. 이미지 입력 + 합성곱 구조의 대표 예제이고,
사용자가 자기 모델을 올릴 때 따라 할 **기준 템플릿**이다 ([docs/USER_MODEL_GUIDE.md](../../docs/USER_MODEL_GUIDE.md)).

## 구조

| 층 | 출력 | 파라미터 |
| --- | --- | --- |
| 입력 | 28×28×1 int8 | |
| Conv2D 8, 3×3, ReLU | 26×26×8 | 80 |
| MaxPool 2×2 | 13×13×8 | |
| Conv2D 16, 3×3, ReLU | 11×11×16 | 1,168 |
| MaxPool 2×2 | 5×5×16 | |
| Flatten → Dense 10, softmax | 10 int8 | 4,010 |
| 합계 | | 5,258 (약 19만 MAC) |

tflite op: CONV_2D, MAX_POOL_2D, SHAPE, STRIDED_SLICE, PACK, RESHAPE, FULLY_CONNECTED, SOFTMAX (모두 tflm_runtime 지원).
SHAPE/STRIDED_SLICE/PACK은 Keras `Flatten`이 만드는 동적 reshape다.

## 학습과 양자화

`pc/models/cnn_mnist/train.py` (요구 패키지 `pc/models/cnn_mnist/requirements.txt`)

- 데이터: 공식 분할 60,000 학습 / 10,000 테스트. 학습 중 검증은 학습 데이터의 10%
- 전처리: 픽셀 / 255 (0~1 float). 그 외 없음
- Adam 1e-3, batch 128, 8 epoch, seed 42, TF op determinism
- int8 전체 양자화: 대표 데이터 학습 이미지 500장, 입력·출력 int8
- **재현성:** 같은 환경(TF 2.21.0, numpy 2.2.6, Python 3.10.11, Windows CPU)에서 다시 학습하면 tflite가 바이트 단위로 같다 (sha256 `dc6d5cef…bfab`)

입력 양자화는 scale 1/255, zero_point −128이라 **int8 입력값 = 픽셀값(0~255) − 128**이다.
서버·사용자는 MNIST 형식 이미지(흰 글씨, 검은 배경)를 이렇게만 바꿔 테스트 파일을 만들면 된다.

## 출처와 라이선스

| 항목 | 내용 |
| --- | --- |
| 데이터셋 | MNIST, `tf.keras.datasets.mnist.load_data()` (https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz) |
| 데이터 라이선스 | **CC BY-SA 3.0**, 저작권 Yann LeCun, Corinna Cortes (Keras 문서 기준). NIST 데이터의 파생물 |
| 인용 | Y. LeCun, C. Cortes, C. J. C. Burges, *The MNIST database of handwritten digits* |
| 모델 | 이 저장소에서 직접 설계·학습 (외부 사전학습 모델 없음) |

패키지의 평가·데모 샘플과 `test_data.npz`는 MNIST 테스트 이미지를 양자화한 파생물이므로 CC BY-SA 3.0을 따른다
(출처 표시, 같은 조건으로 배포).

## 샘플

- 평가: 테스트 분할에서 클래스마다 앞쪽 20장, 총 200장 (클래스 균등). 패키지 169,216 B 중 156,800 B
- 데모: 테스트 인덱스 139 (정답 4, int8 점수 0.996), 평가 샘플과 겹치지 않음
- `test_data.npz`: 평가 샘플 200장과 정답 (서버 연동 예제)

## 결과

| 항목 | 호스트 | 보드 (ESP32 240 MHz, tflm_runtime b2a7e8c, 2026-10-06) |
| --- | --- | --- |
| float 테스트 정확도 (10,000장) | 98.28% | — |
| **int8 테스트 정확도 (10,000장)** | **98.26%** (목표 98% 이상) | — |
| int8 평가 샘플 정확도 (200장) | 199/200 (TF Lite, TFLM 모두) | **199/200 — 일치** |
| 틀린 샘플 | 평가 인덱스 186: 정답 9 → 5로 예측 (점수 0.77) | 같음 |
| 데모 | 4 (0.996) | 4 (0.996) |
| tflite / 패키지 크기 | 10,552 B / 169,216 B | |
| 아레나 | 9,600 B (호스트 64비트) | **8,524 B** 사용 (최대 155,648 B) |
| 지연 `b` (92회, warm) | | 평균 **21.63 ms** (최소 21.63, 최대 21.65) ≈ 46 FPS |
| 지연 `a` (200장) | | 평균 21.74 ms, 전체 4.4초 |
| 지연 스트리밍 (200장) | | 평균 22.56 ms, 921600 baud 전체 7.3초, 재전송 0 |
| 주기 metrics | | 0.2초 예산에 10회 |

참고: 기존 MLP(8×8 입력, 64-16-10, 1,184 MAC)는 0.1 ms(160 MHz)다. 이 CNN은 MAC 수가 약 160배이고 시간은 약 210배다.
1 MAC당 약 27 사이클로, 채널 수가 작은 층에서 ESP-NN generic 커널(ESP32는 S3의 SIMD 없음)의
층별 고정 비용이 큰 것으로 추정한다. 펌웨어를 고치지 않는 범위라 층별 프로파일링은 하지 않았다.

## 다시 만들기

```powershell
python pc/models/cnn_mnist/train.py
python pc/bori_package.py build models/cnn_mnist_int8/manifest.json
python pc/bori_package.py check models/cnn_mnist_int8/manifest.json --update-manifest
python pc/bori_package.py make-test models/cnn_mnist_int8/default_model.bin models/cnn_mnist_int8/test_data.npz
```
