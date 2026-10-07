# 내 모델 올리기 (사용자 가이드)

보드 클라우딩에서 빌린 ESP32 보드에 **직접 학습한 모델**을 올려 성능(추론 속도, 메모리, 정확도, 에러율)을 측정하는 방법이다.
모델을 **BTF1 패키지(.bin)** 로 만들어 GitHub에 올리고, 그 파일 링크를 서비스에 제출하면 된다.
제출한 모델은 검수 없이 그대로 보드에 올라가므로, 아래 제한을 지키지 않으면 보드에서 오류가 난다 (보드가 망가지지는 않는다).

템플릿: 손글씨 숫자 CNN [`pc/models/cnn_mnist/train.py`](../pc/models/cnn_mnist/train.py) → [`models/cnn_mnist_int8/`](../models/cnn_mnist_int8/)

## 1. 보드와 제한

| 항목 | 제한 |
| --- | --- |
| 보드 | ESP32 (240 MHz, 듀얼코어), **PSRAM 없음**, 가속기 없음 |
| 런타임 | TensorFlow Lite Micro (esp-tflite-micro 1.4.1 + ESP-NN) |
| 패키지 크기 | 최대 **2 MB** (운영자가 더 작게 정할 수 있음) |
| 작업 메모리(아레나) | 최대 **155,648 B (152 KB)**. 중간 결과 텐서가 모두 여기에 들어가야 함 |
| 입력·출력 | **입력 1개, 출력 1개**, 둘 다 **int8** (int8 전체 양자화) |
| op | [지원 op 목록](ADDING_A_MODEL.md#지원-op)에 있는 것만. CUSTOM op, IF/WHILE 불가 |
| 작업 종류 | `classification`(클래스 점수), `binary_score`(점수 1개 + 임계값), `regression`(값 최대 64개) |

모델 가중치(tflite)는 플래시에서 바로 읽으므로 아레나 제한과 무관하다. 아레나를 차지하는 것은 **층 사이의 활성값(feature map)** 이다.
예: 96×96×8 int8 feature map 하나가 73,728 B. 입력 해상도와 앞쪽 층의 채널 수가 아레나를 좌우한다.

참고 실측값 (같은 보드, tflm_runtime 58964e0):

| 모델 | 크기 | 아레나 | 1회 추론 |
| --- | --- | --- | --- |
| 사인 회귀 (FC 3층) | 2.7 KB | 716 B | 0.04 ms |
| MNIST CNN (Conv 2층, 5k 파라미터, 19만 MAC) | 10.5 KB | 8.5 KB | 21.4 ms |
| 동작 인식 1D CNN (128×9 입력) | 10.5 KB | 5.9 KB | 16.0 ms |
| 키워드 인식 tiny_conv (49×40 입력) | 18.8 KB | 6.8 KB | 12.2 ms |
| 사람 감지 MobileNet v1 0.25 (96×96 흑백) | 293.5 KB | 80.4 KB | 382.7 ms |
| 이상 탐지 오토인코더 (32×640 입력, Dense 10층) | 311.7 KB | 74.9 KB | 685.3 ms |

## 2. 모델 만들기

### 2.1 학습

자유롭게 학습하되 위 제한 안에 들게 설계한다. 지원 op만 쓰려면 Keras 기본 층
(Conv2D, DepthwiseConv2D, Dense, MaxPooling, AveragePooling, GlobalAveragePooling, Flatten, Reshape, ReLU/ReLU6, softmax, sigmoid, tanh, Add, Concatenate 등)을 쓴다.

### 2.2 int8 전체 양자화 (필수)

```python
def representative():
    for x in calibration_images[:500]:          # 학습 데이터 일부, float32, 모델 입력과 같은 전처리
        yield [x[None].astype("float32")]

conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = representative
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8      # 빠뜨리면 입력이 float32로 남음
conv.inference_output_type = tf.int8
open("model.tflite", "wb").write(conv.convert())
```

### 2.3 샘플 준비 (선택이지만 권장)

패키지 안에 넣는 샘플이다. 모두 **양자화된 int8 값**으로 `.npy`에 저장한다.

| 파일 | shape | 용도 |
| --- | --- | --- |
| `demo_input.npy` | 입력 shape 그대로 (샘플 1개) | `i`(1회 추론), 벤치마크·주기 측정의 입력. 없으면 0 입력 |
| `demo_expected.npy` | `[1]` (클래스 번호) 또는 회귀 값 | 데모 정답 |
| `eval_inputs.npy` | `[N, *입력 shape]` | 보드 정확도 측정(`a`) |
| `eval_expected.npy` | 분류 `[N]`, 회귀 `[N, 출력 수]` | 정답 |

float 데이터 양자화: `q = clip(round(x / scale) + zero_point, -128, 127)` (scale, zero_point는 tflite 입력 텐서 값).
manifest에 `"quantize_inputs": true`를 쓰면 float `.npy`를 자동으로 양자화해 준다.

## 3. 패키지 만들기

저장소를 받아 `pip install -r pc/requirements.txt` 후, 폴더를 만든다 (`model_id`는 소문자·숫자·`_`, 31자 이내):

```
my_model/
  model.tflite
  manifest.json
  samples/*.npy
```

`manifest.json` (최소):

```json
{
  "model_id": "my_digits_v1",
  "name": "My digit classifier",
  "task": "classification",
  "files": {"tflite": "model.tflite", "package": "default_model.bin"},
  "build": {
    "arena_bytes": 0,
    "labels": ["0","1","2","3","4","5","6","7","8","9"],
    "demo_input": "samples/demo_input.npy",
    "demo_expected": "samples/demo_expected.npy",
    "eval_inputs": "samples/eval_inputs.npy",
    "eval_expected": "samples/eval_expected.npy"
  }
}
```

- `arena_bytes`: **0(자동) 권장.** 보드가 쓸 수 있는 가장 큰 메모리를 잡는다.
- `labels`: 분류면 출력 순서대로. 개수 = 출력 클래스 수.
- `binary_score`는 `"task_param": 0.5`(판정 임계값), `regression`은 허용 오차.

```
python pc/bori_package.py build    my_model/manifest.json
python pc/bori_package.py validate my_model/default_model.bin
```

`validate`가 `VALID`를 출력하면 보드에서 형식 오류가 나지 않는다. 출력의 `checksum=0x...`가 보드에 올라간 모델을 확인하는 값이다.
Linux(또는 WSL)에서는 `python pc/bori_package.py check my_model/manifest.json`으로 TFLM 인터프리터의 정확도와 아레나 추정치도 미리 볼 수 있다.

## 4. GitHub에 올리고 링크 제출

1. 공개(public) 저장소에 `default_model.bin`을 커밋한다.
2. GitHub에서 그 **파일**을 열고 주소를 복사해 제출한다.

| 링크 | 가능 여부 |
| --- | --- |
| `https://github.com/<user>/<repo>/blob/<branch>/path/default_model.bin` | ✅ (자동으로 raw 링크로 바꿈) |
| `https://raw.githubusercontent.com/<user>/<repo>/<branch>/path/default_model.bin` | ✅ |
| 릴리스 첨부 파일 링크 (`.../releases/download/...`) | ✅ |
| 저장소·폴더 링크 (`github.com/<user>/<repo>`, `.../tree/...`) | ❌ 파일 링크가 아님 |
| 비공개 저장소 | ❌ 내려받을 수 없음 |
| **Git LFS로 올린 파일** | ❌ raw 링크가 실제 파일이 아니라 LFS 포인터(텍스트)를 줌 → 배포 전 검사에서 거부 |

`https://`만 허용한다. 크기 제한을 넘는 파일은 내려받는 중에 거부된다.

## 5. 테스트 파일 (서버 성능 측정용)

서비스에서 "테스트 실행"을 할 때 주는 입력 데이터 파일이다. 형식: [TEST_DATA_FORMAT.md](TEST_DATA_FORMAT.md)

```python
np.savez("my_test.npz", x=x_int8, y=labels)    # x: [N, *입력 shape] int8, y: 정답(선택)
```

- **전처리는 서비스가 하지 않는다.** `x`는 모델 입력에 그대로 들어갈 int8 값이어야 한다.
- `y`가 있으면 정확도까지, 없으면 속도·메모리·에러율만 측정한다.
- 패키지의 평가 샘플로 예제 파일 만들기: `python pc/bori_package.py make-test my_model/default_model.bin my_test.npz`

## 6. 자주 나는 오류

파일 형식·체크섬은 보드에 쓰기 전에 라즈베리파이가 검사해 거부한다("배포 전 거부").
나머지는 보드가 로드할 때 잡아내며, 보드는 재부팅하지 않고 오류 코드를 남긴다.

| 오류 코드 / 증상 | 원인 | 해결 |
| --- | --- | --- |
| `unsupported_op` (메시지에 op 이름) | 런타임에 없는 op | 그 op를 만드는 층을 지원 op로 바꾼다. 예: `tf.image.resize`·사용자 정의 층 제거, `Lambda` 대신 기본 층. SELECT_TF_OPS 사용 금지 |
| `CUSTOM(...)` | TF 연산이 그대로 들어감 | 변환 시 `supported_ops = [TFLITE_BUILTINS_INT8]`만 지정, 해당 층 교체 |
| `arena_alloc_failed` | 중간 텐서가 152 KB에 안 들어감 | 입력 해상도 축소, 앞쪽 층 채널 축소, stride 2로 일찍 줄이기, width multiplier 축소 |
| `arena_too_small` | manifest의 `arena_bytes`가 작음 | 메시지의 "model needs N bytes" 이상으로 하거나 0(자동) |
| `package_invalid` … `input tensor does not match` | 패키지를 만든 뒤 tflite만 바꿈 | `bori_package.py build`로 다시 만든다 |
| `package_invalid` … `1 input and 1 output` | 입력/출력이 여러 개 | 모델을 단일 입력·단일 출력으로 |
| 배포 전 거부: `unknown model format` | .bin이 BTF1 패키지가 아님 (tflite 그대로, LFS 포인터, HTML 페이지) | `bori_package.py build` 결과물을 일반 Git으로 커밋, 파일 링크 확인 |
| 배포 전 거부: `checksum_mismatch` | 파일 손상 (부분 다운로드, 편집) | 다시 빌드·업로드 |
| `validate`의 경고 "input tensor is float32" | 양자화 누락 (`inference_input_type` 미지정) | 2.2의 변환 설정을 모두 지정 |
| 정확도가 학습 때보다 크게 낮음 | 대표 데이터의 전처리가 학습과 다름, 또는 테스트 파일을 float/다른 스케일로 만듦 | 대표 데이터·테스트 파일 모두 학습과 같은 전처리 후 입력 scale/zero_point로 양자화 |
| `model_crashed` | 이전 부팅에서 이 모델을 실행하다 보드가 리셋됨 | 다른 모델을 올리면 정상 동작. 재현되면 운영자에게 알림 |
| `test_report`의 `bad test file` | 테스트 파일 dtype/shape가 입력과 다름 | `x`를 int8, `[N, *입력 shape]`로 |

## 7. 측정 결과 읽기

| 지표 | 의미 |
| --- | --- |
| `avg_us` (bench/metrics) | 같은 입력을 연속 실행한 평균 추론 시간 (캐시가 데워진 상태) |
| `avg_us` (테스트 실행) | 샘플을 하나씩 받아 1회씩 실행한 평균. 작은 모델은 bench보다 크게 나온다 |
| `arena_used_bytes` | 모델이 실제로 쓴 작업 메모리 |
| `accuracy` | 정답이 있는 샘플 중 맞힌 비율 (회귀는 허용 오차 안의 비율, `mean_abs_err` 함께) |
| `invoke_errors`, `error_rate` | 추론 실패 수와 비율 |
