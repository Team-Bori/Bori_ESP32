# 서버 테스트 파일 형식

서버가 보드에서 모델을 시험할 때 주는 입력 데이터 파일이다. 흐름:

```
서버 ──(테스트 파일)──▶ 라즈베리파이: esp_monitor.py run-test ──(시리얼 프레임)──▶ ESP32 추론
서버 ◀──(test_report JSON POST)── 라즈베리파이 ◀──(test_result / test_summary)── ESP32
```

- **전처리는 어디에서도 하지 않는다.** 파일에는 모델 입력 텐서에 그대로 들어갈 값(대개 int8로 양자화된 값)을 담는다.
  이미지 리사이즈, 흑백 변환, 음성 → 특징값 변환, 정규화·양자화는 파일을 만드는 쪽(서버·사용자)이 끝낸 상태여야 한다.
- 모델 파일과 별개다. 보드에 이미 올라간 모델(`info.model`)의 입력 형식에 맞는지 라즈베리파이가 보내기 전에 검사한다.

## 형식: NumPy `.npz` (또는 `.npy`)

| 배열 | 필수 | dtype | shape | 내용 |
| --- | --- | --- | --- | --- |
| `x` | 예 | 모델 입력 dtype (예: `int8`) | `[N, *입력 shape]` | 입력 샘플 N개 |
| `y` | 아니오 | 분류: 정수 또는 문자열 / 회귀: float | 분류 `[N]`, 회귀 `[N, 출력 값 수]` | 정답 |

- `.npy`는 `x`만 담은 파일로 취급한다 (정답 없음).
- 입력 shape의 앞쪽 배치 차원 1은 생략해도 된다. 모델 입력이 `[1, 28, 28, 1]`이면 `x`는 `[N, 1, 28, 28, 1]`과 `[N, 28, 28, 1]` 모두 허용.
  샘플이 하나면 `[28, 28, 1]`도 된다.
- dtype이 다르면 거부한다 (예: float32 `x`를 int8 모델에 보내면 `bad test file`). 자동 변환은 하지 않는다.
- 분류 `y`는 클래스 번호(0부터) 또는 레이블 문자열(모델 레이블과 정확히 같은 이름). `binary_score`는 0/1.
- `allow_pickle=False`로 읽는다. object 배열은 쓸 수 없다.
- 샘플 수 제한은 없다 (샘플을 하나씩 보내므로 보드 메모리와 무관). 시간은 [아래](#소요-시간)를 참고.

### 정답이 있을 때와 없을 때

| | 정답 있음 | 정답 없음 |
| --- | --- | --- |
| 지연 시간 (avg/min/max), 메모리, 에러율 | ✅ | ✅ |
| 정확도 (`accuracy`, 회귀는 `mean_abs_err`도) | ✅ | ❌ (`accuracy: null`) |

사용자가 올린 모델(명세 1.6)처럼 정답을 모르는 경우에도 성능 측정은 된다.

## 만드는 예 (Python)

```python
import numpy as np
# x_q: 이미 int8로 양자화된 입력, 모델 입력 [1, 28, 28, 1]
np.savez("mnist_test.npz", x=x_q.astype(np.int8), y=labels.astype(np.int64))
```

float 데이터를 양자화하는 방법 (입력 텐서의 scale, zero_point는 `info.model.input` 또는 모델 manifest에 있다):

```python
q = np.clip(np.round(x_float / scale) + zero_point, -128, 127).astype(np.int8)
```

패키지에 든 평가 샘플로 예제 파일 만들기 / 파일 검사:

```
python pc/bori_package.py make-test  models/<id>/default_model.bin models/<id>/test_data.npz
python pc/bori_package.py check-test models/<id>/test_data.npz models/<id>/default_model.bin
```

각 모델 폴더의 `test_data.npz`가 서버 연동 테스트용 예제다.

## 실행 (라즈베리파이)

```
python3 rpi/esp_monitor.py /dev/ttyUSB0 run-test test.npz --post http://<server>/api/metrics
```

| 옵션 | 기본 | 설명 |
| --- | --- | --- |
| `--baud` | 921600 | 전송 중 속도. 0이면 115200 유지. 실패하면 자동으로 115200으로 돌아감 |
| `--no-labels` | | `y`를 무시 (지연·메모리만) |
| `--limit N` | | 앞의 N개만 |
| `--report-outputs` | | 샘플마다 출력 값도 받기 |
| `--emit-samples` | | 샘플별 `test_result`를 stdout에 출력 (서버로는 보내지 않음) |
| `--sample-timeout` | 10 | 샘플당 허용 시간(초, 전송 시간에 더해짐) |

`--post`가 있으면 서버로 가는 것은 마지막 `test_report` 한 건이다 (샘플별 결과는 보내지 않는다).

### test_report (서버로 POST)

```json
{"type":"test_report","ok":true,"file":"mnist_test.npz","model":"cnn_mnist_int8","task":"classification",
 "samples":100,"labeled":100,"correct":98,"accuracy":0.98,
 "avg_us":1830.0,"min_us":1820,"max_us":1900,"invoke_errors":0,"error_rate":0.0,
 "memory":{"heap_free":0,"heap_min_free":0,"heap_total":0,"internal_free":0,"stack_free":0},"arena_used_bytes":0,
 "baud":921600,"transport_retries":0,"wall_s":2.4,"firmware_id":"tflm_runtime","cpu_freq_mhz":240,
 "port":"/dev/ttyUSB0","ts":1790000000.0}
```
(숫자는 형식 설명용 예시)

- 값이 없는 필드(정답 없음의 `accuracy`, 분류의 `mean_abs_err` 등)는 빠진다.
- 파일이 잘못되면 `{"type":"test_report","ok":false,"error":"bad test file: ..."}`이고 종료 코드 5.
- `transport_retries`: 시리얼 재전송 횟수 (통신 품질 지표).

## 소요 시간

샘플당 시간 ≈ 전송 시간 + 추론 시간 + 왕복 지연(수 ms). 전송 시간 = 샘플 바이트 × 10 / baud (계산값):

| 입력 | 바이트 | 115200 | 921600 |
| --- | --- | --- | --- |
| 28×28×1 int8 | 784 | 68 ms | 8.5 ms |
| 128×9 int8 | 1,152 | 100 ms | 12.5 ms |
| 49×40 int8 | 1,960 | 170 ms | 21 ms |
| 96×96×1 int8 | 9,216 | 800 ms | 100 ms |

보드 실측 (2026-10-05, CH9102, Windows COM6): 9,216 B 프레임이 115200에서 812.5 ms(10개), 921600에서 101.9 ms(40개)로
계산값과 거의 같았고 CRC 오류·재전송은 0건이었다. 속도 전환은 0.05~0.08초, 호스트가 따라오지 않으면 보드가 3.03초 후 115200으로 복귀했다.
