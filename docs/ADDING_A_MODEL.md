# 모델 추가 가이드 (tflm_runtime)

`tflm_runtime` 펌웨어 위에 새 모델을 올리는 방법이다. **펌웨어는 고치지 않는다.**
할 일은 "학습/변환 → int8 양자화 → 샘플 준비 → 패키징 → 호스트 검증 → (사용자) 보드 검증 → 기록"이다.
런타임에 문제가 있으면(필요한 op가 없음, 아레나 부족 등) 직접 고치지 말고 [요청 사항](#펌웨어-변경이-필요할-때)으로 정리한다.

관련 문서: [PACKAGE_FORMAT.md](PACKAGE_FORMAT.md) · [PROTOCOL_v2.md](PROTOCOL_v2.md) · [TEST_DATA_FORMAT.md](TEST_DATA_FORMAT.md)

## 메모리·크기 예산

| 항목 | 값 | 근거 |
| --- | --- | --- |
| 패키지(.bin) 최대 크기 | **2,097,152 B (2 MB)** | 모델 파티션 0x200000~0x3FFFFF |
| tflite 위치 | 플래시 (mmap) | RAM을 쓰지 않음. 크기는 패키지 한도만 신경 쓰면 됨 |
| 최대 아레나 | **155,648 B (152 KB)** | 보드 실측 `info.model.max_arena_bytes` (2026-10-05, tflm_runtime 66a9038-dirty, 패키지 미로드 상태) |
| 기준 보드 | ESP32 rev 3.1, 240 MHz, PSRAM 없음, 플래시 4MB | `info.board` |

- 패키지를 올린 상태에서도 `max_arena_bytes`는 155,648 B로 같았다. 자동 아레나는 여기서 1KB를 남긴 154,624 B를 할당한다.
  모델 로드 실패 시 아레나는 반환된다 (힙 여유 287,464 B로 복귀 확인).
- **지연 시간 지표는 두 가지다.** `b`/`metrics`는 같은 입력을 연속 실행한 warm 캐시 값, 스트리밍(`run-test`)과 `i`는
  샘플당 1회 실행이라 플래시 캐시 미스가 포함된다. sine_regression에서는 `b` 41.7 µs, 스트리밍 188 µs로 4.5배 차이가 났다.
  아주 작은 모델은 빌드(코드 배치)만 바뀌어도 20% 안팎 달라지므로 실측값에는 펌웨어 버전을 함께 적는다.
  큰 모델일수록 비율 차이는 줄어들 것으로 예상(추정)하며, README에 둘 다 기록한다.
- **최대 아레나**는 부팅 후 아레나 할당 직전, 내부 RAM의 가장 큰 연속 블록이다 (`heap_caps_get_largest_free_block`).
  가용 힙 합계(약 300KB)가 아니라 **가장 큰 한 덩어리**가 기준이다.
- 아레나가 여기에 들어가지 않으면 `arena_alloc_failed`. 이 경우 모델을 줄여야 한다 (입력 해상도·채널·width multiplier 축소 등).

### 아레나 크기 정하기

1. 호스트 추정: `bori_package.py check`가 TFLM 레퍼런스 커널(x86_64)로 잰 `host arena`를 알려 준다.
2. **보드 실측이 최종값이다.** ESP32의 ESP-NN 최적화 커널은 conv/depthwise/fully_connected/softmax에서 scratch 버퍼를
   아레나에 추가로 요청하므로 호스트 값보다 커질 수 있다. 또 포인터 크기(64/32비트) 차이로 구조체 크기도 다르다.
3. manifest의 `build.arena_bytes`:
   - `0` (기본, 권장): **자동**. 보드가 가장 큰 블록을 아레나로 잡는다. 실제 사용량은 `info.model.arena_used_bytes`.
   - 숫자: 정확히 그만큼 할당. 모자라면 `arena_too_small` 오류 메시지에 보드가 잰 **실제 필요량**이 나온다.
   2단계 모델은 자동(0)으로 두고, 보드 실측 `arena_used_bytes`를 manifest `board`에 기록한다.

## 지원 op

`esp32_code/tflm_runtime/main/ops.def` (71개, 펌웨어와 호스트 검사가 같은 파일을 읽는다):

`ABS`, `ADD`, `ADD_N`, `ARG_MAX`, `ARG_MIN`, `AVERAGE_POOL_2D`, `BATCH_MATMUL`, `BATCH_TO_SPACE_ND`, `CAST`, `CONCATENATION`,
`CONV_2D`, `DEPTH_TO_SPACE`, `DEPTHWISE_CONV_2D`, `DEQUANTIZE`, `DIV`, `ELU`, `EQUAL`, `EXP`, `EXPAND_DIMS`, `FILL`, `FLOOR`,
`FULLY_CONNECTED`, `GATHER`, `GREATER`, `HARD_SWISH`, `L2_NORMALIZATION`, `LEAKY_RELU`, `LESS`, `LOG`, `LOG_SOFTMAX`, `LOGISTIC`,
`MAX_POOL_2D`, `MAXIMUM`, `MEAN`, `MINIMUM`, `MIRROR_PAD`, `MUL`, `NEG`, `PACK`, `PAD`, `PADV2`, `PRELU`, `QUANTIZE`,
`REDUCE_MAX`, `RELU`, `RELU6`, `RESHAPE`, `RESIZE_BILINEAR`, `RESIZE_NEAREST_NEIGHBOR`, `RSQRT`, `SHAPE`, `SLICE`, `SOFTMAX`,
`SPACE_TO_BATCH_ND`, `SPACE_TO_DEPTH`, `SPLIT`, `SPLIT_V`, `SQRT`, `SQUARE`, `SQUARED_DIFFERENCE`, `SQUEEZE`, `STRIDED_SLICE`,
`SUB`, `SUM`, `SVDF`, `TANH`, `TRANSPOSE`, `TRANSPOSE_CONV`, `UNIDIRECTIONAL_SEQUENCE_LSTM`, `UNPACK`, `ZEROS_LIKE`

- ESP-NN 최적화가 적용되는 op: ADD, MUL, CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, MAX/AVERAGE_POOL_2D, SOFTMAX (int8).
  나머지는 TFLM 레퍼런스 커널.
- Keras `Conv1D`는 변환 시 보통 `EXPAND_DIMS`/`RESHAPE` + `CONV_2D` + `SQUEEZE`/`RESHAPE`로 바뀐다.
  `GlobalAveragePooling1D/2D`는 `MEAN`. 모두 등록되어 있다. 실제 결과는 `bori_package.py inspect`의 `ops`로 확인한다.
- CUSTOM op, 여러 subgraph(IF/WHILE)는 지원하지 않는다.
- 런타임 버전: `espressif/esp-tflite-micro` 1.4.1 + `esp-nn` 1.4.0 (`esp32_code/tflm_runtime/main/idf_component.yml`, `dependencies.lock`).

## 모델 요건

- **입력 1개, 출력 1개.** 둘 다 int8 (int8 전체 양자화: `inference_input_type = inference_output_type = tf.int8`)
- 입출력 텐서는 per-tensor 양자화 (가중치는 per-channel 가능)
- 작업 종류(task): `classification`(출력 = 클래스 점수, 가장 큰 값이 예측), `binary_score`(출력 1개, 임계값 판정),
  `regression`(출력 값 그대로, 최대 64개)

## 폴더 구성

```
pc/models/<name>/            학습·변환·평가 스크립트 (prepare.py / train.py / evaluate.py ...), requirements.txt
models/<model_id>/
  model.tflite               int8 모델 (검토한 결과물만 커밋)
  manifest.json              모델 정보 + 패키지 빌드 설정 (아래)
  samples/*.npy              데모·평가 샘플 (패키지 빌드 입력)
  default_model.bin          패키지 (bori_package.py build 결과)
  test_data.npz              서버 연동용 예제 테스트 파일 (bori_package.py make-test)
  README.md                  출처·라이선스·구조·정확도·실측 결과
```

`model_id`: 소문자·숫자·`_`, 31자 이내 (예: `cnn_mnist_int8`). 폴더 이름과 같게 한다.

## manifest.json

`sine_regression`이 실제 예시다 ([models/sine_regression/manifest.json](../models/sine_regression/manifest.json)).

```json
{
  "model_id": "cnn_mnist_int8",
  "name": "MNIST small CNN (int8)",
  "description": "28x28 손글씨 숫자 분류",
  "task": "classification",
  "firmware": "tflm_runtime",
  "package_format": "BTF1",
  "files": {"tflite": "model.tflite", "package": "default_model.bin"},
  "build": {
    "arena_bytes": 0,
    "labels": ["0","1","2","3","4","5","6","7","8","9"],
    "demo_input": "samples/demo_input.npy",
    "demo_expected": [7],
    "eval_inputs": "samples/eval_inputs.npy",
    "eval_expected": "samples/eval_expected.npy"
  },
  "source": {"url": "...", "commit": "...", "license": "..."},
  "dataset": {"name": "...", "url": "...", "version": "...", "license": "..."},
  "host": {},
  "board": {}
}
```

| 키 | 누가 | 설명 |
| --- | --- | --- |
| `model_id`, `name`, `description`, `task`, `files`, `source`, `dataset` | 직접 | |
| `build.arena_bytes` | 직접 | 0 = 자동 (권장) |
| `build.task_param` | 직접 | `binary_score` 임계값(기본 0.5), `regression` 허용 오차 |
| `build.labels` | 직접 | 분류면 출력 순서대로 |
| `build.demo_input` | 직접 | `.npy` 경로, 샘플 1개, **입력 텐서 dtype 그대로** (int8) |
| `build.demo_expected` | 직접 | 값 목록 또는 `.npy`. 분류는 클래스 번호 |
| `build.eval_inputs` / `eval_expected` | 직접 | `[N, *입력 shape]` int8 / 분류 `[N]`, 회귀 `[N, 출력 수]` |
| `build.quantize_inputs` | 선택 | true면 float `.npy`를 입력 scale/zero_point로 양자화해서 넣음 |
| `firmware`, `package_format`, `input`, `output`, `ops`, `package` | `build`가 채움 | |
| `host` | `check --update-manifest`가 채움 | 호스트 아레나 추정, 호스트 int8 정확도 |
| `board` | 보드 검증 후 직접 | 아래 [기록](#결과-기록) |

## 절차

### 0. 환경

```powershell
pip install -r pc/requirements.txt          # numpy, tflite(flatbuffer 파서) 등
```

학습에 TensorFlow가 필요하면 모델 폴더의 `pc/models/<name>/requirements.txt`에 따로 적는다.

**호스트 검사(check)용 WSL 환경** (pip `tflite-micro`는 Linux x86_64 휠만 있다. 한 번만):

```powershell
wsl -d Ubuntu-22.04 -- bash -lc "python3 -m pip install --user virtualenv && python3 -m virtualenv ~/bori-tflm && ~/bori-tflm/bin/pip install tflite-micro numpy"
```

`bori_package.py check`가 Windows에서 자동으로 WSL(`Ubuntu-22.04`, `~/bori-tflm/bin/python`)을 호출한다.
다른 배포판/경로면 환경변수 `BORI_WSL_DISTRO`, `BORI_WSL_PYTHON`으로 바꾼다. Linux에서는 그 자리에서 실행한다.

### 1. 모델 준비

- 학습: 시드 고정, 재현 가능한 스크립트. 원본 사전학습 모델이면 출처·커밋을 고정해 내려받고 sha256을 검사한다
  (`pc/models/sine_regression/prepare.py` 참고).
- 변환: int8 전체 양자화 + 대표 데이터셋.

  ```python
  conv = tf.lite.TFLiteConverter.from_keras_model(model)
  conv.optimizations = [tf.lite.Optimize.DEFAULT]
  conv.representative_dataset = rep_data
  conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  conv.inference_input_type = tf.int8
  conv.inference_output_type = tf.int8
  ```
- 샘플: 평가 샘플은 테스트 분할에서 클래스 균등하게. 패키지 크기와 `a` 실행 시간(샘플 수 × 1회 추론 시간)을 고려한다.

### 2. 패키징과 호스트 검증

```powershell
python pc/bori_package.py build    models/<id>/manifest.json
python pc/bori_package.py validate models/<id>/default_model.bin
python pc/bori_package.py check    models/<id>/manifest.json --update-manifest
python pc/bori_package.py make-test models/<id>/default_model.bin models/<id>/test_data.npz
```

- `build`: tflite에서 입출력 텐서를 읽어 패키지를 만들고 manifest를 갱신한다. 마지막 줄 `checksum=0x...`가 보드 검증에 쓰인다.
- `validate`: 형식·crc·텐서 일치·**지원 op**·크기. 실패하면 보드에 올릴 필요가 없다.
- `check`: 호스트 TFLM으로 데모/평가 샘플을 돌려 **호스트 int8 정확도**와 아레나 추정치를 낸다.
  보드 `eval`은 이 값과 같아야 한다 (같은 샘플, 같은 판정 규칙).
- `inspect`: 패키지 내용 확인.

### 3. 보드 검증 (사용자가 실행)

에이전트는 플래시하지 않는다. 사용자에게 아래 명령을 안내하고 결과 로그를 받는다.
보드에 `tflm_runtime` 펌웨어가 올라가 있어야 한다.

Windows 개발 PC (ESP-IDF 환경을 활성화한 PowerShell, 포트 COM6):

```powershell
python -m esptool --chip esp32 -p COM6 -b 921600 write_flash 0x200000 models\<id>\default_model.bin
python rpi\esp_monitor.py COM6 verify --expect-checksum 0x........ --expect-firmware tflm_runtime
python rpi\esp_monitor.py COM6 info
python rpi\esp_monitor.py COM6 infer
python rpi\esp_monitor.py COM6 bench
python rpi\esp_monitor.py COM6 eval
python rpi\esp_monitor.py COM6 run-test models\<id>\test_data.npz
python rpi\esp_monitor.py COM6 listen --duration 20
```

라즈베리파이: `rpi/flash_local_model.sh /dev/ttyUSB0 models/<id>/default_model.bin` (펌웨어 호환 검사 + 쓰기 + verify).

확인할 것:
- `verify`가 `ok:true` (부팅, 재부팅 없음, 체크섬 일치)
- `info.model.arena_used_bytes`, `max_arena_bytes`
- `infer`가 데모 정답과 일치
- `eval.accuracy`가 호스트 `check`와 일치. 다르면 원인(양자화 파라미터, 입력 바이트 순서, 커널 차이)을 분석해 보고
- `run-test`의 `accuracy`가 `eval`과 일치 (같은 샘플이면)
- `listen` 중 `metrics`가 5초마다 오고 명령 응답이 늦지 않은지

### 결과 기록

manifest `board`에 실측값을 적는다 (추정치는 넣지 않는다):

```json
"board": {
  "measured_at": "2026-10-xx",
  "firmware": "tflm_runtime <fw_version>",
  "cpu_freq_mhz": 240,
  "latency_us": {"avg": 0, "min": 0, "max": 0, "source": "bench"},
  "arena_used_bytes": 0,
  "max_arena_bytes": 0,
  "eval_samples": 0,
  "eval_accuracy": 0.0,
  "eval_wall_ms": 0
}
```

그리고 카탈로그를 다시 만든다: `python pc/build_catalog.py` (`models/catalog.json`은 손으로 고치지 않는다).

## 펌웨어 변경이 필요할 때

직접 고치지 말고 PR 설명(또는 보고서)에 아래를 정리해 1단계 담당에게 요청한다.
- 필요한 op 이름과 그 op를 쓰는 레이어, `ops.def`에 추가할 줄 (`BORI_OP(Add..., NAME)`)
- 아레나: 보드 `max_arena_bytes` 대비 필요량(`arena_too_small` 메시지 또는 호스트 추정)과 시도한 축소안
- 그 밖의 런타임 제약 (입출력 2개 이상 등)

## PR 체크리스트

- [ ] 브랜치 `model/<name>`, 커밋 메시지 한국어
- [ ] 출처 URL·라이선스·원본 버전(커밋/릴리스)·데이터셋 인용을 README와 manifest에 기록, 재배포 허용 확인
- [ ] `validate` 통과, `check --update-manifest`로 호스트 정확도 기록
- [ ] `default_model.bin`, `test_data.npz`, `samples/`, `model.tflite`, 스크립트 커밋 (원본 데이터셋은 커밋하지 않음)
- [ ] 보드 실측(지연·아레나·정확도)을 manifest `board`와 README에 기록, 보드 `eval` = 호스트 정확도 확인
- [ ] `python pc/build_catalog.py` 실행 후 `models/catalog.json` 커밋
- [ ] 펌웨어(`esp32_code/`) 변경 없음
