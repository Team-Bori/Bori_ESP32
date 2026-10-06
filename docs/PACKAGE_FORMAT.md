# 모델 패키지 형식 (BTF1)

`tflm_runtime` 펌웨어가 모델 파티션에서 읽는 파일 형식이다. 하나의 `.bin`에 tflite 모델, 입출력 텐서 정보,
레이블, 데모 입력, 정확도 평가 샘플이 모두 들어 있다.

- 만드는 법: `python pc/bori_package.py build models/<id>/manifest.json` ([ADDING_A_MODEL.md](ADDING_A_MODEL.md))
- 참조 구현: 펌웨어 `esp32_code/tflm_runtime/main/package.c`, 호스트 `pc/btf_format.py` (둘의 검사 규칙은 같다)
- 기존 MLP 형식(`MLP1`, `pc/model_format.py`)은 그대로 유지되며 `mlp` 펌웨어만 읽는다.
  `pc/validate_model.py`가 앞 4바이트(magic)로 두 형식을 구분한다.

## 플래시 배치

| 이름 | 종류 | 오프셋 | 크기 |
| --- | --- | --- | --- |
| nvs | data/nvs | 0x9000 | 0x6000 |
| phy_init | data/phy | 0xF000 | 0x1000 |
| factory | app | 0x10000 | 0x1F0000 (1,984 KB) |
| model | data/0x40 | 0x200000 | 0x200000 (2 MB) |

- 두 펌웨어(`mlp_esp32`, `tflm_runtime`)가 같은 파티션 표를 쓴다. `rpi/common.sh`의
  `MODEL_OFFSET`/`MODEL_PARTITION_BYTES`/`FIRMWARE_LIMIT_BYTES`, `pc/btf_format.py`의 상수와 맞춰야 한다.
- 패키지는 항상 `0x200000`에 쓴다. 최대 크기는 2,097,152 바이트.
- **파티션 표 변경(0x1E0000/128KB → 0x200000/2MB) 이전에 배포된 보드는 새 펌웨어를 0x0부터 다시 써야 한다.**
  예전 펌웨어 이미지는 `pc/validate_firmware.py`가 거부한다 (배포 스크립트가 쓰기 전에 검사).
  이전 파티션 표의 보드에 새 위치(0x200000)로 모델만 쓰면 펌웨어가 찾지 못한다.
- 펌웨어는 패키지를 `esp_partition_mmap`으로 주소 공간에 매핑해 쓴다. tflite 모델은 **RAM에 복사되지 않는다**.

## 전체 구조

모든 정수는 리틀엔디언. 오프셋은 패키지 시작 기준.

```
+--------------------+ 0
| 헤더 (64 B)        |
+--------------------+ 64
| 섹션 표 (16 B × n) |
+--------------------+
| 섹션 데이터 ...     |  tflite 섹션은 16바이트 정렬, 나머지는 4바이트 정렬 (사이는 0으로 채움)
+--------------------+ total_size
```

### 헤더 (64 바이트)

| 오프셋 | 크기 | 필드 | 설명 |
| --- | --- | --- | --- |
| 0 | 4 | magic | `BTF1` |
| 4 | 2 | format_version | 1 |
| 6 | 2 | header_size | 64 (펌웨어는 ≥64를 허용, 섹션 표는 header_size 위치에서 시작) |
| 8 | 4 | total_size | 패키지 전체 바이트 수 |
| 12 | 4 | crc32 | **오프셋 16부터 total_size까지** 의 CRC-32 (IEEE, Python `zlib.crc32`와 같음) |
| 16 | 32 | model_id | `[a-z0-9_]` 1~31자, 나머지 0 (NUL 종료 필수) |
| 48 | 1 | task | 0 `classification`, 1 `regression`, 2 `binary_score` |
| 49 | 1 | section_count | 섹션 표 항목 수 |
| 50 | 2 | label_dim | 샘플당 정답 값 개수. 분류·이진 = 1, 회귀 = 출력 원소 수 (1~64) |
| 52 | 4 | arena_bytes | 텐서 아레나 크기. **0 = 자동** (보드에서 할당 가능한 가장 큰 블록) |
| 56 | 4 | task_param (float32) | `binary_score`: 판정 임계값(역양자화한 점수 ≥ 값이면 1), `regression`: 허용 오차(정답 판정), `classification`: 0 |
| 60 | 4 | reserved | 0 |

CRC가 magic·버전·크기(0~15)를 제외하고 나머지 헤더(model_id, task, arena 등)와 모든 섹션을 덮는다.
`info.model.checksum`과 배포 스크립트의 `--expect-checksum` 값이 이 crc32다.

### 섹션 표 항목 (16 바이트)

| 오프셋 | 크기 | 필드 |
| --- | --- | --- |
| 0 | 4 | type |
| 4 | 4 | offset (패키지 시작 기준) |
| 8 | 4 | size (바이트) |
| 12 | 4 | count (항목 수, 섹션마다 의미가 다름) |

같은 type이 두 번 나오면 오류. **모르는 type은 건너뛴다** (형식을 확장해도 예전 펌웨어가 읽을 수 있게).

| type | 이름 | 필수 | count | 내용 |
| --- | --- | --- | --- | --- |
| 1 | tflite | 예 | 0 | `.tflite` 파일 그대로. **offset은 16의 배수**, 데이터 4~8번째 바이트가 `TFL3` |
| 2 | tensors | 예 | 2 | 텐서 기술자 2개(입력, 출력), 각 40바이트 |
| 3 | labels | 아니오 | 레이블 수 | UTF-8 문자열을 NUL로 끝내 이어 붙임. 분류면 개수 = 출력 원소 수 |
| 4 | demo_input | 아니오 | 1 | 입력 텐서 바이트 그대로 (크기 = 입력 텐서 bytes) |
| 5 | demo_expected | 아니오 | 1 | float32 × label_dim. demo_input이 있어야 함 |
| 6 | eval_inputs | 아니오 | N | 입력 텐서 바이트 × N (이어 붙임) |
| 7 | eval_expected | eval_inputs가 있으면 예 | N | float32 × label_dim × N |

정답 값은 모두 float32로 저장한다. 분류·이진은 클래스 번호(0, 1, 2, ...)를 float으로 담는다 (2^24까지 정확).

### 텐서 기술자 (40 바이트)

| 오프셋 | 크기 | 필드 | 설명 |
| --- | --- | --- | --- |
| 0 | 1 | dtype | 1 int8, 2 uint8, 3 int16, 4 int32, 5 float32 |
| 1 | 1 | ndim | 1~6 |
| 2 | 2 | reserved | 0 |
| 4 | 24 | dims | int32 × 6 (ndim 이후는 0) |
| 28 | 4 | scale | float32 (float32 텐서는 0) |
| 32 | 4 | zero_point | int32 |
| 36 | 4 | bytes | 원소 수 × dtype 크기 (검사함) |

펌웨어는 로드 후 TFLM 인터프리터의 `input(0)`/`output(0)`과 이 값(dtype, shape, bytes, scale, zero_point)을 비교한다.
다르면 `package_invalid`. 즉 **패키지에 적힌 입출력 정보가 실제 모델과 다를 수 없다**.
(scale은 상대오차 1e-6까지 같은 값으로 본다.)

## 런타임 제약 (현재 tflm_runtime)

- 입력 1개, 출력 1개인 모델만 지원 (`package_invalid`)
- tflite의 모든 op가 펌웨어 op 리졸버에 등록되어 있어야 함 (`unsupported_op`). 목록: [ADDING_A_MODEL.md](ADDING_A_MODEL.md#지원-op)
- 사용자 정의(CUSTOM) op 미지원
- 아레나는 내부 RAM에서 16바이트 정렬로 할당 (PSRAM 없음)
- 입출력 dtype은 int8을 기본으로 한다 (uint8/int16/int32/float32도 형식상 담을 수 있고 펌웨어도 처리하지만,
  `bori_package.py`가 경고를 낸다)

## 출력 해석 (펌웨어와 `pc/btf_format.py`가 같은 규칙을 쓴다)

출력 값은 모두 `(raw - zero_point) × scale`로 역양자화한 float32로 다룬다.

| task | 예측 | 정답 판정 (`correct`) | 집계 |
| --- | --- | --- | --- |
| classification | 가장 큰 출력의 번호 (같으면 앞 번호), score = 그 값 | 예측 == round(정답) | accuracy |
| binary_score | score = 출력[0], 예측 = score ≥ task_param ? 1 : 0 | 예측 == round(정답) | accuracy |
| regression | 출력 값 전체 | max\|출력 - 정답\| ≤ task_param | accuracy(허용 오차 안의 비율), mean_abs_err |

## 오류 코드 (패키지 관련)

| code | 원인 |
| --- | --- |
| `no_package` | 모델 파티션이 비어 있음 (첫 4바이트가 0xFF) |
| `package_invalid` | magic/버전/크기/섹션/텐서 기술자/flatbuffer 검증/입출력 개수/텐서 불일치 |
| `checksum_mismatch` | crc32 불일치 (전송 중 손상, 쓰다 만 패키지 등) |
| `unsupported_op` | 등록되지 않은 op 사용. 메시지에 op 이름 |
| `arena_alloc_failed` | 요청한 아레나를 할당할 수 없음, 또는 자동 아레나로도 모델이 들어가지 않음 |
| `arena_too_small` | 패키지의 arena_bytes가 모자람. 메시지에 **보드에서 잰 실제 필요량** (`model needs N bytes`) |
| `model_crashed` | 직전 부팅이 이 패키지를 로드/첫 실행하다 패닉·워치독으로 리셋됨. 반복 재부팅을 막으려고 로드를 건너뜀 |

어떤 경우든 보드는 재부팅하지 않고 명령(`m` 등)에 응답한다. `info.model.load_error`로도 확인할 수 있다.

## 예시 (`sine_regression`, 3,272 바이트)

`python pc/bori_package.py inspect models/sine_regression/default_model.bin`의 섹션 부분:

| type | offset | size | count |
| --- | --- | --- | --- |
| tflite | 160 (0xA0, 16의 배수) | 2,704 | 0 |
| tensors | 2,864 | 80 | 2 |
| demo_input | 2,944 | 1 | 1 |
| demo_expected | 2,948 | 4 | 1 |
| eval_inputs | 2,952 | 64 | 64 |
| eval_expected | 3,016 | 256 | 64 |

헤더: model_id `sine_regression`, task `regression`, label_dim 1, arena_bytes 0(자동), task_param 0.15, crc32 `0xfc8a647d`.
