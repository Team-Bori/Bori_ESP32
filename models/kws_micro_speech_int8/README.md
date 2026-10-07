# kws_micro_speech_int8

TensorFlow Lite Micro 공식 예제 **micro_speech**의 키워드 인식 모델이다. 1초 음성에서 만든 스펙트로그램 특징으로
`silence` / `unknown` / `yes` / `no`를 분류한다. 오디오 계열 대표 모델이다.

**보드에 마이크가 없으므로** 이번 범위는 "특징값(49×40 int8) 입력 → 키워드 분류"까지다.
음성 → 특징 변환은 호스트(PC/서버)에서 원본 예제와 같은 방식으로 한다. 보드에서 특징 추출까지 하는 것은 향후 과제다.

## 출처와 라이선스

| 항목 | 내용 |
| --- | --- |
| 모델 | https://github.com/tensorflow/tflite-micro `tensorflow/lite/micro/examples/micro_speech/models/micro_speech_quantized.tflite` — **수정 없이 사용** |
| 커밋 | `22c2469a233981012ac16ad849f59c7105655aa9` (2026-10-01) |
| sha256 | 모델 `09e5e2a9…2a83`, 전처리 모델 `audio_preprocessor_int8.tflite` `278949d1…40e5` |
| 라이선스 | Apache-2.0 (Copyright The TensorFlow Authors) |
| 학습 데이터 (원본) | Speech Commands v0.02, `tiny_conv` 구조, wanted_words `yes,no` (원본 학습 노트북 기준) |
| 평가 데이터 | Speech Commands v0.02 **테스트 세트** (`speech_commands_test_set_v0.02.tar.gz`, 4,890 클립, sha256 `cc2a00c1…51df`) |
| 데이터 라이선스 | **CC BY 4.0** — P. Warden, *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition*, arXiv:1804.03209, 2018 |

명세 1.3에 따라 원본은 `prepare.py`가 고정 커밋에서 내려받아 해시를 확인한다. 우리 저장소에는 검토한 모델 파일과 가공한 샘플만 둔다.
패키지의 평가·데모 샘플과 `test_data.npz`는 위 테스트 세트 음성에서 계산한 특징값(파생물)이므로 CC BY 4.0의 출처 표시를 따른다.
원본 음성은 커밋하지 않는다 (`~/.cache/bori/speech_commands`).

## 특징 추출 (호스트)

`pc/models/kws_micro_speech/features.py`

이 커밋의 예제는 예전 `micro_frontend`가 아니라 **`audio_preprocessor` 모델 방식**이다. 그래서 원본 예제의
`audio_preprocessor_int8.tflite`(Signal 라이브러리 op: Hann window → FFT auto scale → RFFT → energy → 40밴드 filter bank →
square root → spectral subtraction → PCAN → log)를 **그대로** TFLM 인터프리터로 프레임마다 실행한다. 다시 구현하지 않았다.

| 항목 | 값 |
| --- | --- |
| 오디오 | 16 kHz, mono, int16, 1초 (16,000 샘플) |
| 프레임 | 30 ms 창(480 샘플), 20 ms 간격(320 샘플) → 49 프레임 |
| 특징 | 프레임당 40채널 int8 → `[1, 1960]` (49×40 펼침) |
| 상태 | 잡음 추정·PCAN 상태는 클립마다 초기화 (원본 `evaluate.py`와 같음) |
| 양자화 | 전처리 모델의 int8 출력이 곧 분류 모델 입력 (scale 0.1017157, zero_point −128). 추가 변환 없음 |

**원본과 비트 단위 일치 검증** (`features.py --verify`):

| 확인 | 결과 |
| --- | --- |
| `no_30ms.wav` 특징 vs `micro_speech_test.cc` NoFeatureTest 기대값 40개 | **비트 단위 일치** (최대 차이 0) |
| `yes_30ms.wav` 특징 vs YesFeatureTest 기대값 40개 | **비트 단위 일치** (최대 차이 0) |
| `yes/no/silence/noise_1000ms.wav` 분류 | yes / no / silence / silence — 원본 테스트 기대값과 같음 |

## 모델

| 항목 | 값 |
| --- | --- |
| 구조 | tiny_conv: DEPTHWISE_CONV_2D → FULLY_CONNECTED → SOFTMAX (+ RESHAPE) |
| 입력 | `[1, 1960]` int8 (scale 0.1017157, zero_point −128) |
| 출력 | `[1, 4]` int8, 레이블 순서 원본 그대로 `silence, unknown, yes, no` |
| tflite | 18,800 B |

## 정확도 (int8, Speech Commands v0.02 테스트 세트 4,890 클립)

테스트 세트 폴더 → 모델 범주: `_silence_` → silence, `yes` → yes, `no` → no, `_unknown_`과 나머지 8단어 → unknown.

| 범주 | 클립 | 재현율 |
| --- | --- | --- |
| silence | 408 | 98.28% |
| unknown | 3,658 | 72.39% |
| yes | 419 | 95.23% |
| no | 405 | 91.36% |
| **전체 정확도** | 4,890 | **78.08%** (unknown이 75%를 차지) |
| **범주 평균(balanced)** | | **89.32%** |

혼동 행렬 (행 = 정답, 열 = 예측: silence / unknown / yes / no):

| | silence | unknown | yes | no |
| --- | --- | --- | --- | --- |
| silence | **401** | 5 | 2 | 0 |
| unknown | 27 | **2,648** | 323 | 660 |
| yes | 3 | 9 | **399** | 8 |
| no | 2 | 31 | 2 | **370** |

- 오류의 대부분은 **다른 단어를 yes/no로 오인**하는 것이다. 특히 `go`(26%), `left`(53%), `down`(55%)가 낮다 (`go`는 `no`와 발음이 비슷).
- **원본과 비교:** 원본 저장소(README, 학습 노트북)에는 정확도 수치가 기록되어 있지 않다 (노트북은 실행 결과 없이 저장됨).
  원본 README는 이 모델이 "정확도가 높지 않고, 항상 켜 둔 저전력 1단계 감지용"이라고 설명한다.
  원본 학습 노트북은 silence·unknown 샘플 수를 각각 yes+no 수의 25%로 맞춰(`SILENT_PERCENTAGE = UNKNOWN_PERCENTAGE = 25`)
  테스트하므로 구성이 대략 yes:no:silence:unknown = 2:2:1:1이다. 우리 재현율을 이 비율로 가중하면 **약 90.6%** 로,
  원본 방식에 가까운 비교 값은 이쪽(또는 범주 평균 89.3%)이다. 전체 정확도 78.1%는 unknown이 75%인 테스트 세트 구성 탓이다.

## 패키지 샘플

- 평가: 범주마다 40개, 총 160개. 각 범주 안에서 고르게 뽑아 unknown은 `_unknown_`과 8단어가 섞여 있다 (단어당 4~5개)
- 데모: 원본 예제의 `testdata/yes_1000ms.wav` 특징 (정답 yes)
- 패키지 335,284 B (평가 특징 160 × 1,960 B가 대부분)

## 결과 요약

| 항목 | 호스트 (TFLM 레퍼런스 커널) | 보드 (ESP32 240 MHz, tflm_runtime b2a7e8c, 2026-10-06) |
| --- | --- | --- |
| 평가 샘플 정확도 (160개) | 143/160 (89.38%) | **143/160 — 일치** |
| 데모 | yes (0.996) | yes (0.996) |
| 아레나 | 7,568 B (64비트) | **6,812 B** 사용 (최대 155,648 B) |
| 지연 `b` (153회, warm) | | 평균 **12.97 ms** (최소 12.96, 최대 12.99) ≈ 77 FPS |
| 지연 `a` (160개) | | 평균 13.22 ms, 전체 2.2초 |
| 지연 스트리밍 (160개) | | 평균 13.98 ms, 921600 baud 전체 6.6초, 재전송 0 |
| 주기 metrics | | 0.2초 예산에 16회 |

참고: 실시간 키워드 인식은 20 ms마다 새 프레임이 들어오므로, 보드에서 특징 추출까지 하려면 분류(13 ms) + 전처리 1프레임이
20 ms 안에 들어와야 한다. 전처리 모델의 보드 시간은 측정하지 않았다 (Signal op은 tflm_runtime에 등록되어 있지 않음, 향후 과제).


> **재측정 (tflm_runtime 58964e0, 플래시 QIO 80 MHz, 2026-10-07):** `b` 평균 12.25 ms (이전 b2a7e8c 12.97 ms). 정확도·아레나는 같다. 최신 값은 `manifest.json`의 `board`, 이전 값은 `board_history`.

## 다시 만들기

```powershell
python pc/models/kws_micro_speech/prepare.py
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/c/project/Bori_ESP32 && export BORI_CACHE=/mnt/c/Users/user/.cache/bori && ~/bori-tflm/bin/python pc/models/kws_micro_speech/features.py --verify && ~/bori-tflm/bin/python pc/models/kws_micro_speech/evaluate.py"
python pc/bori_package.py build models/kws_micro_speech_int8/manifest.json
python pc/bori_package.py check models/kws_micro_speech_int8/manifest.json --update-manifest
python pc/bori_package.py make-test models/kws_micro_speech_int8/default_model.bin models/kws_micro_speech_int8/test_data.npz
```
