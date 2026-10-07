# mlperf_kws_dscnn_int8

**MLPerf Tiny** 벤치마크의 키워드 인식(KWS) 참조 모델 DS-CNN이다. 1초 음성의 MFCC(49 프레임 × 10)로 12개 범주를 분류한다.
MCU용 AI 성능 비교의 업계 표준 벤치마크라 다른 보드의 공개 결과와 비교할 수 있다.

범주 (출력 순서 그대로): `down, go, left, no, off, on, right, stop, up, yes, silence, unknown`

보드에 마이크가 없으므로 음성 → MFCC 변환은 호스트에서 한다 (`kws_micro_speech_int8`과 같은 방식).

## 출처와 라이선스

| 항목 | 내용 |
| --- | --- |
| 모델 | https://github.com/mlcommons/tiny `benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite` — **수정 없이 사용** |
| 커밋 | `4addd0fa08d216e20637637874e084895f289da4` (2026-07-13), sha256 `aeea4368…0bd0ae` |
| 라이선스 | Apache-2.0 (MLCommons) |
| 논문 | C. Banbury et al., *MLPerf Tiny Benchmark*, NeurIPS 2021 Datasets and Benchmarks (arXiv:2106.07597) |
| 데이터 | Speech Commands v0.02 테스트 세트 (4,890클립, 12범주), **CC BY 4.0** — P. Warden, arXiv:1804.03209 |

패키지의 평가 샘플·데모와 `test_data.npz`는 위 음성에서 계산한 MFCC(파생물, CC BY 4.0)다.
검증에 쓴 EEMBC의 공식 벤치마크 입력(`eembc/benchmark-runner-ml` `datasets/kws01`, 커밋 `cf7c2f26`)은 비교에만 썼고 저장소에 넣지 않았다.
같은 MLPerf Tiny의 ResNet-8(이미지 분류)은 평가 데이터 CIFAR-10에 명시적 라이선스가 없어 사용자 결정으로 제외했다.

## 모델

| 항목 | 값 |
| --- | --- |
| 구조 | DS-CNN (depthwise separable CNN), 파라미터 38.6K (논문) |
| op | CONV_2D, DEPTHWISE_CONV_2D, AVERAGE_POOL_2D, RESHAPE, FULLY_CONNECTED, SOFTMAX |
| 입력 | `[1, 49, 10, 1]` int8 (scale 0.5847029, zero_point 83) |
| 출력 | `[1, 12]` int8 |
| tflite | 53,936 B |

## 특징 추출 (호스트, `pc/models/mlperf_kws/features.py`)

참조 구현(`get_dataset.py`의 mfcc 경로, `eval_quantized_model.py`)과 같은 TensorFlow 연산을 쓴다.

1. int16 음성 → float를 `reduce_max(audio)`로 나눈다 (가장 큰 샘플값, 절댓값 최대가 아님) → 16,000 샘플로 0 채움
2. `tf.signal.stft` (창 480 = 30 ms, 간격 320 = 20 ms, FFT 512, Hann) → 크기
3. mel 40밴드 (20~4000 Hz) → `log(x + 1e-6)` → `mfccs_from_log_mel_spectrograms` → 앞 10개 → `[49, 10, 1]`
4. 양자화: 참조 구현처럼 **버림**(`np.array(x/scale + zp, dtype=np.int8)`). 다만 int8 범위로 먼저 잘라 넘침이 돌아 감기지 않게 했다

**검증** (`features.py --verify`): EEMBC 공식 벤치마크 입력 1,000개와 우리 특징을 바이트 단위로 비교했다.

| 결과 | 개수 |
| --- | --- |
| 바이트 단위 일치 (같은 레이블) | **919** |
| 값 몇 개가 ±1 다름 | 80 |
| 우리 테스트 세트에 없는 클립 | 1 |

±1 차이는 대부분 우리 값 83.0 vs 공식 82다. 무음·0 채움 프레임에서 정확히 0이어야 할 MFCC 계수(=zero_point 83)가
참조 실행에서는 −1e-8 같은 아주 작은 음수로 계산되어 버림에서 82가 된 것이다. TensorFlow 버전에 따른 부동소수점 잡음이며
(oneDNN을 꺼도 그대로), 파이프라인 차이가 아니다. 정확도 영향은 아래 표처럼 999클립 중 예측 1개 차이다.

## 결과

### 정확도 (int8)

| 측정 | 정확도 |
| --- | --- |
| **Speech Commands v0.02 테스트 세트 전체 4,890클립** (우리 특징, TF Lite) | **91.66%** — MLPerf Tiny 논문의 참조 91.6%와 일치, 품질 목표 90% 통과 |
| EEMBC 공식 입력 1,000개 (MLPerf 정확도 세트, 공식 파일 그대로) | 90.20% |
| 같은 999클립: 공식 입력 / 우리 특징 | 90.29% / 90.39% (예측 다른 것 1개) |

범주별 재현율 (전체 테스트 세트): down 87.9, go 85.6, left 93.5, no 89.9, off 91.0, on 88.6, right 90.7, stop 92.5, up 89.7, yes 92.8, silence 98.8, unknown 98.8 (%)

### 보드 (ESP32 240 MHz, tflm_runtime 58964e0, 2026-10-07)

| 항목 | 호스트 | 보드 |
| --- | --- | --- |
| 평가 샘플 (범주당 15, 180개) | TFLM 166/180 (92.22%), TF Lite 165/180 (91.67%) | **166/180 = 호스트 TFLM** |
| 데모 (yes 클립) | yes (0.996) | yes (0.996) |
| 아레나 | 24,256 B (64비트) | **22,780 B** |
| 지연 `b` (13회) | | 평균 **153.73 ms** (최소 153.72, 최대 153.75) ≈ 6.5 FPS |
| `a` (180개) | | 평균 153.81 ms, 전체 27.9초 |
| 스트리밍 (180개, 921600 baud) | | 92.22%, 전체 29.4초, 재전송 0 |

호스트의 두 인터프리터(TF Lite / TFLM 레퍼런스)가 한 샘플에서 다른 범주를 고른다 (점수가 거의 같은 경우의 int8 반올림 차이). 보드는 TFLM과 같다.

## 다른 보드와 비교할 때

- 정확도는 MLPerf 정의(테스트 세트 Top-1)로 비교할 수 있다: 91.66%.
- **지연 시간은 그대로 비교하지 말 것.** MLPerf Tiny 공식 결과는 EEMBC 러너가 정해진 방식(입력 전송·반복·중앙값)으로 잰 값이고,
  우리 `b`는 같은 입력을 2초 동안 반복한 평균이다. 공식 결과표: https://mlcommons.org/benchmarks/inference-tiny/
  참조 보드(NUCLEO-L4R5ZI) + TFLM의 지연 수치는 확인 가능한 출처를 찾지 못해 여기 적지 않았다.
- 참고로 ESP32는 ESP-NN의 SIMD 커널이 없어(ESP32-S3에만 있음) generic 커널로 돈다.

## 다시 만들기

```powershell
python pc/models/kws_micro_speech/prepare.py           # Speech Commands 테스트 세트 (공유 캐시)
python pc/models/mlperf_kws/prepare.py
C:\Users\user\.venvs\bori-train\Scripts\python.exe pc/models/mlperf_kws/features.py --verify   # EEMBC 입력이 캐시에 있을 때
C:\Users\user\.venvs\bori-train\Scripts\python.exe pc/models/mlperf_kws/evaluate.py
python pc/bori_package.py build models/mlperf_kws_dscnn_int8/manifest.json
python pc/bori_package.py check models/mlperf_kws_dscnn_int8/manifest.json --update-manifest
python pc/bori_package.py make-test models/mlperf_kws_dscnn_int8/default_model.bin models/mlperf_kws_dscnn_int8/test_data.npz
```
모델은 `https://raw.githubusercontent.com/mlcommons/tiny/4addd0fa…/benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite`에서 받아 sha256을 확인한다.
