# sine_regression

TensorFlow Lite Micro 공식 예제 **hello_world**의 int8 모델(사인파 회귀)이다.
실사용 가치는 낮고, `tflm_runtime` 펌웨어의 동작 확인용 첫 모델이다
(패키지 로드, 회귀 경로, `i`/`b`/`a`/`m`/`p`, 주기 `metrics`, 테스트 스트리밍).
`tflm_runtime` 보드의 기본 패키지 후보이기도 하다 (`rpi/esp32.conf`의 `DEFAULT_MODEL`).

## 출처와 라이선스

| 항목 | 값 |
| --- | --- |
| 원본 | https://github.com/tensorflow/tflite-micro `tensorflow/lite/micro/examples/hello_world/models/hello_world_int8.tflite` |
| 커밋 | `22c2469a233981012ac16ad849f59c7105655aa9` (2026-10-02 기준 main) |
| sha256 | `505ee4fae7fa46ab67bea4c08b4969eb3eb8b9114c50595ec4a29d9a27993202` |
| 라이선스 | Apache-2.0 (재배포 허용, 저작권 고지 유지) — Copyright The TensorFlow Authors |
| 데이터 | 합성 데이터 y = sin(x) (라이선스 해당 없음) |

`pc/models/sine_regression/prepare.py`가 고정 커밋에서 모델을 받아 sha256을 확인하고 샘플을 만든다.
모델은 수정하지 않았다.

## 모델

| 항목 | 값 |
| --- | --- |
| 구조 | FULLY_CONNECTED 3층 (1 → 16 → 16 → 1) |
| 입력 | `[1, 1]` int8, scale 0.0244801, zero_point -128 (x ∈ [0, 2π)) |
| 출력 | `[1, 1]` int8, scale 0.00829096, zero_point 5 (sin x) |
| tflite | 2,704 B |
| 패키지 | 3,272 B, crc32 `0xfc8a647d` |
| op | FULLY_CONNECTED |

## 샘플과 판정

- 데모: x = π/2 (양자화 값 -64 → 역양자화 1.5667), 정답 sin = 0.99999
- 평가: [0, 2π)를 64등분한 x를 양자화한 64개. 정답은 **역양자화한 x**의 sin 값
- 정답 판정: |출력 - 정답| ≤ 0.15 (`task_param`). hello_world는 노이즈(표준편차 0.1)를 섞은 sin으로 학습한
  예제라 오차가 0.1 안팎이다. 판정 기준은 "모델이 원래 성능대로 도는지" 확인용이다.

## 결과

보드: ESP32 rev 3.1, 240 MHz, 2026-10-06 tflm_runtime b2a7e8c(`firmware/tflm_runtime.bin`)로 재확인.
지연 시간은 같은 날 두 번째 빌드(test_summary seq, info 1KB 수정) 값이다. 첫 빌드에서는 `b` 평균 53.5 µs, 스트리밍 160.5 µs로,
이 정도로 작은 모델은 코드 배치에 따라 20% 안팎 달라진다. 정확도·아레나는 두 빌드가 같았다.

| 항목 | 호스트 (TFLM 레퍼런스 커널, x86_64) | 보드 |
| --- | --- | --- |
| 평가 정확도 (±0.15, `a`) | 64/64 (1.0) | 64/64 (1.0) — 일치 |
| 평균 절대오차 | 0.020747 | 0.0207465 — 일치 |
| 최대 절대오차 | 0.10046 | (보고 항목 아님) |
| 데모 출력 (`i`) | 1.0032 (오차 0.0032) | 1.00321 (오차 0.00321) — 일치 |
| 아레나 사용량 | 1,456 B (64비트) | **716 B** (`arena_used_bytes`) |
| 지연 (`b`, 1000회, 캐시 warm) | — | 평균 41.7 µs, 최소 41, 최대 48 |
| 지연 (`a`, 64개) | — | 평균 43.6 µs |
| 지연 (스트리밍 64개, 샘플당 1회) | — | 평균 188.2 µs, 최소 180, 최대 236 |
| 지연 (`i`, 1회) | — | 164 µs |
| 테스트 파일 64개 전송+추론 (`run-test`) | — | 921600 baud 0.28초, 115200 baud 1.03초(첫 빌드), 재전송 0 |
| 힙 여유 (로드 후) | — | 132,616 B (자동 아레나 154,624 B 할당) |

스트리밍·단발 추론이 `b`보다 느린 것은 이 모델이 매우 작아(53 µs) 매번 플래시 캐시 미스 비용이 그대로 드러나기 때문이다
(샘플을 받는 동안 UART 코드가 캐시를 차지함). `b`는 같은 입력을 연속 실행한 warm 값이다.

## 다시 만들기

```powershell
python pc/models/sine_regression/prepare.py
python pc/bori_package.py build models/sine_regression/manifest.json
python pc/bori_package.py check models/sine_regression/manifest.json --update-manifest
python pc/bori_package.py make-test models/sine_regression/default_model.bin models/sine_regression/test_data.npz
```
