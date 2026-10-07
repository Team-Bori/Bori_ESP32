# person_detection_int8

TensorFlow Lite Micro 공식 예제 **person_detection** 모델이다. 96×96 흑백 이미지에 사람이 있는지(no_person / person) 판별한다.
다른 모델보다 크기·메모리·연산량이 훨씬 커서 **PSRAM 없는 ESP32에서 어디까지 가능한지 확인하는 한계 시험** 성격이 있다.

**결론: 이 보드에서 동작한다.** 아레나 82,300 B(가능한 최대 155,648 B의 53%), 1회 추론 456.7 ms(2.2 FPS),
보드 정확도는 호스트와 일치. 펌웨어 변경이 필요한 문제는 없었다.

보드에 카메라가 없으므로 입력은 호스트(PC/서버)에서 이미지를 96×96 흑백 int8로 바꾼 값이다.

## 출처와 라이선스

| 항목 | 내용 |
| --- | --- |
| 모델 | https://github.com/tensorflow/tflite-micro `tensorflow/lite/micro/models/person_detect.tflite` — **수정 없이 사용** |
| 커밋 | `22c2469a233981012ac16ad849f59c7105655aa9` (2026-10-01), sha256 `808cfdfc…2e9d` |
| 모델 라이선스 | Apache-2.0 (Copyright The TensorFlow Authors) |
| 구조 | MobileNet v1, depth multiplier 0.25, 96×96 흑백 (원본 `training_a_model.md`) |
| 평가 데이터 | Visual Wake Words val 분할 = COCO 2014 minival 8,059장 (A. Chowdhery et al., arXiv:1906.05721) |
| COCO | T.-Y. Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014. 주석 파일 CC BY 4.0 |
| minival 목록 | tensorflow/models `930a6f98` `research/object_detection/data/mscoco_minival_ids.txt` |

### 이미지 라이선스 필터 (패키지에 넣는 이미지)

COCO 이미지는 이미지마다 Flickr 라이선스가 다르다. 패키지의 평가·데모 샘플과 `test_data.npz`는 원본을 96×96 흑백으로 줄인 **파생물**이므로,
재배포와 변경이 허용되는 라이선스만 썼다.

| COCO license id | 라이선스 | 사용 |
| --- | --- | --- |
| 1, 2, 3 | CC BY-NC-SA / BY-NC / BY-NC-ND (비상업) | ❌ 제외 |
| 6 | CC BY-ND (변경 금지) | ❌ 제외 (축소·흑백 변환이 변경에 해당) |
| 4 | CC BY 2.0 | ✅ 40장 |
| 5 | CC BY-SA 2.0 | ✅ 20장 (파생물도 CC BY-SA) |
| 7 | No known copyright restrictions (Flickr Commons) | ✅ 1장 |
| 8 | United States Government Work | 허용 (minival에는 해당 이미지 없음) |

minival 8,059장 중 허용 라이선스는 2,067장이다. 그중 클래스별로 seed 42 무작위 추출을 했다.
**사용한 61장의 이미지 ID·라이선스·출처(Flickr 사진 페이지) 목록: [image_licenses.csv](image_licenses.csv)**
(CC BY / BY-SA 출처 표시는 이 목록으로 한다.) 원본 이미지와 주석은 저장소에 넣지 않는다 (`~/.cache/bori/coco`).

호스트 정확도 측정용 2,000장은 라이선스와 무관하게 골랐다. 재배포하지 않고 PC에서 측정에만 쓴다.

## 정답 레이블과 전처리

- 레이블 (VWW 정의): 이미지에 `person` 상자가 하나라도 있고 그 넓이가 이미지의 0.5%를 넘으면 person, 아니면 no_person
- 전처리 (`pc/models/person_detection/preprocess.py`): 원본 변환 스크립트(`training_a_model.md`의 representative_dataset)와 같은 단계
  1. Pillow `image.resize((96, 96))`: 이미지 전체를 줄인다. 비율 유지 안 함, Pillow 12.2.0 기본 보간
  2. `convert('L')`: 흑백
  3. `x = pixel / 127.5 − 1` (−1~1)
  4. `q = clip(round(x / 0.0078431377) − 1, −128, 127)`: 입력 양자화 (scale 1/127.5, zero_point −1)
- 원본 예제 테스트 이미지 확인: `person.bmp` → person (113 vs −113), `no_person.bmp` → no_person. 원본 테스트 기대와 같다.
  이 bmp는 int8 값을 바이트로 담고 있다 (원본은 `tobytes()` 그대로 memcpy).

## 결과

### 호스트 정확도 (int8, TFLM 인터프리터, VWW val 무작위 2,000장: person 930 / no_person 1,070)

| 전처리 | 정확도 | no_person 재현율 | person 재현율 |
| --- | --- | --- | --- |
| **전체 resize (원본 변환 방식, 패키지에 사용)** | **77.60%** | 78.79% | 76.24% |
| 중앙 87.5% crop 후 resize (TF-slim 평가 방식) | 78.35% | 78.41% | 78.28% |

- **원본 보고치와 비교:** 원본 문서는 "완전히 학습한 모델이면 VWW val에서 약 84%"라고 적고 있다. 이 tflite 파일 자체의 측정치는 기록되어 있지 않다.
  우리가 같은 val 분할에서 잰 int8 정확도는 77.6~78.4%로 약 6%p 낮다.
  원인은 확인하지 못했다. 후보로는 이 tflite가 1M step까지 학습한 모델이 아닐 가능성, int8 양자화 손실, 흑백 변환·보간 방식 차이가 있다.

혼동 행렬 (전체 resize, 행 = 정답, 열 = 예측):

| | no_person | person |
| --- | --- | --- |
| no_person | **843** | 227 |
| person | 221 | **709** |

### 보드 (ESP32 240 MHz, PSRAM 없음, tflm_runtime b2a7e8c, 2026-10-06)

| 항목 | 결과 |
| --- | --- |
| 로드 | 성공 (verify 통과, op 5종 모두 지원) |
| **아레나** | **82,300 B 사용** (자동 할당 154,624 B, 보드 최대 155,648 B의 53%). 호스트 추정 85,264 B보다 약간 작다. ESP-NN scratch 포함 |
| 플래시 | 패키지 863,264 B (tflite 300,568 B + 평가 샘플 60 × 9,216 B) |
| **지연** | `b` 평균 **456.65 ms** (5회, 최소 456.64, 최대 456.67) ≈ 2.19 FPS |
| `a` 평가 (60장, 클래스당 30) | **52/60 (86.67%) = 호스트 52/60**, 전체 27.6초 |
| 데모 (CC BY 사람 이미지 #250368) | person (0.953) = 호스트 |
| 스트리밍 (60장) | 52/60, 샘플당 456.74 ms, 921600 baud 전체 37.2초(샘플당 0.62초), 재전송 0. 115200 baud는 샘플당 1.28초 |
| 주기 metrics | 시간 예산(0.2초)에 맞춰 1회만 측정, 약 5.5초 간격, seq 누락 없음 |
| 명령 응답 (`m` 25회, 주기 보고 동작 중) | 25/25 응답, 중앙값 93 ms, 최대 500 ms (주기 측정 1회와 겹칠 때) |
| 힙 여유 (로드 후) | 132,616 B |

평가 샘플 수 60개는 `a` 실행이 30초 안에 끝나도록 정했다 (1회 0.46초).

## 한계와 개선안

| 항목 | 현황 | 개선안 |
| --- | --- | --- |
| 지연 456.7 ms (b2a7e8c) | Espressif 자료(ESP32 240 MHz + ESP-NN) 380 ms보다 약 20% 느렸음 | **해결 (58964e0):** 플래시 DIO 40 MHz → QIO 80 MHz로 **382.7 ms (−16.2%)**, Espressif 자료와 거의 같다 |
| 정확도 77.6% | 원본 보고 84%보다 낮음 | 중앙 crop 전처리 +0.8%p. 큰 개선은 재학습(더 긴 학습, 컬러 입력, 해상도 상향)이 필요 |
| 메모리 | 아레나 53% 사용 | 여유 약 73 KB. 입력을 128×128로 키우면 활성값이 약 1.8배라 한도에 근접(추정). MobileNet v2 등 큰 구조는 별도 측정 필요 |
| 명령 응답 지연 | 주기 측정과 겹치면 최대 0.5초 대기했음 (b2a7e8c) | **해결 (58964e0):** 주기 측정을 두 번째 코어 태스크로 옮겨 `m` 응답 최대 94 ms (중앙값 78 ms) |
| 실시간 카메라 | 보드에 카메라 없음 | 범위 밖. 서버 테스트 파일로 측정 |


> **재측정 (tflm_runtime 58964e0, 플래시 QIO 80 MHz, 2026-10-07):** `b` 평균 382.65 ms (이전 b2a7e8c 456.65 ms). 정확도·아레나는 같다. 최신 값은 `manifest.json`의 `board`, 이전 값은 `board_history`.

## 다시 만들기

```powershell
python pc/models/person_detection/prepare.py                 # 모델, COCO 주석, minival 목록, 이미지 2,061장 (약 3분)
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/c/project/Bori_ESP32 && export BORI_CACHE=/mnt/c/Users/user/.cache/bori && ~/bori-tflm/bin/python pc/models/person_detection/evaluate.py"
python pc/bori_package.py build models/person_detection_int8/manifest.json
python pc/bori_package.py check models/person_detection_int8/manifest.json --update-manifest
python pc/bori_package.py make-test models/person_detection_int8/default_model.bin models/person_detection_int8/test_data.npz
```
`evaluate.py`는 tflite-micro와 Pillow 12.2.0이 필요하다 (WSL 가상환경).
