# anomaly_ae_int8

기계(팬) 소리 10초 클립이 정상인지 이상인지 판정하는 **이상 탐지 오토인코더**다. 구조는 DCASE 2023 Task 2 베이스라인을 따랐다.
정상 소리만으로 학습하고, 재구성 오차가 크면 이상으로 본다 (비지도 학습). 보드는 클립마다 **이상 점수 하나**를 내고, 임계값으로 판정한다.

> ⚠️ **비상업 데이터.** 데이터셋이 CC BY-NC-SA 4.0이라, 패키지의 평가·데모 샘플과 `test_data.npz`(이 데이터에서 계산한 특징)는
> **비상업적 용도로만**, 같은 라이선스로 배포할 수 있다. 사용자 결정으로 이 데이터셋을 썼다 (아래 "라이선스" 참고).

## 출처와 라이선스

| 항목 | 내용 |
| --- | --- |
| 데이터셋 | DCASE 2023 Challenge Task 2 Development Dataset, 기계 종류 **fan** (MIMII DG). Zenodo [10.5281/zenodo.7882613](https://doi.org/10.5281/zenodo.7882613) v3.0, `dev_fan.zip` (md5 `9348591e…62fc`) |
| 데이터 라이선스 | **CC BY-NC-SA 4.0** (Hitachi, Ltd., NTT Corporation). 주의: Zenodo 메타데이터에는 CC BY 4.0으로 적혀 있지만 데이터셋 본문 "Condition of use"는 CC BY-NC-SA 4.0이다. 더 엄격한 본문을 따른다 |
| 인용 (데이터셋이 요구하는 3편) | N. Harada et al., *First-shot anomaly detection for machine condition monitoring: A domain generalization baseline*, arXiv:2303.00455, 2023 · K. Dohi et al., *MIMII DG*, DCASE 2022 Workshop · N. Harada et al., *ToyADMOS2*, DCASE 2021 Workshop |
| 모델 | 이 저장소에서 학습 (`pc/models/anomaly_ae/train.py`). 구조와 학습 설정은 DCASE 2023 Task 2 오토인코더 베이스라인 |

다른 후보와 제외 이유:
- DCASE 2020·2024 Task 2: CC BY-NC-SA, 2020은 기계당 1~1.9 GB
- MIMII 2019: CC BY-SA 4.0으로 상업 이용이 가능하지만, 파일 하나가 7~11 GB이고 비교할 공식 수치 체계가 다르다

원본 음성은 커밋하지 않는다 (`~/.cache/bori/dcase2023`).

## 데이터 (fan, section 00)

| 분할 | 구성 |
| --- | --- |
| 학습 | 정상 1,000클립 (source 도메인 990 + target 도메인 10), 각 10초, 16 kHz mono |
| 테스트 | 200클립 = source(정상 50 + 이상 50) + target(정상 50 + 이상 50) |

target 도메인은 녹음 조건이 다른 환경이다 (도메인 일반화 과제). 학습 데이터가 10클립뿐이라 어렵다.

## 특징과 모델

| 항목 | 값 |
| --- | --- |
| 특징 | log-mel 128밴드 (n_fft 1024 = 64 ms, hop 512, power 2, `10·log10`), 연속 5프레임 연결 → 640차원 벡터, 10초에 309개 (librosa 0.11.0, DCASE 베이스라인과 같음) |
| 정규화 | **전체 값 하나의 평균·표준편차** (평균 −30.93 dB, 표준편차 9.49 dB, `train_report.json`). 차원별 정규화를 하지 않아 MSE 순위가 베이스라인(원시 dB의 MSE)과 같다 |
| 오토인코더 | 640 → 128 → 128 → 128 → 128 → **8** → 128 → 128 → 128 → 128 → 640 (Dense + BatchNorm + ReLU), 파라미터 269,992 |
| 학습 | Adam 1e-3, batch 256, 100 epoch, 검증 10% (베이스라인 설정), seed 42 |
| **보드 입력** | 클립의 309개 벡터 중 **고르게 고른 32개** → int8 `[32, 640]` (20,480 B) |
| **보드 출력** | 모델 그래프 안에서 이상 점수 계산: `mean((x − AE(x))²)` → int8 `[1, 1]` |
| tflite op | FULLY_CONNECTED, SQUARED_DIFFERENCE, MEAN (BatchNorm은 Dense에 합쳐지고 ReLU는 융합됨) |
| tflite 크기 | 319,176 B |

**왜 32개 벡터인가:** 클립 전체(309 × 640 int8 = 198 KB)는 보드 아레나 최대치(152 KB)에 들어가지 않는다.
32개만 써도 정확도 손실은 없었다 (float 기준 source AUC 79.66% → 79.76%).

**int8 점수 보정:** 정상 클립만으로 양자화를 보정하면 점수 범위가 정상 수준(최대 약 0.12)에 맞춰져, 이상 점수가 모두 최댓값에 포화된다 (AUC 75% → 62%).
그래서 보정 데이터의 절반에 가우시안 잡음(σ 0.1~0.7)을 섞어 큰 오차까지 범위에 넣었다 (출력 범위 0~0.56, 단위 0.0022).

## 판정 (binary_score)

- 점수 ≥ 임계값 **0.1004** 이면 `anomaly`
- 임계값은 학습 정상 클립 1,000개의 점수(float, 같은 32벡터)에 감마 분포를 맞추고 90% 지점을 잡았다 (DCASE 베이스라인 방식).
  위치 모수는 0으로 고정했다 (`floc=0`). 자유롭게 두면 scipy가 위치를 최솟값에 붙이고 모양 모수 0.19의 퇴화한 분포를 내서
  90% 지점이 데이터 밖(0.60)으로 나간다. 고정한 결과(0.1075)는 경험적 90% 지점(0.1038)과 맞았다 (1 epoch 시험 기준).

## 결과 (호스트, 테스트 200클립)

DCASE 2023 정의를 썼다. 도메인별 AUC = 그 도메인의 정상 클립 vs **모든** 이상 클립. pAUC = 전체 정상 vs 전체 이상, p = 0.1.
pAUC는 공식 평가기와 같이 scikit-learn `roc_auc_score(max_fpr=0.1)`(McClish 표준화)로 계산했다.

| | source AUC | target AUC | pAUC |
| --- | --- | --- | --- |
| DCASE 2023 공식 베이스라인 (MSE), fan | 80.19% | 36.18% | 59.04% |
| 우리 모델 float, 309벡터 전체 | **79.66%** | 28.38% | 53.74% |
| 우리 모델 float, 32벡터 | 79.76% | 28.42% | 53.74% |
| **우리 모델 int8, 32벡터 (보드와 같은 모델)** | **79.33%** | 28.40% | 53.71% |

- source 도메인은 공식 베이스라인과 0.9%p 차이로 재현됐다. 32벡터와 int8 변환에 따른 손실은 0.5%p 미만이다.
- target 도메인은 베이스라인처럼 약하다 (AUC 50% 미만은 target 정상 클립이 이상보다 점수가 높게 나온다는 뜻).
  pAUC도 5%p 낮다. 정확한 원인은 확인하지 못했다 (학습 무작위성, BatchNorm 구현 차이 등이 후보).

### 임계값 판정 정확도 (int8)

| 도메인 | 정확도 | 정상을 정상으로 | 이상을 이상으로 |
| --- | --- | --- | --- |
| source | 61% | 62% | 60% |
| target | 50% | **0%** | 100% |
| 전체 | 55.5% | | |

점수 분포(int8)를 보면 이유가 드러난다. 학습 정상 클립 중앙값 0.088, 테스트 정상 0.132, 테스트 이상 0.184.
**테스트의 정상 소리, 특히 target 도메인이 학습 정상보다 점수가 높아서** 학습 데이터로 정한 임계값으로는 target 정상이 모두 이상으로 판정된다.
DCASE 과제가 다루는 도메인 이동 문제 그 자체이며, 같은 임계값 방식인 베이스라인에도 해당한다.
AUC(순위)가 주 지표이고, 판정 정확도는 참고용이다.

## 패키지 샘플

- 평가: (source, target) × (normal, anomaly)마다 12클립 = 48클립, 각 32 × 640 int8. 호스트 30/48 (62.5%)
- 데모: 평가에 없는 source 이상 클립 중 점수가 가장 높은 것 (`section_00_source_test_anomaly_0021`, 점수 0.230 → anomaly)
- 패키지 1,323,164 B (tflite 319 KB + 샘플 48 × 20,480 B)

## 보드

보드 실측 대기 (보드가 연결되면 측정).
호스트 TFLM 아레나 추정은 78,256 B다 (입력 20 KB + 재구성 20 KB + 제곱오차 20 KB 등). 보드 최대 155,648 B 안에 들어갈 것으로 예상한다.

## 다시 만들기

```powershell
C:\Users\user\.venvs\bori-train\Scripts\python.exe pc/models/anomaly_ae/train.py --machine fan --k 32   # 특징 캐시 후 약 70분
python pc/bori_package.py build models/anomaly_ae_int8/manifest.json
python pc/bori_package.py check models/anomaly_ae_int8/manifest.json --update-manifest
python pc/bori_package.py make-test models/anomaly_ae_int8/default_model.bin models/anomaly_ae_int8/test_data.npz
```
`--reuse`를 주면 캐시된 학습 가중치(`~/.cache/bori/dcase2023/fan_ae.weights.h5`)로 변환·평가만 다시 한다.
`pc/models/anomaly_ae/requirements.txt`: tensorflow, librosa, scipy, scikit-learn.
