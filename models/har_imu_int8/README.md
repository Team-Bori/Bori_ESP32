# har_imu_int8

스마트폰 IMU(가속도·자이로) 원시 신호 2.56초 창으로 사람의 동작 6가지를 분류하는 1D CNN이다.
MCU에서 가장 흔한 **센서 시계열** 계열의 대표 모델이다.

보드에는 IMU가 없으므로, 입력은 데이터셋의 원시 신호 창을 정규화·양자화한 int8 값이다
(서버 테스트 파일도 같은 형식, [TEST_DATA_FORMAT.md](../../docs/TEST_DATA_FORMAT.md)).

## 출처와 라이선스

| 항목 | 내용 |
| --- | --- |
| 데이터셋 | UCI *Human Activity Recognition Using Smartphones* — https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones |
| 버전 | 2012-12-09 기증본, DOI [10.24432/C54S4K](https://doi.org/10.24432/C54S4K). zip sha256 `c00b8030…1031` |
| 라이선스 | **CC BY 4.0** (출처 표시 필요) |
| 인용 | D. Anguita, A. Ghio, L. Oneto, X. Parra, J. L. Reyes-Ortiz. *A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN 2013. |
| 모델 | 이 저장소에서 직접 설계·학습 |

패키지의 평가·데모 샘플과 `test_data.npz`는 이 데이터셋 테스트 분할의 원시 신호를 정규화·양자화한 파생물이며 CC BY 4.0을 따른다.
원본 데이터는 저장소에 넣지 않는다 (`train.py`가 `~/.cache/bori/uci_har`에 내려받는다).

## 입력

| 항목 | 값 |
| --- | --- |
| 창 | 128 샘플 × 50 Hz = 2.56초 (데이터셋이 50% 겹침으로 잘라 둔 창 그대로) |
| 채널 (9) | body_acc x/y/z, body_gyro x/y/z, total_acc x/y/z |
| 단위 | 가속도 g(중력 단위), 자이로 rad/s |
| 정규화 | 채널별 `(x − mean) / std`, 학습 피험자(검증 피험자 제외)로 계산. 값은 `manifest.json`의 `preprocessing` |
| 양자화 | `q = clip(round(x_norm / 0.0827583) + 7, −128, 127)` |
| 입력 텐서 | `[1, 128, 9]` int8, 1,152 B |

**561개 수작업 특징은 쓰지 않는다.** 보드에서 특징 공학 없이 바로 돌릴 수 있도록 원시 관성 신호(`Inertial Signals`)만 쓴다.

## 구조

| 층 | 출력 | 비고 |
| --- | --- | --- |
| Conv1D 16, k=5, ReLU | 124×16 | tflite: EXPAND_DIMS → CONV_2D → RESHAPE |
| MaxPool1D 2 | 62×16 | MAX_POOL_2D |
| Conv1D 32, k=5, ReLU | 58×32 | CONV_2D |
| GlobalAveragePooling1D | 32 | MEAN |
| Dropout 0.3 (학습만) → Dense 6, softmax | 6 int8 | FULLY_CONNECTED, SOFTMAX |

파라미터 3,526개, tflite 10,520 B. **Conv1D는 TFLM에서 EXPAND_DIMS + CONV_2D + RESHAPE로 변환되며 모두 tflm_runtime 지원 op다.**

## 학습

`pc/models/har_imu/train.py` (`pc/models/har_imu/requirements.txt`)

- 공식 분할 유지: 학습 21명 / 테스트 9명 (피험자 분리). 테스트 분할은 최종 수치에만 사용
- 조기 종료는 학습 피험자 중 3명(28, 29, 30)을 통째로 떼어 낸 검증 세트로 판단 (같은 사람의 겹치는 창이 학습·검증에 동시에 들어가지 않게)
- Adam 1e-3, batch 64, 최대 100 epoch, patience 15 (최적 31 epoch), seed 42, TF op determinism
- int8 전체 양자화, 대표 데이터 학습 창 500개

## 채널 수 선택

같은 구조·설정으로 채널 수만 바꿔 학습했다 (`train.py --compare`, 결과 `channel_compare.json`, 테스트 2,947창):

| 채널 | 신호 | 파라미터 | tflite | float | int8 |
| --- | --- | --- | --- | --- | --- |
| 3 | total_acc (가속도계만) | 3,046 | 10,040 B | 85.14% | **84.02%** (목표 미달) |
| 6 | total_acc + body_gyro (가속도+자이로) | 3,286 | 10,416 B | 91.25% | 90.97% |
| **9** | body_acc + body_gyro + total_acc | 3,526 | 10,656 B | 92.03% | **91.82%** |

**9채널을 선택했다.** 정확도가 가장 높고, 크기 차이는 0.6 KB로 무시할 만하다.
(비교 실행은 한 프로세스에서 연달아 학습해 tflite의 층 이름이 달라 크기가 최종 모델(10,520 B)과 조금 다르다. 정확도는 같다.)

실제 센서를 붙일 때의 참고:
- **가속도 3축만으로는 88% 목표에 못 미친다 (84%).** 주로 걷기 계열과 앉기/서기 구분이 떨어진다.
- 가속도+자이로(6축 IMU)면 91%로 충분하다. 9채널의 body_acc는 total_acc에서 중력 성분(저역 통과)을 뺀 값이라
  6축 IMU만 있어도 보드에서 계산할 수 있다 (필터 구현 필요).

## 결과

| 항목 | 목표 | 호스트 | 보드 |
| --- | --- | --- | --- |
| int8 테스트 정확도 (2,947창) | 88% 이상 | **91.82%** (float 92.03%) | — |
| int8 평가 샘플 (150창, 클래스당 25, 테스트 피험자 9명 모두 포함) | 보드와 일치 | 139/150 (92.67%, TF Lite·TFLM 같음) | 보드 실측 대기 |
| tflite | 32 KB 이하 | 10,520 B | |
| 아레나 | | 7,120 B (호스트 64비트) | 보드 실측 대기 |
| 지연 | 실측 보고 | | 보드 실측 대기 |

### 혼동 행렬 (int8, 테스트 2,947창, 행 = 정답, 열 = 예측)

| | 걷기 | 계단↑ | 계단↓ | 앉기 | 서기 | 눕기 |
| --- | --- | --- | --- | --- | --- | --- |
| 걷기 | **492** | 0 | 4 | 0 | 0 | 0 |
| 계단 오르기 | 13 | **435** | 23 | 0 | 0 | 0 |
| 계단 내려가기 | 1 | 13 | **406** | 0 | 0 | 0 |
| 앉기 | 0 | 7 | 0 | **398** | 86 | 0 |
| 서기 | 0 | 0 | 0 | 91 | **441** | 0 |
| 눕기 | 0 | 3 | 0 | 0 | 0 | **534** |

- 오류의 대부분(177/241)이 **앉기 ↔ 서기** 혼동이다. 둘 다 정지 자세이고, 차이는 허리에 찬 폰의 기울기(중력 방향)의 작은 차이뿐이라 원시 신호 2.56초로는 구분이 어렵다 (HAR 연구에서 흔히 보고되는 오류다).
- 그다음은 걷기 ↔ 계단 오르기/내려가기 (54건).
- 눕기는 거의 완벽 (중력 방향이 확연히 다름).

## 다시 만들기

```powershell
python pc/models/har_imu/train.py --compare      # 채널 비교 (약 6분)
python pc/models/har_imu/train.py --channels 9
python pc/bori_package.py build models/har_imu_int8/manifest.json
python pc/bori_package.py check models/har_imu_int8/manifest.json --update-manifest
python pc/bori_package.py make-test models/har_imu_int8/default_model.bin models/har_imu_int8/test_data.npz
```
