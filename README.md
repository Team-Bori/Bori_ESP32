# ESP32 AI 모델 및 코드

보드 클라우딩 서비스의 ESP32 파트. 서버 → 라즈베리파이(USB 시리얼) → ESP32 구조로
모델을 올리고, 서버가 준 테스트 파일로 추론해 성능 지표(지연, 메모리, 정확도, 에러율)를 서버로 보낸다.

## 펌웨어

| 펌웨어 | 프로토콜 | 모델 형식 | 용도 |
| --- | --- | --- | --- |
| `tflm_runtime` | v2 | BTF1 패키지 (TFLite Micro int8 모델) | 범용 런타임. 모델은 패키지만 바꿔 쓴다 |
| `mlp` | v1 | MLP1 (64-16-10 MLP 고정) | 기존 펌웨어 (하위 호환 유지) |

두 펌웨어는 같은 파티션 표를 쓴다: 앱 0x10000 (1.94MB), **모델 0x200000 (2MB)**.
이 배치 이전에 배포된 보드는 펌웨어를 0x0부터 다시 써야 한다.

## 파일 구조

### 1. esp32_code
- `tflm_runtime/`: TFLite Micro 범용 런타임 (ESP-IDF 5.3.1, esp-tflite-micro 1.4.1)
- `mlp_esp32/`: 기존 MLP 펌웨어 (`PROTOCOL.md` = 프로토콜 v1)
- `components/bori_common/`: 두 펌웨어 공용 코드 (부팅·메모리·보드 정보 JSON, CRC32)

### 2. firmware
빌드한 병합 이미지(0x0에 쓰는 `<firmware_id>.bin` + `.sha256`). 빌드 시간을 줄이려고 저장한다.

### 3. models
모델별 폴더(`manifest.json`, 패키지 `default_model.bin`, 예제 테스트 파일 `test_data.npz`, README)와
전체 목록 `catalog.json` (`pc/build_catalog.py`로 생성).

### 4. pc
모델 학습·변환·패키징 도구. `bori_package.py`(BTF1 패키지 build/inspect/validate/check),
`validate_model.py`(MLP1/BTF1 검증), `validate_firmware.py`(펌웨어 이미지 검사), `pc/models/<name>/`(모델별 스크립트).

### 5. rpi
라즈베리파이 스크립트. `esp_monitor.py`(info/infer/bench/eval/run-test/verify/listen/monitor),
배포(`flash_*.sh`)와 반납 초기화(`reset_board.sh`), 설정(`esp32.conf`).

### 6. docs
- [PACKAGE_FORMAT.md](docs/PACKAGE_FORMAT.md): 모델 패키지(BTF1)와 파티션
- [PROTOCOL_v2.md](docs/PROTOCOL_v2.md): 시리얼 프로토콜 v2
- [TEST_DATA_FORMAT.md](docs/TEST_DATA_FORMAT.md): 서버 테스트 파일
- [ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md): 모델 추가 가이드
