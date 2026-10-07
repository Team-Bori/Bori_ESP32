# ESP32 시리얼 프로토콜 v2

`tflm_runtime` 펌웨어의 프로토콜이다. **v1([mlp_esp32/PROTOCOL.md](../esp32_code/mlp_esp32/PROTOCOL.md))의 상위 호환**:
v1 필드는 지우거나 의미를 바꾸지 않았고, v1 클라이언트(`m`/`i`/`b`/`p`, `metrics`)는 그대로 동작한다.
호스트 구현: `rpi/esp_monitor.py` (v1·v2 모두 지원).

- UART0, 8N1, 부팅 시 **115200 baud**. 테스트 데이터 전송 중에만 더 빠른 속도로 바꿀 수 있다 ([속도 변경](#속도-변경)).
- 보드 → 호스트: **JSON 객체 한 줄**(`\n` 종료, 실제로는 `\r\n`). `{`로 시작하지 않는 줄(부트로더, ESP_LOG, TFLM 메시지)은 무시한다.
- 호스트 → 보드: **1바이트 명령** 또는 **바이너리 프레임**(`0xA5 0x5A`로 시작).
- 모든 메시지에 `type`이 있다. 한 줄은 1KB를 넘지 않게 만든다 (출력 값은 최대 16개, 레이블 목록은 200자까지만 싣는다).
- 이 문서의 JSON 예시에 나오는 숫자는 **형식 설명용**이다. 실측값은 각 모델의 README/manifest에 있다.

## 펌웨어 구분

| 펌웨어 | `boot.proto` | `boot.firmware_id` | 모델 형식 |
| --- | --- | --- | --- |
| mlp | 1 | `mlp` (이전 빌드는 필드 없음 → `mlp`로 간주) | MLP1 |
| tflm_runtime | 2 | `tflm_runtime` | BTF1 ([PACKAGE_FORMAT.md](PACKAGE_FORMAT.md)) |

## 1바이트 명령

| 명령 | 응답 `type` | 설명 |
| --- | --- | --- |
| `m` | `info` | 보드 사양, 로드된 모델, 텐서·아레나 정보, 메모리 |
| `i` | `inference` | 데모 입력으로 1회 추론. 패키지에 데모가 없으면 0(양자화한 0.0) 입력 |
| `b` | `bench` | 시간 예산 벤치마크 (최소 1회, 약 2초, 최대 1000회) |
| `a` | `eval` | **v2 신규.** 패키지의 평가 샘플 전체 추론 → 정확도·지연 |
| `l` | `labels` | **v2 신규 (b2a7e8c 이후 빌드).** 로드된 모델의 레이블 목록 (`info`에서 빠진 경우용) |
| `p` | `periodic` | 주기 보고(`metrics`) 켜기/끄기 토글. 부팅 시 켜짐 |
| 그 외 출력 가능한 ASCII | `error` (`unknown_command`) | |
| 공백·제어 문자·0x7F 이상 | 응답 없음 | (0xA5는 프레임 시작) |
| 터미널 이스케이프 시퀀스 (`ESC [ …`, `ESC O …`: 방향키 등) | 응답 없음 | 시퀀스 전체를 버린다 (b2a7e8c 이후 빌드) |

명령은 순서대로 하나씩 처리된다. 실행 중에 들어온 바이트는 수신 버퍼(4KB)에 쌓였다가 다음에 처리된다.

### 반복 횟수와 시간 예산

모델마다 1회 추론 시간이 수십 µs에서 1초 가까이까지 다르므로 고정 횟수 대신 시간 예산을 쓴다.

| 용도 | 예열 | 측정 |
| --- | --- | --- |
| `b` (bench) | 최대 10회 / 0.1초 | 최소 1회, 2초가 지나면 중단, 최대 1000회 |
| `metrics` | 없음 | 최소 1회, 0.2초가 지나면 중단, 최대 100회 |

- 지연 시간은 `Invoke()` 시간만 잰다 (입력 복사·출력 해석 제외).
- 0.5초마다 1틱 쉬어 태스크 워치독이 걸리지 않게 한다 (쉬는 시간은 측정에 포함되지 않음).
- **주기 보고의 측정은 두 번째 코어의 별도 태스크가 한다** (b2a7e8c 이후 빌드). 측정 중에도 모델을 쓰지 않는
  명령(`m`, `l`, `p`, 속도 변경·PING 프레임)은 바로 응답한다. 모델을 쓰는 명령(`i`, `b`, `a`, 테스트 샘플)은
  진행 중인 측정(최대 1회 추론)이 끝난 뒤 실행된다. 출력은 항상 메인 태스크가 하므로 JSON 줄이 섞이지 않는다.
  (b2a7e8c 빌드는 측정을 메인 루프에서 해 느린 모델에서 명령이 최대 1회 추론 시간만큼 늦었다: person_detection 0.5초.)

## 부팅 시 자동 출력

1. `boot`
2. 모델 로드 실패 시 `error` (코드는 [오류 코드](#오류-코드) 참고). 보드는 계속 동작한다.
3. `ready`
4. 5초마다 `metrics` (주기 보고가 켜져 있고 테스트 세션·속도 변경 중이 아닐 때)

```json
{"type":"boot","proto":2,"firmware_id":"tflm_runtime","fw_version":"66a9038","idf_version":"v5.3.1","reset_reason":"poweron","fault_reset":false}
{"type":"ready","model_loaded":true,"commands":"i,b,m,p,a,l","frames":true,"periodic":true,"interval_ms":5000}
```

## 메시지 형식

### info (`m`)

```json
{"type":"info","proto":2,"firmware_id":"tflm_runtime",
 "board":{"chip":"ESP32","cores":2,"revision":301,"cpu_freq_mhz":240,"flash_bytes":4194304,"psram_bytes":0,
          "wifi":true,"bt":true,"ai_accelerator":false,"fw_version":"...","idf_version":"v5.3.1"},
 "baud":115200,"registered_ops":71,
 "supported_models":["sine_regression"],
 "model":{"runtime":"tflm","loaded":true,"id":"sine_regression","task":"regression",
          "input":{"shape":[1,1],"dtype":"int8","scale":0.0244801,"zero_point":-128,"bytes":1},
          "output":{"shape":[1,1],"dtype":"int8","scale":0.00829096,"zero_point":5,"bytes":1},
          "arena_bytes":0,"arena_alloc_bytes":154624,"arena_used_bytes":716,
          "eval_samples":64,"ops":1,"size_bytes":3272,"checksum":"0xfc8a647d",
          "max_arena_bytes":155648,"partition_offset":2097152,"partition_bytes":2097152,"label_count":0},
 "memory":{"heap_free":132616,"heap_min_free":132616,"heap_total":315776,"internal_free":209804,"stack_free":7952}}
```
(sine_regression 보드 실측값)

`info`는 가장 긴 응답이라 1KB 제한을 펌웨어가 직접 지킨다: 레이블을 `model`의 마지막에 두고,
그때까지 출력한 길이 + 레이블 + 나머지(약 160자)가 1,023자를 넘으면 레이블 목록을 빼고 `labels_truncated:true`를 보낸다.

| 필드 | 설명 |
| --- | --- |
| `supported_models` | 로드된 모델 id (v1 호환 필드). 로드 실패면 `[]` |
| `model.label_count` / `model.labels` | 레이블 개수 / 목록 (분류 모델). 1KB에 안 들어가면 목록 대신 `labels_truncated:true` |
| `model.arena_bytes` | 패키지가 요구한 아레나 (0 = 자동) |
| `model.arena_alloc_bytes` | 실제로 할당한 아레나 |
| `model.arena_used_bytes` | **실측** 사용량 (TFLM `arena_used_bytes()`). 메모리 지표로 쓴다 |
| `model.max_arena_bytes` | 아레나 할당 직전 내부 RAM의 가장 큰 연속 블록 = 이 보드에서 가능한 최대 아레나 |
| `model.ops` | 모델이 쓰는 op 종류 수 |
| `model.load_error` | 로드 실패 시 오류 코드 (`loaded:false`일 때만) |
| `model.checksum` | 패키지 crc32 (배포 검증 `--expect-checksum`) |

### inference (`i`)

```json
{"type":"inference","input":"demo","task":"classification","pred":3,"label":"3","score":0.996,"expected":3,"correct":true,
 "latency_us":1830,"output":[0.0,0.0,0.0,0.996,0.0,0.0,0.0,0.0,0.004,0.0],"output_count":10,"memory":{...}}
```
- 회귀는 `pred`/`label`/`score` 대신 `expected`(배열)와 `abs_err`.
- 패키지에 데모가 없으면 `"input":"zeros"`이고 `expected`/`correct`가 없다.
- v1의 `logits`는 v2에서 `output`(역양자화한 값)으로 바뀌었다. v1 펌웨어는 그대로 `logits`를 보낸다.

### bench (`b`)

```json
{"type":"bench","iterations":1000,"total_us":110000,"avg_us":110.0,"min_us":105,"max_us":160,"fps":9090.9,
 "budget_ms":2000,"heap_before":0,"heap_after":0,"memory":{...}}
```
v1 필드(`iterations`, `total_us`, `avg_us`, `fps`, `heap_before`, `heap_after`)는 그대로이고 `min_us`, `max_us`, `budget_ms`가 추가됐다.
`iterations`는 고정값이 아니라 시간 예산으로 정해진다.

### labels (`l`, 신규)

```json
{"type":"labels","count":10,"labels":["0","1","2","3","4","5","6","7","8","9"],"shown":10,"truncated":false}
```
- 1KB 안에 들어가는 만큼만 싣고, 다 못 실으면 `truncated:true`, `shown` = 실은 개수.
- 모델이 없으면 `error` `model_not_loaded`.

### eval (`a`, 신규)

```json
{"type":"eval","source":"package","model":"cnn_mnist_int8","task":"classification",
 "samples":150,"labeled":150,"correct":148,"accuracy":0.9867,"invoke_errors":0,"error_rate":0.0,
 "total_us":270000,"avg_us":1800.0,"min_us":1790,"max_us":1850,"wall_ms":300,"memory":{...}}
```
- 회귀는 `mean_abs_err`가 추가되고 `accuracy`는 허용 오차(`task_param`) 안에 든 비율.
- 평가 샘플이 없으면 `error` `no_eval_samples`.

### metrics (5초 주기)

```json
{"type":"metrics","seq":3,"uptime_ms":15340,"model_loaded":true,
 "iterations":100,"avg_us":110.0,"fps":9090.9,"min_us":105,"max_us":160,
 "invoke_errors":0,"arena_used_bytes":1200,"memory":{...}}
```
- `invoke_errors`: 부팅 후 실패한 `Invoke()` 누적 수 (에러율 계산용).
- 모델이 없으면 `model_loaded:false`이고 추론 관련 필드가 없다.

### memory (공통)

v1과 같다: `heap_free`, `heap_min_free`, `heap_total`, `internal_free`, `stack_free` (바이트).

### error

```json
{"type":"error","code":"arena_too_small","message":"package arena 256 bytes, model needs 1200 bytes"}
{"type":"error","code":"crc_error","seq":7,"message":"frame crc 0x..., computed 0x..."}
```
프레임에 대한 오류에는 그 프레임의 `seq`가 붙는다 (헤더를 못 읽었으면 -1).

## 바이너리 프레임

서버가 준 테스트 입력처럼 바이너리 데이터를 보내거나 통신 속도를 바꿀 때 쓴다.

```
0xA5 0x5A | type (1) | seq (1) | length (4, LE) | payload (length) | crc32 (4, LE)
```
- crc32는 `type`부터 payload 끝까지 (sync 2바이트 제외), Python `zlib.crc32`와 같다.
- 바이트 사이 간격이 1초를 넘으면 `frame_timeout`. 오류가 나면 보드는 **회선이 0.1초 조용해질 때까지 입력을 버린다**
  (깨진 프레임의 payload가 명령으로 해석되지 않게). 호스트는 오류 응답을 받은 뒤 같은 프레임을 다시 보내면 된다.
- 모든 프레임은 응답 한 줄을 받는다 (응답의 `seq` = 요청의 `seq`). 응답을 받기 전에 다음 프레임을 보내지 않는다 (흐름 제어).

| type | 이름 | payload | 응답 |
| --- | --- | --- | --- |
| 0x01 | PING | 없음 | `pong` |
| 0x02 | SET_BAUD | u32 baud | `baud` (이전 속도로 보낸 뒤 전환) |
| 0x10 | TEST_BEGIN | u32 예상 샘플 수(0 = 모름), u8 정답 포함 여부, u8 출력 값 보고 여부, u16 0 | `test_begin` |
| 0x11 | TEST_SAMPLE | 입력 텐서 바이트 [+ float32 × label_dim 정답] | `test_result` |
| 0x12 | TEST_END | 없음 | `test_summary` |

payload 최대 길이: TEST_SAMPLE은 `input.bytes + label_dim×4`와 **정확히** 같아야 하고(`input_size_mismatch`),
나머지는 32바이트 이하(`frame_too_large`).

### 테스트 세션 (서버 테스트 파일 실행)

서버가 준 테스트 파일([TEST_DATA_FORMAT.md](TEST_DATA_FORMAT.md))을 샘플 단위로 보내 추론하고 성능 지표를 돌려받는다.
샘플은 **입력 텐서 메모리로 바로 수신**되므로 추가 RAM이 들지 않고, 파일 크기에도 제한이 없다.

```
호스트                                   보드
TEST_BEGIN(n=100, labels=1)  ───────▶   {"type":"test_begin","seq":0,"input_bytes":784,"label_dim":1,"has_labels":true,"max_payload":788}
TEST_SAMPLE(seq=1, 784+4 B)  ───────▶   {"type":"test_result","seq":1,"index":0,"ok":true,"latency_us":1830,"pred":7,"label":"7","score":0.99,"expected":7,"correct":true}
TEST_SAMPLE(seq=2, ...)      ───────▶   ...
TEST_END                     ───────▶   {"type":"test_summary","source":"stream","model":"cnn_mnist_int8","task":"classification",
                                          "expected_samples":100,"samples":100,"labeled":100,"correct":98,"accuracy":0.98,
                                          "invoke_errors":0,"error_rate":0.0,"total_us":183000,"avg_us":1830.0,"min_us":1820,"max_us":1900,
                                          "memory":{...}}
```
- `test_summary`의 `seq`는 TEST_END에 대한 응답일 때만 붙는다 (세션 시간 초과·새 세션으로 끝난 경우는 없음).
  66a9038 기반 첫 빌드는 TEST_END 응답에도 `seq`가 없었고, `esp_monitor.py`는 두 경우를 모두 받는다.
- `seq`는 1바이트로 순환한다. **직전 샘플과 같은 seq로 다시 보내면 추론하지 않고 직전 결과를 그대로 다시 보낸다**
  (응답이 유실돼 재전송해도 두 번 집계되지 않음).
- 정답 없이 보내면 `correct`/`accuracy`가 없고(`accuracy:null`) 지연·메모리·에러율만 나온다.
- 출력 값 보고를 켜면(또는 회귀 모델이면) `test_result`에 `output`(최대 16개)이 붙는다.
- `Invoke()`가 실패하면 그 샘플은 `{"ok":false,"error":"invoke_failed"}`이고 `invoke_errors`에 집계된다.
- 세션 중에는 주기 `metrics`를 보내지 않는다. 60초 동안 프레임이 없으면 세션을 끝내고 `test_summary`에 `"aborted":true`.
- 세션 중에도 1바이트 명령은 처리된다.

### 속도 변경

```
호스트 (115200)  SET_BAUD(921600)  ──▶  {"type":"baud","seq":0,"baud":921600,"previous":115200,"confirm_ms":3000}  (115200으로 전송 후 전환)
호스트 (921600)  PING              ──▶  {"type":"pong","seq":1,"baud":921600}
```
- 지원 속도: 115200, 230400, 460800, 921600 (그 외 `bad_baud`).
- 전환 후 **3초 안에** 유효한 프레임이나 명령이 오지 않으면 보드는 이전 속도로 되돌아가고
  `{"type":"baud","baud":115200,"reverted":true}`를 보낸다. 호스트도 PING이 실패하면 이전 속도로 돌아간다.
- 리셋하면 항상 115200으로 시작한다. 테스트가 끝나면 호스트가 115200으로 되돌린다 (`esp_monitor.py run-test`가 자동 처리).
- 115200에서 1바이트 ≈ 87µs (8N1 기준 계산값). 9,216바이트(96×96 흑백) 샘플 전송에 약 0.8초, 921600이면 약 0.1초.

## 오류 코드

| code | 언제 |
| --- | --- |
| `no_package` | 부팅 시 모델 파티션이 비어 있음 |
| `package_invalid` | 패키지 형식 오류, 텐서 불일치, 입출력 개수 ≠ 1, MLP1 파일 등 |
| `checksum_mismatch` | 패키지 crc32 불일치 |
| `unsupported_op` | 펌웨어에 등록되지 않은 op |
| `arena_alloc_failed` | 아레나 메모리 할당 실패 (메시지에 가장 큰 여유 블록) |
| `arena_too_small` | 패키지 arena_bytes 부족 (메시지에 실측 필요량) |
| `model_crashed` | 직전 부팅이 이 패키지 로드/첫 실행 중 크래시 → 로드를 건너뜀 |
| `model_not_loaded` | 모델 없이 `i`/`b`/`a`/테스트 요청 |
| `invoke_failed` | `Invoke()` 실패 (`i`/`b`) |
| `no_eval_samples` | 평가 샘플 없는 패키지에 `a` |
| `unknown_command` | 알 수 없는 1바이트 명령 |
| `crc_error` | 프레임 crc 불일치 (재전송) |
| `frame_timeout` | 프레임이 중간에 끊김 (재전송) |
| `frame_too_large` | payload가 허용 길이 초과 |
| `input_size_mismatch` | 샘플 크기가 입력 텐서(+정답)와 다름 |
| `test_not_started` | TEST_BEGIN 없이 샘플/종료 |
| `bad_baud` | 지원하지 않는 속도 |
| `unknown_frame` | 모르는 프레임 type |

v1의 `model_load_failed`는 v2에서 원인별 코드(`no_package`, `package_invalid`, ...)로 나뉘었다.

## 에러/실패율 집계 (명세 1.5)

- 추론 실패: `metrics.invoke_errors`(누적), `eval`/`test_summary`의 `invoke_errors`, `error_rate`
- 비정상 리셋: `boot.fault_reset`, `reset_reason` (v1과 같음)
- 통신 실패: `esp_monitor.py`의 `monitor_event`, `test_report.transport_retries`
