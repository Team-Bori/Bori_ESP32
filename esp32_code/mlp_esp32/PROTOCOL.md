# ESP32 MLP 시리얼 프로토콜 (v1)

- UART0, 115200 baud, 8N1
- 호스트(라즈베리파이)는 **1바이트 명령**을 보내고, 보드는 **JSON 객체 한 줄**(`\n` 종료)로 응답한다.
- `{`로 시작하지 않는 줄(부트로더 메시지, ESP_LOG 출력)은 무시한다.
- 모든 메시지에는 `type` 필드가 있다.

## 명령

| 명령 | 응답 `type` | 설명 |
| --- | --- | --- |
| `m` | `info` | 보드 사양, 지원 모델, 로드된 모델, 메모리 |
| `i` | `inference` | 데모 입력으로 1회 추론 |
| `b` | `bench` | 1000회 추론 벤치마크 |
| `p` | `periodic` | 주기 보고(`metrics`) 켜기/끄기 토글. 부팅 시 기본값은 켜짐 |
| 그 외 출력 가능한 ASCII 문자 | `error` (`unknown_command`) | 메시지에 받은 바이트 값을 포함 |
| 공백·제어 문자·비ASCII(0x7F 이상) | 응답 없음 | 터미널 이스케이프 시퀀스나 포트 연결 시 잡음은 무시 |

## 자동 출력 (부팅 시)

1. `boot`: 펌웨어 버전, 리셋 원인
2. 모델 로드 실패 시 `error` (`model_load_failed`)
3. `ready`: 명령 수신 가능
4. 이후 5초(`interval_ms`)마다 `metrics` (주기 보고가 켜져 있는 동안)

## 메시지 형식

### boot
```json
{"type":"boot","proto":1,"fw_version":"c6c4415","idf_version":"v5.3.1","reset_reason":"poweron","fault_reset":false}
```
- `reset_reason`: `poweron` `external` `software` `panic` `int_wdt` `task_wdt` `wdt` `deepsleep` `brownout` `sdio` `unknown`
- `fault_reset`: 직전 리셋이 패닉·워치독·브라운아웃이면 `true`. 에러/실패율 집계에 사용한다.

### ready
```json
{"type":"ready","model_loaded":true,"commands":"i,b,m,p","periodic":true,"interval_ms":5000}
```

### metrics (주기 보고)
```json
{"type":"metrics","seq":3,"uptime_ms":15340,"model_loaded":true,
 "iterations":100,"avg_us":103.1,"fps":9699.0,"memory":{...}}
```
- `seq`: 부팅 후 1부터 증가. 값이 건너뛰면 보고가 유실된 것이다.
- 매 보고마다 100회 추론으로 지연 시간을 측정한다 (약 10ms 소요).
- 모델이 없으면 `model_loaded:false`이고 `iterations`/`avg_us`/`fps`는 빠진다.

### periodic (`p` 응답)
```json
{"type":"periodic","enabled":false,"interval_ms":5000}
```

### info
```json
{"type":"info","proto":1,
 "board":{"chip":"ESP32","cores":2,"revision":301,"cpu_freq_mhz":160,"flash_bytes":4194304,"psram_bytes":0,
          "wifi":true,"bt":true,"ai_accelerator":false,"fw_version":"c6c4415","idf_version":"v5.3.1"},
 "supported_models":["mlp_64_16_10_int8"],
 "model":{"id":"mlp_64_16_10_int8","loaded":true,"dims":[64,16,10],"quantization":"int8",
          "size_bytes":1340,"checksum":"0x12550a83","partition_offset":1966080,"partition_bytes":131072},
 "memory":{...}}
```
- `model.size_bytes`, `model.checksum`은 모델이 로드된 경우에만 있다.
- `revision`: 칩 리비전 (예: 301 = v3.1)

### inference
```json
{"type":"inference","input":"demo","pred":5,"expected":5,"correct":true,"latency_us":211,
 "logits":[1583,-4548,-8418,-251,-3430,5636,-245,-834,1433,3256],"memory":{...}}
```

### bench
```json
{"type":"bench","iterations":1000,"total_us":103106,"avg_us":103.106,"fps":9698.757,
 "heap_before":303480,"heap_after":303480,"memory":{...}}
```

### memory (공통 필드)
```json
{"heap_free":303480,"heap_min_free":303456,"heap_total":319760,"internal_free":387840,"stack_free":2080}
```
- 단위는 모두 바이트. `stack_free`는 메인 태스크 스택의 최소 여유량(high water mark).

### error
```json
{"type":"error","code":"model_not_loaded","message":"..."}
```
| code | 의미 |
| --- | --- |
| `model_load_failed` | 부팅 시 모델 파티션에 유효한 model.bin이 없음 |
| `model_not_loaded` | 모델 없이 `i`/`b`를 요청함 |
| `unknown_command` | 알 수 없는 명령 |
