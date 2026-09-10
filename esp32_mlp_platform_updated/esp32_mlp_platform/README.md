# ESP32 MLP AI Experience Platform - firmware/weight separated prototype

## 1. Architecture

The ESP32 application and the neural-network weights are separated.

```text
GitHub
├── firmware/mlp.bin
│      -> bootloader + partition table + MLP application
└── models/mlp/default_model.bin
       -> example 64 -> 16 -> 10 INT8 model

Raspberry Pi
├── flash firmware once
└── replace only model.bin for each user model
```

The ESP32 does **not** contain multiple AI models simultaneously.
The selected AI has one firmware image, and the user's weights are stored in a separate `model` data partition.

Current partition map:

```text
0x00009000  NVS
0x0000F000  PHY init
0x00010000  factory application
0x001E0000  model data (128 KB)
```

The model partition is intentionally separate so a weight update does not rebuild or replace the application firmware.

## 2. Project tree

```text
esp32_mlp_platform/
├── mlp_esp32/
│   ├── CMakeLists.txt
│   ├── partitions.csv
│   ├── sdkconfig.defaults
│   ├── release.sh
│   └── main/
│       ├── CMakeLists.txt
│       ├── main.c
│       ├── mlp.c
│       ├── mlp.h
│       ├── model.c
│       └── model.h
├── pc/
│   ├── model_format.py
│   ├── train.py
│   ├── validate_model.py
│   └── requirements.txt
├── models/
│   └── mlp/
│       └── default_model.bin
├── firmware/
└── rpi/
    ├── flash_firmware_from_github.sh
    ├── flash_model_from_github.sh
    ├── flash_from_github.sh
    └── flash_local_model.sh
```

## 3. Build a model on the PC

```bash
cd pc
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

The default output is:

```text
models/mlp/default_model.bin
```

The trainer no longer generates `mlp_model.c` or `mlp_model.h`.
Weights, biases and quantization scales are serialized into `model.bin`.

Validate a model before flashing:

```bash
python validate_model.py ../models/mlp/default_model.bin
```

Expected:

```text
VALID
size=1340
payload_size=1288
dims=64->16->10
checksum=0x...
```

## 4. Model file format

`model.bin` consists of:

```text
52-byte header
+
1288-byte payload

Payload:
W1  = 64 x 16 int8   = 1024 bytes
B1  = 16 int32       =   64 bytes
W2  = 16 x 10 int8   =  160 bytes
B2  = 10 int32       =   40 bytes
```

The header stores:

- magic/version
- model dimensions
- input/weight/activation scales
- payload sizes
- FNV-1a checksum

The ESP32 validates the header, dimensions and checksum before using the model.

For the current firmware, uploaded weights must match exactly:

```text
64 -> 16 -> 10
ReLU hidden layer
INT8 W1/W2
INT32 B1/B2
```

A future model version can introduce a different model ID/version and a different firmware.

## 5. Build the ESP32 firmware

Use ESP-IDF on the development PC, not on the Raspberry Pi.

```bash
cd mlp_esp32
idf.py set-target esp32
idf.py build
```

The project uses `partitions.csv` and `sdkconfig.defaults` to create the custom model partition.

Create a GitHub-ready merged firmware image:

```bash
./release.sh
```

Output:

```text
firmware/mlp.bin
firmware/mlp.bin.sha256
```

`mlp.bin` is the complete firmware image and is flashed from address `0x0`.
It does not contain the user's model data.

## 6. First hardware test

For the first test, flash the firmware with ESP-IDF:

```bash
idf.py -p /dev/ttyUSB0 flash monitor
```

Then flash the example model to the separate model partition:

```bash
esptool --chip esp32 -p /dev/ttyUSB0 -b 921600 \
  write-flash --flash-mode dio --flash-size detect \
  0x1E0000 ../models/mlp/default_model.bin
```

After reboot:

```text
i -> one inference
b -> 1000 inference benchmark
```

The serial output reports prediction, latency, FPS, model bytes and heap usage.

If the model partition is missing or invalid, the firmware prints:

```text
MLP_ERROR model_not_loaded
```

## 7. Raspberry Pi: firmware update

Install `curl` and `esptool`.

```bash
chmod +x rpi/*.sh
```

Firmware:

```bash
./rpi/flash_firmware_from_github.sh \
  /dev/ttyUSB0 \
  https://raw.githubusercontent.com/USER/REPO/main/firmware/mlp.bin
```

The Raspberry Pi does not build the ESP-IDF project.

## 8. Raspberry Pi: model-only update

This is the important path for web uploads.

GitHub model:

```bash
./rpi/flash_model_from_github.sh \
  /dev/ttyUSB0 \
  https://raw.githubusercontent.com/USER/REPO/main/models/mlp/default_model.bin
```

Local uploaded model:

```bash
./rpi/flash_local_model.sh \
  /dev/ttyUSB0 \
  /home/pi/uploads/user_model.bin
```

The local script first validates the binary and then writes it only to the model partition.
The firmware itself is not replaced.

## 9. Full first-time deployment from GitHub

```bash
./rpi/flash_from_github.sh \
  /dev/ttyUSB0 \
  https://raw.githubusercontent.com/USER/REPO/main/firmware/mlp.bin \
  https://raw.githubusercontent.com/USER/REPO/main/models/mlp/default_model.bin
```

This performs:

```text
firmware.bin -> 0x000000
model.bin    -> 0x1E0000
```

## 10. Web upload integration

The web server should not send arbitrary files directly to `esptool`.
The intended flow is:

```text
Web user
   |
   | upload trained weights
   v
Raspberry Pi server
   |
   | validate_model.py
   | check dimensions / format / checksum
   v
validated model.bin
   |
   | flash_local_model.sh
   v
ESP32 model partition (0x1E0000)
   |
   v
ESP32 reboots and loads model.bin
```

The web server therefore controls the uploaded model file, while `firmware/mlp.bin` remains a fixed trusted executable.

## 11. Current first model

Model:

```text
64 -> 16 -> 10 MLP
activation: ReLU
quantization: INT8 weights + INT32 biases
```

PC verification from the current sample training run:

```text
FP32 test accuracy : 0.9583
INT8 test accuracy : 0.9556
INT8 payload       : 1288 bytes
model.bin total    : 1340 bytes
```
