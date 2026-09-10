import argparse
from pathlib import Path

from model_format import read_model_bin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    args = ap.parse_args()
    info = read_model_bin(args.model)
    print("VALID")
    print(f"size={info['size']}")
    print(f"payload_size={info['payload_size']}")
    print(f"dims={info['dims'][0]}->{info['dims'][1]}->{info['dims'][2]}")
    print(f"checksum=0x{info['checksum']:08x}")


if __name__ == "__main__":
    main()
