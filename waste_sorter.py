import argparse
import os
import sys
import time

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


MODEL_NAME = "watersplash/waste-classification"
SERVO_ORGANIC_COMMAND = "ORGANIC"
SERVO_INORGANIC_COMMAND = "INORGANIC"

INORGANIC_RECYCLABLE = {"cardboard", "paper"}
INORGANIC_MATERIALS = {"glass", "metal", "plastic"}
RESIDUAL_TRASH = {"trash"}


def map_to_waste_group(fine_grained_label: str) -> tuple[str, str]:
    label = fine_grained_label.lower().strip()

    if label in INORGANIC_RECYCLABLE:
        return SERVO_INORGANIC_COMMAND, "VO CO - giay / bia cung tai che"
    if label in INORGANIC_MATERIALS:
        return SERVO_INORGANIC_COMMAND, "VO CO - chai lo / kim loai / nhua"
    if label in RESIDUAL_TRASH:
        return SERVO_INORGANIC_COMMAND, "VO CO - rac sinh hoat con lai"

    return SERVO_ORGANIC_COMMAND, "HUU CO - rac de phan huy sinh hoc"


def load_image(image_path: str) -> Image.Image:
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Khong tim thay anh '{image_path}'. Hay dat anh vao project hoac truyen --image."
        )

    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def predict_waste(image_path: str) -> dict:
    raw_image = load_image(image_path)

    print("Dang tai model tu Hugging Face...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)

    inputs = processor(images=raw_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    predicted_class_idx = logits.argmax(-1).item()

    labels = model.config.id2label
    original_label = labels[predicted_class_idx]
    confidence_score = probabilities[predicted_class_idx].item()
    command, final_category = map_to_waste_group(original_label)

    return {
        "image_path": image_path,
        "original_label": original_label,
        "confidence": confidence_score,
        "command": command,
        "final_category": final_category,
        "probabilities": {labels[idx]: score.item() for idx, score in enumerate(probabilities)},
    }


def auto_detect_port() -> str | None:
    if list_ports is None:
        return None

    preferred_keywords = ("usb", "uart", "ch340", "cp210", "silicon labs", "jtag")

    for port in list_ports.comports():
        description = f"{port.device} {port.description} {port.manufacturer}".lower()
        if any(keyword in description for keyword in preferred_keywords):
            return port.device

    return None


def send_command_to_esp32(port: str, command: str, baudrate: int, reset_after: float) -> None:
    if serial is None:
        raise RuntimeError("Chua cai pyserial. Cai bang lenh: python -m pip install pyserial")

    print(f"Mo cong serial {port} @ {baudrate} baud...")
    with serial.Serial(port, baudrate, timeout=2) as connection:
        time.sleep(2)

        connection.write(f"{command}\n".encode("utf-8"))
        connection.flush()
        print(f"Da gui lenh: {command}")

        if reset_after > 0:
            time.sleep(reset_after)
            connection.write(b"CENTER\n")
            connection.flush()
            print("Da dua servo ve CENTER")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chay model Python de phan loai rac va dieu khien servo ESP32."
    )
    parser.add_argument("--image", default="img.png", help="Duong dan anh dau vao.")
    parser.add_argument("--port", default=None, help="Cong serial ESP32, vi du COM5.")
    parser.add_argument("--baudrate", type=int, default=115200, help="Baudrate serial.")
    parser.add_argument(
        "--no-serial",
        action="store_true",
        help="Chi chay model, khong gui lenh den ESP32.",
    )
    parser.add_argument(
        "--test-command",
        choices=[SERVO_ORGANIC_COMMAND, SERVO_INORGANIC_COMMAND, "CENTER"],
        help="Gui lenh test servo truc tiep, khong chay model.",
    )
    parser.add_argument(
        "--reset-after",
        type=float,
        default=0,
        help="So giay cho truoc khi gui CENTER. Mac dinh 0 la giu nguyen vi tri.",
    )
    return parser


def resolve_port(requested_port: str | None) -> str:
    port = requested_port or auto_detect_port()
    if not port:
        raise RuntimeError("Khong tim thay ESP32. Hay cam board va truyen --port COMx.")
    return port


def print_prediction(result: dict) -> None:
    print("\n" + "=" * 50)
    print(f"PHAN LOAI CUOI CUNG: {result['final_category']}")
    print(
        f"Nhan model: {result['original_label']} "
        f"(do tin cay: {result['confidence']:.2%})"
    )
    print(f"Lenh servo: {result['command']}")
    print("=" * 50)

    print("\nXac suat tung nhan:")
    for label, score in result["probabilities"].items():
        print(f"  - {label}: {score:.2%}")

    if result["command"] == SERVO_INORGANIC_COMMAND:
        print("\nKet qua: rac VO CO -> servo quay sang phai.")
    else:
        print("\nKet qua: rac HUU CO -> servo quay 90 do sang trai.")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.test_command:
            port = resolve_port(args.port)
            send_command_to_esp32(port, args.test_command, args.baudrate, args.reset_after)
            return 0

        result = predict_waste(args.image)
        print_prediction(result)

        if args.no_serial:
            return 0

        port = resolve_port(args.port)
        send_command_to_esp32(port, result["command"], args.baudrate, args.reset_after)
        return 0
    except Exception as exc:
        print(f"Loi: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
