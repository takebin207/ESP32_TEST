import argparse
import os
import sys
import time
from typing import Any

# The HF Xet downloader can stall on some Windows networks. Plain HTTP is slower
# but more predictable for this small local dashboard.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


REMOTE_MODEL_NAME = "watersplash/waste-classification"
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "waste-classification")
MODEL_NAME = LOCAL_MODEL_DIR
SERVO_ORGANIC_COMMAND = "ORGANIC"
SERVO_INORGANIC_COMMAND = "INORGANIC"

ORGANIC_LABELS = {"biological"}
INORGANIC_RECYCLABLE = {"cardboard", "paper"}
INORGANIC_MATERIALS = {
    "battery",
    "brown-glass",
    "clothes",
    "glass",
    "green-glass",
    "metal",
    "plastic",
    "shoes",
    "white-glass",
}
RESIDUAL_TRASH = {"trash"}


def map_to_waste_group(fine_grained_label: str) -> tuple[str, str]:
    label = fine_grained_label.lower().strip()

    if label in ORGANIC_LABELS:
        return SERVO_ORGANIC_COMMAND, "HUU CO - rac de phan huy sinh hoc"
    if label in INORGANIC_RECYCLABLE:
        return SERVO_INORGANIC_COMMAND, "VO CO - giay / bia cung tai che"
    if label in INORGANIC_MATERIALS:
        return SERVO_INORGANIC_COMMAND, "VO CO - vat lieu kho phan huy"
    if label in RESIDUAL_TRASH:
        return SERVO_INORGANIC_COMMAND, "VO CO - rac sinh hoat con lai"

    return SERVO_INORGANIC_COMMAND, "VO CO - chua ro nhom chi tiet"


def local_model_ready() -> bool:
    has_config = os.path.exists(os.path.join(LOCAL_MODEL_DIR, "config.json"))
    has_processor = os.path.exists(os.path.join(LOCAL_MODEL_DIR, "preprocessor_config.json"))
    has_weights = any(
        os.path.exists(os.path.join(LOCAL_MODEL_DIR, filename))
        for filename in ("model.safetensors", "pytorch_model.bin")
    )
    return has_config and has_processor and has_weights


def load_model() -> tuple[Any, Any]:
    if not local_model_ready():
        raise RuntimeError(
            "Chua co model local. Hay chay lenh: python download_model.py"
        )

    print(f"Dang tai model local tu: {LOCAL_MODEL_DIR}")
    processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
    model.eval()
    return processor, model


def load_image(image_path: str) -> Image.Image:
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Khong tim thay anh '{image_path}'. Hay dat anh vao project hoac truyen --image."
        )

    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def capture_camera_frame(camera_index: int, warmup_frames: int) -> Image.Image:
    if cv2 is None:
        raise RuntimeError("Chua cai opencv-python. Cai bang lenh: python -m pip install -r requirements.txt")

    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise RuntimeError(f"Khong mo duoc camera laptop index {camera_index}.")

    try:
        frame = None
        for _ in range(max(1, warmup_frames)):
            ok, frame = camera.read()
            if not ok:
                frame = None
            time.sleep(0.05)

        if frame is None:
            raise RuntimeError("Khong doc duoc frame tu camera.")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)
    finally:
        camera.release()


def predict_image(image: Image.Image, processor: Any, model: Any) -> dict:
    inputs = processor(images=image, return_tensors="pt")

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
        "original_label": original_label,
        "confidence": confidence_score,
        "command": command,
        "final_category": final_category,
        "probabilities": {labels[idx]: score.item() for idx, score in enumerate(probabilities)},
    }


def predict_waste_from_file(image_path: str, processor: Any, model: Any) -> dict:
    return predict_image(load_image(image_path), processor, model)


def predict_waste_from_camera(
    camera_index: int,
    warmup_frames: int,
    processor: Any,
    model: Any,
    save_frame: str | None,
) -> dict:
    image = capture_camera_frame(camera_index, warmup_frames)
    if save_frame:
        image.save(save_frame)
        print(f"Da luu frame camera vao: {save_frame}")
    return predict_image(image, processor, model)


def auto_detect_port() -> str | None:
    if list_ports is None:
        return None

    preferred_keywords = ("usb", "uart", "ch340", "cp210", "silicon labs", "jtag")

    for port in list_ports.comports():
        description = f"{port.device} {port.description} {port.manufacturer}".lower()
        if any(keyword in description for keyword in preferred_keywords):
            return port.device

    return None


def send_command_to_esp32(
    port: str,
    command: str,
    baudrate: int,
    reset_after: float,
    response_wait: float = 1.0,
    boot_wait: float = 2.0,
) -> list[str]:
    if serial is None:
        raise RuntimeError("Chua cai pyserial. Cai bang lenh: python -m pip install -r requirements.txt")

    print(f"Mo cong serial {port} @ {baudrate} baud...")
    responses = []
    with serial.Serial(port, baudrate, timeout=2) as connection:
        if boot_wait > 0:
            time.sleep(boot_wait)

        connection.write(f"{command}\n".encode("utf-8"))
        connection.flush()
        print(f"Da gui lenh: {command}")
        responses.extend(read_serial_lines(connection, response_wait))

        if reset_after > 0:
            time.sleep(reset_after)
            connection.write(b"CENTER\n")
            connection.flush()
            print("Da dua servo ve CENTER")
            responses.extend(read_serial_lines(connection, response_wait))

    return responses


def read_serial_lines(connection: Any, duration: float) -> list[str]:
    deadline = time.monotonic() + duration
    lines = []

    while time.monotonic() < deadline:
        raw_line = connection.readline()
        if not raw_line:
            continue

        line = raw_line.decode("utf-8", errors="replace").strip()
        if line:
            print(f"ESP32: {line}")
            lines.append(line)

    return lines


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


def run_live_camera(args: argparse.Namespace, processor: Any, model: Any) -> None:
    if cv2 is None:
        raise RuntimeError("Chua cai opencv-python. Cai bang lenh: python -m pip install -r requirements.txt")

    port = None if args.no_serial else resolve_port(args.port)
    camera = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise RuntimeError(f"Khong mo duoc camera laptop index {args.camera_index}.")

    last_sent_at = 0.0
    last_command = None

    print("Dang nhan dien live. Nhan Ctrl+C de dung.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Khong doc duoc frame tu camera.")

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            result = predict_image(image, processor, model)
            print_prediction(result)

            now = time.monotonic()
            can_send = result["confidence"] >= args.min_confidence
            cooldown_done = now - last_sent_at >= args.cooldown
            command_changed = result["command"] != last_command

            if port and can_send and (cooldown_done or command_changed):
                send_command_to_esp32(port, result["command"], args.baudrate, args.reset_after)
                last_sent_at = now
                last_command = result["command"]

            time.sleep(args.interval)
    finally:
        camera.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chay model Python de phan loai rac tu anh hoac camera laptop va dieu khien servo ESP32."
    )
    parser.add_argument("--image", default="img.png", help="Duong dan anh dau vao.")
    parser.add_argument("--camera", action="store_true", help="Dung camera laptop thay vi anh file.")
    parser.add_argument("--camera-index", type=int, default=0, help="Index camera laptop, mac dinh la 0.")
    parser.add_argument("--warmup-frames", type=int, default=10, help="So frame bo qua de camera on dinh.")
    parser.add_argument("--save-frame", default="camera_frame.jpg", help="Noi luu frame camera vua chup.")
    parser.add_argument("--live", action="store_true", help="Nhan dien lien tuc tu camera laptop.")
    parser.add_argument("--interval", type=float, default=2.0, help="So giay giua moi lan nhan dien live.")
    parser.add_argument("--cooldown", type=float, default=3.0, help="So giay toi thieu giua moi lan gui lenh servo.")
    parser.add_argument("--min-confidence", type=float, default=0.35, help="Nguong tin cay de gui lenh servo live.")
    parser.add_argument("--port", default=None, help="Cong serial ESP32, vi du COM5.")
    parser.add_argument("--baudrate", type=int, default=115200, help="Baudrate serial.")
    parser.add_argument("--no-serial", action="store_true", help="Chi chay model, khong gui lenh den ESP32.")
    parser.add_argument(
        "--test-command",
        choices=[SERVO_ORGANIC_COMMAND, SERVO_INORGANIC_COMMAND, "CENTER"],
        help="Gui lenh test servo truc tiep, khong chay model.",
    )
    parser.add_argument(
        "--reset-after",
        type=float,
        default=1.0,
        help="So giay cho truoc khi gui CENTER. Dat 0 de giu nguyen vi tri.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.test_command:
            port = resolve_port(args.port)
            send_command_to_esp32(port, args.test_command, args.baudrate, args.reset_after)
            return 0

        processor, model = load_model()

        if args.live:
            run_live_camera(args, processor, model)
            return 0

        if args.camera:
            result = predict_waste_from_camera(
                args.camera_index,
                args.warmup_frames,
                processor,
                model,
                args.save_frame,
            )
        else:
            result = predict_waste_from_file(args.image, processor, model)

        print_prediction(result)

        if args.no_serial:
            return 0

        port = resolve_port(args.port)
        send_command_to_esp32(port, result["command"], args.baudrate, args.reset_after)
        return 0
    except KeyboardInterrupt:
        print("\nDa dung chuong trinh.")
        return 0
    except Exception as exc:
        print(f"Loi: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
