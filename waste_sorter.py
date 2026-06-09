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
KERAS_REMOTE_MODEL_NAME = "03Komalpreet/WASTE_CLASSIFICATION"
KERAS_LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "komalpreet-waste-classification")
KERAS_MODEL_FILE = os.path.join(KERAS_LOCAL_MODEL_DIR, "waste_classification_model.h5")
FATHIMA_REMOTE_MODEL_NAME = "fathima-ai/waste_classifier2"
FATHIMA_LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "fathima-waste-classifier")
FATHIMA_MODEL_FILE = os.path.join(FATHIMA_LOCAL_MODEL_DIR, "waste_classifier_model.h5")
MODEL_NAME = LOCAL_MODEL_DIR
SERVO_ORGANIC_COMMAND = "ORGANIC"
SERVO_INORGANIC_COMMAND = "INORGANIC"

KERAS_LABELS = [
    "battery",
    "biological",
    "brown-glass",
    "cardboard",
    "clothes",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
    "white-glass",
]

ORGANIC_LABELS = {"biological", "cardboard", "paper"}
HAZARDOUS_LABELS = {"battery"}
INORGANIC_RECYCLABLE_LABELS = {
    "brown-glass",
    "green-glass",
    "metal",
    "plastic",
    "white-glass",
}
INORGANIC_RESIDUAL_LABELS = {
    "clothes",
    "shoes",
    "trash",
}
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

LABEL_INFO = {
    "battery": ("Pin / rac nguy hai", "NGUY HAI - pin can thu gom rieng, tam dua sang ben vo co"),
    "biological": ("Rac sinh hoc / thuc pham", "HUU CO - rac de phan huy sinh hoc"),
    "brown-glass": ("Thuy tinh nau", "VO CO - thuy tinh tai che duoc"),
    "cardboard": ("Bia carton", "HUU CO - bia carton co the phan huy"),
    "clothes": ("Quan ao / vai", "VO CO - vat lieu vai tong hop hoac can phan loai rieng"),
    "green-glass": ("Thuy tinh xanh", "VO CO - thuy tinh tai che duoc"),
    "metal": ("Kim loai", "VO CO - kim loai tai che duoc"),
    "paper": ("Giay", "HUU CO - giay co the phan huy"),
    "plastic": ("Nhua", "VO CO - nhua kho phan huy"),
    "shoes": ("Giay dep", "VO CO - vat lieu tong hop kho phan huy"),
    "trash": ("Rac sinh hoat con lai", "VO CO - rac con lai / khong tai che"),
    "white-glass": ("Thuy tinh trang", "VO CO - thuy tinh tai che duoc"),
    "organic": ("Rac huu co", "HUU CO - model nhi phan phat hien organic"),
    "recyclable": ("Rac tai che", "VO CO - model nhi phan phat hien recyclable"),
}


def map_to_waste_group(fine_grained_label: str) -> tuple[str, str]:
    label = fine_grained_label.lower().strip()

    if label in LABEL_INFO:
        detail = LABEL_INFO[label][1]
        if label in ORGANIC_LABELS or label == "organic":
            return SERVO_ORGANIC_COMMAND, detail
        return SERVO_INORGANIC_COMMAND, detail

    if label in ORGANIC_LABELS:
        if label in {"cardboard", "paper"}:
            return SERVO_ORGANIC_COMMAND, "HUU CO - giay / bia co the phan huy"
        return SERVO_ORGANIC_COMMAND, "HUU CO - rac de phan huy sinh hoc"
    if label in HAZARDOUS_LABELS:
        return SERVO_INORGANIC_COMMAND, "NGUY HAI - pin / rac can tach rieng"
    if label in INORGANIC_RECYCLABLE_LABELS:
        return SERVO_INORGANIC_COMMAND, "VO CO - tai che duoc"
    if label in INORGANIC_MATERIALS:
        return SERVO_INORGANIC_COMMAND, "VO CO - vat lieu kho phan huy"
    if label in RESIDUAL_TRASH:
        return SERVO_INORGANIC_COMMAND, "VO CO - rac sinh hoat con lai"

    return SERVO_INORGANIC_COMMAND, "VO CO - chua ro nhom chi tiet"


def display_label(label: str) -> str:
    normalized = label.lower().strip()
    return LABEL_INFO.get(normalized, (label, ""))[0]


def label_group(label: str) -> str:
    normalized = label.lower().strip()
    if normalized in ORGANIC_LABELS:
        return "organic"
    if normalized in HAZARDOUS_LABELS:
        return "hazardous"
    return "inorganic"


def classify_group(probabilities: dict[str, float]) -> tuple[str, str, str, float, dict[str, float]]:
    group_scores = {"organic": 0.0, "inorganic": 0.0, "hazardous": 0.0}

    for label, score in probabilities.items():
        group_scores[label_group(label)] += score

    best_group = max(group_scores, key=group_scores.get)

    if best_group == "organic":
        return (
            SERVO_ORGANIC_COMMAND,
            "HUU CO - phu hop u phan / biogas",
            best_group,
            group_scores[best_group],
            group_scores,
        )
    if best_group == "hazardous":
        return (
            SERVO_INORGANIC_COMMAND,
            "NGUY HAI - can tach rieng, tam dua sang ben vo co",
            best_group,
            group_scores[best_group],
            group_scores,
        )

    return (
        SERVO_INORGANIC_COMMAND,
        "VO CO - tai che / khong tai che tuy vat lieu",
        best_group,
        group_scores[best_group],
        group_scores,
    )


def local_model_ready() -> bool:
    has_config = os.path.exists(os.path.join(LOCAL_MODEL_DIR, "config.json"))
    has_processor = os.path.exists(os.path.join(LOCAL_MODEL_DIR, "preprocessor_config.json"))
    has_weights = any(
        os.path.exists(os.path.join(LOCAL_MODEL_DIR, filename))
        for filename in ("model.safetensors", "pytorch_model.bin")
    )
    return has_config and has_processor and has_weights


def keras_model_ready() -> bool:
    return os.path.exists(KERAS_MODEL_FILE) and os.path.getsize(KERAS_MODEL_FILE) > 0


def fathima_model_ready() -> bool:
    return os.path.exists(FATHIMA_MODEL_FILE) and os.path.getsize(FATHIMA_MODEL_FILE) > 0


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


def load_keras_model() -> tuple[None, Any]:
    if not keras_model_ready():
        raise RuntimeError(
            "Chua co model Keras local. Hay chay lenh: python download_keras_model.py"
        )

    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError(
            "Chua cai TensorFlow. Hay chay: python -m pip install tensorflow"
        ) from exc

    print(f"Dang tai Keras model local tu: {KERAS_MODEL_FILE}")
    model = keras.models.load_model(KERAS_MODEL_FILE, compile=False)
    return None, model


def load_fathima_model() -> tuple[None, Any]:
    if not fathima_model_ready():
        raise RuntimeError(
            "Chua co model Fathima local. Hay chay lenh: python download_fathima_model.py"
        )

    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError(
            "Chua cai TensorFlow. Hay chay: python -m pip install tensorflow"
        ) from exc

    print(f"Dang tai Fathima model local tu: {FATHIMA_MODEL_FILE}")
    model = keras.models.load_model(FATHIMA_MODEL_FILE, compile=False)
    return None, model


def load_selected_model(model_backend: str) -> tuple[str, Any, Any]:
    if model_backend == "fathima":
        processor, model = load_fathima_model()
        return "fathima", processor, model

    if model_backend == "keras":
        processor, model = load_keras_model()
        return "keras", processor, model

    processor, model = load_model()
    return "transformers", processor, model


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

    camera = cv2.VideoCapture(camera_index)
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

    labels = model.config.id2label
    probability_map = {labels[idx]: score.item() for idx, score in enumerate(probabilities)}
    
    # An (hide) phan loai bia carton de tranh nham lan voi background
    if "cardboard" in probability_map:
        probability_map["cardboard"] = 0.0

    original_label = max(probability_map, key=probability_map.get)
    original_confidence = probability_map[original_label]
    _, _, _, _, group_scores = classify_group(probability_map)
    command, final_category = map_to_waste_group(original_label)
    group = label_group(original_label)
    confidence_score = max(original_confidence, group_scores[group])

    return {
        "original_label": original_label,
        "display_label": display_label(original_label),
        "original_confidence": original_confidence,
        "confidence": confidence_score,
        "command": command,
        "group": group,
        "final_category": final_category,
        "group_scores": group_scores,
        "probabilities": probability_map,
    }


def predict_keras_image(image: Image.Image, model: Any) -> dict:
    import numpy as np

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    height = input_shape[1] or 224
    width = input_shape[2] or 224

    resized = image.convert("RGB").resize((width, height))
    array = np.asarray(resized, dtype="float32") / 255.0
    batch = np.expand_dims(array, axis=0)

    outputs = model.predict(batch, verbose=0)
    probabilities = outputs[0]

    probability_map = {
        KERAS_LABELS[idx]: float(probabilities[idx])
        for idx in range(min(len(KERAS_LABELS), len(probabilities)))
    }
    
    # An (hide) phan loai bia carton de tranh nham lan voi background
    if "cardboard" in probability_map:
        probability_map["cardboard"] = 0.0

    original_label = max(probability_map, key=probability_map.get)
    original_confidence = probability_map[original_label]
    _, _, _, _, group_scores = classify_group(probability_map)
    command, final_category = map_to_waste_group(original_label)
    group = label_group(original_label)
    confidence_score = max(original_confidence, group_scores[group])

    return {
        "original_label": original_label,
        "display_label": display_label(original_label),
        "original_confidence": original_confidence,
        "confidence": confidence_score,
        "command": command,
        "group": group,
        "final_category": final_category,
        "group_scores": group_scores,
        "probabilities": probability_map,
    }


def predict_fathima_image(image: Image.Image, model: Any) -> dict:
    import numpy as np

    resized = image.convert("RGB").resize((150, 150))
    array = np.asarray(resized, dtype="float32") / 255.0
    batch = np.expand_dims(array, axis=0)

    output = model.predict(batch, verbose=0)[0]
    recyclable_score = float(output[0])
    organic_score = 1.0 - recyclable_score
    probability_map = {
        "organic": organic_score,
        "recyclable": recyclable_score,
    }

    if organic_score >= recyclable_score:
        original_label = "organic"
        original_confidence = organic_score
        command = SERVO_ORGANIC_COMMAND
        group = "organic"
        final_category = "HUU CO - model Fathima phat hien organic"
    else:
        original_label = "recyclable"
        original_confidence = recyclable_score
        command = SERVO_INORGANIC_COMMAND
        group = "inorganic"
        final_category = "VO CO - model Fathima phat hien recyclable"

    group_scores = {
        "organic": organic_score,
        "inorganic": recyclable_score,
        "hazardous": 0.0,
    }

    return {
        "original_label": original_label,
        "display_label": display_label(original_label),
        "original_confidence": original_confidence,
        "confidence": original_confidence,
        "command": command,
        "group": group,
        "final_category": final_category,
        "group_scores": group_scores,
        "probabilities": probability_map,
    }

def is_blank_image(image: Image.Image) -> bool:
    import numpy as np
    import logging

    if not hasattr(is_blank_image, "reference_bg"):
        is_blank_image.reference_bg = None

    gray = np.array(image.convert("L"), dtype=np.float32)

    if is_blank_image.reference_bg is None:
        is_blank_image.reference_bg = gray
        logging.info("Set initial frame as reference background.")
        return True

    # Compensate for global brightness changes (auto-exposure)
    diff_raw = gray - is_blank_image.reference_bg
    median_shift = np.median(diff_raw)
    
    diff_abs = np.abs(diff_raw - median_shift)
    changed_pixels = np.sum(diff_abs > 35) # threshold for local changes
    changed_ratio = changed_pixels / gray.size

    # Compute edge density as a secondary check
    if cv2 is not None:
        edges = cv2.Canny(np.array(image.convert("L")), 30, 100)
        edge_density = np.sum(edges > 0) / edges.size
    else:
        edge_density = 1.0

    if changed_ratio < 0.04 or edge_density < 0.015:
        # Update background
        is_blank_image.reference_bg = 0.9 * is_blank_image.reference_bg + 0.1 * gray
        return True

    logging.info(f"Image not blank. changed_ratio={changed_ratio:.4f}, edge_density={edge_density:.4f}")
    return False


def predict_with_backend(image: Image.Image, backend: str, processor: Any, model: Any) -> dict:
    if is_blank_image(image):
        return {
            "original_label": "nothing",
            "display_label": "Khong co rac",
            "original_confidence": 1.0,
            "confidence": 1.0,
            "command": "CENTER",
            "group": "nothing",
            "final_category": "TRONG - Khong phat hien rac",
            "group_scores": {"organic": 0.0, "inorganic": 0.0, "hazardous": 0.0, "nothing": 1.0},
            "probabilities": {"nothing": 1.0},
        }

    if backend == "fathima":
        return predict_fathima_image(image, model)

    if backend == "keras":
        return predict_keras_image(image, model)

    return predict_image(image, processor, model)


def predict_waste_from_file(image_path: str, backend: str, processor: Any, model: Any) -> dict:
    return predict_with_backend(load_image(image_path), backend, processor, model)


def predict_waste_from_camera(
    camera_index: int,
    warmup_frames: int,
    backend: str,
    processor: Any,
    model: Any,
    save_frame: str | None,
) -> dict:
    image = capture_camera_frame(camera_index, warmup_frames)
    if save_frame:
        image.save(save_frame)
        print(f"Da luu frame camera vao: {save_frame}")
    return predict_with_backend(image, backend, processor, model)


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
    serial_timeout = max(0.05, min(response_wait, 0.2))
    connection = serial.Serial()
    connection.port = port
    connection.baudrate = baudrate
    connection.timeout = serial_timeout
    connection.write_timeout = 1
    connection.dtr = False
    connection.rts = False

    with connection:
        # Avoid toggling ESP32 auto-reset lines every time the web app sends a command.
        connection.setDTR(False)
        connection.setRTS(False)
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
        f"Nhan model: {result.get('display_label', result['original_label'])} / {result['original_label']} "
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


def run_live_camera(args: argparse.Namespace, backend: str, processor: Any, model: Any) -> None:
    if cv2 is None:
        raise RuntimeError("Chua cai opencv-python. Cai bang lenh: python -m pip install -r requirements.txt")

    port = None if args.no_serial else resolve_port(args.port)
    camera = cv2.VideoCapture(args.camera_index)
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
            result = predict_with_backend(image, backend, processor, model)
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
    parser.add_argument(
        "--model-backend",
        choices=["transformers", "keras", "fathima"],
        default="transformers",
        help="Chon model: transformers hien tai, keras cua 03Komalpreet, hoac fathima binary organic/recyclable.",
    )
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
        default=1.5,
        help="So giay cho truoc khi gui CENTER sau khi quay trai/phai.",
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

        backend, processor, model = load_selected_model(args.model_backend)

        if args.live:
            run_live_camera(args, backend, processor, model)
            return 0

        if args.camera:
            result = predict_waste_from_camera(
                args.camera_index,
                args.warmup_frames,
                backend,
                processor,
                model,
                args.save_frame,
            )
        else:
            result = predict_waste_from_file(args.image, backend, processor, model)

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
