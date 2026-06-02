import os
import shutil
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

from huggingface_hub import hf_hub_download

from waste_sorter import KERAS_LOCAL_MODEL_DIR, KERAS_MODEL_FILE, KERAS_REMOTE_MODEL_NAME, keras_model_ready


def main() -> int:
    if keras_model_ready():
        print(f"Keras model da co san o: {KERAS_MODEL_FILE}")
        return 0

    os.makedirs(KERAS_LOCAL_MODEL_DIR, exist_ok=True)

    print(f"Dang tai Keras model {KERAS_REMOTE_MODEL_NAME} ve local:")
    print(f"  {KERAS_MODEL_FILE}")

    try:
        cached_path = hf_hub_download(
            repo_id=KERAS_REMOTE_MODEL_NAME,
            filename="waste_classification_model.h5",
        )
        shutil.copy2(cached_path, KERAS_MODEL_FILE)
    except Exception as exc:
        print(f"Loi khi tai Keras model: {exc}", file=sys.stderr)
        return 1

    size_mb = os.path.getsize(KERAS_MODEL_FILE) / (1024 * 1024)
    print(f"Tai Keras model thanh cong ({size_mb:.1f} MB).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
