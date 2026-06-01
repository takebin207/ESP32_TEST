import os
import shutil
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

from huggingface_hub import hf_hub_download

from waste_sorter import LOCAL_MODEL_DIR, REMOTE_MODEL_NAME, local_model_ready


FILES = (
    "config.json",
    "preprocessor_config.json",
    "model.safetensors",
)


def copy_file(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    if local_model_ready():
        print(f"Model da co san o: {LOCAL_MODEL_DIR}")
        return 0

    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

    print(f"Dang tai model {REMOTE_MODEL_NAME} ve local:")
    print(f"  {LOCAL_MODEL_DIR}")

    for index, filename in enumerate(FILES, start=1):
        dst = os.path.join(LOCAL_MODEL_DIR, filename)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"[{index}/{len(FILES)}] Da co {filename}")
            continue

        print(f"[{index}/{len(FILES)}] Dang tai {filename} ...")
        try:
            cached_path = hf_hub_download(
                repo_id=REMOTE_MODEL_NAME,
                filename=filename,
            )
            copy_file(cached_path, dst)
            size_mb = os.path.getsize(dst) / (1024 * 1024)
            print(f"    OK: {filename} ({size_mb:.1f} MB)")
        except Exception as exc:
            print(f"Loi khi tai {filename}: {exc}", file=sys.stderr)
            print("Neu mang cham, hay chay lai lenh nay. File da tai se duoc dung lai.", file=sys.stderr)
            return 1

    if not local_model_ready():
        print("Tai xong nhung model local van chua day du.", file=sys.stderr)
        return 1

    print("Tai model thanh cong. Tu gio web/CLI se load model tu local.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
