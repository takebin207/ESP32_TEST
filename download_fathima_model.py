import os
import sys

from huggingface_hub import hf_hub_download

from waste_sorter import FATHIMA_LOCAL_MODEL_DIR, FATHIMA_MODEL_FILE, FATHIMA_REMOTE_MODEL_NAME


def main() -> int:
    os.makedirs(FATHIMA_LOCAL_MODEL_DIR, exist_ok=True)

    print(f"Dang tai model {FATHIMA_REMOTE_MODEL_NAME} ve local:")
    print(f"  {FATHIMA_LOCAL_MODEL_DIR}")

    if os.path.exists(FATHIMA_MODEL_FILE) and os.path.getsize(FATHIMA_MODEL_FILE) > 0:
        print(f"Da co {FATHIMA_MODEL_FILE}")
        return 0

    try:
        downloaded_path = hf_hub_download(
            repo_id=FATHIMA_REMOTE_MODEL_NAME,
            filename="waste_classifier_model.h5",
            repo_type="space",
            local_dir=FATHIMA_LOCAL_MODEL_DIR,
        )
    except Exception as exc:
        print(f"Loi khi tai model Fathima: {exc}", file=sys.stderr)
        print("Neu mang cham, hay chay lai lenh nay. File da tai se duoc dung lai.")
        return 1

    print(f"Da tai xong: {downloaded_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
