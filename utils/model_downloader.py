import gdown
import os

def download_all_models():
    os.makedirs("Main_py", exist_ok=True)

    files = {
        "cnn_model.keras": "14Bj_If0o4HT1txDs7rzlW2u8t_EXI2R3",
        "banana_vit_model.keras": "1xxMwa7GM0lRQYJbmvpYtS78uYgb28zFt",
        "scaler.pkl": "1tT0CIfYUdEEmXkyKRxLM48UxAJVf9g5Q",
        "mlp_model.pkl": "1B0Oe4WqfkfwrMTX94WyzsiPPm3mfF7ky",
        "outlier_detector.pkl": "1byA3KawqZYCkz8U-t1od0bshEE7jzRGe",
        "label_encoder.pkl": "1xbf38OzK3gss0piszLEK01ekFh3PAPIW",
        "kb_data_image.json": "1VSRg-3t1_yoSHVqmy6VHtpj2eVq52M8A",
        "kb_data_text.json": "1ZhI9je0TnUbk1qb8h5j8uZF0BR8pgVQh"
    }

    for name, file_id in files.items():
        url = f"https://drive.google.com/uc?id={file_id}"
        output = f"Main_py/{name}"
        if not os.path.exists(output):
            print(f"🔽 Downloading {name}...")
            gdown.download(url, output, quiet=False)
        else:
            print(f"✅ {name} already exists.")
