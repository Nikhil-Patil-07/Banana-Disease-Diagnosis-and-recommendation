{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "48282e52-4969-4d2f-852d-d18c93d147d8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import gdown\n",
    "\n",
    "def download_file_from_drive(file_id, save_path):\n",
    "    if not os.path.exists(save_path):\n",
    "        print(f\"📥 Downloading {save_path} ...\")\n",
    "        url = f\"https://drive.google.com/uc?id={file_id}\"\n",
    "        gdown.download(url, save_path, quiet=False)\n",
    "\n",
    "def download_all_models():\n",
    "    os.makedirs(\"Main_py\", exist_ok=True)\n",
    "\n",
    "    files_to_download = [\n",
    "        # (file_id, destination_path)\n",
    "        (\"14Bj_If0o4HT1txDs7rzlW2u8t_EXI2R3\", \"Main_py/cnn_model.keras\"),           # CNN model\n",
    "        (\"1xxMwa7GM0lRQYJbmvpYtS78uYgb28zFt\", \"Main_py/banana_vit_model.keras\"),   # ViT model\n",
    "        (\"1tT0CIfYUdEEmXkyKRxLM48UxAJVf9g5Q\", \"Main_py/scaler.pkl\"),               # Scaler\n",
    "        (\"1B0Oe4WqfkfwrMTX94WyzsiPPm3mfF7ky\", \"Main_py/mlp_model.pkl\"),            # MLP model\n",
    "        (\"1byA3KawqZYCkz8U-t1od0bshEE7jzRGe\", \"Main_py/outlier_detector.pkl\"),     # Outlier model\n",
    "        (\"1xbf38OzK3gss0piszLEK01ekFh3PAPIW\", \"Main_py/label_encoder.pkl\"),        # Label encoder\n",
    "        (\"1VSRg-3t1_yoSHVqmy6VHtpj2eVq52M8A\", \"Main_py/kb_data_image.json\"),       # KB for image\n",
    "        (\"1ZhI9je0TnUbk1qb8h5j8uZF0BR8pgVQh\", \"Main_py/kb_data_text.json\"),        # KB for text\n",
    "    ]\n",
    "\n",
    "    for file_id, save_path in files_to_download:\n",
    "        download_file_from_drive(file_id, save_path)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.18"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
