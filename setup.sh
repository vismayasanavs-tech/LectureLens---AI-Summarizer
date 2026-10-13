#!/usr/bin/env bash
# One-shot setup script for LectureLens
set -e
echo "=== LectureLens Setup ==="

# 0. Create project structure
echo "[+] Creating project directories..."
mkdir -p templates utils models
touch utils/__init__.py models/__init__.py __init__.py

# 1. ffmpeg
if ! command -v ffmpeg &>/dev/null; then
  echo "[+] Installing ffmpeg..."
  sudo apt-get update -qq && sudo apt-get install -y ffmpeg
else
  echo "[✓] ffmpeg already installed."
fi

# 2. Python deps
echo "[+] Installing Python packages..."
pip install -r requirements.txt

# 3. Pre-download Whisper 'base' model (140 MB, best CPU default)
echo "[+] Pre-downloading Whisper 'base' model (~140 MB)..."
python -c "
from faster_whisper import WhisperModel
WhisperModel('base', device='cpu', compute_type='int8')
print('Whisper base ready.')
"

# 4. Pre-download DistilBART summarizer (matches pipeline.py default, ~600 MB)
echo "[+] Pre-downloading DistilBART summarizer (~600 MB)..."
python -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
m = 'sshleifer/distilbart-cnn-12-6'
AutoTokenizer.from_pretrained(m)
AutoModelForSeq2SeqLM.from_pretrained(m)
print('DistilBART ready.')
"

echo ""
echo "=== Setup complete! ==="
echo "Run:  python app.py"
echo "Open: http://localhost:5000"
echo ""
echo "To use a better (but slower) Whisper model:"
echo "  WHISPER_MODEL=small python app.py"
echo "  WHISPER_MODEL=large-v3-turbo python app.py   # needs >4 GB RAM"
