@echo off
REM ============================================================
REM  LectureLens Setup Script — Windows
REM  Run this once from the lectrelens folder:
REM    setup.bat
REM ============================================================

echo === LectureLens Setup (Windows) ===

REM 1. Create project structure
echo [+] Creating project directories...
if not exist templates   mkdir templates
if not exist utils       mkdir utils
if not exist models      mkdir models
if not exist utils\__init__.py    type nul > utils\__init__.py
if not exist models\__init__.py   type nul > models\__init__.py
if not exist __init__.py          type nul > __init__.py

REM 2. Upgrade pip silently
echo [+] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM 3. Install Python packages
echo [+] Installing Python packages (this may take a few minutes)...
pip install -r requirements.txt

REM 4. Pre-download Whisper base model
echo [+] Pre-downloading Whisper base model (~140 MB)...
python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8'); print('Whisper base ready.')"

REM 5. Pre-download DistilBART summarizer
echo [+] Pre-downloading DistilBART summarizer (~600 MB)...
python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; m='sshleifer/distilbart-cnn-12-6'; AutoTokenizer.from_pretrained(m); AutoModelForSeq2SeqLM.from_pretrained(m); print('DistilBART ready.')"

echo.
echo === Setup complete! ===
echo Run:   python app.py
echo Open:  http://localhost:5000
echo.
echo To use a better Whisper model:
echo   set WHISPER_MODEL=small ^&^& python app.py
echo   set WHISPER_MODEL=large-v3-turbo ^&^& python app.py
pause
