import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")

python_bin = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable
app_file = os.path.join(BASE_DIR, "app.py")

print("==================================================")
print(" 🌿 AI Crop Disease Detector")
print(f" Using Python: {python_bin}")
print(" Starting web server at: http://127.0.0.1:5000")
print(" Press Ctrl+C in terminal to stop.")
print("==================================================")

os.chdir(BASE_DIR)
subprocess.run([python_bin, app_file])
