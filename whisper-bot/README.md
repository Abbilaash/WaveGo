# WaveGo Whisper Bot Backend

An offline, autonomous robotic system that combines local Speech-to-Text (STT), Semantic Intent Classification, local Retrieval-Augmented Generation (RAG) using a Small Language Model (SLM), Computer Vision (Object, Face, and Color Tracking), and Bluetooth Audio connectivity.

---

## 📋 System Requirements

### Hardware
* **Chassis & Motors**: [Waveshare WaveGo Quadruped Robot](https://www.waveshare.com/wavego.htm)
* **Controller**: Raspberry Pi 4 (or Pi 5) flashed via [Raspberry Pi Imager](https://www.raspberrypi.com/software/) or Ubuntu Server ([ubuntu.com](https://ubuntu.com/)) flashed with Rufus/balenaEtcher.
* **Locomotion Board**: WaveGo ESP32 Driver Board (connected via `/dev/ttyS0` serial interface)
* **Audio Hardware**: USB Microphone / USB Soundcard + Bluetooth speaker or headphones
* **Vision**: Raspberry Pi Camera Module or compatible USB webcam

### Software Architecture
* **Backend**: Python 3 / Flask
* **Models**:
  * **STT (Speech-to-Text)**: Vosk voice transcriber (run offline in memory)
  * **Embeddings**: Sentence-Transformers MiniLM-L6-V2 ONNX model (offline semantic matching)
  * **SLM / RAG (Small Language Model)**: Google Gemma3-270M-IT ONNX model (local text generation)
  * **Handwriting Digit Recognition**: LeNet-5 ONNX model (offline CNN character visualizer)

---

## ⚡ Quick Start & Deployment

### Step 1: Clone the Repository
Clone the codebase into the target directory on your Raspberry Pi:
```bash
git clone https://github.com/Abbilaash/WaveGo.git
cd WaveGo/whisper-bot
```

### Step 2: Run the Automated Setup
The installer script automates system packages installation and public model asset downloads:
```bash
chmod +x install.sh
./install.sh
```

> [!IMPORTANT]
> The setup script adds your user account to the `dialout` group to allow serial communications. You **must log out and log back in** (or run `su - $USER`) to activate these group permissions.

### Step 3: Run the Server
Launch the Flask backend directly:
```bash
python3 webServer.py
```
The server will start on port `5000` (e.g., `http://192.168.4.215:5000/`).

---

## 🤖 Systemd Background Automation

To configure the bot to start automatically on system boot and recover from crashes, deploy the systemd service unit:

1. **Copy the service file**:
   ```bash
   sudo cp whisper-bot.service /etc/systemd/system/
   ```
2. **Reload and Enable**:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable whisper-bot.service
   ```
3. **Control Commands**:
   * *Start service*: `sudo systemctl start whisper-bot.service`
   * *Stop service*: `sudo systemctl stop whisper-bot.service`
   * *Check status*: `sudo systemctl status whisper-bot.service`
   * *View logs*: `journalctl -u whisper-bot.service -f`

---

## 🔊 Bluetooth Audio Connection

1. **Scan and Connect**: Tap the Bluetooth dropdown inside the mobile app or web dashboard interface to discover surrounding Bluetooth audio speakers.
2. **Output Routing**: Once paired and connected, the Linux OS routes the generated offline Text-to-Speech (TTS) synthesizer (which uses `pyttsx3`/`espeak` under the hood) directly to the speaker output.

---

## 📊 REST API Specifications

The Flask backend exposes the following primary endpoints on port `5000`:

* **`POST /api/chatbot/command`**: Processes chatbot actions from raw text inputs.
* **`POST /api/chatbot/audio`**: Transcribes voice recordings (WAV) and triggers actions.
* **`POST /api/detect_digit`**: Predicts a handwritten digit and returns CNN activations/weights for the visualizer.
* **`POST /api/move/<action>`**: Drives robot locomotion (e.g. `forward`, `backward`, `left`, `right`).
* **`POST /api/camera/tilt`**: Tilts camera gimbal up, down, or centers it.
* **`POST /api/face/detect/<action>`**: Controls Open-CV face recognition pipelines (`learn`, `detect`, `follow`).
* **`POST /api/color/follow/<color>`**: Commences real-time color target tracking (`red`, `green`, `blue`).
