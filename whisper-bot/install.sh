#!/bin/bash
# ==============================================================================
# WaveGo Whisper Bot Automated Installer Script
# Targets: Debian/Raspberry Pi OS (Linux)
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status

echo "======================================================================"
echo "🧠 Starting WaveGo Whisper Bot installation..."
echo "======================================================================"

# 1. Update system packages and install prerequisites
echo "📦 Updating apt packages and installing system dependencies..."
sudo apt-get update -y
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-opencv \
    libgl1 \
    libglib2.0-0 \
    espeak \
    bluetooth \
    bluez \
    bluez-tools \
    gfortran \
    libatlas-base-dev \
    v4l-utils \
    wget \
    curl \
    unzip \
    libcap-dev \
    build-essential \
    python3-dev

# 2. Grant serial port permissions for ESP32 locomotion control
echo "🔌 Configuring hardware serial interface (/dev/ttyS0)..."
sudo usermod -a -G dialout $USER

# Disable the serial console logins to release /dev/ttyS0 for Python
echo "🚫 Disabling login terminal shell on ttyS0..."
sudo systemctl stop serial-getty@ttyS0.service || true
sudo systemctl disable serial-getty@ttyS0.service || true

# 3. Install Python Dependencies
echo "🐍 Installing Python pip requirements..."
pip3 install -r requirements.txt --break-system-packages || pip3 install -r requirements.txt

# 4. Fetch speech models and public ONNX networks
echo "📥 Downloading neural network model assets from Hugging Face..."

HF_BASE="https://huggingface.co/Abbilaash/Whisper-Bot/resolve/main"

# Vosk Speech Recognition Model (external dependency)
if [ ! -d "whisper/vosk-model-small-en-us-0.15" ]; then
    echo "  -> Downloading Vosk speech model..."
    mkdir -p whisper
    wget -q --show-progress -O whisper/vosk-model-small-en-us-0.15.zip \
        https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
    
    echo "  -> Extracting Vosk speech model..."
    unzip -q whisper/vosk-model-small-en-us-0.15.zip -d whisper/
    rm whisper/vosk-model-small-en-us-0.15.zip
else
    echo "  -> Vosk speech model already present."
fi

# MobileNet Vision Embeddings model (root file)
if [ ! -f "mobilenetv3_embedding.onnx" ]; then
    echo "  -> Downloading MobileNet embeddings model..."
    wget -q --show-progress -O mobilenetv3_embedding.onnx \
        "$HF_BASE/mobilenetv3_embedding.onnx"
else
    echo "  -> MobileNet embeddings model already present."
fi

# LeNet-5 MNIST digit classification model
if [ ! -f "lenet5/mnist-12.onnx" ]; then
    echo "  -> Downloading LeNet-5 MNIST model..."
    mkdir -p lenet5
    wget -q --show-progress -O lenet5/mnist-12.onnx \
        "$HF_BASE/lenet5/mnist-12.onnx"
else
    echo "  -> LeNet-5 MNIST model already present."
fi

# MiniLM Text Embeddings models and configs
echo "  -> Downloading MiniLM text embeddings model and configs..."
mkdir -p MiniLM
MINILM_FILES=("all-MiniLM.onnx" "config.json" "tokenizer.json" "vocab.txt" "intents.yaml" "intent_db.pkl")
for file in "${MINILM_FILES[@]}"; do
    if [ ! -f "MiniLM/$file" ]; then
        echo "     Downloading $file..."
        wget -q --show-progress -O "MiniLM/$file" \
            "$HF_BASE/MiniLM/$file"
    else
        echo "     $file already present."
    fi
done

# Knowledge (Gemma3) models and external weights
echo "  -> Downloading Gemma3 SLM models and data..."
mkdir -p knowledge
GEMMA_FILES=("gemma3.onnx" "model.onnx_data" "config.json" "generation_config.json" "tokenizer.json" "tokenizer_config.json")
for file in "${GEMMA_FILES[@]}"; do
    if [ ! -f "knowledge/$file" ]; then
        echo "     Downloading $file..."
        wget -q --show-progress -O "knowledge/$file" \
            "$HF_BASE/knowledge/$file"
    else
        echo "     $file already present."
    fi
done

# FollowObject vision model
if [ ! -f "FollowObject/best4.onnx" ]; then
    echo "  -> Downloading FollowObject YOLO network..."
    mkdir -p FollowObject
    wget -q --show-progress -O FollowObject/best4.onnx \
        "$HF_BASE/FollowObject/best4.onnx"
else
    echo "  -> FollowObject YOLO network already present."
fi

echo "======================================================================"
echo "🎉 Installation completed successfully!"
echo "======================================================================"
echo "⚠️ IMPORTANT NOTES FOR DEPLOYMENT:"
echo "1. Group Permissions: Please log out and log back in, or run 'su - $USER'"
echo "   to apply the new dialout group permissions."
echo "2. Running the server: Start the bot server using python3:"
echo "   python3 webServer.py"
echo "======================================================================"
