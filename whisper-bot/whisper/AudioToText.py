import sys
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Load the local persistent Vosk package
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VOSK_PKG_DIR = os.path.join(THIS_DIR, "vosk_package")
if os.path.exists(VOSK_PKG_DIR) and VOSK_PKG_DIR not in sys.path:
	sys.path.insert(0, VOSK_PKG_DIR)

from vosk import Model, KaldiRecognizer
import wave
import json
import struct
import numpy as np

VOSK_SAMPLE_RATE = 16000

class AudioToTextTranscriber:
    def __init__(self, model_dir):
        """
        Initializes the Vosk transcriber.
        """
        self.model_dir = model_dir
        vosk_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vosk-model-small-en-us-0.15")
        
        if not os.path.exists(vosk_model_path):
            raise FileNotFoundError(
                f"Vosk model not found at: {vosk_model_path}. "
                "Please download and extract it there."
            )
            
        self.model = Model(vosk_model_path)

    @staticmethod
    def _read_wav_as_float(wf):
        """Read all frames from a wave.Wave_read and return as float32 numpy array."""
        n_frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        n_channels = wf.getnchannels()
        raw = wf.readframes(n_frames)

        if sampwidth == 1:
            samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        elif sampwidth == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        # Convert stereo (or multi-channel) to mono by averaging channels
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        return samples

    @staticmethod
    def _resample(samples, orig_rate, target_rate):
        """Resample a float32 numpy array from orig_rate to target_rate."""
        if orig_rate == target_rate:
            return samples
        duration = len(samples) / orig_rate
        target_len = int(round(duration * target_rate))
        resampled = np.interp(
            np.linspace(0, len(samples) - 1, target_len),
            np.arange(len(samples)),
            samples
        )
        return resampled.astype(np.float32)

    @staticmethod
    def _float_to_pcm16(samples):
        """Convert float32 [-1, 1] samples to int16 PCM bytes."""
        clipped = np.clip(samples, -1.0, 1.0)
        pcm = (clipped * 32767).astype(np.int16)
        return pcm.tobytes()

    def transcribe(self, wav_path):
        """
        Transcribes a WAV file. Automatically converts to mono 16kHz PCM
        regardless of what the browser sent.
        """
        with wave.open(wav_path, 'rb') as wf:
            orig_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            print(
                f"[AudioToText] WAV info: channels={n_channels}, "
                f"sampwidth={sampwidth}, rate={orig_rate}, frames={n_frames}, "
                f"duration={n_frames/orig_rate:.2f}s"
            )
            samples = self._read_wav_as_float(wf)

        print(f"[AudioToText] Samples read: {len(samples)}, min={samples.min():.4f}, max={samples.max():.4f}, rms={float(np.sqrt(np.mean(samples**2))):.4f}")

        # Resample to 16kHz
        samples = self._resample(samples, orig_rate, VOSK_SAMPLE_RATE)
        print(f"[AudioToText] After resample: {len(samples)} samples at {VOSK_SAMPLE_RATE}Hz")

        # Convert back to int16 PCM bytes for Vosk
        pcm_bytes = self._float_to_pcm16(samples)

        # Run Vosk recognizer on the full PCM buffer
        rec = KaldiRecognizer(self.model, VOSK_SAMPLE_RATE)
        rec.SetWords(True)

        full_text = []
        chunk_size = 8000 * 2  # 8000 int16 samples = 0.5s chunks at 16kHz
        n_chunks = 0
        n_accepted = 0

        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i : i + chunk_size]
            n_chunks += 1
            if rec.AcceptWaveform(chunk):
                n_accepted += 1
                res = json.loads(rec.Result())
                text_part = res.get("text", "")
                print(f"[AudioToText] Mid-result: '{text_part}'")
                if text_part:
                    full_text.append(text_part)

        final_json = json.loads(rec.FinalResult())
        text_part = final_json.get("text", "")
        print(f"[AudioToText] Chunks={n_chunks}, accepted={n_accepted}, FinalResult='{text_part}'")
        if text_part:
            full_text.append(text_part)

        result = " ".join(full_text).strip()
        print(f"[AudioToText] Final transcription: '{result}'")
        return result
