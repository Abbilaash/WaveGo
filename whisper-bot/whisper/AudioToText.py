import os
import wave
import json
from vosk import Model, KaldiRecognizer

class AudioToTextTranscriber:
    def __init__(self, model_dir):
        """
        Initializes the Vosk transcriber.
        """
        self.model_dir = model_dir
        vosk_model_path = os.path.join(model_dir, "vosk-model-small-en-us-0.15")
        
        if not os.path.exists(vosk_model_path):
            raise FileNotFoundError(f"Vosk model not found at: {vosk_model_path}. Please download and extract it there.")
            
        self.model = Model(vosk_model_path)

    def transcribe(self, wav_path):
        """
        Transcribes a WAV file.
        """
        full_text = []
        with wave.open(wav_path, 'rb') as wf:
            framerate = wf.getframerate()
            
            # Vosk expects 16-bit mono.
            rec = KaldiRecognizer(self.model, framerate)
            rec.SetWords(True)
            
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    text_part = res.get("text", "")
                    if text_part:
                        full_text.append(text_part)
            
            res = json.loads(rec.FinalResult())
            text_part = res.get("text", "")
            if text_part:
                full_text.append(text_part)
            
        return " ".join(full_text).strip()
