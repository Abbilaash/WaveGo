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
        try:
            with wave.open(wav_path, 'rb') as wf:
                framerate = wf.getframerate()
                
                # Vosk expects 16-bit mono.
                rec = KaldiRecognizer(self.model, framerate)
                rec.SetWords(True)
                
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    rec.AcceptWaveform(data)
                
                result = json.loads(rec.FinalResult())
                text = result.get("text", "")
                print(text)
                
                return text.strip()
        except Exception as e:
            print(f"Error in transcription: {e}")
            return ""
        finally:
            if os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception as e:
                    print(f"Failed to remove wav file {wav_path}: {e}")
