import os
import wave
import json
import audioop
from vosk import Model, KaldiRecognizer

VOSK_SAMPLE_RATE = 16000

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
        Transcribes a WAV file. Automatically resamples to 16kHz if needed.
        The browser typically records at 44100 or 48000 Hz. Vosk only works at 16000 Hz.
        """
        full_text = []
        with wave.open(wav_path, 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            
            print(f"[AudioToText] WAV info: channels={n_channels}, sampwidth={sampwidth}, rate={framerate}")

            rec = KaldiRecognizer(self.model, VOSK_SAMPLE_RATE)
            rec.SetWords(True)

            resample_state = None

            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break

                # Convert stereo to mono by averaging channels
                if n_channels > 1:
                    data = audioop.tomono(data, sampwidth, 0.5, 0.5)

                # Resample to 16000 Hz if the browser sent a different rate
                if framerate != VOSK_SAMPLE_RATE:
                    data, resample_state = audioop.ratecv(
                        data, sampwidth, 1, framerate, VOSK_SAMPLE_RATE, resample_state
                    )

                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    text_part = res.get("text", "")
                    if text_part:
                        full_text.append(text_part)

            res = json.loads(rec.FinalResult())
            text_part = res.get("text", "")
            if text_part:
                full_text.append(text_part)

        result = " ".join(full_text).strip()
        print(f"[AudioToText] Transcribed: '{result}'")
        return result
