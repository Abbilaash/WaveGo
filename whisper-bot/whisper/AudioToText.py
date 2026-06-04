import os
import wave
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

class AudioToTextTranscriber:
    def __init__(self, model_dir):
        """
        Initializes the Whisper ONNX transcriber.
        Expects the ONNX models, tokenizer.json, and mel_filters.npz to be in the model_dir.
        """
        self.model_dir = model_dir
        
        # Load local Tokenizer (using lightweight tokenizers package)
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"tokenizer.json not found at: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Load precomputed Mel filters matrix from OpenAI's mel_filters.npz
        mel_filters_path = os.path.join(model_dir, "mel_filters.npz")
        if not os.path.exists(mel_filters_path):
            raise FileNotFoundError(f"mel_filters.npz not found at: {mel_filters_path}. Please download it from: https://github.com/openai/whisper/raw/main/whisper/assets/mel_filters.npz")
            
        with np.load(mel_filters_path) as f:
            # Whisper base uses 80 Mel bands
            self.mel_filters = f["mel_80"]
            
        # Paths to encoder and decoder ONNX models (supporting both default and quantized names)
        encoder_path = os.path.join(model_dir, "encoder_model.onnx")
        if not os.path.exists(encoder_path):
            encoder_path = os.path.join(model_dir, "encoder_model_quantized.onnx")
            
        decoder_path = os.path.join(model_dir, "decoder_model.onnx")
        if not os.path.exists(decoder_path):
            decoder_path = os.path.join(model_dir, "decoder_model_quantized.onnx")
            
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"ONNX Encoder model not found in: {model_dir}")
        if not os.path.exists(decoder_path):
            raise FileNotFoundError(f"ONNX Decoder model not found in: {model_dir}")
            
        # Initialize ONNX inference sessions (using CPU execution provider)
        self.encoder_session = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
        self.decoder_session = ort.InferenceSession(decoder_path, providers=["CPUExecutionProvider"])
        
        # Determine session input names dynamically
        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        
        self.decoder_input_ids_name = "input_ids"
        self.decoder_encoder_states_name = "encoder_hidden_states"
        for inp in self.decoder_session.get_inputs():
            name = inp.name
            if "ids" in name or ("input" in name and "hidden" not in name):
                self.decoder_input_ids_name = name
            elif "hidden" in name or "encoder" in name or "states" in name:
                self.decoder_encoder_states_name = name
                
        # Retrieve Whisper special tokens
        self.start_token = self.tokenizer.token_to_id("<|startoftranscript|>")
        self.en_token = self.tokenizer.token_to_id("<|en|>")
        self.transcribe_token = self.tokenizer.token_to_id("<|transcribe|>")
        self.notimestamps_token = self.tokenizer.token_to_id("<|notimestamps|>")
        self.eos_token_id = self.tokenizer.token_to_id("<|endoftext|>")

    def transcribe(self, wav_path):
        """
        Transcribes a WAV file.
        Expects 16kHz mono PCM audio format.
        """
        try:
            # 1. Read WAV audio file
            with wave.open(wav_path, 'rb') as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                data = wf.readframes(n_frames)
                
                # Convert binary audio buffer to float32 NumPy array normalized to [-1.0, 1.0]
                if sampwidth == 2:
                    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                elif sampwidth == 1:
                    audio = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                elif sampwidth == 4:
                    audio = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    raise ValueError(f"Unsupported sample width: {sampwidth}")
                    
                # Convert stereo to mono if needed
                if n_channels > 1:
                    audio = audio.reshape(-1, n_channels).mean(axis=1)
                    
            # 2. Replicate Whisper Log-Mel Spectrogram extraction using pure NumPy
            # Pad or trim audio to exactly 30s (480,000 samples at 16kHz)
            n_samples = 480000
            if len(audio) > n_samples:
                audio = audio[:n_samples]
            else:
                audio = np.pad(audio, (0, n_samples - len(audio)))
                
            # Pad audio with reflection (200 samples at the beginning and end) for window centering
            audio_padded = np.pad(audio, 200, mode='reflect')
            
            # Compute STFT (Window size 400, Hop 160, 3000 frames)
            window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(400) / 400))
            stft = []
            for i in range(3000):
                start = i * 160
                frame = audio_padded[start : start + 400] * window
                fft = np.fft.rfft(frame)
                stft.append(np.abs(fft) ** 2)
                
            stft_matrix = np.array(stft).T # Shape: (201, 3000)
            
            # Apply Mel filterbank matrix (80, 201)
            mel_spec = np.dot(self.mel_filters, stft_matrix)
            
            # Log Scaling
            log_mel_spec = np.log10(np.maximum(mel_spec, 1e-10))
            
            # Dynamic Range Compression
            log_mel_spec = np.maximum(log_mel_spec, log_mel_spec.max() - 8.0)
            
            # Scale to [-1, 1]
            log_mel_spec = (log_mel_spec + 4.0) / 4.0
            
            # Add batch dimension: Shape becomes (1, 80, 3000)
            input_features = np.expand_dims(log_mel_spec, axis=0).astype(np.float32)
            
            # 3. Run Encoder Session
            encoder_outputs = self.encoder_session.run(None, {self.encoder_input_name: input_features})
            last_hidden_state = encoder_outputs[0]
            
            # 4. Autoregressive Decoder Loop (Greedy Search)
            seq = [self.start_token]
            if self.en_token is not None:
                seq.append(self.en_token)
            if self.transcribe_token is not None:
                seq.append(self.transcribe_token)
            if self.notimestamps_token is not None:
                seq.append(self.notimestamps_token)
                
            decoder_input_ids = np.array([seq], dtype=np.int64)
            generated_tokens = []
            max_tokens = 100
            
            for _ in range(max_tokens):
                decoder_inputs = {
                    self.decoder_input_ids_name: decoder_input_ids,
                    self.decoder_encoder_states_name: last_hidden_state
                }
                decoder_outputs = self.decoder_session.run(None, decoder_inputs)
                logits = decoder_outputs[0]
                
                # Predict the next token (greedy)
                next_token_logits = logits[0, -1, :]
                next_token_id = int(np.argmax(next_token_logits))
                
                if next_token_id == self.eos_token_id:
                    break
                    
                generated_tokens.append(next_token_id)
                decoder_input_ids = np.append(decoder_input_ids, [[next_token_id]], axis=1)
                
            # 5. Decode generated token IDs to text string using tokenizers
            transcription = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return transcription.strip()
        finally:
            if os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception as e:
                    print(f"Failed to remove wav file {wav_path}: {e}")
