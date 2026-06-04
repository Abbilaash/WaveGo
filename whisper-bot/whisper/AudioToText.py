import os
import wave
import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor, WhisperTokenizer

class AudioToTextTranscriber:
    def __init__(self, model_dir):
        """
        Initializes the Whisper ONNX transcriber.
        Expects the ONNX models and Hugging Face configuration files to be in the model_dir.
        """
        self.model_dir = model_dir
        
        # Load local configuration and tokenizer from model_dir
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_dir)
        self.tokenizer = WhisperTokenizer.from_pretrained(model_dir)
        
        # Paths to encoder and decoder ONNX models (standard or quantized names)
        encoder_path = os.path.join(model_dir, "encoder_model.onnx")
        if not os.path.exists(encoder_path):
            encoder_path = os.path.join(model_dir, "encoder_model_quantized.onnx")
            
        decoder_path = os.path.join(model_dir, "decoder_model.onnx")
        if not os.path.exists(decoder_path):
            decoder_path = os.path.join(model_dir, "decoder_model_quantized.onnx")
        
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"ONNX Encoder model not found (tried encoder_model.onnx and encoder_model_quantized.onnx) in: {model_dir}")
        if not os.path.exists(decoder_path):
            raise FileNotFoundError(f"ONNX Decoder model not found (tried decoder_model.onnx and decoder_model_quantized.onnx) in: {model_dir}")
            
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
        self.start_token = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        self.en_token = self.tokenizer.convert_tokens_to_ids("<|en|>")
        self.transcribe_token = self.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        self.notimestamps_token = self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        self.eos_token_id = self.tokenizer.eos_token_id

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
                    
            # 2. Extract Log-Mel Spectrogram features
            # Pads/truncates the audio to a standard 30s window (3000 frames) and extracts 80 channels
            features = self.feature_extractor(audio, sampling_rate=16000, return_tensors="np")
            input_features = features.input_features.astype(np.float32)
            
            # 3. Run Encoder Session
            encoder_outputs = self.encoder_session.run(None, {self.encoder_input_name: input_features})
            last_hidden_state = encoder_outputs[0]
            
            # 4. Autoregressive Decoder Loop (Greedy Search)
            # Sequence format: [<|startoftranscript|>, <|en|>, <|transcribe|>, <|notimestamps|>]
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
                
                # Predict the next token (greedy, argmax of the last step's logits)
                next_token_logits = logits[0, -1, :]
                next_token_id = int(np.argmax(next_token_logits))
                
                if next_token_id == self.eos_token_id:
                    break
                    
                generated_tokens.append(next_token_id)
                decoder_input_ids = np.append(decoder_input_ids, [[next_token_id]], axis=1)
                
            # 5. Decode generated token IDs to text string
            transcription = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return transcription.strip()
        finally:
            if os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception as e:
                    print(f"Failed to remove wav file {wav_path}: {e}")
