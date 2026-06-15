#!/usr/bin/env python3
"""
Diagnostic step 2: check model integrity and save frame for cross-platform testing.
    cd /home/rpi/WaveGo/whisper-bot
    python3 diag_detect2.py
"""
import os, sys, hashlib
import cv2
import numpy as np

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, THIS_DIR)

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    model_path = os.path.join(THIS_DIR, "FollowObject", "best.onnx")

    # ---- Model hash ----
    pi_hash = md5(model_path)
    laptop_hash = "eaa0174d082628f9201f14e75de6c567"
    print("=" * 60)
    print("MODEL FILE INTEGRITY")
    print("=" * 60)
    print(f"  Pi MD5     : {pi_hash}")
    print(f"  Laptop MD5 : {laptop_hash}")
    if pi_hash == laptop_hash:
        print("  >> MATCH — model file is intact")
    else:
        print("  >> MISMATCH — model file is CORRUPTED! Re-copy best.onnx to the Pi.")
        return

    # ---- Capture frame and save as .npy for laptop testing ----
    from picamera2 import Picamera2
    import time

    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
    )
    picam2.start()
    time.sleep(2)  # let auto-exposure settle longer

    frame = picam2.capture_array()
    picam2.stop()

    npy_path = os.path.join(THIS_DIR, "pi_frame.npy")
    np.save(npy_path, frame)
    print(f"\nSaved raw frame to: {npy_path}")
    print(f"  shape: {frame.shape}, dtype: {frame.dtype}")
    print(">> Copy pi_frame.npy to your laptop and run diag_laptop.py on it")

    # ---- Also test with a pure numpy synthetic tensor ----
    # This tests if onnxruntime itself works correctly on ARM
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]

    # Create a random input tensor
    random_input = np.random.rand(1, 3, 416, 416).astype(np.float32)
    outputs = sess.run(None, {inp.name: random_input})
    out = outputs[0][0]  # [6, 3549]
    out_t = np.transpose(out)  # [3549, 6]
    scores = out_t[:, 4:]
    print(f"\n  Random input max score : {scores.max():.6f}")
    print(f"  Random input min score : {scores.min():.6f}")
    print(f"  Random input mean score: {scores.mean():.6f}")
    print(f"  Output shape           : {outputs[0].shape}")

    # Now test with an all-zeros tensor (should give near-zero scores)
    zero_input = np.zeros((1, 3, 416, 416), dtype=np.float32)
    outputs_z = sess.run(None, {inp.name: zero_input})
    out_z = np.transpose(outputs_z[0][0])
    scores_z = out_z[:, 4:]
    print(f"\n  Zeros input max score  : {scores_z.max():.6f}")
    print(f"  Zeros input mean score : {scores_z.mean():.6f}")

    # Now test with the actual frame, preprocessed manually step by step
    print("\n" + "=" * 60)
    print("MANUAL PREPROCESSING TEST")
    print("=" * 60)
    resized = cv2.resize(frame, (416, 416))
    print(f"  resized shape: {resized.shape}, dtype: {resized.dtype}")

    # Test BGR->RGB path
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, 0)
    blob = np.ascontiguousarray(blob, dtype=np.float32)
    print(f"  blob shape: {blob.shape}, contiguous: {blob.flags['C_CONTIGUOUS']}")
    print(f"  blob min/max: {blob.min():.4f} / {blob.max():.4f}")
    print(f"  blob channel means: R={blob[0,0].mean():.4f} G={blob[0,1].mean():.4f} B={blob[0,2].mean():.4f}")

    out_real = sess.run(None, {inp.name: blob})
    out_real_t = np.transpose(out_real[0][0])
    scores_real = out_real_t[:, 4:]
    print(f"  BGR->RGB max score     : {scores_real.max():.6f}")

    # Test no-conversion path (pass as-is)
    blob2 = resized.astype(np.float32) / 255.0
    blob2 = np.transpose(blob2, (2, 0, 1))
    blob2 = np.expand_dims(blob2, 0)
    blob2 = np.ascontiguousarray(blob2, dtype=np.float32)
    print(f"  blob2 channel means: ch0={blob2[0,0].mean():.4f} ch1={blob2[0,1].mean():.4f} ch2={blob2[0,2].mean():.4f}")

    out_real2 = sess.run(None, {inp.name: blob2})
    out_real2_t = np.transpose(out_real2[0][0])
    scores_real2 = out_real2_t[:, 4:]
    print(f"  As-is max score        : {scores_real2.max():.6f}")

    print("\n>> If random input gives similar scores as real frame, onnxruntime may have an ARM issue.")
    print(">> If random input gives much HIGHER scores, the model IS working but doesn't see a ball.")

if __name__ == "__main__":
    main()
