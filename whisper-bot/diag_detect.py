#!/usr/bin/env python3
"""
Standalone diagnostic: run this directly on the Pi via SSH.
    cd /home/rpi/WaveGo/whisper-bot
    python3 diag_detect.py

It captures one frame from Picamera2 (the same way camera_opencv.py does),
prints every detail, and runs detection with both color modes.
"""
import os, sys, time
import cv2
import numpy as np

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, THIS_DIR)

from FollowObject.detect import detect

def main():
    from picamera2 import Picamera2

    # ---- same configuration as camera_opencv.py ----
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
    )
    picam2.start()
    time.sleep(1)  # let auto-exposure settle

    frame = picam2.capture_array()
    picam2.stop()

    # ---- frame diagnostics ----
    print("=" * 60)
    print("FRAME DIAGNOSTICS")
    print("=" * 60)
    print(f"  shape        : {frame.shape}")
    print(f"  dtype        : {frame.dtype}")
    print(f"  contiguous   : {frame.flags['C_CONTIGUOUS']}")
    print(f"  writeable    : {frame.flags['WRITEABLE']}")
    print(f"  strides      : {frame.strides}")
    print(f"  pixel [0,0]  : {frame[0, 0]}")       # first pixel
    print(f"  pixel [240,320]: {frame[240, 320]}")  # center pixel
    print(f"  min/max      : {frame.min()} / {frame.max()}")

    # ---- save raw frame for visual inspection ----
    # cv2.imwrite treats input as BGR; if frame is actually RGB the saved
    # JPEG will have R-B swapped. Compare visually to reality.
    cv2.imwrite(os.path.join(THIS_DIR, "diag_frame_as_bgr.jpg"), frame)
    # also save with explicit RGB→BGR swap
    cv2.imwrite(os.path.join(THIS_DIR, "diag_frame_as_rgb.jpg"),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print("\nSaved diag_frame_as_bgr.jpg  (correct colors if frame is BGR)")
    print("Saved diag_frame_as_rgb.jpg  (correct colors if frame is RGB)")
    print(">> Open both on the Pi and see which one has CORRECT colors.\n")

    # ---- run detection with BOTH color modes ----
    model_path = os.path.join(THIS_DIR, "FollowObject", "best.onnx")

    # make a guaranteed contiguous copy
    frame_c = np.ascontiguousarray(frame)

    print("=" * 60)
    print("TEST 1: input_is_rgb=False  (treats frame as BGR, converts to RGB)")
    print("=" * 60)
    r1 = detect(frame_c, model_path=model_path, input_is_rgb=False)
    print(f"  success    : {r1['success']}")
    print(f"  detections : {len(r1.get('detections', []))}")
    for d in r1.get("detections", []):
        print(f"    -> {d['class_name']}  conf={d['conf']:.4f}  "
              f"box=[{d['x1']:.0f},{d['y1']:.0f},{d['x2']:.0f},{d['y2']:.0f}]")

    print()
    print("=" * 60)
    print("TEST 2: input_is_rgb=True  (treats frame as RGB, no conversion)")
    print("=" * 60)
    r2 = detect(frame_c, model_path=model_path, input_is_rgb=True)
    print(f"  success    : {r2['success']}")
    print(f"  detections : {len(r2.get('detections', []))}")
    for d in r2.get("detections", []):
        print(f"    -> {d['class_name']}  conf={d['conf']:.4f}  "
              f"box=[{d['x1']:.0f},{d['y1']:.0f},{d['x2']:.0f},{d['y2']:.0f}]")

    print()
    print("=" * 60)
    print("TEST 3: Manually swapped channels (frame[:,:,::-1])")
    print("=" * 60)
    frame_swapped = np.ascontiguousarray(frame[:, :, ::-1])
    r3 = detect(frame_swapped, model_path=model_path, input_is_rgb=False)
    print(f"  success    : {r3['success']}")
    print(f"  detections : {len(r3.get('detections', []))}")
    for d in r3.get("detections", []):
        print(f"    -> {d['class_name']}  conf={d['conf']:.4f}  "
              f"box=[{d['x1']:.0f},{d['y1']:.0f},{d['x2']:.0f},{d['y2']:.0f}]")

    print()
    print("=" * 60)
    print("TEST 4: cv2.VideoCapture(0) — standard webcam path like test.py")
    print("=" * 60)
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        time.sleep(1)
        ret, vcap_frame = cap.read()
        cap.release()
        if ret and vcap_frame is not None:
            print(f"  vcap shape      : {vcap_frame.shape}")
            print(f"  vcap contiguous : {vcap_frame.flags['C_CONTIGUOUS']}")
            print(f"  vcap pixel[0,0] : {vcap_frame[0, 0]}")
            r4 = detect(vcap_frame, model_path=model_path, input_is_rgb=False)
            print(f"  success    : {r4['success']}")
            print(f"  detections : {len(r4.get('detections', []))}")
            for d in r4.get("detections", []):
                print(f"    -> {d['class_name']}  conf={d['conf']:.4f}  "
                      f"box=[{d['x1']:.0f},{d['y1']:.0f},{d['x2']:.0f},{d['y2']:.0f}]")
        else:
            print("  Could not read frame from VideoCapture")
    else:
        print("  cv2.VideoCapture(0) could not open camera")

    print()
    print("=" * 60)
    print("TEST 5: ONNX model file integrity check")
    print("=" * 60)
    fsize = os.path.getsize(model_path)
    print(f"  model path : {model_path}")
    print(f"  file size  : {fsize} bytes")
    
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print(f"  input name : {inp.name}")
    print(f"  input shape: {inp.shape}")
    print(f"  input type : {inp.type}")
    print(f"  output name: {out.name}")
    print(f"  output shape: {out.shape}")
    print(f"  ort version: {ort.__version__}")
    print(f"  numpy ver  : {np.__version__}")
    print(f"  cv2 ver    : {cv2.__version__}")

    print("\n>> Share the full output above so we can diagnose the issue.")


if __name__ == "__main__":
    main()
