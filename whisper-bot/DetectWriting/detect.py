import cv2

# Initialize PaddleOCR lazily only when needed
ocr_instance = None

def detect(frame):
    global ocr_instance
    if frame is None:
        return []
        
    try:
        from paddleocr import PaddleOCR
    except ImportError as e:
        print("[DetectWriting] paddleocr is not installed.")
        raise e
        
    if ocr_instance is None:
        print("[DetectWriting] Initializing PaddleOCR instance...")
        ocr_instance = PaddleOCR(use_angle_cls=True, lang="en")
        
    results = ocr_instance.ocr(frame)
    extracted_text = []
    
    if results and results[0]:
        for line in results[0]:
            text = line[1][0]
            confidence = float(line[1][1])
            print(f"[DetectWriting] Found text: '{text}' (conf: {confidence:.2f})")
            extracted_text.append((text, confidence))
            
    return extracted_text
