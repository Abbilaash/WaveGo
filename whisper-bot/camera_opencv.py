import os
import cv2
from base_camera import BaseCamera
import numpy as np
import robot
import datetime
import time
import threading
import imutils
from FollowObject.detect import detect

curpath = os.path.realpath(__file__)
thisPath = os.path.dirname(curpath)

faceCascade = cv2.CascadeClassifier(thisPath + '/haarcascade_frontalface_default.xml')

upperGlobalIP = 'UPPER IP'

linePos_1 = 440
linePos_2 = 380
lineColorSet = 255
frameRender = 1
findLineError = 20

colorUpper = np.array([44, 255, 255])
colorLower = np.array([24, 100, 100])

speedMove = 100



class CVThread(threading.Thread):
    font = cv2.FONT_HERSHEY_SIMPLEX

    cameraDiagonalW = 64
    cameraDiagonalH = 48
    videoW = 640
    videoH = 480
    tor = 27
    aspd = 0.005


    def __init__(self, *args, **kwargs):
        self.CVThreading = 0
        self.CVMode = 'none'
        self.imgCV = None
        self.faces = None
        self.detected_objects = []
        self.detected_balls = []
        self.object_templates = None
        self.mobilenet_session = None
        self.mobilenet_input_name = None

        self.mov_x = None
        self.mov_y = None
        self.mov_w = None
        self.mov_h = None

        self.radius = 0
        self.box_x = None
        self.box_y = None
        self.drawing = 0

        self.findColorDetection = 0

        self.left_Pos1 = None
        self.right_Pos1 = None
        self.center_Pos1 = None

        self.left_Pos2 = None
        self.right_Pos2 = None
        self.center_Pos2 = None

        self.center = None

        self.ball_search_state = 'init'
        self.ball_search_start_time = 0.0

        super(CVThread, self).__init__(*args, **kwargs)
        self.__flag = threading.Event()
        self.__flag.clear()

        self.avg = None
        self.motionCounter = 0
        self.lastMovtionCaptured = datetime.datetime.now()
        self.frameDelta = None
        self.thresh = None
        self.cnts = None

        self.CVCommand = 'forward'


    def mode(self, invar, imgInput):
        if invar == 'ballSearch' and self.CVMode != 'ballSearch':
            self.ball_search_state = 'init'
            self.ball_search_start_time = 0.0
        self.CVMode = invar
        self.imgCV = imgInput.copy() if imgInput is not None else None
        self.resume()


    def elementDraw(self,imgInput):
        if self.CVMode == 'none':
            pass

        elif self.CVMode == 'faceDetection':
            if self.faces and len(self.faces):
                cv2.putText(imgInput, f'{len(self.faces)} Face(s) Detected', (40,60), CVThread.font, 0.5, (255,255,255), 1, cv2.LINE_AA)
                for face_info in self.faces:
                    try:
                        x, y, w, h = face_info["box"]
                        name = face_info["name"]
                        color = (74, 222, 128) if name != "Unknown" else (64, 128, 255) # Green for matched, orange for unknown
                        cv2.rectangle(imgInput, (x, y), (x + w, y + h), color, 2)
                        label = f"Hello {name}" if name != "Unknown" else "Unknown"
                        cv2.putText(imgInput, label, (x, y - 10), CVThread.font, 0.5, color, 1, cv2.LINE_AA)
                    except (KeyError, TypeError):
                        pass
            else:
                cv2.putText(imgInput, 'Face Detecting', (40,60), CVThread.font, 0.5, (255,255,255), 1, cv2.LINE_AA)

        elif self.CVMode == 'faceFollowing':
            target_name = Camera.followName.strip().lower()
            if self.faces and len(self.faces):
                cv2.putText(imgInput, f'Following: {Camera.followName}', (40,60), CVThread.font, 0.5, (255,255,255), 1, cv2.LINE_AA)
                for face_info in self.faces:
                    try:
                        x, y, w, h = face_info["box"]
                        name = face_info["name"]
                        if name.lower() == target_name:
                            color = (74, 222, 128) # Active bright green for the target being followed
                            label = f"Target: {name}"
                            cv2.rectangle(imgInput, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(imgInput, label, (x, y - 10), CVThread.font, 0.5, color, 1, cv2.LINE_AA)
                        else:
                            color = (128, 128, 128) # Muted gray for other recognized/unknown faces
                            label = name
                            cv2.rectangle(imgInput, (x, y), (x + w, y + h), color, 1)
                            cv2.putText(imgInput, label, (x, y - 10), CVThread.font, 0.4, color, 1, cv2.LINE_AA)
                    except (KeyError, TypeError):
                        pass
            else:
                cv2.putText(imgInput, f'Searching for {Camera.followName}...', (40,60), CVThread.font, 0.5, (255,255,255), 1, cv2.LINE_AA)

        elif self.CVMode == 'objectDetection':
            if hasattr(self, 'detected_objects') and self.detected_objects:
                cv2.putText(imgInput, f'{len(self.detected_objects)} Object(s) Detected', (40,60), CVThread.font, 0.5, (255,255,255), 1, cv2.LINE_AA)
                for obj in self.detected_objects:
                    try:
                        x, y, w, h = obj["box"]
                        name = obj["name"]
                        similarity = obj["similarity"]
                        
                        # Draw green bounding box around the detected object
                        cv2.rectangle(imgInput, (x, y), (x + w, y + h), (74, 222, 128), 2)
                        
                        # Put label near the top-left corner of the box
                        label = f"{name} ({similarity*100:.1f}%)"
                        cv2.putText(imgInput, label, (x, y - 10), CVThread.font, 0.5, (74, 222, 128), 1, cv2.LINE_AA)
                    except Exception:
                        pass
            else:
                cv2.putText(imgInput, 'Object Detecting', (40,60), CVThread.font, 0.5, (255,255,255), 1, cv2.LINE_AA)

        elif self.CVMode == 'ballSearch':
            search_info = getattr(Camera, 'ball_search_info', None)
            if search_info and isinstance(search_info, dict):
                box = search_info.get('box')
                if box:
                    x = int(box.get('x', 0))
                    y = int(box.get('y', 0))
                    w = int(box.get('w', 0))
                    h = int(box.get('h', 0))
                    color = (74, 222, 128)
                    cv2.rectangle(imgInput, (x, y), (x + w, y + h), color, 2)
                else:
                    cv2.putText(imgInput, 'Ball search active, no detection yet', (40,60), CVThread.font, 0.5, (255,255,255), 1, cv2.LINE_AA)
            else:
                cv2.putText(imgInput, 'Ball search active', (40,60), CVThread.font, 0.5, (255,255,255), 1, cv2.LINE_AA)

        elif self.CVMode == 'findColor':
            if self.findColorDetection:
                cv2.putText(imgInput,'Target Detected',(40,60), CVThread.font, 0.5,(255,255,255),1,cv2.LINE_AA)
                self.drawing = 1
            else:
                cv2.putText(imgInput,'Target Detecting',(40,60), CVThread.font, 0.5,(255,255,255),1,cv2.LINE_AA)
                self.drawing = 0

            if self.radius > 10 and self.drawing:
                cv2.rectangle(imgInput,(int(self.box_x-self.radius),int(self.box_y+self.radius)),(int(self.box_x+self.radius),int(self.box_y-self.radius)),(255,255,255),1)

        elif self.CVMode == 'followColor':
            if hasattr(self, 'follow_circle') and self.follow_circle:
                x, y, radius, action = self.follow_circle
                cv2.circle(imgInput, (int(x), int(y)), int(radius), (74, 222, 128), 2)
                cv2.circle(imgInput, (int(x), int(y)), 3, (64, 128, 255), -1)
                cv2.putText(imgInput, f"{action}  R={radius}", (40, 60), CVThread.font, 0.6, (74, 222, 128), 2, cv2.LINE_AA)
            else:
                target_color = getattr(Camera, 'followColor', 'none').upper()
                cv2.putText(imgInput, f"SEARCHING {target_color}", (40, 60), CVThread.font, 0.6, (64, 128, 255), 2, cv2.LINE_AA)

        elif self.CVMode == 'findlineCV':
            if frameRender:
                imgInput = cv2.cvtColor(imgInput, cv2.COLOR_BGR2GRAY)
                retval_bw, imgInput =  cv2.threshold(imgInput, 0, 255, cv2.THRESH_OTSU)
                imgInput = cv2.erode(imgInput, None, iterations=6)
            try:
                if lineColorSet == 255:
                    cv2.putText(imgInput,('Following White Line'),(30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
                    cv2.putText(imgInput,('Following White Line'),(230,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0),1,cv2.LINE_AA)
                else:
                    cv2.putText(imgInput,('Following Black Line'),(30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
                    cv2.putText(imgInput,('Following Black Line'),(230,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0),1,cv2.LINE_AA)

                cv2.putText(imgInput,(self.CVCommand),(30,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(imgInput,(self.CVCommand),(230,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0),1,cv2.LINE_AA)

                cv2.line(imgInput,(self.left_Pos1,(linePos_1+30)),(self.left_Pos1,(linePos_1-30)),(255,255,255),1)
                cv2.line(imgInput,((self.left_Pos1+1),(linePos_1+30)),((self.left_Pos1+1),(linePos_1-30)),(0,0,0),1)

                cv2.line(imgInput,(self.right_Pos1,(linePos_1+30)),(self.right_Pos1,(linePos_1-30)),(255,255,255),1)
                cv2.line(imgInput,((self.right_Pos1-1),(linePos_1+30)),((self.right_Pos1-1),(linePos_1-30)),(0,0,0),1)

                cv2.line(imgInput,(0,linePos_1),(640,linePos_1),(255,255,255),1)
                cv2.line(imgInput,(0,linePos_1+1),(640,linePos_1+1),(0,0,0),1)

                cv2.line(imgInput,(320-findLineError,0),(320-findLineError,480),(255,255,255),1)
                cv2.line(imgInput,(320+findLineError,0),(320+findLineError,480),(255,255,255),1)

                cv2.line(imgInput,(320-findLineError+1,0),(320-findLineError+1,480),(0,0,0),1)
                cv2.line(imgInput,(320+findLineError-1,0),(320+findLineError-1,480),(0,0,0),1)

                cv2.line(imgInput,(self.left_Pos2,(linePos_2+30)),(self.left_Pos2,(linePos_2-30)),(255,255,255),1)
                cv2.line(imgInput,(self.right_Pos2,(linePos_2+30)),(self.right_Pos2,(linePos_2-30)),(255,255,255),1)
                cv2.line(imgInput,(0,linePos_2),(640,linePos_2),(255,255,255),1)

                cv2.line(imgInput,(self.left_Pos2+1,(linePos_2+30)),(self.left_Pos2+1,(linePos_2-30)),(0,0,0),1)
                cv2.line(imgInput,(self.right_Pos2-1,(linePos_2+30)),(self.right_Pos2-1,(linePos_2-30)),(0,0,0),1)
                cv2.line(imgInput,(0,linePos_2+1),(640,linePos_2+1),(0,0,0),1)

                cv2.line(imgInput,((self.center-20),int((linePos_1+linePos_2)/2)),((self.center+20),int((linePos_1+linePos_2)/2)),(0,0,0),1)
                cv2.line(imgInput,((self.center),int((linePos_1+linePos_2)/2+20)),((self.center),int((linePos_1+linePos_2)/2-20)),(0,0,0),1)

                cv2.line(imgInput,((self.center-20),int((linePos_1+linePos_2)/2+1)),((self.center+20),int((linePos_1+linePos_2)/2+1)),(255,255,255),1)
                cv2.line(imgInput,((self.center+1),int((linePos_1+linePos_2)/2+20)),((self.center+1),int((linePos_1+linePos_2)/2-20)),(255,255,255),1)
            except:
                pass

        elif self.CVMode == 'watchDog':
            if self.drawing:
                cv2.putText(imgInput,'Motion Detected',(40,60), CVThread.font, 0.5,(255,255,255),1,cv2.LINE_AA)
                robot.buzzerCtrl(1, 0)
                robot.lightCtrl('red', 0)
                cv2.rectangle(imgInput, (self.mov_x, self.mov_y), (self.mov_x + self.mov_w, self.mov_y + self.mov_h), (128, 255, 0), 1)
            else:
                cv2.putText(imgInput,'Motion Detecting',(40,60), CVThread.font, 0.5,(255,255,255),1,cv2.LINE_AA)
                robot.buzzerCtrl(0, 0)
                robot.lightCtrl('blue', 0)

        return imgInput


    def watchDog(self, imgInput):
        timestamp = datetime.datetime.now()
        gray = cv2.cvtColor(imgInput, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.avg is None:
            print("[INFO] starting background model...")
            self.avg = gray.copy().astype("float")
            return 'background model'

        cv2.accumulateWeighted(gray, self.avg, 0.5)
        self.frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg))

        # threshold the delta image, dilate the thresholded image to fill
        # in holes, then find contours on thresholded image
        self.thresh = cv2.threshold(self.frameDelta, 5, 255,
            cv2.THRESH_BINARY)[1]
        self.thresh = cv2.dilate(self.thresh, None, iterations=2)
        self.cnts = cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        self.cnts = imutils.grab_contours(self.cnts)
        # print('x')
        # loop over the contours
        for c in self.cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 2000:
                continue
     
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (self.mov_x, self.mov_y, self.mov_w, self.mov_h) = cv2.boundingRect(c)
            self.drawing = 1
            
            self.motionCounter += 1

            self.lastMovtionCaptured = timestamp

        if (timestamp - self.lastMovtionCaptured).seconds >= 0.5:
            self.drawing = 0
            robot.buzzerCtrl(0, 0)

        self.pause()


    def findLineTest(self, posInput, setCenter):#2
        if not posInput:
            robot.robotCtrl.moveStart(speedMove, 'no', 'no')
            return

        if posInput > (setCenter + findLineError):
            self.CVCommand = 'Turning Right'

        elif posInput < (setCenter - findLineError):
            self.CVCommand = 'Turning Left'

        else:
            self.CVCommand = 'Forward'


    def findLineCtrl(self, posInput, setCenter):#2
        if not posInput:
            robot.robotCtrl.moveStart(speedMove, 'no', 'no')
            return

        if posInput > (setCenter + findLineError):
            #turnRight
            robot.right()
            self.CVCommand = 'Turning Right'
            print('Turning Right')

        elif posInput < (setCenter - findLineError):
            #turnLeft
            robot.left()
            self.CVCommand = 'Turning Left'
            print('Turning Left')

        else:
            #forward
            robot.forward()
            self.CVCommand = 'Forward'
            print('Forward')


    def findlineCV(self, frame_image):
        frame_findline = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
        retval, frame_findline =  cv2.threshold(frame_findline, 0, 255, cv2.THRESH_OTSU)
        frame_findline = cv2.erode(frame_findline, None, iterations=6)
        colorPos_1 = frame_findline[linePos_1]
        colorPos_2 = frame_findline[linePos_2]
        try:
            lineColorCount_Pos1 = np.sum(colorPos_1 == lineColorSet)
            lineColorCount_Pos2 = np.sum(colorPos_2 == lineColorSet)

            lineIndex_Pos1 = np.where(colorPos_1 == lineColorSet)
            lineIndex_Pos2 = np.where(colorPos_2 == lineColorSet)

            if lineColorCount_Pos1 == 0:
                lineColorCount_Pos1 = 1
            if lineColorCount_Pos2 == 0:
                lineColorCount_Pos2 = 1

            self.left_Pos1 = lineIndex_Pos1[0][lineColorCount_Pos1-1]
            self.right_Pos1 = lineIndex_Pos1[0][0]
            self.center_Pos1 = int((self.left_Pos1+self.right_Pos1)/2)

            self.left_Pos2 = lineIndex_Pos2[0][lineColorCount_Pos2-1]
            self.right_Pos2 = lineIndex_Pos2[0][0]
            self.center_Pos2 = int((self.left_Pos2+self.right_Pos2)/2)

            self.center = int((self.center_Pos1+self.center_Pos2)/2)
        except:
            center = None
            pass

        if Camera.CVMode == 'run':
            self.findLineCtrl(self.center, 320)
        else:
            self.findLineTest(self.center, 320)
        self.pause()


    def findColor(self, frame_image):
        hsv = cv2.cvtColor(frame_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, colorLower, colorUpper)#1
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        if len(cnts) > 0:
            X_LOCK = 0
            Y_LOCK = 0
            self.findColorDetection = 1
            c = max(cnts, key=cv2.contourArea)
            ((self.box_x, self.box_y), self.radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            X = int(self.box_x)
            Y = int(self.box_y)
            error_Y = abs(240 - Y)
            error_X = abs(320 - X)

            if Y < 240 - CVThread.tor:
                # error_Y*CVThread.aspd
                robot.lookUp()
            elif Y > 240 + CVThread.tor:
                robot.lookDown()
            else:
                Y_LOCK = 1

            if X < 320 - CVThread.tor:
                robot.lookLeft()
            elif X > 320 + CVThread.tor:
                robot.lookRight()
            else:
                X_LOCK = 1

            if X_LOCK == 1 and Y_LOCK == 1:
                robot.buzzerCtrl(1, 0)
                robot.lightCtrl('red', 0)
            else:
                robot.buzzerCtrl(0, 0)
                robot.lightCtrl('blue', 0)

        else:
            self.findColorDetection = 0
        self.pause()


    def faceDetectCV(self, frame_image):
        import face_detection
        self.faces = face_detection.recognize_faces(frame_image)
        if len(self.faces):
            robot.lightCtrl('red', 0)
        else:
            robot.lightCtrl('blue', 0)
        self.pause()


    def ballSearchCV(self, frame_image):
        try:
            # Run detection on BGR image
            result = detect(frame_image, conf_threshold=0.0, input_is_rgb=False)
            
            # Filter detections by class_id == 1 (ball) and confidence >= 0.50
            ball_detections = []
            if result.get("success", False) and result.get("detections"):
                ball_detections = [d for d in result["detections"] if d.get("class_id") == 1 and d.get("conf") >= 0.50]
                
            # Log state & detections
            print(f"[ballSearchCV] State: {self.ball_search_state}, Detections: {len(ball_detections)}")
            
            # --- State Machine ---
            if self.ball_search_state == 'init':
                # Transition to searching, start rotating 360 degrees right
                self.ball_search_state = 'searching'
                self.ball_search_start_time = time.time()
                robot.right()
                print("[ballSearchCV] Initializing search: rotating 360 degrees right...")
                Camera.ball_search_info = None
                
            elif self.ball_search_state == 'searching':
                if ball_detections:
                    # Found the ball! Stop turning and follow
                    self.ball_search_state = 'following'
                    robot.stopLR()
                    print("[ballSearchCV] Ball found! Transitioning to following state.")
                else:
                    # Check if 360 turn timeout reached (e.g. 6.0 seconds)
                    elapsed = time.time() - self.ball_search_start_time
                    if elapsed > 6.0:
                        self.ball_search_state = 'not_found'
                        robot.stopLR()
                        print("[ballSearchCV] 360 turn complete. Ball not found.")
                    # Otherwise, continue rotating
                    Camera.ball_search_info = None
                    
            elif self.ball_search_state == 'following':
                if ball_detections:
                    best_det = ball_detections[0]
                    x1 = int(best_det["x1"])
                    y1 = int(best_det["y1"])
                    w_box = int(best_det["x2"] - best_det["x1"])
                    h_box = int(best_det["y2"] - best_det["y1"])
                    
                    Camera.ball_search_info = {
                        "box": {"x": x1, "y": y1, "w": w_box, "h": h_box},
                        "confidence": best_det["conf"]
                    }
                    
                    # Compute center and radius
                    cx = x1 + w_box / 2
                    cy = y1 + h_box / 2
                    radius = (w_box + h_box) / 4
                    
                    # Centering horizontally (center_x = 320)
                    error_x = cx - 320
                    
                    # Distance Control: target radius is 80 pixels
                    if radius > 80:
                        # Close enough, stop movement
                        robot.stopFB()
                        robot.stopLR()
                        print(f"[ballSearchCV] Target distance reached (radius={radius:.1f} > 80). Stopping.")
                    else:
                        # Adjust angle to keep the ball centered
                        if abs(error_x) < 50:
                            # Centered horizontally, move forward
                            robot.stopLR()
                            robot.forward()
                            print(f"[ballSearchCV] Centered (error={error_x:.1f}). Moving forward.")
                        elif error_x < 0:
                            # Turn left
                            robot.stopFB()
                            robot.left()
                            print(f"[ballSearchCV] Off-center left (error={error_x:.1f}). Turning left.")
                        else:
                            # Turn right
                            robot.stopFB()
                            robot.right()
                            print(f"[ballSearchCV] Off-center right (error={error_x:.1f}). Turning right.")
                            
                    # Center vertically using camera tilt
                    tor = CVThread.tor
                    if cy < 240 - tor:
                        robot.lookUp()
                    elif cy > 240 + tor:
                        robot.lookDown()
                    else:
                        robot.lookStopUD()
                        
                else:
                    # Ball is lost! Go back to searching state
                    print("[ballSearchCV] Ball lost. Transitioning back to searching.")
                    self.ball_search_state = 'searching'
                    self.ball_search_start_time = time.time()
                    robot.right()
                    Camera.ball_search_info = None
                    
            elif self.ball_search_state == 'not_found':
                # Stop robot
                robot.stopLR()
                robot.stopFB()
                robot.lookStopUD()
                Camera.ball_search_info = None
                
                # If we see the ball again, recover and follow
                if ball_detections:
                    self.ball_search_state = 'following'
                    print("[ballSearchCV] Ball detected in not_found state. Resuming follow.")
                    
        except Exception as e:
            print("Error in ballSearchCV:", e)
            Camera.ball_search_info = None
        self.pause()


    def faceFollowingCV(self, frame_image):
        import face_detection
        self.faces = face_detection.recognize_faces(frame_image)
        
        target_face = None
        target_name = Camera.followName.strip().lower()
        
        for face in self.faces:
            name = face.get("name", "Unknown")
            if name.lower() == target_name:
                target_face = face
                break
                
        if target_face:
            robot.lightCtrl('red', 0)
            x, y, w, h = target_face["box"]
            X = int(x + w / 2)
            Y = int(y + h / 2)
            
            # Deadzone tolerance
            tor = CVThread.tor
            
            # Control Y axis (tilt up / down / stop)
            if Y < 240 - tor:
                robot.lookUp()
            elif Y > 240 + tor:
                robot.lookDown()
            else:
                robot.lookStopUD()
                
            # Control X axis (pan left / right / stop)
            if X < 320 - tor:
                robot.lookLeft()
            elif X > 320 + tor:
                robot.lookRight()
            else:
                robot.lookStopLR()
        else:
            robot.lightCtrl('blue', 0)
            robot.lookStopUD()
            robot.lookStopLR()
            
        self.pause()


    def load_object_templates(self):
        import glob
        import os
        import pickle
        
        self.object_templates = []
        object_db_dir = os.path.join(thisPath, "object_db")
        if not os.path.exists(object_db_dir):
            return
            
        # Initialize ONNX session if needed
        if not hasattr(self, 'mobilenet_session') or self.mobilenet_session is None:
            try:
                import onnxruntime as ort
                model_path = os.path.normpath(os.path.join(thisPath, "mobilenetv3_embedding.onnx"))
                if not os.path.exists(model_path):
                    print(f"Error: mobilenetv3_embedding.onnx not found at {model_path}")
                    return
                self.mobilenet_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                session = get_session(model_path)
                input_name = session.get_inputs()[0].name
                input_shape = session.get_inputs()[0].shape  # [1, 3, 416, 416]
                print(f"[Detect] Model loaded from {model_path}, input shape {input_shape}")
            except Exception as e:
                print(f"Error initializing ONNX runtime in CVThread: {e}")
                return

        pkl_files = glob.glob(os.path.join(object_db_dir, "*.pkl"))
        for pkl_p in pkl_files:
            try:
                with open(pkl_p, "rb") as f:
                    data = pickle.load(f)
                    if "name" in data and "embeddings" in data:
                        self.object_templates.append({
                            "name": data["name"],
                            "embeddings": data["embeddings"]
                        })
            except Exception as e:
                print(f"Error loading template {pkl_p}: {e}")

    def get_crop_embedding(self, crop):
        if not hasattr(self, 'mobilenet_session') or self.mobilenet_session is None:
            return None
            
        import cv2
        import numpy as np

        try:
            # Resize to 224x224
            resized = cv2.resize(crop, (224, 224))
            # Convert BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1]
            normalized = rgb.astype(np.float32) / 255.0
            # ImageNet mean & std
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (normalized - mean) / std
            # HWC to CHW
            chw = np.transpose(normalized, (2, 0, 1))
            # Add batch dim NCHW
            nchw = np.expand_dims(chw, axis=0)

            # Inference
            outputs = self.mobilenet_session.run(None, {self.mobilenet_input_name: nchw})
            embedding = outputs[0][0]

            # L2 Normalize
            norm = np.linalg.norm(embedding)
            if norm > 1e-10:
                embedding = embedding / norm
            return embedding
        except Exception as e:
            print(f"Error running inference in get_crop_embedding: {e}")
            return None

    def objectDetectCV(self, frame_image):
        import cv2
        import numpy as np
        
        self.detected_objects = []
        if not self.object_templates:
            self.pause()
            return
            
        # Ensure ONNX session is loaded
        if not hasattr(self, 'mobilenet_session') or self.mobilenet_session is None:
            self.load_object_templates()
            if not hasattr(self, 'mobilenet_session') or self.mobilenet_session is None:
                self.pause()
                return

        try:
            # Proposal generation via Canny and contours
            gray = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 50, 150)
            
            # Dilate the edges to merge near contours
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated = cv2.dilate(edged, kernel, iterations=1)
            
            cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            import imutils
            cnts = imutils.grab_contours(cnts)
            
            # Sort contours by area in descending order and limit to top 10
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
            
            best_detections = []
            
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 1500 or area > 200000:
                    continue
                    
                (x, y, w, h) = cv2.boundingRect(c)
                if w < 40 or h < 40:
                    continue
                    
                # Crop and get embedding
                crop = frame_image[y : y + h, x : x + w]
                crop_emb = self.get_crop_embedding(crop)
                if crop_emb is None:
                    continue
                    
                best_match_name = None
                best_match_sim = 0.0
                
                # Match against all templates
                for template in self.object_templates:
                    name = template["name"]
                    tem_embs = template["embeddings"]
                    
                    for tem_emb in tem_embs:
                        # Cosine similarity is dot product of normalized embeddings
                        sim = float(np.dot(crop_emb, tem_emb))
                        if sim > best_match_sim:
                            best_match_sim = sim
                            best_match_name = name
                            
                # Match threshold is 0.70
                if best_match_sim >= 0.70:
                    best_detections.append({
                        "box": (x, y, w, h),
                        "name": best_match_name,
                        "similarity": best_match_sim
                    })
            
            # Post-process detections: NMS based on IoU
            if best_detections:
                best_detections.sort(key=lambda x: x["similarity"], reverse=True)
                keep_detections = []
                
                def get_iou(boxA, boxB):
                    xA = max(boxA[0], boxB[0])
                    yA = max(boxA[1], boxB[1])
                    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
                    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
                    
                    interArea = max(0, xB - xA) * max(0, yB - yA)
                    boxAArea = boxA[2] * boxA[3]
                    boxBArea = boxB[2] * boxB[3]
                    
                    iou = interArea / float(boxAArea + boxBArea - interArea)
                    return iou
                    
                for det in best_detections:
                    overlap = False
                    for keep in keep_detections:
                        if get_iou(det["box"], keep["box"]) > 0.3:
                            overlap = True
                            break
                    if not overlap:
                        keep_detections.append(det)
                        
                self.detected_objects = keep_detections
                
                if self.detected_objects:
                    robot.lightCtrl('green', 0)
                else:
                    robot.lightCtrl('blue', 0)
            else:
                robot.lightCtrl('blue', 0)
                
        except Exception as err:
            print(f"Error in objectDetectCV: {err}")
            
        self.pause()


    def followColorCV(self, frame_image):
        target_color = getattr(Camera, 'followColor', 'none').lower()
        
        COLORS = {
            "red": [
                ((0, 120, 70), (10, 255, 255)),
                ((170, 120, 70), (180, 255, 255))
            ],
            "green": [
                ((35, 50, 50), (85, 255, 255))
            ],
            "blue": [
                ((90, 50, 50), (130, 255, 255))
            ],
            "yellow": [
                ((20, 100, 100), (35, 255, 255))
            ]
        }
        
        if target_color not in COLORS or target_color == "none":
            self.follow_circle = None
            robot.stopFB()
            robot.stopLR()
            robot.lookStopUD()
            self.pause()
            return
            
        try:
            hsv = cv2.cvtColor(frame_image, cv2.COLOR_BGR2HSV)
            masks = []
            for lower, upper in COLORS[target_color]:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                masks.append(mask)
                
            if not masks:
                self.follow_circle = None
                robot.lookStopUD()
                self.pause()
                return
                
            mask = masks[0]
            for m in masks[1:]:
                mask |= m
                
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=2)
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            # Find contours to locate contiguous color blobs
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            import imutils
            cnts = imutils.grab_contours(cnts)
            
            action = "SEARCH"
            if len(cnts) > 0:
                # Select the contour representing the maximum concentration of the target color
                largest_contour = max(cnts, key=cv2.contourArea)
                
                # Minimum area threshold (e.g., 200 pixels) to avoid tracking tiny background noise
                if cv2.contourArea(largest_contour) > 200:
                    ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
                    x, y, radius = int(x), int(y), int(radius)
                    
                    center_x = frame_image.shape[1] // 2
                    error = x - center_x
                    
                    # STOP_RADIUS = 100, CENTER_TOLERANCE = 50
                    if radius > 100:
                        action = "STOP"
                        robot.stopFB()
                        robot.stopLR()
                    else:
                        if abs(error) < 50:
                            action = "FORWARD"
                            robot.forward()
                        elif error < 0:
                            action = "LEFT"
                            robot.left()
                        else:
                            action = "RIGHT"
                            robot.right()
                            
                    # Control Y axis (tilt up / down / stop) to keep the target centered vertically
                    tor = CVThread.tor
                    if y < 240 - tor:
                        robot.lookUp()
                    elif y > 240 + tor:
                        robot.lookDown()
                    else:
                        robot.lookStopUD()
                            
                    self.follow_circle = (x, y, radius, action)
                else:
                    action = "SEARCH"
                    robot.left() # slow rotation to search for the ball
                    robot.lookStopUD()
                    self.follow_circle = None
            else:
                action = "SEARCH"
                robot.left() # slow rotation to search for the ball
                robot.lookStopUD()
                self.follow_circle = None
        except Exception as e:
            print("Error in followColorCV:", e)
            self.follow_circle = None
            robot.lookStopUD()
            
        self.pause()


    def pause(self):
        self.__flag.clear()

    def resume(self):
        self.__flag.set()

    def run(self):
        while 1:
            if self.CVMode == 'none':
                robot.buzzerCtrl(0, 0)

            self.__flag.wait()
            if self.CVMode == 'none':
                self.object_templates = None
                robot.stopLR()
                robot.stopFB()
                robot.lookStopUD()
                robot.lookStopLR()
                robot.buzzerCtrl(0, 0)
                robot.lightCtrl('blue', 0)
                self.pause()
                robot.buzzerCtrl(0, 0)
                continue

            elif self.CVMode == 'findColor':
                self.CVThreading = 1
                self.findColor(self.imgCV)
                self.CVThreading = 0

            elif self.CVMode == 'findlineCV':
                self.CVThreading = 1
                self.findlineCV(self.imgCV)
                self.CVThreading = 0

            elif self.CVMode == 'watchDog':
                self.CVThreading = 1
                self.watchDog(self.imgCV)
                self.CVThreading = 0

            elif self.CVMode == 'faceDetection':
                self.CVThreading = 1
                self.faceDetectCV(self.imgCV)
                self.CVThreading = 0

            elif self.CVMode == 'faceFollowing':
                self.CVThreading = 1
                self.faceFollowingCV(self.imgCV)
                self.CVThreading = 0

            elif self.CVMode == 'objectDetection':
                self.CVThreading = 1
                if self.object_templates is None:
                    self.load_object_templates()
                self.objectDetectCV(self.imgCV)
                self.CVThreading = 0

            elif self.CVMode == 'followColor':
                self.CVThreading = 1
                self.followColorCV(self.imgCV)
                self.CVThreading = 0

            elif self.CVMode == 'ballSearch':
                self.CVThreading = 1
                self.ballSearchCV(self.imgCV)
                self.CVThreading = 0


class Camera(BaseCamera):
    modeSelect = 'none'
    followName = ''
    followColor = 'none'
    ball_search_info = None
    latest_bgr_frame = None
    # modeSelect = 'findlineCV'
    # modeSelect = 'findColor'
    # modeSelect = 'watchDog'
    # add # modeSelect = 'faceDetection'

    CVMode = 'run'
    # CVMode = 'no'

    def __init__(self):
        super(Camera, self).__init__()

    def robotStop(self):
        robot.robotCtrl.moveStart(speedMove, 'no', 'no')
        time.sleep(0.1)
        robot.robotCtrl.moveStart(speedMove, 'no', 'no')

    def colorFindSet(self, invarH, invarS, invarV):
        global colorUpper, colorLower
        HUE_1 = invarH+15
        HUE_2 = invarH-15
        if HUE_1>180:HUE_1=180
        if HUE_2<0:HUE_2=0

        SAT_1 = invarS+150
        SAT_2 = invarS-150
        if SAT_1>255:SAT_1=255
        if SAT_2<0:SAT_2=0

        VAL_1 = invarV+150
        VAL_2 = invarV-150
        if VAL_1>255:VAL_1=255
        if VAL_2<0:VAL_2=0

        colorUpper = np.array([HUE_1, SAT_1, VAL_1])
        colorLower = np.array([HUE_2, SAT_2, VAL_2])
        print('HSV_1:%d %d %d'%(HUE_1, SAT_1, VAL_1))
        print('HSV_2:%d %d %d'%(HUE_2, SAT_2, VAL_2))
        print(colorUpper)
        print(colorLower)

    def modeSet(self, invar):
        Camera.modeSelect = invar

    def upperIP(self, invar):
        global upperGlobalIP
        upperGlobalIP = invar

    def CVRunSet(self, invar):
        global CVRun
        CVRun = invar

    def linePosSet_1(self, invar):
        global linePos_1
        linePos_1 = invar

    def linePosSet_2(self, invar):
        global linePos_2
        linePos_2 = invar

    def colorSet(self, invar):
        global lineColorSet
        lineColorSet = invar

    def randerSet(self, invar):
        global frameRender
        frameRender = invar

    def errorSet(self, invar):
        global findLineError
        findLineError = invar

    @staticmethod
    def frames():
        from picamera2 import Picamera2

        picam2 = Picamera2()
        picam2.configure(
            picam2.create_video_configuration(
                main={"format": "RGB888", "size": (640, 480)}
            )
        )
        picam2.start()

        cvt = CVThread()
        cvt.start()

        print("DEBUG: Running frames() from whisper-bot/camera_opencv.py - Using frame directly")
        while True:
            frame = picam2.capture_array()
            if frame is None:
                raise RuntimeError('Camera started but could not read frames. Check the Pi camera and Picamera2 configuration.')

            img = frame
            Camera.latest_bgr_frame = img.copy() if isinstance(img, np.ndarray) else None

            if Camera.modeSelect == 'none':
                cvt.pause()
                robot.buzzerCtrl(0, 0)
            else:
                if cvt.CVThreading:
                    pass
                else:
                    # Print debug status to see if cvt is alive and receiving the mode Select
                    print(f"[DEBUG Camera.frames] modeSelect={Camera.modeSelect}, CVMode={cvt.CVMode}, CVThreading={cvt.CVThreading}, cvt.is_alive={cvt.is_alive()}")
                    cvt.mode(Camera.modeSelect, img)
                    cvt.resume()
                try:
                    img = cvt.elementDraw(img)
                except Exception as e:
                    import traceback
                    print("Error in elementDraw:")
                    traceback.print_exc()

            # encode as a jpeg image and return it
            try:
                yield cv2.imencode('.jpg', img)[1].tobytes()
            except:
                pass


def commandAct(act, inputA):
    global speedMove
    if act == 'forward':
        robot.forward(speedMove)
    elif act == 'backward':
        robot.backward(speedMove)
    elif act == 'left':
        robot.left(speedMove)
    elif act == 'right':
        robot.right(speedMove)
    elif act == 'DS':
        robot.stopFB()
    elif act == 'TS':
        robot.stopLR()

    elif 'wsB' in act:
        speedMove = int(act.split()[1])
        if(speedMove > 1 and speedMove <= 100):
            robot.speedSet(speedMove)

    elif act == 'up':
        robot.lookUp()
    elif act == 'down':
        robot.lookDown()
    elif act == 'UDstop':
        robot.lookStopUD()
    elif act == 'lookleft':
        robot.lookLeft()
    elif act == 'lookright':
        robot.lookRight()
    elif act == 'LRstop':
        robot.lookStopLR()

    elif act == 'jump':
        robot.jump()
    elif act == 'handshake':
        robot.handShake()
    elif act == 'steady':
        robot.steadyMode()
    elif act == 'steadyOff':
        robot.steadyMode()

    # openCV ctrl.
    elif act == 'faceDetection':
        Camera.modeSelect = 'faceDetection'
    elif act == 'faceDetectionOff':
        Camera.modeSelect = 'none'
        robot.buzzerCtrl(0, 0)
    elif 'trackLine' == act:
        Camera.modeSelect = 'findlineCV'
        Camera.CVMode = 'run'
    elif 'trackLineOff' == act:
        Camera.modeSelect = 'none'
        time.sleep(0.05)
        robot.stopLR()
        time.sleep(0.05)
        robot.stopFB()
        robot.buzzerCtrl(0, 0)