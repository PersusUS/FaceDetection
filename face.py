import cv2
from ultralytics import YOLO
from deepface import DeepFace  # Import DeepFace for emotion recognition

# Load the YOLOv8 models
person_model = YOLO('face/yolov8n.pt')  # Pretrained YOLO model for person detection
face_model = YOLO('face/yolov8n-face.pt')  # Pretrained YOLO model for face detection

# Define confidence thresholds
CONFIDENCE_THRESHOLD = 0.75
CONFIDENCE_THRESHOLD_FACE = 0.50

# Start video capture
video_capture = cv2.VideoCapture(0)

# Initial flags
detect_person = True
detect_face = False

while True:
    ret, frame = video_capture.read()
    
    if not ret:
        break
    
    if detect_person:
        # Run YOLO model for person detection
        results = person_model(frame)
        
        for result in results:
            boxes = result.boxes  # Extract the bounding box information
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cls = int(box.cls[0])
                
                if confidence > CONFIDENCE_THRESHOLD and cls == 0:  # Assuming class '0' is for 'person'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Conf: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    if detect_face:
                        # Optional: Crop and run face detection on the ROI
                        roi_frame = frame[y1:y2, x1:x2]
                        face_results = face_model(roi_frame)
                        
                        for face_result in face_results:
                            face_boxes = face_result.boxes
                            
                            for face_box in face_boxes:
                                fx1, fy1, fx2, fy2 = map(int, face_box.xyxy[0])
                                face_confidence = face_box.conf[0]
                                face_cls = int(face_box.cls[0])
                                
                                if face_confidence > CONFIDENCE_THRESHOLD_FACE and face_cls == 0:
                                    fx1, fy1, fx2, fy2 = fx1 + x1, fy1 + y1, fx2 + x1, fy2 + y1
                                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
                                    cv2.putText(frame, f'Face Conf: {face_confidence:.2f}', (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                    
                                    # Extract the face ROI for emotion detection
                                    face_roi = frame[fy1:fy2, fx1:fx2]
                                    
                                    # Debug: Check the dimensions of the face ROI
                                    if face_roi.size == 0:
                                        print("Face ROI has zero size, skipping emotion detection.")
                                        continue

                                    # Detect emotion using DeepFace
                                    try:
                                        # Analyze the emotion using DeepFace and print the output for debugging
                                        emotion_result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                                        print("Emotion result:", emotion_result)  # Debugging line

                                        dominant_emotion = emotion_result[0]['dominant_emotion']  # Fix: Index result with [0]
                                        
                                        # Display the emotion on the frame
                                        cv2.putText(frame, f'Emotion: {dominant_emotion}', (fx1, fy2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                    except Exception as e:
                                        print(f"Emotion detection failed: {e}")

    elif detect_face:
        # Run YOLO model for face detection on the entire frame
        results = face_model(frame)
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cls = int(box.cls[0])
                
                if confidence > CONFIDENCE_THRESHOLD_FACE and cls == 0:  # Assuming class '0' is for 'face'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f'Face Conf: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Extract the face ROI for emotion detection
                    face_roi = frame[y1:y2, x1:x2]
                    
                    # Debug: Check the dimensions of the face ROI
                    if face_roi.size == 0:
                        print("Face ROI has zero size, skipping emotion detection.")
                        continue

                    # Detect emotion using DeepFace
                    try:
                        # Analyze the emotion using DeepFace and print the output for debugging
                        emotion_result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                        print("Emotion result:", emotion_result)  # Debugging line

                        dominant_emotion = emotion_result[0]['dominant_emotion']  # Fix: Index result with [0]
                        
                        # Display the emotion on the frame
                        cv2.putText(frame, f'Emotion: {dominant_emotion}', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    except Exception as e:
                        print(f"Emotion detection failed: {e}")

    # Display the resulting frame
    cv2.imshow('Video Feed', frame)

    # Save the frame to a file
    cv2.imwrite('output_frame.jpg', frame)

    # Key press to toggle detection modes and exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('f'):
        detect_face = not detect_face  # Toggle face detection mode
    elif key == ord('p'):
        detect_person = not detect_person  # Toggle person detection mode
    elif key == ord('q'):
        break  # Exit the loop

# Release resources
video_capture.release()
cv2.destroyAllWindows()
