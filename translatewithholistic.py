import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(1)

with mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1) as holistic:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        # Voltear la cámara
        image = cv2.flip(image, 1)
    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Dibujar las conexiones de la cara
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0,138,255), thickness=1, circle_radius=1))
            
        #mano izquierda
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255,255,0), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=1))
        
        #mano derecha
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(57,143,0), thickness=2, circle_radius=1))
        
        #pose
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(128,0,255), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1))
        
        face_landmarks=results.face_landmarks
        pose_landmarks=results.pose_landmarks
        left_landmarks=results.left_hand_landmarks
        right_landmarks=results.right_hand_landmarks
        
        
        cv2.imshow('Translate', image)
        
        #plasmar puntos en el espacio:
        # mp_drawing.plot_landmarks(
        #     results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
