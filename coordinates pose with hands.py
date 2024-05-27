import cv2
import mediapipe as mp

# Inicializar MediaPipe Hands y Pose
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands()
pose = mp_pose.Pose()

# Capturar video desde la cámara
cap = cv2.VideoCapture(1)

def get_hand_orientation(landmarks):
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]

    # Promedio de las posiciones Y de los MCPs
    mcp_y_average = (index_mcp.y + middle_mcp.y + ring_mcp.y + pinky_mcp.y) / 4

    if wrist.y > mcp_y_average:
        return "Arriba"
    elif wrist.y < mcp_y_average:
        return "Tumbado"
    else:
        return "Lado"

def get_hand_position(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    # Comparar la posición X del pulgar y del meñique
    if thumb_tip.x < pinky_tip.x:
        return "Palma de Frente"
    else:
        return "Palma volteada"

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    # Voltear la cámara
    image = cv2.flip(image, 1)

    # Convertir la imagen a RGB para MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(image_rgb)
    pose_results = pose.process(image_rgb)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            # Dibujar los puntos de referencia y conexiones en la imagen
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detectar la orientación de la mano
            orientation = get_hand_orientation(hand_landmarks.landmark)
            position = get_hand_position(hand_landmarks.landmark)
            handedness_label = handedness.classification[0].label
            hand_label = f'{handedness_label}, {orientation}, {position}'

            # Obtener punto de referencia para imprimir tipo de mano
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_x, wrist_y = int(wrist.x * image.shape[1]), int(wrist.y * image.shape[0])
            cv2.putText(image, hand_label, (wrist_x, wrist_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    if pose_results.pose_landmarks:
        # Dibujar los puntos de referencia y conexiones de pose en la imagen
        mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Imprimir las coordenadas de cada punto de referencia en la consola
        for id, landmark in enumerate(pose_results.pose_landmarks.landmark):
            height, width, _ = image.shape
            cx, cy = int(landmark.x * width), int(landmark.y * height)
            print(f'Punto {id}: ({cx}, {cy})')

            # Mostrar las coordenadas en la imagen del video
            cv2.putText(image, f'({cx}, {cy})', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Mostrar la imagen en una ventana
    cv2.imshow('Sistema de coordenadas', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
