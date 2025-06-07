import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def normalize(v):
    norm = np.linalg.norm(v)
    return v / (norm + 1e-6)

def draw_axis(image, origin, x_axis, y_axis, z_axis, scale=80):
    origin = np.int32(origin)
    end_x = origin + np.int32(scale * x_axis[:2])
    end_y = origin + np.int32(scale * y_axis[:2])
    end_z = origin + np.int32(scale * z_axis[:2])

    cv2.line(image, tuple(origin), tuple(end_x), (0, 0, 255), 3)  # X - RED
    cv2.line(image, tuple(origin), tuple(end_y), (0, 255, 0), 3)  # Y - GREEN
    cv2.line(image, tuple(origin), tuple(end_z), (255, 0, 0), 3)  # Z - BLUE

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, _ = img.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # sintetagmenes 3D twn landmark
            landmarks = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in hand_landmarks.landmark])

            origin = landmarks[1]  #vasi antixeira
            x_axis = normalize(landmarks[2] - landmarks[1])  #kata mikos antixeira
            temp = landmarks[5] - landmarks[1]  # pros deikti
            z_axis = normalize(np.cross(x_axis, temp))  #kathetos sto epipedo antixeira-deikti
            y_axis = np.cross(z_axis, x_axis)  

            draw_axis(img, origin[:2], x_axis, y_axis, z_axis)

            # Debug
            print(f"X: {x_axis}, Y: {y_axis}, Z: {z_axis}")

            #sxedio xeriou
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Coordinate Frame", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
