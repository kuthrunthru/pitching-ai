import cv2
import mediapipe as mp

print("Starting MediaPipe webcam test...")

# Try to open the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam. Check that a camera is connected and not used by another app.")
    input("Press Enter to exit...")
    raise SystemExit

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

print("Webcam opened successfully. Press ESC in the video window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read a frame from the webcam.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("MediaPipe Pose Test", frame)

    # Press ESC to quit
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        print("ESC pressed, closing.")
        break

cap.release()
cv2.destroyAllWindows()
print("Done. Closing program.")
input("Press Enter to exit...")
