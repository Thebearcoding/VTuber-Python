import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpPoses = mp.solutions.pose
# 获取mp接口的pose
poses = mpPoses.Pose()
hands = mp.solutions.hands.Hands()
mpDraw = mp.solutions.drawing_utils
poseLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 0), thickness=30)
poseConStyle = mpDraw.DrawingSpec(thickness=5)
# 点和线的风格
while True:
    ret, img = cap.read()
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Pose接收RGB图像
        result = poses.process(imgRGB)
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        hands.process(imgRGB)
        if result.pose_landmarks:
            poseLms = result.pose_landmarks
            # for poseLms in result.pose_landmarks:
            mpDraw.draw_landmarks(img, result.pose_landmarks, mpPoses.POSE_CONNECTIONS, poseLmsStyle, poseConStyle)
            for i, lm in enumerate(poseLms.landmark):
                xPos = round(lm.x * imgWidth)
                yPos = round(lm.y * imgHeight)
                cv2.putText(img, str(i), (xPos - 25, yPos + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                print(i, xPos, yPos)

        # print(result.pose_landmarks)

        cv2.imshow("img", img)
    if cv2.waitKey(1) == ord('q'):
        break
