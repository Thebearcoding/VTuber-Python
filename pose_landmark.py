import cv2
import mediapipe as mp


class PoseMeshDetector:
    def __init__(self,
                 static_image_mode=False,
                 max_num_faces=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpPoses = mp.solutions.pose
        self.pose_mesh = self.mpPoses.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        self.poseLmsStyle = self.mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=30)
        self.poseConStyle = self.mp_drawing.DrawingSpec(thickness=5)
        self.poses = []

    def find_pose_mesh(self, Img, draw=True):
        img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
        result = self.pose_mesh.process(img)
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]

        if result.pose_landmarks:
            poseLms = result.pose_landmarks
            if draw:
                # for poseLms in result.pose_landmarks:
                self.mp_drawing.draw_landmarks(img, result.pose_landmarks, self.mpPoses.POSE_CONNECTIONS,
                                               self.poseLmsStyle,
                                               self.poseConStyle)
            pose = []
            for i, lm in enumerate(poseLms.landmark):
                xPos = round(lm.x * imgWidth)
                yPos = round(lm.y * imgHeight)
                zPos = round(lm.z)
                # if i == 24 or i == 14 or i == 12 or i == 16 or i == 23 or i == 13 or i == 15:
                cv2.putText(img, str(i), (xPos - 25, yPos + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                pose.append([xPos, yPos, zPos])
            self.poses = pose
            # cv2.imshow("img", img)
        # print(result.pose_landmarks)
        return img, self.poses


def main():
    detector = PoseMeshDetector()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, faces = detector.find_pose_mesh(imgRGB)
        # Pose接收RGB图像
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 点和线的风格
    cap.release()


if __name__ == "__main__":
    main()
