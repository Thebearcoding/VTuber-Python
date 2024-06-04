import numpy as np
import math
import cv2
from pose_landmark import PoseMeshDetector
from body_rotation import BodyRotationCalculator
from argparse import ArgumentParser
import socket
from facial_landmark import FaceMeshDetector
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer
from facial_features import FacialFeatures, Eyes
import sys

# global variable
port = 5066  # have to be same as unity
alpha = 0.6


# init TCP connection with unity
# return the socket connected
def init_TCP():
    port = args.port

    # '127.0.0.1' = 'localhost' = your computer internal data transmission IP
    address = ('127.0.0.1', port)
    # address = ('192.168.0.107', port)

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(address)
        print("Connected to address:", socket.gethostbyname(socket.gethostname()) + ":" + str(port))
        return s
    except OSError as e:
        print("Error while connecting :: %s" % e)
        sys.exit()


def send_info_to_unity(s, args):
    msg = '%.4f ' * len(args) % args
    try:
        s.send(bytes(msg, "utf-8"))
    except socket.error as e:
        print("Error while sending :: " + str(e))
        sys.exit()


def print_debug_msg(args):
    msg = '%.4f ' * len(args) % args
    print(msg)


# Stabilizer for body
kf_rk = Stabilizer(state_num=2, measure_num=1, cov_process=1e-5, cov_measure=1e-2)
kf_rh = Stabilizer(state_num=2, measure_num=1, cov_process=1e-5, cov_measure=1e-2)
kf_lk = Stabilizer(state_num=2, measure_num=1, cov_process=1e-5, cov_measure=1e-2)
kf_lh = Stabilizer(state_num=2, measure_num=1, cov_process=1e-5, cov_measure=1e-2)

kf_le = Stabilizer(state_num=2, measure_num=1, cov_process=1e-5, cov_measure=1e-2)
kf_ls = Stabilizer(state_num=2, measure_num=1, cov_process=1e-5, cov_measure=1e-2)
kf_re = Stabilizer(state_num=2, measure_num=1, cov_process=1e-5, cov_measure=1e-2)
kf_rs = Stabilizer(state_num=2, measure_num=1, cov_process=1e-5, cov_measure=1e-2)


def get_leg_ro(poses):
    rhu_kne_vec = np.array([poses[26][0] - poses[24][0], poses[26][1] - poses[24][1]])
    rkn_ank_vec = np.array([poses[28][0] - poses[26][0], poses[28][1] - poses[26][1]])
    rsh_hu_vec = np.array([poses[12][0] - poses[24][0], poses[12][1] - poses[24][1]])
    mod_rhu_kne = np.sqrt(rhu_kne_vec.dot(rhu_kne_vec)) + 1e-6
    mod_rkn_ank = np.sqrt(rkn_ank_vec.dot(rkn_ank_vec)) + 1e-6
    mod_rsh_hu = np.sqrt(rsh_hu_vec.dot(rsh_hu_vec)) + 1e-6
    theta_rk = math.acos(rhu_kne_vec.dot(rkn_ank_vec) / mod_rhu_kne / mod_rkn_ank)
    theta_rh = math.acos(rsh_hu_vec.dot(rhu_kne_vec) / mod_rsh_hu / mod_rhu_kne)

    kf_rk.update([theta_rk])
    kf_rh.update([theta_rh])
    theta_rk = kf_rk.state[0]
    theta_rh = kf_rh.state[0]

    roRThigh = alpha * (np.rad2deg(theta_rk) / 3)
    roRShoulderHip = alpha * ((180 - np.rad2deg(theta_rh)) / 3)

    lhu_kne_vec = np.array([poses[25][0] - poses[23][0], poses[25][1] - poses[23][1]])
    lkn_ank_vec = np.array([poses[27][0] - poses[25][0], poses[27][1] - poses[25][1]])
    lsh_hu_vec = np.array([poses[11][0] - poses[23][0], poses[11][1] - poses[23][1]])
    mod_lhu_kne = np.sqrt(lhu_kne_vec.dot(lhu_kne_vec)) + 1e-6
    mod_lkn_ank = np.sqrt(lkn_ank_vec.dot(lkn_ank_vec)) + 1e-6
    mod_lsh_hu = np.sqrt(lsh_hu_vec.dot(lsh_hu_vec)) + 1e-6
    theta_lk = math.acos(lhu_kne_vec.dot(lkn_ank_vec) / mod_lhu_kne / mod_lkn_ank)
    theta_lh = math.acos(lsh_hu_vec.dot(lhu_kne_vec) / mod_lsh_hu / mod_lhu_kne)

    kf_lk.update([theta_lk])
    kf_lh.update([theta_lh])
    theta_lk = kf_lk.state[0]
    theta_lh = kf_lh.state[0]

    roLThigh = alpha * (np.rad2deg(theta_lk) / 3)
    roLShoulderHip = alpha * ((180 - np.rad2deg(theta_lh)) / 3)

    return roRThigh, roLThigh, roRShoulderHip, roLShoulderHip


def get_arm_ro(poses):
    lse_vec = np.array([poses[13][0] - poses[11][0], poses[13][1] - poses[11][1]])
    lew_vec = np.array([poses[13][0] - poses[15][0], poses[13][1] - poses[15][1]])
    lhs_vec = np.array([poses[11][0] - poses[23][0], poses[11][1] - poses[23][1]])
    mod_lse = np.sqrt(lse_vec.dot(lse_vec)) + 1e-6
    mod_lew = np.sqrt(lew_vec.dot(lew_vec)) + 1e-6
    mod_lhs = np.sqrt(lhs_vec.dot(lhs_vec)) + 1e-6
    tmp = lse_vec.dot(lew_vec) / mod_lse / mod_lew
    tem = lhs_vec.dot(lse_vec) / mod_lhs / mod_lse
    theta_le = math.acos(tmp)
    theta_ls = math.acos(tem)

    kf_le.update([theta_le])
    kf_ls.update([theta_ls])
    theta_le = kf_le.state[0]
    theta_ls = kf_ls.state[0]

    roL4Arm = alpha * ((180 - np.rad2deg(theta_le)) / 10)
    roLUArm = alpha * ((180 - np.rad2deg(theta_ls)) / 10)

    rse_vec = np.array([poses[14][0] - poses[12][0], poses[14][1] - poses[12][1]])
    rew_vec = np.array([poses[14][0] - poses[16][0], poses[14][1] - poses[16][1]])
    rhs_vec = np.array([poses[12][0] - poses[24][0], poses[12][1] - poses[24][1]])
    mod_rse = np.sqrt(rse_vec.dot(rse_vec)) + 1e-6
    mod_rew = np.sqrt(rew_vec.dot(rew_vec)) + 1e-6
    mod_rhs = np.sqrt(rhs_vec.dot(rhs_vec)) + 1e-6
    tmp = rse_vec.dot(rew_vec) / mod_rse / mod_rew
    tem = rhs_vec.dot(rse_vec) / mod_rhs / mod_rse
    theta_re = math.acos(tmp)
    theta_rs = math.acos(tem)

    kf_re.update([theta_re])
    kf_rs.update([theta_rs])
    theta_re = kf_re.state[0]
    theta_rs = kf_rs.state[0]

    roR4Arm = alpha * ((180 - np.rad2deg(theta_re)) / 10)
    roRUArm = alpha * ((180 - np.rad2deg(theta_rs)) / 10)

    return roL4Arm, roLUArm, roR4Arm, roRUArm


def main():
    global s, roL4Arm, roLUArm, roR4Arm, roLShoulderHip, roRUArm, roLThigh, roRShoulderHip, roRThigh, body_yaw

    cap = cv2.VideoCapture(args.cam)

    face_detector = FaceMeshDetector()
    pose_detector = PoseMeshDetector()
    success, img = cap.read()

    pose_estimator = PoseEstimator((img.shape[0], img.shape[1]))
    image_points = np.zeros((pose_estimator.model_points_full.shape[0], 2))

    iris_image_points = np.zeros((10, 2))
    rotation_calculator = BodyRotationCalculator(alpha=alpha)

    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    eyes_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    eye_left_stabilizer = Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1)

    eye_right_stabilizer = Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1)

    mouth_dist_stabilizer = Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1
    )

    body_yaw_stabilizer = Stabilizer(state_num=2, measure_num=1, cov_process=1e-5, cov_measure=1e-2)

    if args.connect:
        s = init_TCP()

    while cap.isOpened():
        success, img = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pose_mesh, poses = pose_detector.find_pose_mesh(imgRGB)
        cv2.imshow("Pose Landmark", img_pose_mesh)

        img_facemesh, faces = face_detector.findFaceMesh(img)
        img = cv2.flip(img, 1)

        if faces and poses:
            roL4Arm, roLUArm, roR4Arm, roRUArm = get_arm_ro(poses)
            roRThigh, roLThigh, roRShoulderHip, roLShoulderHip = get_leg_ro(poses)
            body_yaw = rotation_calculator.calculate_body_rotation(poses)
            body_yaw_stabilizer.update([body_yaw])
            body_yaw = body_yaw_stabilizer.state[0]
            for i in range(len(image_points)):
                image_points[i, 0] = faces[0][i][0]
                image_points[i, 1] = faces[0][i][1]

            for j in range(len(iris_image_points)):
                iris_image_points[j, 0] = faces[0][j + 468][0]
                iris_image_points[j, 1] = faces[0][j + 468][1]

            pose = pose_estimator.solve_pose_by_all_points(image_points)

            x_ratio_left, y_ratio_left = FacialFeatures.detect_iris(image_points, iris_image_points, Eyes.LEFT)
            x_ratio_right, y_ratio_right = FacialFeatures.detect_iris(image_points, iris_image_points, Eyes.RIGHT)

            eye_left = FacialFeatures.eye_aspect_ratio(image_points, Eyes.LEFT)
            eye_right = FacialFeatures.eye_aspect_ratio(image_points, Eyes.RIGHT)

            eye_left_stabilizer.update([eye_left])
            eye_right_stabilizer.update([eye_right])
            eye_left = eye_left_stabilizer.state[0]
            eye_right = eye_right_stabilizer.state[0]

            pose_eye = [eye_left, eye_right, x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right]

            mar = FacialFeatures.mouth_aspect_ratio(image_points)
            mouth_distance = FacialFeatures.mouth_distance(image_points)

            steady_pose = []
            pose_np = np.array(pose).flatten()

            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                steady_pose.append(ps_stb.state[0])

            steady_pose = np.reshape(steady_pose, (-1, 3))

            steady_pose_eye = []
            for value, ps_stb in zip(pose_eye, eyes_stabilizers):
                ps_stb.update([value])
                steady_pose_eye.append(ps_stb.state[0])

            mouth_dist_stabilizer.update([mouth_distance])
            steady_mouth_dist = mouth_dist_stabilizer.state[0]

            roll = np.clip(np.degrees(steady_pose[0][1]), -90, 90)
            pitch = np.clip(-(180 + np.degrees(steady_pose[0][0])), -90, 90)
            yaw = np.clip(np.degrees(steady_pose[0][2]), -90, 90)
            print(steady_pose)
            print("Roll: %.2f, Pitch: %.2f, Yaw: %.2f" % (roll, pitch, yaw))

            if args.connect:
                send_info_to_unity(s,
                                   (roL4Arm, roLUArm, roR4Arm, roRUArm, roLShoulderHip, roLThigh, roRShoulderHip,
                                    roRThigh, body_yaw, roll, pitch, yaw,
                                    eye_left, eye_right, x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right,
                                    mar, mouth_distance)
                                   )

            if args.debug:
                print_debug_msg((roL4Arm, roLUArm, roR4Arm, roRUArm, roLShoulderHip, roLThigh, roRShoulderHip, roRThigh,
                                 body_yaw, roll, pitch, yaw,
                                 eye_left, eye_right, x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right,
                                 mar, mouth_distance))

            pose_estimator.draw_axes(img_facemesh, steady_pose[0], steady_pose[1])

        else:
            pose_estimator = PoseEstimator((img_facemesh.shape[0], img_facemesh.shape[1]))

        cv2.imshow('Facial landmark', img_facemesh)
        cv2.imshow("Pose Landmark", img_pose_mesh)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--connect", action="store_true",
                        help="connect to unity character",
                        default=False)

    parser.add_argument("--port", type=int,
                        help="specify the port of the connection to unity. Have to be the same as in Unity",
                        default=5066)

    parser.add_argument("--cam", type=int,
                        help="specify the camera number if you have multiple cameras",
                        default=0)

    parser.add_argument("--debug", action="store_true",
                        help="showing raw values of detection in the terminal",
                        default=False)

    args = parser.parse_args()

    main()
