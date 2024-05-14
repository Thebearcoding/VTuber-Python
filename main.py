import cv2
import mediapipe as mp
from pose_landmark import PoseMeshDetector
from argparse import ArgumentParser
import socket
import math
import sys
import numpy as np

port = 5066
alpha = 0.6


def init_tcp():
    port = args.port
    address = ('127.0.0.1', port)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(address)
        # print(socket.gethostbyname(socket.gethostname()) + "::" + str(port))
        print("Connected to address:", socket.gethostbyname(socket.gethostname()) + ":" + str(port))
        return s
    except OSError as e:
        print("Error while connecting :: %s" % e)

        # quit the script if connection fails (e.g. Unity server side quits suddenly)
        sys.exit()


def send_info_to_unity(s, info):
    msg = '%.4f ' * len(info) % info
    try:
        print(msg)
        s.send(bytes(msg, "utf-8"))
    except socket.error as e:
        print("error while sending :: " + str(e))

        # quit the script if connection fails (e.g. Unity server side quits suddenly)
        sys.exit()


def print_debug_msg(info):
    msg = '%.4f ' * len(info) % info
    # print(msg)


def get_leg_ro(poses, alpha):
    # 右腿
    rhu_kne_vec = np.array([poses[26][0] - poses[24][0], poses[26][1] - poses[24][1]])  # 右臀膝向量
    rkn_ank_vec = np.array([poses[28][0] - poses[26][0], poses[28][1] - poses[26][1]])  # 右膝踝向量
    rsh_hu_vec = np.array([poses[12][0] - poses[24][0], poses[12][1] - poses[24][1]])  # 右肩到右臀向量
    mod_rhu_kne = np.sqrt(rhu_kne_vec.dot(rhu_kne_vec)) + 1e-6
    mod_rkn_ank = np.sqrt(rkn_ank_vec.dot(rkn_ank_vec)) + 1e-6
    mod_rsh_hu = np.sqrt(rsh_hu_vec.dot(rsh_hu_vec)) + 1e-6
    theta_rk = math.acos(rhu_kne_vec.dot(rkn_ank_vec) / mod_rhu_kne / mod_rkn_ank)
    theta_rh = math.acos(rsh_hu_vec.dot(rhu_kne_vec) / mod_rsh_hu / mod_rhu_kne)
    roRThigh = alpha * ((180 - np.rad2deg(theta_rk)) / 10)
    roRShoulderHip = alpha * ((180 - np.rad2deg(theta_rh)) / 10)

    print("theta_rhkna=" + str(np.rad2deg(theta_rk)) + ", roRThigh=" + str(roRThigh))
    print("theta_rshu=" + str(np.rad2deg(theta_rh)) + ", roRShoulderHip=" + str(roRShoulderHip))

    # 左腿
    lhu_kne_vec = np.array([poses[25][0] - poses[23][0], poses[25][1] - poses[23][1]])  # 左臀膝向量
    lkn_ank_vec = np.array([poses[27][0] - poses[25][0], poses[27][1] - poses[25][1]])  # 左膝踝向量
    lsh_hu_vec = np.array([poses[11][0] - poses[23][0], poses[11][1] - poses[23][1]])  # 左肩到左臀向量
    mod_lhu_kne = np.sqrt(lhu_kne_vec.dot(lhu_kne_vec)) + 1e-6
    mod_lkn_ank = np.sqrt(lkn_ank_vec.dot(lkn_ank_vec)) + 1e-6
    mod_lsh_hu = np.sqrt(lsh_hu_vec.dot(lsh_hu_vec)) + 1e-6
    theta_lk = math.acos(lhu_kne_vec.dot(lkn_ank_vec) / mod_lhu_kne / mod_lkn_ank)
    theta_lh = math.acos(lsh_hu_vec.dot(lhu_kne_vec) / mod_lsh_hu / mod_lhu_kne)
    roLThigh = alpha * ((180 - np.rad2deg(theta_lk)) / 10)
    roLShoulderHip = alpha * ((180 - np.rad2deg(theta_lh)) / 10)

    print("theta_lhkna=" + str(np.rad2deg(theta_lk)) + ", roLThigh=" + str(roLThigh))
    print("theta_lshu=" + str(np.rad2deg(theta_lh)) + ", roLShoulderHip=" + str(roLShoulderHip))

    return roRThigh, roLThigh, roRShoulderHip, roLShoulderHip


def get_arm_ro(poses):
    # left arm
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
    print("theta_lsew=" + str(np.rad2deg(theta_le)) + " theta_lhse=" + str(np.rad2deg(theta_ls)))
    # print(np.rad2deg(theta_s))
    roL4Arm = alpha * ((180 - np.rad2deg(theta_le)) / 10)
    roLUArm = alpha * ((180 - np.rad2deg(theta_ls)) / 10)
    print("roL4Arm=" + str(roL4Arm) + " " + "roLUArm=" + str(roLUArm))
    # right arm
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
    print("theta_rsew=" + str(np.rad2deg(theta_re)) + " theta_rhse=" + str(np.rad2deg(theta_rs)))
    roR4Arm = alpha * ((180 - np.rad2deg(theta_re)) / 10)
    roRUArm = alpha * ((180 - np.rad2deg(theta_rs)) / 10)
    print("roR4Arm=" + str(roR4Arm) + " " + "roRUArm=" + str(roRUArm))
    return roL4Arm, roLUArm, roR4Arm, roRUArm


def main():
    cap = cv2.VideoCapture(args.cam)
    detector = PoseMeshDetector()
    success, img = cap.read()
    s = init_tcp()
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pose_mesh, poses = detector.find_pose_mesh(imgRGB)
        if poses:
            roL4Arm, roLUArm, roR4Arm, roRUArm = get_arm_ro(poses)
            if args.connect:
                send_info_to_unity(s, (roL4Arm, roLUArm, roR4Arm, roRUArm))
        # if poses:
        #     # print(poses)
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
