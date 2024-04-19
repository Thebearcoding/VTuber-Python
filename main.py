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
    print(msg)


def get_ro(poses):
    # left arm
    lse_vec = np.array([poses[13][0] - poses[11][0], poses[13][1] - poses[11][1]])
    lew_vec = np.array([poses[13][0] - poses[15][0], poses[15][1] - poses[15][1]])
    lhs_vec = np.array([poses[11][0] - poses[23][0], poses[11][1] - poses[23][1]])
    mod_lse = np.sqrt(lse_vec.dot(lse_vec)) + 1e-6
    mod_lew = np.sqrt(lew_vec.dot(lew_vec)) + 1e-6
    mod_lhs = np.sqrt(lhs_vec.dot(lhs_vec)) + 1e-6
    tmp = lse_vec.dot(lew_vec) / mod_lse / mod_lew
    tem = lhs_vec.dot(lse_vec) / mod_lhs / mod_lse
    theta_le = math.acos(tmp)

    theta_ls = math.acos(tem)
    print("theta_sew=" + str(np.rad2deg(theta_le)) + " theta_hse=" + str(np.rad2deg(theta_ls)))
    # print(np.rad2deg(theta_s))
    roL4Arm = alpha * (np.rad2deg(theta_le) / 10)
    roLUArm = alpha * ((180 - np.rad2deg(theta_ls)) / 10)
    print("roL4Arm=" + str(roL4Arm) + " " + "roLUArm=" + str(roLUArm))
    # right arm
    rse_vec = np.array([poses[14][0] - poses[12][0], poses[14][1] - poses[14][1]])
    rew_vec = np.array([poses[14][0] - poses[16][0], poses[14][1] - poses[16][1]])
    rhs_vec = np.array([poses[12][0] - poses[24][0], poses[12][1] - poses[24][1]])
    mod_rse = np.sqrt(rse_vec.dot(rse_vec)) + 1e-5
    mod_rew = np.sqrt(rew_vec.dot(rew_vec)) + 1e-5
    mod_rhs = np.sqrt(rhs_vec.dot(rhs_vec)) + 1e-5
    tmp = rse_vec.dot(rew_vec) / mod_rse / mod_rew
    tem = rhs_vec.dot(rse_vec) / mod_rhs / mod_rse
    theta_re = math.acos(tmp)
    theta_rs = math.acos(tem)
    print("theta_rsew=" + str(np.rad2deg(theta_re)) + " theta_rhse=" + str(np.rad2deg(theta_rs)))
    # print(np.rad2deg(theta_s))
    roR4Arm = alpha * (np.rad2deg(theta_re) / 10)
    roRUArm = alpha * ((180 - np.rad2deg(theta_rs)) / 10)
    print("roR4Arm=" + str(roR4Arm) + " " + "roRUArm=" + str(roRUArm))
    return roL4Arm, roLUArm, roR4Arm, roLUArm


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
            roL4Arm, roLUArm, roR4Arm, roRUArm = get_ro(poses)
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
