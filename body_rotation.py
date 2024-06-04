import numpy as np
import math


class BodyRotationCalculator:
    def __init__(self, alpha=0.6):
        self.alpha = alpha

    def calculate_body_rotation(self, poses):
        left_shoulder = np.array([poses[11][0], poses[11][1]])
        right_shoulder = np.array([poses[12][0], poses[12][1]])
        left_hip = np.array([poses[23][0], poses[23][1]])
        right_hip = np.array([poses[24][0], poses[24][1]])

        mid_shoulder = (left_shoulder + right_shoulder) / 2
        mid_hip = (left_hip + right_hip) / 2

        # 计算肩膀和胯部连线的向量
        vector_shoulder_to_hip = mid_hip - mid_shoulder
        print("Vector from shoulder to hip:", vector_shoulder_to_hip)

        # 垂直向下的向量
        down_vector = np.array([0, -1])
        if mid_shoulder[0] > mid_hip[0]:
            # 计算向量之间的夹角
            angle = self._calculate_angle_between_vectors(vector_shoulder_to_hip, down_vector) - 180
        else:
            angle = 180 - self._calculate_angle_between_vectors(vector_shoulder_to_hip, down_vector)
        # 将角度映射到[-10, 10]范围
        mapped_angle = np.clip(self.alpha*angle, -10, 10)

        print("Angle: %.2f, Mapped Angle: %.2f" % (angle, mapped_angle))
        return mapped_angle

    def _calculate_angle_between_vectors(self, v1, v2):
        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)
        dot_product = np.dot(unit_v1, unit_v2)
        angle = np.degrees(np.arccos(dot_product))
        return angle

    def _map_to_range(self, value, left_min, left_max, right_min, right_max):
        # 线性映射函数
        left_span = left_max - left_min
        right_span = right_max - right_min
        value_scaled = (value - left_min) / left_span
        return right_min + (value_scaled * right_span)

# 示例使用
