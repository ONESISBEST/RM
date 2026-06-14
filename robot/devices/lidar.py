import math

import utilities
from utilities import divide_into_chunks

from robot.devices.sensor import TimedSensor
from data_structures.angle import Angle
from data_structures.vectors import Vector2D

class Lidar(TimedSensor):
    """
    Webots 라이다 센서를 관리하는 클래스입니다.

    360도 수평 스캔으로 주변 장애물까지의 거리를 측정하고,
    결과를 2D 포인트 클라우드(x, y 좌표 목록)로 변환합니다.

    - in_bounds_point_cloud: 감지 범위 안에 있는 장애물 점들 (→ 벽으로 등록)
    - out_of_bounds_point_cloud: 감지 범위를 초과한 방향 점들 (→ 열린 공간으로 등록)
    - distance_detections: 방향 벡터(Vector2D) 목록 (→ 조난자 위치 추정에 사용)
    """
    def __init__(self, webots_device, time_step, step_counter, layers_used=range(4)):
        super().__init__(webots_device, time_step, step_counter)
        self.x = 0
        self.y = 0
        self.z = 0
        self.orientation = Angle(0)   # 로봇의 현재 방향 (포인트 좌표 계산에 사용)

        self.horizontal_fov = self.device.getFov()                  # 수평 시야각(rad), 보통 2π(360도)
        self.vertical_fov = self.device.getVerticalFov()            # 수직 시야각(rad)
        self.horizontal_resolution = self.device.getHorizontalResolution()  # 수평 방향 감지 포인트 수
        self.vertical_resolution = self.device.getNumberOfLayers()          # 수직 레이어 수

        self.radian_per_detection_horizontally = self.horizontal_fov / self.horizontal_resolution
        self.radian_per_layer_vertically = self.vertical_fov / self.vertical_resolution

        self.rotation_offset = 0

        self.max_detection_distance = 0.06 * 8   # 최대 감지 거리(m) = 약 0.48m
        self.min_detection_distance = 0.06 * 0.6 # 최소 감지 거리(m) = 약 0.036m (너무 가까운 노이즈 제거)

        self.is_point_close = False              # 로봇 바로 앞에 장애물이 있는지 여부
        self.is_point_close_threshold = 0.03     # '가깝다'고 판단하는 거리 임계값(m)
        self.is_point_close_range = (0, 360)     # 가까움 판단에 사용할 각도 범위(도)

        self.distance_bias = 0.005               # 라이다 거리 보정값(m)

        self.layers_used = layers_used           # 처리할 수직 레이어 번호 목록 (예: (2,))

        self.__point_cloud = None
        self.__out_of_bounds_point_cloud = None
        self.__distance_detections = None

    def get_point_cloud(self):
        """StepCounter 주기일 때 감지 범위 내 포인트 클라우드를 반환합니다."""
        if self.step_counter.check():
            return self.__point_cloud

    def get_out_of_bounds_point_cloud(self):
        """StepCounter 주기일 때 감지 범위 초과(열린 방향) 포인트 클라우드를 반환합니다."""
        if self.step_counter.check():
            return self.__out_of_bounds_point_cloud

    def get_detections(self):
        """StepCounter 주기일 때 방향+거리 벡터(Vector2D) 목록을 반환합니다."""
        if self.step_counter.check():
            return self.__distance_detections

    def set_orientation(self, angle):
        """로봇 방향이 변경될 때 호출하여 포인트 좌표 변환에 반영합니다."""
        self.orientation = angle

    def update(self):
        """매 타임스텝 호출: StepCounter 주기마다 포인트 클라우드를 갱신합니다."""
        super().update()
        if self.step_counter.check():
            self.__update_point_clouds()

    def __update_point_clouds(self):
        """라이다 원시 데이터를 읽어 포인트 클라우드로 변환하는 핵심 처리 메서드입니다."""
        self.is_point_close = False

        self.__point_cloud = []
        self.__out_of_bounds_point_cloud = []
        self.__distance_detections = []

        total_depth_array = self.device.getRangeImage()
        # 전체 1D 배열을 레이어별로 분할 (각 레이어 = 수평 360도 스캔 한 줄)
        total_depth_array = divide_into_chunks(total_depth_array, self.horizontal_resolution)

        for layer_number, layer_depth_array in enumerate(total_depth_array):
            if layer_number not in self.layers_used:
                continue

            vertical_angle = layer_number * self.radian_per_layer_vertically + self.vertical_fov / 2
            # 로봇 방향을 반영하여 수평 각도 시작점 설정
            horizontal_angle = self.rotation_offset + ((2 * math.pi) - self.orientation.radians)

            for item in layer_depth_array:
                if item >= self.max_detection_distance or item == float("inf") or item == float("inf") * -1:
                    # 범위 초과 → 열린 공간 방향으로 max 거리 포인트 등록
                    distance = self.__normalize_distance(self.max_detection_distance, vertical_angle)
                    point = utilities.getCoordsFromRads(horizontal_angle, distance)
                    self.__out_of_bounds_point_cloud.append(self.__normalize_point(point))
                else:
                    if item >= self.min_detection_distance:
                        # 유효한 감지 → 포인트 클라우드에 추가
                        distance = self.__normalize_distance(item, vertical_angle)
                        point = utilities.getCoordsFromRads(horizontal_angle, distance)
                        self.__point_cloud.append(self.__normalize_point(point))

                        v = Vector2D(Angle(horizontal_angle), distance)
                        v.direction = Angle(math.pi) - v.direction
                        v.direction.normalize()
                        self.__distance_detections.append(v)

                        # 전방 근거리 장애물 감지
                        if self.__in_range_for_close_point(horizontal_angle) and distance < self.is_point_close_threshold:
                            self.is_point_close = True

                horizontal_angle += self.radian_per_detection_horizontally

        # 빈 클라우드 방지 (mapper가 None 처리 하지 않도록 더미 포인트 삽입)
        if len(self.__out_of_bounds_point_cloud) == 0:
            self.__out_of_bounds_point_cloud = [[0, 0]]

        if len(self.__point_cloud) == 0:
            self.__point_cloud = [[0, 0]]

    def __in_range_for_close_point(self, horizontal_angle):
        """지정된 각도 범위 내에 있는지 확인 (전방 근거리 감지용)."""
        return utilities.degsToRads(self.is_point_close_range[0]) > horizontal_angle > utilities.degsToRads(self.is_point_close_range[1])

    def __normalize_distance(self, distance, vertical_angle):
        """수직 경사각 보정 + 거리 오프셋 보정을 적용하여 2D 평면 거리로 변환합니다."""
        distance = distance * math.cos(vertical_angle)  # 수직 경사 제거 → 수평 거리만 추출
        distance += self.distance_bias                  # 하드웨어 오프셋 보정
        return distance

    def __normalize_point(self, point):
        """Webots 좌표계(Y축 반전)에서 일반 좌표계로 변환합니다."""
        return [point[0], point[1] * -1]
