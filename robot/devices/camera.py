import numpy as np
from robot.devices.sensor import TimedSensor
import cv2 as cv

from flow_control.step_counter import StepCounter

from data_structures.angle import Angle

from dataclasses import dataclass

import math

@dataclass
class CameraData:
    """카메라의 현재 상태 메타데이터: 해상도, 시야각, 공간적 위치 등을 담습니다."""
    height: int                           # 이미지 세로 픽셀 수
    width: int                            # 이미지 가로 픽셀 수
    vertical_fov: Angle                   # 수직 시야각
    horizontal_fov: Angle                 # 수평 시야각
    relative_vertical_orientation: Angle  # 로봇 기준 수직 방향
    relative_horizontal_orientation: Angle # 로봇 기준 수평 방향 (정면=0, 우=270, 좌=90)
    vertical_orientation: Angle           # 전역 기준 수직 방향
    horizontal_orientation: Angle         # 전역 기준 수평 방향
    distance_from_center: float           # 로봇 중심에서 카메라까지의 거리(m)

class CameraImage:
    """캡처된 카메라 이미지와 해당 시점의 메타데이터를 함께 보관하는 컨테이너입니다."""
    def __init__(self) -> None:
        self.image: np.ndarray = None   # BGR+Alpha 형식의 numpy 이미지 배열
        self.data: CameraData = None    # 이미지 촬영 당시의 카메라 메타데이터

class Camera(TimedSensor):
    """
    Webots 카메라 장치를 관리하는 클래스입니다.

    StepCounter를 통해 n 타임스텝에 한 번만 이미지를 갱신하여 성능을 최적화합니다.
    로봇에는 전방(center), 우측(right), 좌측(left) 3개의 카메라가 있습니다.
    """
    def __init__(self, webots_device, time_step, step_counter: StepCounter,
                 orientation: Angle, distance_from_center: float, rotate180=False):
        super().__init__(webots_device, time_step, step_counter)
        self.rotate180 = rotate180                  # 좌측 카메라처럼 180도 회전이 필요한 경우
        self.height = self.device.getHeight()       # 이미지 세로 해상도
        self.width = self.device.getWidth()         # 이미지 가로 해상도
        self.horizontal_fov = Angle(self.device.getFov())  # 수평 시야각(rad)
        # 수직 FOV는 수평 FOV와 종횡비로부터 계산
        self.vertical_fov = Angle(2 * math.atan(math.tan(self.horizontal_fov * 0.5) * (self.height / self.width)))
        self.image = CameraImage()

        self.horizontal_orientation_in_robot = orientation  # 로봇 기준 장착 방향
        self.vertical_orientation_in_robot = Angle(0)

        self.horizontal_orientation = orientation   # 전역 기준 현재 방향 (로봇 회전 반영)
        self.vertical_orientation = Angle(0)
        self.distance_from_center = distance_from_center

    def get_image(self):
        """StepCounter 주기가 돌아왔을 때만 이미지를 반환합니다."""
        if self.step_counter.check():
            return self.image

    def get_last_image(self):
        """주기와 무관하게 마지막으로 캡처된 이미지를 반환합니다."""
        return self.image

    def get_data(self):
        """현재 카메라 상태의 메타데이터(CameraData)를 생성하여 반환합니다."""
        data = CameraData(self.height,
                          self.width,
                          self.vertical_fov,
                          self.horizontal_fov,
                          self.vertical_orientation_in_robot,
                          self.horizontal_orientation_in_robot,
                          self.vertical_orientation,
                          self.horizontal_orientation,
                          self.distance_from_center)
        return data

    def update(self, robot_orientation: Angle):
        """
        매 타임스텝 호출:
        - 전역 방향 = 로봇 기준 장착 방향 + 로봇 현재 방향
        - StepCounter 주기마다 Webots 버퍼에서 이미지를 읽어 numpy 배열로 변환합니다.
        """
        super().update()

        # 로봇이 회전하면 카메라의 전역 방향도 함께 갱신
        self.horizontal_orientation = self.horizontal_orientation_in_robot + robot_orientation

        if self.step_counter.check():
            image_data = self.device.getImage()
            # RGBA 형식으로 버퍼에서 읽어 (height, width, 4) 배열로 변환
            self.image.image = np.array(np.frombuffer(image_data, np.uint8).reshape((self.height, self.width, 4)))

            # 좌측 카메라는 180도 회전 보정
            if self.rotate180:
                self.image.image = np.rot90(self.image.image, 2, (0, 1))

            self.image.orientation = self.horizontal_orientation
            self.image.data = self.get_data()
