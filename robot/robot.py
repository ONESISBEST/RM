from controller import Robot as WebotsRobot

from flow_control.step_counter import StepCounter

from data_structures.angle import Angle
from data_structures.vectors import Position2D, Vector2D

# 디바이스 모듈들
from robot.devices.wheel import Wheel
from robot.devices.camera import Camera
from robot.devices.lidar import Lidar
from robot.devices.gps import Gps
from robot.devices.gyroscope import Gyroscope
from robot.devices.comunicator import Comunicator

from robot.pose_manager import PoseManager

from robot.drive_base import DriveBase, Criteria

import cv2 as cv


class Robot:
    """
    Webots 로봇의 모든 하드웨어를 추상화하는 최상위 인터페이스 클래스입니다.

    이 클래스는 Executor에게 깔끔한 고수준 API를 제공합니다:
    - 위치/방향 (position, orientation)
    - 이동 명령 (move_wheels, rotate_to_angle, move_to_coords)
    - 센서 데이터 (get_point_cloud, get_camera_images)
    - 서버 통신 (comunicator)

    내부적으로는 DriveBase, PoseManager, Lidar, Camera, GPS, Gyroscope 등을
    모두 생성하고 연결합니다.
    """
    def __init__(self, time_step):
        self.time_step = time_step        # 기본 시뮬레이션 타임스텝(ms), 예: 32ms
        self.__start_time = 0             # 미션 시작 시각
        self.__time = 0                   # 현재 시뮬레이션 시각

        self.diameter = 0.074             # 로봇 직경(m) = 7.4cm

        self.robot = WebotsRobot()        # Webots 컨트롤러 API 객체

        # GPS와 자이로스코프 초기화
        self.gps = Gps(self.robot.getDevice("gps"), self.time_step)
        self.gyroscope = Gyroscope(self.robot.getDevice("gyro"), 1, self.time_step)

        # 위치/방향 통합 관리자
        self.pose_manager = PoseManager(self.gps, self.gyroscope)

        # 라이다: 6 타임스텝마다 갱신, 레이어 2번만 사용 (수평 스캔)
        lidar_interval = 6
        self.lidar = Lidar(webots_device = self.robot.getDevice("lidar"),
                           time_step = self.time_step * lidar_interval,
                           step_counter = StepCounter(lidar_interval),
                           layers_used=(2,))

        # 카메라: 3 타임스텝마다 갱신, 전방/우측/좌측 3개
        self.camera_distance_from_center = 0.0310  # 카메라와 로봇 중심 간 거리(m)
        camera_interval = 3
        self.center_camera = Camera(webots_device = self.robot.getDevice("camera1"),
                                    time_step = self.time_step * camera_interval,
                                    step_counter = StepCounter(camera_interval),
                                    orientation=Angle(0, Angle.DEGREES),         # 전방
                                    distance_from_center=self.camera_distance_from_center)

        self.right_camera = Camera(webots_device = self.robot.getDevice("camera2"),
                                   time_step = self.time_step * camera_interval,
                                   step_counter = StepCounter(camera_interval),
                                   orientation=Angle(270, Angle.DEGREES),        # 우측
                                   distance_from_center=self.camera_distance_from_center)

        self.left_camera = Camera(webots_device = self.robot.getDevice("camera3"),
                                  time_step = self.time_step * camera_interval,
                                  step_counter = StepCounter(camera_interval),
                                  orientation=Angle(90, Angle.DEGREES),          # 좌측
                                  distance_from_center=self.camera_distance_from_center,
                                  rotate180=True)                                 # 좌측은 이미지 180도 반전

        # 서버 통신 (에미터 + 리시버)
        self.comunicator = Comunicator(self.robot.getDevice("emitter"),
                                       self.robot.getDevice("receiver"),
                                       self.time_step)

        # 구동 베이스: 최대 각속도 6.28 rad/s
        max_wheel_speed = 6.28
        self.drive_base = DriveBase(
            left_wheel = Wheel(self.robot.getDevice("wheel1 motor"), max_wheel_speed),
            right_wheel = Wheel(self.robot.getDevice("wheel2 motor"), max_wheel_speed),
            max_wheel_velocity = max_wheel_speed)

    def update(self):
        """
        매 타임스텝 호출: 모든 하드웨어 컴포넌트를 순서대로 갱신합니다.
        1. 현재 시뮬레이션 시간 갱신
        2. 위치/방향 갱신 (PoseManager)
        3. DriveBase에 최신 위치/방향 전달
        4. 라이다 방향 설정 및 갱신
        5. 카메라 3개 갱신
        """
        self.__time = self.robot.getTime()

        self.pose_manager.update(self.drive_base.get_wheel_average_angular_velocity(),
                                 self.drive_base.get_wheel_velocity_difference())

        # DriveBase가 현재 위치/방향을 알아야 올바른 이동 제어가 가능
        self.drive_base.orientation = self.orientation
        self.drive_base.position = self.position

        # 라이다는 포인트 좌표 변환에 로봇 방향을 사용
        self.lidar.set_orientation(self.orientation)
        self.lidar.update()

        self.right_camera.update(self.orientation)
        self.left_camera.update(self.orientation)
        self.center_camera.update(self.orientation)

    def do_loop(self):
        """시뮬레이션을 한 타임스텝 전진시킵니다. 시뮬레이션이 실행 중이면 True를 반환합니다."""
        return self.robot.step(self.time_step) != -1

    def set_start_time(self):
        """미션 시작 시각을 기록합니다. 이후 self.time은 경과 시간을 반환합니다."""
        self.__start_time = self.robot.getTime()

    @property
    def time(self):
        """미션 시작 이후 경과 시간(초)을 반환합니다."""
        return self.__time - self.__start_time

    # -------- DriveBase 래퍼 --------
    @property
    def max_wheel_speed(self):
        return self.drive_base.max_wheel_velocity

    def move_wheels(self, left_ratio, right_ratio):
        """좌/우 바퀴를 비율(-1.0~1.0)로 구동합니다."""
        self.drive_base.move_wheels(left_ratio, right_ratio)

    def rotate_to_angle(self, angle, direction=Criteria.CLOSEST):
        """지정 각도(도 단위)로 회전합니다. 완료 시 True 반환."""
        return self.drive_base.rotate_to_angle(Angle(angle, Angle.DEGREES), direction)

    def rotate_slowly_to_angle(self, angle, direction=Criteria.CLOSEST):
        """느린 속도로 지정 각도(도 단위)로 회전합니다."""
        return self.drive_base.rotate_slowly_to_angle(angle, direction)

    def move_to_coords(self, targetPos):
        """지정 좌표까지 이동합니다. 완료 시 True 반환."""
        return self.drive_base.move_to_position(Position2D(targetPos[0], targetPos[1]))

    # -------- 라이다 래퍼 --------
    @property
    def point_is_close(self) -> bool:
        """전방 근거리 장애물 감지 여부를 반환합니다."""
        return self.lidar.is_point_close

    def get_point_cloud(self):
        """감지 범위 내 포인트 클라우드(장애물 위치 목록)를 반환합니다."""
        return self.lidar.get_point_cloud()

    def get_out_of_bounds_point_cloud(self):
        """감지 범위 초과(열린 방향) 포인트 클라우드를 반환합니다."""
        return self.lidar.get_out_of_bounds_point_cloud()

    def get_lidar_detections(self):
        """방향+거리 벡터(Vector2D) 목록을 반환합니다."""
        return self.lidar.get_detections()

    # -------- 카메라 래퍼 --------
    def get_camera_images(self):
        """카메라 갱신 주기일 때 [우측, 전방, 좌측] 3개의 CameraImage를 반환합니다."""
        if self.center_camera.step_counter.check():
            return [self.right_camera.get_image(),
                    self.center_camera.get_image(),
                    self.left_camera.get_image()]

    def get_last_camera_images(self):
        """주기와 무관하게 마지막으로 캡처된 카메라 이미지 3개를 반환합니다."""
        return [self.right_camera.get_last_image(),
                self.center_camera.get_last_image(),
                self.left_camera.get_last_image()]

    # -------- 위치/방향 래퍼 (PoseManager로 위임) --------
    @property
    def position(self):
        """오프셋 보정이 적용된 현재 위치를 반환합니다."""
        return self.pose_manager.position

    @property
    def raw_position(self):
        """GPS 원시 위치를 반환합니다 (서버 통신용)."""
        return self.pose_manager.raw_position

    @property
    def previous_position(self):
        """이전 타임스텝의 위치를 반환합니다 (StuckDetector 사용)."""
        return self.pose_manager.previous_position

    @property
    def position_offsets(self):
        return self.pose_manager.position_offsets

    @position_offsets.setter
    def position_offsets(self, value):
        self.pose_manager.position_offsets = value

    @property
    def orientation(self):
        """현재 방향(Angle 객체)을 반환합니다."""
        return self.pose_manager.orientation

    @property
    def previous_orientation(self):
        return self.pose_manager.previous_orientation

    @property
    def auto_decide_orientation_sensor(self):
        """방향 센서 자동 선택 여부를 반환합니다."""
        return self.pose_manager.automatically_decide_orientation_sensor

    @auto_decide_orientation_sensor.setter
    def auto_decide_orientation_sensor(self, value):
        self.pose_manager.automatically_decide_orientation_sensor = value

    @property
    def orientation_sensor(self):
        return self.pose_manager.orientation_sensor

    @orientation_sensor.setter
    def orientation_sensor(self, value):
        self.pose_manager.orientation_sensor = value

    @property
    def GPS(self):
        """GPS 센서 상수를 반환합니다 (PoseManager.GPS)."""
        return PoseManager.GPS

    @property
    def GYROSCOPE(self):
        """자이로스코프 센서 상수를 반환합니다 (PoseManager.GYROSCOPE)."""
        return PoseManager.GYROSCOPE

    def is_shaky(self):
        """로봇이 심하게 흔들리거나 급격히 회전 중인지 반환합니다."""
        return self.pose_manager.is_shaky()
