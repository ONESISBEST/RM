from data_structures.angle import Angle
from robot.devices.sensor import Sensor
import copy


class Gyroscope(Sensor):
    """
    Webots 자이로스코프를 이용해 로봇의 전역 회전 각도를 추적하는 클래스입니다.

    각속도(rad/s)를 누적 적분하여 현재 절대 각도를 계산합니다.
    GPS가 부정확한 상황(회전 중, 늪지대)에서 주로 사용됩니다.
    """
    def __init__(self, webots_device, index, time_step):
        super().__init__(webots_device, time_step)
        self.index = index                      # 자이로 센서 배열에서 Y축 인덱스 (수직축 회전)
        self.orientation = Angle(0)             # 누적된 현재 절대 각도
        self.angular_velocity = Angle(0)        # 현재 각속도 (= 이번 스텝 회전량)
        self.previous_angular_velocity = Angle(0)  # 이전 스텝 각속도 (흔들림 감지에 사용)

    def update(self):
        """매 타임스텝 호출: 각속도를 읽어 현재 방향에 누적합니다."""
        time_elapsed = self.time_step / 1000   # ms → 초 변환
        sensor_y_value = self.device.getValues()[self.index]   # Y축 각속도(rad/s)
        self.previous_angular_velocity = copy.copy(self.angular_velocity)
        self.angular_velocity = Angle(sensor_y_value * time_elapsed)  # 이번 스텝 회전량
        self.orientation += self.angular_velocity  # 누적 각도 업데이트
        self.orientation.normalize()               # 0 ~ 2π 범위로 정규화

    def get_angular_velocity(self):
        """방향 부호 없이 각속도의 크기만 반환합니다."""
        return abs(self.angular_velocity)

    def get_orientation(self):
        """현재 누적 각도(절대 방향)를 반환합니다."""
        return self.orientation

    def set_orientation(self, angle):
        """GPS 기반 각도로 자이로스코프 영점을 교정할 때 사용합니다."""
        self.orientation = angle
