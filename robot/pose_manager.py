from robot.devices.gps import Gps
from robot.devices.gyroscope import Gyroscope
from data_structures.angle import Angle
from data_structures.vectors import Position2D

from flags import SHOW_DEBUG


class PoseManager:
    """
    로봇의 위치(Position)와 방향(Orientation)을 통합 관리하는 클래스입니다.

    GPS와 자이로스코프 두 센서를 상황에 따라 선택적으로 사용합니다.
    - 직진 중 (회전 없음, 고속): GPS가 더 정확 → GPS 사용
    - 회전 중 또는 늪지대: GPS 오차 큼 → 자이로스코프 사용
    이 선택은 robot_is_going_straight() 조건으로 자동 판단하거나 수동 지정할 수 있습니다.
    """
    GPS = 0        # GPS 센서 식별자
    GYROSCOPE = 1  # 자이로스코프 센서 식별자

    def __init__(self, gps: Gps, gyroscope: Gyroscope, position_offsets=Position2D(0, 0)) -> None:
        # 직진 중이라고 판단하는 최대 각속도 임계값
        self.maximum_angular_velocity_for_gps = Angle(1, Angle.DEGREES)
        # 흔들림(shaky) 판단에 사용하는 각속도 변화량 임계값
        self.maximum_angular_velocity_change_for_shaky = Angle(1, Angle.DEGREES)

        self.gps = gps
        self.gyroscope = gyroscope

        self.orientation = Angle(0)
        self.previous_orientation = Angle(0)

        self.__position = Position2D(0, 0)
        self.__previous_position = Position2D(0, 0)

        self.orientation_sensor = self.GYROSCOPE          # 기본값: 자이로스코프 사용
        self.previous_orientation_sensor = self.GYROSCOPE
        self.automatically_decide_orientation_sensor = True  # True면 자동 선택, False면 수동 지정

        self.position_offsets = position_offsets  # 시작점 오프셋 보정값

        self.shaky_threshold = Angle(5, unit=Angle.DEGREES)  # 흔들림 판단 각도 임계값

    def update(self, average_wheel_velocity, wheel_velocity_difference):
        """
        매 타임스텝 호출: GPS와 자이로스코프를 갱신하고 현재 위치·방향을 계산합니다.
        average_wheel_velocity: 양 바퀴 평균 각속도 (직진 여부 판단용)
        wheel_velocity_difference: 좌우 바퀴 속도 차 (회전 여부 판단용)
        """
        self.gps.update()
        self.gyroscope.update()

        # GPS로 전역 위치 갱신
        self.__previous_position = self.__position
        self.__position = self.gps.get_position()

        # 조건에 따라 방향 센서 자동 선택
        if self.automatically_decide_orientation_sensor:
            self.decide_orientation_sensor(average_wheel_velocity, wheel_velocity_difference)

        self.previous_orientation = self.orientation
        self.calculate_orientation()

    def decide_orientation_sensor(self, average_wheel_velocity, wheel_velocity_difference):
        """직진 중이면 GPS, 회전 중이면 자이로스코프를 방향 센서로 선택합니다."""
        if self.robot_is_going_straight(average_wheel_velocity, wheel_velocity_difference):
            self.orientation_sensor = self.GPS
        else:
            self.orientation_sensor = self.GYROSCOPE
            # 직진이 아니면 GPS 방향 기준선을 초기화하여, 기준선이 회전 구간을
            # 가로질러 잘못된 방향을 계산하는 것을 방지합니다.
            self.gps.reset_orientation_baseline()

    def robot_is_going_straight(self, average_wheel_velocity, wheel_velocity_difference) -> bool:
        """로봇이 직진 중인지 판단: 각속도 작고, 속도 충분하고, 좌우 속도 차 작으면 직진"""
        return self.gyroscope.get_angular_velocity() < self.maximum_angular_velocity_for_gps and \
               average_wheel_velocity >= 1 and \
               wheel_velocity_difference < 1

    def calculate_orientation(self):
        """선택된 센서로 현재 방향 각도를 계산합니다. GPS 사용 시 자이로도 동기화합니다."""
        gps_orientation = self.gps.get_orientation()

        if self.orientation_sensor == self.GYROSCOPE or gps_orientation is None:
            self.orientation = self.gyroscope.get_orientation()
            if SHOW_DEBUG: print(f"[방향센서:pose_manager.calculate_orientation] 자이로스코프 사용: 방향={self.orientation.degrees:.1f}°, 각속도={self.gyroscope.get_angular_velocity().degrees:.2f}°/step")
        else:
            self.orientation = gps_orientation
            # GPS 값으로 자이로스코프를 교정하여 드리프트를 보정
            self.gyroscope.set_orientation(self.orientation)
            if SHOW_DEBUG: print(f"[방향센서:pose_manager.calculate_orientation] GPS 사용: 방향={self.orientation.degrees:.1f}°, 위치=({self.__position.x:.4f},{self.__position.y:.4f})m, 자이로 동기화 완료")

    @property
    def position(self):
        """오프셋이 적용된 현재 위치를 반환합니다."""
        return self.__position + self.position_offsets

    @property
    def raw_position(self):
        """오프셋이 적용되지 않은 GPS 원시 위치를 반환합니다 (서버 전송용)."""
        return self.__position

    @property
    def previous_position(self):
        """오프셋이 적용된 이전 타임스텝의 위치를 반환합니다."""
        return self.__previous_position + self.position_offsets

    def is_shaky(self) -> bool:
        """
        로봇이 심하게 흔들리거나 급격히 회전 중인지 판단합니다.
        True이면 라이다 데이터 신뢰도가 낮으므로 맵 업데이트를 일부 생략합니다.
        - 방향 변화가 크거나
        - 각속도 방향이 바뀌었거나
        - 각속도 크기 변화가 크면 → 흔들리는 것으로 판단
        """
        high_orient_diff = self.orientation.get_absolute_distance_to(self.previous_orientation) > self.shaky_threshold
        changed_direction = self.gyroscope.angular_velocity * self.gyroscope.previous_angular_velocity < 0
        high_angular_velocity_difference = self.gyroscope.previous_angular_velocity.get_absolute_distance_to(
            self.gyroscope.angular_velocity) > self.maximum_angular_velocity_change_for_shaky

        return high_orient_diff or changed_direction or high_angular_velocity_difference
