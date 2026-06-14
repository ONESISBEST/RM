from utilities import mapVals
from enum import Enum
from data_structures.angle import Angle
from data_structures.vectors import Position2D, Vector2D
import math
from flags import SHOW_DEBUG

from robot.devices.wheel import Wheel

class Criteria(Enum):
    """회전 방향 선택 기준 열거형"""
    LEFT = 1      # 왼쪽(반시계) 방향으로 회전
    RIGHT = 2     # 오른쪽(시계) 방향으로 회전
    CLOSEST = 3   # 현재 각도에서 목표 각도까지 최단 방향으로 회전
    FARTHEST = 4  # 최단 방향의 반대 방향으로 회전

class DriveBase:
    """
    좌우 바퀴를 통합 관리하며 회전과 목표 좌표 이동을 수행하는 구동 베이스 클래스입니다.

    내부에 RotationManager와 SmoothMovementToCoordinatesManager를 보유하고
    이들에게 실제 제어를 위임합니다.
    """
    def __init__(self, left_wheel, right_wheel, max_wheel_velocity) -> None:
        self.max_wheel_velocity = max_wheel_velocity
        self.left_wheel = left_wheel
        self.right_wheel = right_wheel
        # 일반 회전 관리자
        self.rotation_manager = RotationManager(self.left_wheel, self.right_wheel)
        # 느린 회전 관리자 (조난자 정렬 등에 사용)
        self.slow_rotation_manager = RotationManager(self.left_wheel, self.right_wheel)
        self.slow_rotation_manager.max_velocity = 0.4
        self.slow_rotation_manager.max_velocity_cap = 0.4
        # 부드러운 곡선 경로 이동 관리자
        self.movement_manager = SmoothMovementToCoordinatesManager(self.left_wheel, self.right_wheel)

    def move_wheels(self, left_ratio, right_ratio):
        """좌/우 바퀴를 비율(-1.0~1.0)로 직접 구동합니다."""
        self.left_wheel.move(left_ratio)
        self.right_wheel.move(right_ratio)

    def rotate_to_angle(self, angle: Angle, criteria: Criteria.CLOSEST) -> bool:
        """지정 각도로 회전합니다. 완료 시 True 반환."""
        self.rotation_manager.rotate_to_angle(angle, criteria)
        return self.rotation_manager.finished_rotating

    def rotate_slowly_to_angle(self, angle: Angle, criteria: Criteria.CLOSEST) -> bool:
        """느린 속도로 지정 각도로 회전합니다. 완료 시 True 반환."""
        self.slow_rotation_manager.rotate_to_angle(angle, criteria)
        return self.slow_rotation_manager.finished_rotating

    def move_to_position(self, position: Position2D) -> bool:
        """지정 좌표로 이동합니다. 완료 시 True 반환."""
        self.movement_manager.move_to_position(position)
        return self.movement_manager.finished_moving

    @property
    def position(self) -> Position2D:
        return self.movement_manager.current_position

    @position.setter
    def position(self, value: Position2D):
        self.movement_manager.current_position = value

    @property
    def orientation(self) -> Angle:
        return self.rotation_manager.current_angle

    @orientation.setter
    def orientation(self, value: Angle):
        """방향이 바뀌면 세 관리자 모두에 동기화합니다."""
        self.movement_manager.current_angle = value
        self.rotation_manager.current_angle = value
        self.slow_rotation_manager.current_angle = value

    def get_wheel_average_angular_velocity(self):
        """양 바퀴 평균 각속도를 반환합니다 (PoseManager의 직진 판단에 사용)."""
        if self.right_wheel.velocity + self.left_wheel.velocity == 0:
            return 0
        return (self.right_wheel.velocity + self.left_wheel.velocity) / 2

    def get_wheel_velocity_difference(self):
        """우측 바퀴 - 좌측 바퀴 속도 차를 반환합니다 (회전 여부 판단에 사용)."""
        return self.right_wheel.velocity - self.left_wheel.velocity


class RotationManager:
    """
    지정 각도까지 PID 유사 속도 제어로 회전하는 클래스입니다.
    각도 오차가 클수록 빠르게, 작을수록 느리게 회전합니다.
    """
    def __init__(self, left_wheel, right_wheel) -> None:
        self.Directions = Enum("Directions", ["LEFT", "RIGHT"])

        self.right_wheel = right_wheel
        self.left_wheel = left_wheel

        self.initial_angle = Angle(0)
        self.current_angle = Angle(0)

        self.first_time = True          # 이번 회전 명령의 첫 호출 여부
        self.finished_rotating = True   # 회전 완료 여부

        self.max_velocity_cap = 1       # 속도 상한
        self.min_velocity_cap = 0.2     # 속도 하한 (너무 느려 멈추지 않도록)

        self.max_velocity = 1
        self.min_velocity = 0.2

        self.velocity_reduction_threshold = Angle(10, Angle.DEGREES)  # 이 각도 이하면 속도 절반으로 감속
        self.velocity_reduction_factor = 0.5

        self.accuracy = Angle(2, Angle.DEGREES)  # ±2도 이내면 완료로 판정

    def rotate_to_angle(self, target_angle, criteria=Criteria.CLOSEST):
        """매 타임스텝 호출: 목표 각도를 향해 바퀴를 회전시킵니다."""
        if self.first_time:
            self.initial_angle = self.current_angle
            self.first_time = False
            self.finished_rotating = False

        if self.is_at_angle(target_angle):
            self.finished_rotating = True
            self.first_time = True
            self.left_wheel.move(0)
            self.right_wheel.move(0)
            return

        absolute_difference = self.current_angle.get_absolute_distance_to(target_angle)
        # 오차 크기에 비례하여 속도 결정 (90도 → 최대속도, accuracy → 최소속도)
        velocity = mapVals(absolute_difference.degrees, self.accuracy.degrees, 90, self.min_velocity, self.max_velocity)

        if absolute_difference < self.velocity_reduction_threshold:
            velocity *= self.velocity_reduction_factor  # 목표 근처에서 감속

        velocity = min(velocity, self.max_velocity_cap)
        velocity = max(velocity, self.min_velocity_cap)

        direction = self.__get_direction(target_angle, criteria)

        if direction == self.Directions.RIGHT:
            self.left_wheel.move(velocity * -1)  # 제자리 우회전: 왼쪽 뒤로, 오른쪽 앞으로
            self.right_wheel.move(velocity)
        elif direction == self.Directions.LEFT:
            self.left_wheel.move(velocity)       # 제자리 좌회전: 왼쪽 앞으로, 오른쪽 뒤로
            self.right_wheel.move(velocity * -1)

    def is_at_angle(self, angle) -> bool:
        """현재 각도가 목표 각도로부터 accuracy 범위 안에 있으면 True를 반환합니다."""
        return self.current_angle.get_absolute_distance_to(angle) < self.accuracy

    def __get_direction(self, target_angle, criteria):
        """criteria에 따라 회전 방향(LEFT 또는 RIGHT)을 결정합니다."""
        if criteria == Criteria.CLOSEST:
            angle_difference = self.current_angle - target_angle
            if 180 > angle_difference.degrees > 0 or angle_difference.degrees < -180:
                return self.Directions.RIGHT
            else:
                return self.Directions.LEFT
        elif criteria == Criteria.FARTHEST:
            angle_difference = self.initial_angle - target_angle
            if 180 > angle_difference.degrees > 0 or angle_difference.degrees < -180:
                return self.Directions.LEFT
            else:
                return self.Directions.RIGHT
        elif criteria == Criteria.LEFT: return self.Directions.LEFT
        elif criteria == Criteria.RIGHT: return self.Directions.RIGHT


class MovementToCoordinatesManager:
    """
    단순 회전 → 직진 방식으로 목표 좌표로 이동하는 클래스입니다.
    (현재는 SmoothMovementToCoordinatesManager로 대체되어 사용하지 않습니다.)
    """
    def __init__(self, left_wheel, right_wheel) -> None:
        self.current_position = Position2D()

        self.left_wheel = left_wheel
        self.right_wheel = right_wheel
        self.rotation_manager = RotationManager(self.left_wheel, self.right_wheel)

        self.error_margin = 0.001         # 목표 도달 판정 거리(m)
        self.desceleration_start = 0.5 * 0.12  # 감속 시작 거리(m)

        self.max_velocity_cap = 1
        self.min_velocity_cap = 0.8

        self.max_velocity = 1
        self.min_velocity = 0.1

        self.finished_moving = False

    @property
    def current_angle(self) -> Angle:
        return self.rotation_manager.current_angle

    @current_angle.setter
    def current_angle(self, value):
        self.rotation_manager.current_angle = value

    def move_to_position(self, target_position: Position2D):
        """목표 방향으로 회전 완료 후 직선 이동합니다."""
        dist = abs(self.current_position.get_distance_to(target_position))
        if SHOW_DEBUG: print(f"[이동:drive_base.move_to_position] 목표까지 거리(단순): {dist:.4f}m, 현재=({self.current_position.x:.4f},{self.current_position.y:.4f}), 목표=({target_position.x:.4f},{target_position.y:.4f})")

        if dist < self.error_margin:
            if SHOW_DEBUG: print(f"[이동:drive_base.move_to_position] 완료(단순)! 목표 도달 (오차 {self.error_margin}m 이내)")
            self.finished_moving = True
            self.left_wheel.move(0)
            self.right_wheel.move(0)
        else:
            self.finished_moving = False
            ang = self.current_position.get_angle_to(target_position)

            if self.rotation_manager.is_at_angle(ang):
                velocity = mapVals(dist, 0, self.desceleration_start, self.min_velocity, self.max_velocity)
                velocity = min(velocity, self.max_velocity_cap)
                velocity = max(velocity, self.min_velocity_cap)
                self.right_wheel.move(velocity)
                self.left_wheel.move(velocity)
            else:
                self.rotation_manager.rotate_to_angle(ang)


class SmoothMovementToCoordinatesManager:
    """
    곡선 경로로 부드럽게 목표 좌표로 이동하는 클래스입니다.
    방향 오차에 따라 세 가지 이동 모드로 나뉩니다:
    - 직진: 각도 오차가 angle_error_margin 이하
    - 강한 회전: 각도 오차가 strong_rotation_start 이상 → 제자리 회전
    - 약한 곡선: 각도 오차가 중간 → 좌우 속도 차이를 두어 서서히 방향 조정
    """
    def __init__(self, left_wheel: Wheel, right_wheel: Wheel) -> None:
        self.current_position = Position2D()

        self.left_wheel = left_wheel
        self.right_wheel = right_wheel

        self.current_angle = Angle(0)

        self.error_margin = 0.003    # 목표 도달 판정 거리(m)
        self.velocity = 1            # 기본 전진 속도

        self.distance_weight = 5     # 거리 기반 속도 가중치 (현재 미사용)
        self.angle_weight = 5        # 각도 오차 기반 회전 속도 가중치

        self.min_velocity_cap = 0

        self.turning_speed_multiplier = 0.8

        self.finished_moving = False

        self.angle_error_margin = Angle(3, Angle.DEGREES)     # 직진 판정 각도 오차 한계
        self.strong_rotation_start = Angle(45, Angle.DEGREES) # 강한 회전 시작 각도
        self.light_rotation_start = Angle(30, Angle.DEGREES)  # 약한 곡선 이동 시작 각도

    def move_to_position(self, target_position: Position2D):
        """목표 좌표와의 각도/거리에 따라 세 가지 모드로 이동 제어를 수행합니다."""
        dist = abs(self.current_position.get_distance_to(target_position))
        if SHOW_DEBUG: print(f"[이동:drive_base.move_to_position] 목표까지 거리(부드럽): {dist:.4f}m, 현재=({self.current_position.x:.4f},{self.current_position.y:.4f}), 목표=({target_position.x:.4f},{target_position.y:.4f})")

        if dist < self.error_margin:
            if SHOW_DEBUG: print(f"[이동:drive_base.move_to_position] 완료(부드럽)! 목표 도달 (오차 {self.error_margin}m 이내)")
            self.finished_moving = True
            self.left_wheel.move(0)
            self.right_wheel.move(0)
        else:
            self.finished_moving = False

            angle_to_target = self.current_position.get_angle_to(target_position)
            angle_diff = self.current_angle - angle_to_target
            absolute_angle_diff = self.current_angle.get_absolute_distance_to(angle_to_target)

            if absolute_angle_diff < self.angle_error_margin:
                # 거의 정렬된 상태 → 직진
                self.right_wheel.move(self.velocity)
                self.left_wheel.move(self.velocity)

            elif absolute_angle_diff > self.strong_rotation_start:
                # 크게 틀어진 상태 → 제자리 회전으로 방향 전환
                if 180 > angle_diff.degrees > 0 or angle_diff.degrees < -180:
                    self.right_wheel.move(self.velocity)
                    self.left_wheel.move(self.velocity * -1)
                else:
                    self.right_wheel.move(self.velocity * -1)
                    self.left_wheel.move(self.velocity)

            elif absolute_angle_diff < self.light_rotation_start:
                # 약간 틀어진 상태 → 한쪽 바퀴를 살짝 느리게 하여 곡선 이동
                if 180 > angle_diff.degrees > 0 or angle_diff.degrees < -180:
                    self.right_wheel.move(1)
                    self.left_wheel.move(0.8)
                else:
                    self.right_wheel.move(0.8)
                    self.left_wheel.move(1)

            else:
                # 중간 범위 → 각도 오차 크기에 비례한 동적 속도로 곡선 이동
                distance_speed = abs(dist * -self.distance_weight)
                angle_speed = absolute_angle_diff.radians * self.angle_weight

                speed = angle_speed * self.turning_speed_multiplier
                speed = max(self.min_velocity_cap, speed)

                speed2 = speed * distance_speed
                speed2 = max(speed2, self.min_velocity_cap)

                if 180 > angle_diff.degrees > 0 or angle_diff.degrees < -180:
                    self.right_wheel.move(speed)
                    self.left_wheel.move(speed2)
                else:
                    self.right_wheel.move(speed2)
                    self.left_wheel.move(speed)
