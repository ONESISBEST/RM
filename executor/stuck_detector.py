from data_structures.vectors import Position2D

class StuckDetector:
    """바퀴는 돌고 있지만 로봇이 실제로 이동하지 않을 때(벽에 끼인 상태)를 감지하는 클래스입니다."""
    def __init__(self) -> None:
        self.stuck_counter = 0          # 끼임 상태가 지속된 스텝(타임스텝) 수

        self.stuck_threshold = 50       # 이 값을 초과하면 '끼임 확정'으로 판단
        self.minimum_distance_traveled = 0.001  # 한 스텝에서 이 거리 미만이면 이동 안 한 것으로 판단

        self.__position = Position2D(0, 0)
        self.__previous_position = Position2D(0, 0)
        self.__wheel_direction = 0      # 바퀴의 평균 각속도 (0이면 정지 중)

    def update(self, position, previous_position, wheel_direction):
        """매 타임스텝마다 호출하여 끼임 여부를 업데이트합니다."""
        self.__wheel_direction = wheel_direction
        self.__position = position
        self.__previous_position = previous_position

        # 이번 스텝에 끼인 것으로 보이면 카운터 증가, 아니면 초기화
        if self.__is_stuck_this_step():
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

    def is_stuck(self):
        """카운터가 임계값을 넘으면 True를 반환 → Executor가 이를 보고 'stuck' 상태로 전환"""
        return self.stuck_counter > self.stuck_threshold

    def __is_stuck_this_step(self):
        """이번 한 스텝에서 끼인 상태인지 판단: 바퀴가 돌고 있는데 위치 변화가 거의 없으면 끼인 것"""
        distance_traveled = self.__position.get_distance_to(self.__previous_position)
        is_rotating_wheels = self.__wheel_direction > 0
        return is_rotating_wheels and distance_traveled < self.minimum_distance_traveled
