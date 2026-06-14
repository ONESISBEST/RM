from flags import SHOW_DEBUG


class DelayManager:
    """
    Webots 시뮬레이션을 멈추지 않고 비동기적으로 시간 지연(delay)을 처리하는 클래스입니다.

    time.sleep()은 프로그램 전체를 멈추기 때문에 센서 업데이트가 중단됩니다.
    이 클래스는 Sequencer와 함께 사용하여 "n초 동안 대기"를 메인 루프를 유지하면서 처리합니다.
    """
    def __init__(self) -> None:
        self.time = 0                  # Webots 시뮬레이션 현재 시간 (초)
        self.delay_first_time = True   # 딜레이 시작 시점을 기록하기 위한 플래그
        self.delay_start = 0           # 딜레이 시작 시각

    def update(self, time):
        """매 타임스텝마다 현재 시간을 갱신합니다."""
        self.time = time

    def delay_seconds(self, delay):
        """
        지정한 초(delay)만큼 대기합니다. (Sequencer의 complex_event 방식으로 동작)

        - 처음 호출 시: 시작 시각을 기록하고 False 반환 (아직 대기 중)
        - 이후 호출 시: 경과 시간이 delay를 넘으면 True 반환 (완료)
        """
        if SHOW_DEBUG:
            elapsed = self.time - self.delay_start if not self.delay_first_time else 0.0
            print(f"[딜레이:delay.delay_seconds] 경과: {elapsed:.2f}s / 목표: {delay}s (시뮬 시간: {self.time:.2f}s)")
        if self.delay_first_time:
            self.delay_start = self.time       # 딜레이 시작 시각 기록
            self.delay_first_time = False
        else:
            if self.time - self.delay_start >= delay:  # 지정 시간만큼 경과했으면
                self.delay_first_time = True
                return True                    # 완료 신호 반환
        return False

    def reset_delay(self):
        """딜레이 상태를 초기화합니다. Sequencer의 reset_function으로 사용됩니다."""
        self.delay_first_time = True
