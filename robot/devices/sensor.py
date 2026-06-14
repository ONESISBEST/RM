from abc import ABC, abstractmethod

class Sensor(ABC):
    """모든 Webots 센서 장치의 추상 기반 클래스입니다. 장치를 enable하고 기본 구조를 정의합니다."""
    def __init__(self, webots_device, time_step):
        self.time_step = time_step          # 센서 업데이트 주기(ms)
        self.device = webots_device         # Webots 장치 객체
        self.device.enable(time_step)       # 센서를 지정 주기로 활성화

    def update(self):
        pass  # 서브클래스에서 필요 시 오버라이드

class TimedSensor(Sensor):
    """
    StepCounter를 이용해 n 타임스텝마다 한 번씩 실제 처리를 수행하는 센서 기반 클래스입니다.
    카메라, 라이다 등 처리 비용이 큰 센서에 사용합니다.
    """
    def __init__(self, webots_device, time_step, step_counter):
        super().__init__(webots_device, time_step)
        self.step_counter = step_counter   # 주기 관리를 위한 StepCounter 인스턴스

    def update(self):
        """매 타임스텝 호출: 스텝 카운터를 증가시킵니다."""
        self.step_counter.increase()
