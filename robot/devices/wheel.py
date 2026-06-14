class Wheel:
    """Webots 모터(바퀴) 하나를 제어하는 클래스입니다."""
    def __init__(self, wheel, maxVelocity):
        self.maxVelocity = maxVelocity  # 최대 각속도 (rad/s)
        self.wheel = wheel              # Webots 모터 장치 객체
        self.velocity = 0               # 현재 설정된 속도
        self.wheel.setPosition(float("inf"))  # 위치 제어가 아닌 속도 제어 모드로 설정
        self.wheel.setVelocity(0)       # 초기 속도 0

    def move(self, ratio):
        """
        최대 속도 대비 비율(-1.0 ~ 1.0)로 바퀴를 구동합니다.
        양수 = 전진 방향, 음수 = 후진 방향, 1.0 = 최대 속도
        """
        if ratio > 1:
            ratio = 1
        elif ratio < -1:
            ratio = -1
        self.velocity = ratio * self.maxVelocity
        self.wheel.setVelocity(self.velocity)
