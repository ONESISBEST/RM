class StepCounter:
    """
    n 타임스텝마다 한 번씩 어떤 동작을 실행하도록 제어하는 카운터 클래스입니다.

    라이다나 카메라 같은 무거운 처리를 매 스텝마다 하지 않고
    n 스텝에 한 번만 실행하여 성능을 최적화할 때 사용합니다.
    예) StepCounter(6) → 6 타임스텝에 한 번만 라이다 포인트클라우드 갱신
    """

    def __init__(self, interval):
        self.__current_step = 0      # 현재 스텝 카운트 (0 ~ interval-1 순환)
        self.interval = interval     # 몇 스텝마다 실행할지 지정하는 주기

    def increase(self):
        """매 타임스텝 호출: 카운터를 1 증가시키고, interval에 도달하면 0으로 초기화합니다."""
        self.__current_step += 1
        if self.__current_step == self.interval:
            self.__current_step = 0

    def check(self):
        """카운터가 0일 때(=주기가 돌아왔을 때) True를 반환합니다."""
        return self.__current_step == 0
