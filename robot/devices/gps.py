from data_structures.vectors import Position2D

from robot.devices.sensor import Sensor

class Gps(Sensor):
    """
    Webots GPS 센서를 이용해 로봇의 전역 위치(x, z → x, y)를 추적하는 클래스입니다.

    또한 두 타임스텝 사이의 위치 차이를 이용해 로봇이 직진할 때의 진행 방향(각도)을
    계산할 수 있습니다. 단, 직진 중일 때만 신뢰도가 높습니다.
    """
    def __init__(self, webots_device, time_step, coords_multiplier=1):
        super().__init__(webots_device, time_step)
        self.multiplier = coords_multiplier   # 좌표 스케일링 인수 (기본값 1)
        self.position = self.get_position()   # 현재 위치 초기화

        # v26부터 GPS에 가우시안 노이즈(표준편차 ≈ 2.5mm)가 추가되었다.
        # 연속한 두 위치의 차이는 노이즈가 신호를 압도하므로(이동 ~4mm vs 노이즈차 ~3.5mm),
        # 노이즈보다 충분히 큰 거리를 이동한 두 점으로만 방향을 계산해야 신뢰할 수 있다.
        self.__position_history = [self.position]   # 방향 계산용 위치 이력 (직진 구간만 누적)
        self.__max_history = 40
        self.min_baseline_distance = 0.04   # 방향 계산용 최소 이동 거리(m) ≈ GPS 노이즈의 16배

    def update(self):
        """매 타임스텝 호출: GPS 값을 갱신하고 방향 계산용 위치 이력을 누적합니다."""
        self.position = self.get_position()
        self.__position_history.append(self.position)
        if len(self.__position_history) > self.__max_history:
            self.__position_history.pop(0)

    def reset_orientation_baseline(self):
        """직진이 끊겼을 때(회전 등) 호출하여 방향 계산용 위치 이력을 초기화합니다.
        이력이 회전 구간을 가로지르지 않게 하여 직진 구간만으로 방향을 계산합니다."""
        self.__position_history = [self.position]

    def get_position(self):
        """GPS 센서에서 현재 전역 위치(x, y)를 읽어 반환합니다. (Webots의 z축 → y축으로 변환)"""
        vals = self.device.getValues()
        return Position2D(vals[0] * self.multiplier, vals[2] * self.multiplier)

    def get_orientation(self):
        """
        min_baseline_distance 이상 떨어진 가장 최근 과거 위치에서 현재 위치로의
        방향을 반환합니다. 긴 기준선으로 GPS 노이즈를 평균화하여 신뢰할 수 있는
        방향을 얻습니다. 이동이 부족하면 None을 반환 → PoseManager가 자이로스코프로 대체합니다.
        """
        for past_position in reversed(self.__position_history[:-1]):
            if self.position.get_distance_to(past_position) >= self.min_baseline_distance:
                angle = past_position.get_angle_to(self.position)
                angle.normalize()
                return angle
        return None
