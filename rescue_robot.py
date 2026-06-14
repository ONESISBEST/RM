"""
rescue_robot.py — 초보자용 로봇 제어 인터페이스

복잡한 내부 코드를 몰라도 로봇을 제어할 수 있는 간단한 API를 제공합니다.

사용 예시:
    from rescue_robot import RescueRobot

    robot = RescueRobot()
    robot.run_autonomous()          # 완전 자율 주행 (가장 간단!)

    # 또는 직접 제어:
    while robot.is_running():
        print(f"현재 위치: {robot.location}")
        if robot.victim_visible:
            robot.report_victim()
        robot.go_to_next_target()
"""

from executor.executor import Executor
from mapping.mapper import Mapper
from robot.robot import Robot
from final_matrix_creation.final_matrix_creator import FinalMatrixCreator


class RescueRobot:
    """
    구조 로봇을 쉽게 제어하기 위한 초보자 전용 인터페이스입니다.

    이 클래스 하나로 로봇의 이동, 센서 읽기, 조난자 보고, 지도 전송을 모두 할 수 있습니다.

    빠른 시작:
        robot = RescueRobot()
        robot.run_autonomous()   # 로봇이 알아서 미로를 탐색합니다!

    직접 제어:
        robot = RescueRobot()
        while robot.is_running():
            robot.go_to_next_target()
    """

    def __init__(self):
        """로봇, 지도, 두뇌를 한 번에 초기화합니다. 아무 인자도 필요 없습니다."""
        self._robot = Robot(time_step=32)
        self._mapper = Mapper(
            tile_size=0.12,
            robot_diameter=self._robot.diameter,
            camera_distance_from_center=self._robot.diameter / 2,
        )
        self._executor = Executor(self._mapper, self._robot)
        self._final_matrix_creator = FinalMatrixCreator(
            self._mapper.tile_size, self._mapper.pixel_grid.resolution
        )

    # =========================================================
    # 위치 정보
    # =========================================================

    @property
    def x(self) -> float:
        """로봇의 현재 X 좌표 (단위: 미터)"""
        return self._robot.position.x

    @property
    def y(self) -> float:
        """로봇의 현재 Y 좌표 (단위: 미터)"""
        return self._robot.position.y

    @property
    def direction(self) -> float:
        """로봇이 바라보는 방향 (단위: 도, 0~360°)"""
        return self._robot.orientation.degrees

    @property
    def location(self):
        """현재 위치를 (x, y) 튜플로 반환합니다."""
        return (self.x, self.y)

    @property
    def start_location(self):
        """출발 위치를 (x, y) 튜플로 반환합니다. 아직 등록 전이면 None."""
        sp = self._mapper.start_position
        if sp is None:
            return None
        return (sp.x, sp.y)

    @property
    def distance_to_start(self) -> float:
        """출발점까지의 직선 거리 (단위: 미터). 출발점 미등록 시 매우 큰 값 반환."""
        sp = self._mapper.start_position
        if sp is None:
            return float("inf")
        return self._robot.position.get_distance_to(sp)

    # =========================================================
    # 시간
    # =========================================================

    @property
    def elapsed_time(self) -> float:
        """미션 시작 후 경과 시간 (단위: 초)"""
        return self._robot.time

    @property
    def remaining_time(self) -> float:
        """남은 시간 (단위: 초). 제한 시간은 8분(480초)입니다."""
        return max(0.0, self._executor.max_time_in_run - self.elapsed_time)

    @property
    def is_time_almost_up(self) -> bool:
        """남은 시간이 30초 미만이면 True. 이때는 서둘러 귀환해야 합니다."""
        return self.remaining_time < 30

    # =========================================================
    # 이동 명령
    # =========================================================

    def move_forward(self, speed: float = 1.0):
        """앞으로 이동합니다.

        Args:
            speed: 속도 (0.0 ~ 1.0, 기본값 1.0)
        """
        self._robot.move_wheels(speed, speed)

    def move_backward(self, speed: float = 1.0):
        """뒤로 이동합니다.

        Args:
            speed: 속도 (0.0 ~ 1.0, 기본값 1.0)
        """
        self._robot.move_wheels(-speed, -speed)

    def stop(self):
        """로봇을 즉시 멈춥니다."""
        self._robot.move_wheels(0, 0)

    def turn_left(self, speed: float = 0.5):
        """제자리에서 왼쪽(반시계 방향)으로 회전합니다.

        Args:
            speed: 회전 속도 (0.0 ~ 1.0, 기본값 0.5)
        """
        self._robot.move_wheels(-speed, speed)

    def turn_right(self, speed: float = 0.5):
        """제자리에서 오른쪽(시계 방향)으로 회전합니다.

        Args:
            speed: 회전 속도 (0.0 ~ 1.0, 기본값 0.5)
        """
        self._robot.move_wheels(speed, -speed)

    def go_to(self, x: float, y: float) -> bool:
        """지정한 좌표로 이동합니다. 도착하면 True를 반환합니다.

        Args:
            x: 목표 X 좌표 (미터)
            y: 목표 Y 좌표 (미터)

        Returns:
            도착했으면 True, 아직 이동 중이면 False
        """
        return self._robot.move_to_coords((x, y))

    def face(self, degrees: float) -> bool:
        """지정한 각도 방향으로 로봇을 회전시킵니다. 완료되면 True를 반환합니다.

        Args:
            degrees: 목표 방향 (도, 0°=앞, 90°=왼쪽, 270°=오른쪽)

        Returns:
            회전 완료 시 True, 진행 중이면 False
        """
        return self._robot.rotate_to_angle(degrees)

    def go_to_start(self) -> bool:
        """출발 위치로 돌아갑니다. 도착하면 True를 반환합니다.

        Returns:
            도착했으면 True, 아직 이동 중이면 False
        """
        sp = self._mapper.start_position
        if sp is None:
            return False
        return self._robot.move_to_coords((sp.x, sp.y))

    # =========================================================
    # 센서
    # =========================================================

    @property
    def has_obstacle_ahead(self) -> bool:
        """전방 바로 앞에 장애물(벽)이 있으면 True"""
        return self._robot.point_is_close

    @property
    def is_shaky(self) -> bool:
        """로봇이 심하게 흔들리거나 기울어져 있으면 True (경사면, 충돌 시 발생)"""
        return self._robot.is_shaky()

    @property
    def swamp_nearby(self) -> bool:
        """근처에 늪지대가 있으면 True (늪 위에서는 GPS가 부정확해집니다)"""
        return self._mapper.is_close_to_swamp()

    def get_camera_image(self, camera: str = "center"):
        """카메라 이미지를 numpy 배열로 가져옵니다.

        Args:
            camera: 카메라 방향 - 'center'(전방), 'left'(좌측), 'right'(우측)

        Returns:
            numpy 배열 이미지 (H×W×4, BGRA 형식), 아직 캡처 전이면 None
        """
        if camera == "left":
            return self._robot.left_camera.image.image
        if camera == "right":
            return self._robot.right_camera.image.image
        return self._robot.center_camera.image.image

    # =========================================================
    # 조난자 탐지 및 보고
    # =========================================================

    @property
    def victim_visible(self) -> bool:
        """현재 카메라 화면에 조난자(알파벳 표지)가 보이면 True"""
        for cam in (self._robot.center_camera,
                    self._robot.left_camera,
                    self._robot.right_camera):
            if cam.image.image is not None:
                fixtures = self._executor.fixture_detector.find_fixtures(cam.image.image)
                if len(fixtures) > 0:
                    return True
        return False

    @property
    def victim_letter(self):
        """카메라에서 탐지된 조난자 심볼을 반환합니다 (Φ/Ψ/Ω).

        Returns:
            심볼 문자열, 없으면 None, 이미 보고된 곳이면 None
        """
        for cam in (self._robot.center_camera,
                    self._robot.left_camera,
                    self._robot.right_camera):
            if cam.image.image is not None:
                fixtures = self._executor.fixture_detector.find_fixtures(cam.image.image)
                if len(fixtures) > 0:
                    return self._executor.fixture_detector.classify_fixture(fixtures[0])
        return None

    @property
    def already_reported(self) -> bool:
        """현재 위치에서 이미 조난자를 보고했으면 True (중복 보고 방지용)"""
        return self._mapper.has_detected_victim_from_position()

    def report_victim(self, letter: str = None):
        """발견한 조난자를 서버에 보고합니다.

        Args:
            letter: 보고할 심볼 또는 서버용 글자. None이면 카메라로 자동 판별합니다.
        """
        if letter is None:
            letter = self.victim_letter
        if letter is not None:
            reported_letter = self._to_report_letter(letter)
            if reported_letter is None:
                return
            self._robot.comunicator.send_victim(self._robot.raw_position, reported_letter)
            self._mapper.fixture_mapper.map_detected_fixture(self._robot.position)
            print(f"[RescueRobot] 조난자 보고 완료: 심볼='{letter}', 보고글자='{reported_letter}', 위치={self.location}")

    # =========================================================
    # 지도
    # =========================================================

    @property
    def is_at_start(self) -> bool:
        """출발 위치 근처(4cm 이내)에 있으면 True"""
        return self.distance_to_start < 0.04

    def enable_mapping(self):
        """지도 그리기 기능을 켭니다. 이동 중에 주변 지형을 지도에 기록합니다."""
        self._executor.mapping_enabled = True

    def disable_mapping(self):
        """지도 그리기 기능을 끕니다. 조난자 보고처럼 정밀 작업 중에 사용합니다."""
        self._executor.mapping_enabled = False

    # =========================================================
    # 미션 종료
    # =========================================================

    def send_final_map(self):
        """완성된 지도를 서버로 보냅니다. 미션 중에도 중간 저장 용도로 사용할 수 있습니다."""
        final_matrix = self._final_matrix_creator.pixel_grid_to_final_grid(
            self._mapper.pixel_grid, self._mapper.start_position
        )
        self._robot.comunicator.send_map(final_matrix)
        print("[RescueRobot] 최종 지도 전송 완료")

    def finish_mission(self):
        """지도를 서버로 보내고 미션 종료를 알립니다. 마지막에 한 번만 호출하세요."""
        self.send_final_map()
        self._robot.comunicator.send_end_of_play()
        print("[RescueRobot] 미션 종료 신호 전송 완료")

    # =========================================================
    # 자율 항법 (에이전트)
    # =========================================================

    def go_to_next_target(self) -> bool:
        """에이전트가 계산한 다음 탐색 목표 위치로 이동합니다.

        로봇이 스스로 미탐색 구역을 찾아 이동합니다.

        Returns:
            목표 위치에 도착했으면 True, 이동 중이면 False
        """
        self._executor.agent.update()
        target = self._executor.agent.get_target_position()
        if target is None:
            return False
        return self._robot.move_to_coords((target.x, target.y))

    @property
    def auto_target(self):
        """에이전트가 추천하는 다음 목표 위치 (x, y). 목표 없으면 None."""
        target = self._executor.agent.get_target_position()
        if target is None:
            return None
        return (target.x, target.y)

    @property
    def exploration_complete(self) -> bool:
        """미로 전체 탐색이 완료되어 출발점으로 돌아가야 할 시점이면 True"""
        return self._executor.agent.do_end()

    # =========================================================
    # 루프 제어
    # =========================================================

    def is_running(self) -> bool:
        """시뮬레이션을 한 프레임 앞으로 진행하고 모든 센서를 업데이트합니다.

        while 루프 조건으로 사용합니다. 시뮬레이션이 실행 중이면 True,
        Webots 창이 닫히거나 리셋되면 False를 반환합니다.

        사용 예:
            while robot.is_running():
                robot.go_to_next_target()

        Returns:
            시뮬레이션 계속 실행 중이면 True
        """
        if not self._robot.do_loop():
            return False

        self._executor._tick_real_elapsed()
        self._robot.update()
        self._executor.delay_manager.update(self._robot.time)
        self._executor.stuck_detector.update(
            self._robot.position,
            self._robot.previous_position,
            self._robot.drive_base.get_wheel_average_angular_velocity(),
        )

        # run_autonomous()를 쓰지 않는 수동 루프 모드에서는 state_init을 거치지 않으므로
        # 매핑 활성화와 시작 위치 등록을 여기서 직접 처리합니다.
        if not self._executor.mapping_enabled:
            self._executor.calibrate_position_offsets()
            self._executor.mapping_enabled = True
            self._executor.victim_reporting_enabled = True
        if self._mapper.start_position is None:
            self._mapper.register_start(self._robot.position)

        self._executor.do_mapping()
        self._executor.check_map_sending()
        return True

    def step(self):
        """자율 주행 두뇌(상태 머신)를 한 프레임 실행합니다.

        is_running()과 함께 쓰면 자율 주행에 학생 코드를 더할 수 있습니다.

        사용 예:
            while robot.is_running():
                robot.step()                   # 자율 주행 유지
                print(f"위치: {robot.location}")  # 내가 추가한 코드
        """
        self._executor.state_machine.run()

    def run_autonomous(self):
        """완전 자율 주행 모드로 미션을 처음부터 끝까지 수행합니다.

        이 메서드를 한 줄만 호출하면 로봇이 스스로:
        1. 초기화 및 자세 교정
        2. 미로 탐색 및 조난자 보고
        3. 출발점 복귀
        4. 최종 지도 전송
        을 모두 처리합니다.

        사용 예:
            robot = RescueRobot()
            robot.run_autonomous()
        """
        self._executor.run()

    def _to_report_letter(self, symbol: str):
        """인식된 심볼/문자열을 서버 보고용 글자로 변환합니다."""
        mapping = {
            "Φ": "H",
            "Ψ": "S",
            "Ω": "U",
            "H": "H",
            "S": "S",
            "U": "U",
            "P": "P",
            "F": "F",
            "C": "C",
            "O": "O",
        }
        return mapping.get(symbol)
