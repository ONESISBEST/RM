from data_structures.angle import Angle
from data_structures.vectors import Position2D

from flow_control.sequencer import Sequencer
from flow_control.state_machine import StateMachine
from flow_control.delay import DelayManager
from flow_control.step_counter import StepCounter

from executor.stuck_detector import StuckDetector

from robot.robot import Robot
from robot.drive_base import Criteria as RotationCriteria

from mapping.mapper import Mapper

from agent.agent import Agent

from fixture_detection.fixture_clasification import FixtureClasiffier

from final_matrix_creation.final_matrix_creator import FinalMatrixCreator

from utilities import ColorFilterTuner
from fixture_detection.color_filter import ColorFilter
from fixture_detection.non_fixture_filterer import NonFixtureFilter

from flags import DO_SLOW_DOWN, SLOW_DOWN_S, TUNE_FILTER

import time
import numpy as np
import cv2 as cv
import skimage

class Executor:
    """로봇의 모든 모듈(센서, 모터, 매핑, 길찾기)을 조율하고 전체 미션의 흐름을 제어하는 총괄 클래스입니다."""
    
    def __init__(self, mapper: Mapper, robot: Robot) -> None:
        # --- 1. 핵심 하위 모듈 초기화 ---
        self.agent = Agent(mapper)       # 어디로 가야 할지(목표 좌표)를 계산하는 행동 대장
        self.mapper = mapper             # 지도를 그리는 역할
        self.robot = robot               # 실제 모터와 센서를 다루는 하드웨어 제어기

        self.delay_manager = DelayManager()     # 시간 지연(Delay)을 비동기적으로 관리 (Webots 화면 멈춤 방지)
        self.stuck_detector = StuckDetector()   # 로봇이 벽에 끼어서 헛돌고 있는지 감지

        # --- 2. 상태 머신 (State Machine) 설정 ---
        # 로봇이 현재 어떤 기분/상태인지 정의하고, 상태 간의 전환(Transition)을 관리합니다.
        self.state_machine = StateMachine("init") 
        # create_state("상태이름", 실행할함수, {이동가능한다음상태들})
        self.state_machine.create_state("init", self.state_init, {"explore",}) 
        self.state_machine.create_state("explore", self.state_explore, {"end", "report_fixture", "send_map", "stuck"}) 
        self.state_machine.create_state("end", self.state_end)
        self.state_machine.create_state("report_fixture", self.state_report_fixture, {"explore", "send_map"})
        self.state_machine.create_state("send_map", self.state_send_map, {"explore", "end"})
        self.state_machine.create_state("stuck", self.state_stuck, {"explore", "send_map", "end"})

        # --- 3. 시퀀서 (Sequencer) 설정 ---
        # 로봇에게 "A하고 -> 1초 쉬고 -> B해라" 같은 연속 동작을 지시할 때 사용합니다.
        self.sequencer = Sequencer(reset_function=self.delay_manager.reset_delay) 

        self.fixture_detector = FixtureClasiffier() # 카메라 비전 기반 조난자(심볼) 인식기

        self.final_matrix_creator = FinalMatrixCreator(mapper.tile_size, mapper.pixel_grid.resolution)
        
        # --- 4. 플래그(상태 변수) 및 시퀀서용 함수 매핑 ---
        self.mapping_enabled = False          # 맵핑을 할지 말지 결정
        self.victim_reporting_enabled = False # 조난자 탐지를 할지 말지 결정
        self.fixture_detection_cooldown = 0  # 오탐 후 감지 억제 카운터 (>0이면 감지 스킵)

        # 자주 쓰는 함수들을 시퀀서에서 쉽게 부를 수 있게 래핑(Wrapping)합니다.
        self.seq_print =                   self.sequencer.make_simple_event( print)
        self.seq_move_wheels =             self.sequencer.make_simple_event( self.robot.move_wheels)
        self.seq_rotate_to_angle =         self.sequencer.make_complex_event(self.robot.rotate_to_angle)
        self.seq_rotate_slowly_to_angle =  self.sequencer.make_complex_event(self.robot.rotate_slowly_to_angle)
        self.seq_move_to_coords =          self.sequencer.make_complex_event(self.robot.move_to_coords)
        self.seq_delay_seconds =           self.sequencer.make_complex_event(self.delay_manager.delay_seconds)
        self.seq_align_with_fixture =      self.sequencer.make_complex_event(self.align_with_fixture)

        self.letter_to_report = None          # 발견한 조난자 글자 저장
        self.report_orientation = Angle(0)    # 조난자가 있는 방향
        self.report_timeout_counter = 0       # 조난자 보고 상태 타임아웃 카운터

        self.max_time_in_run = 8 * 60         # 대회 제한 시간 (8분)
        self.map_sent = False                 # 맵 전송 여부

        self.robot.set_start_time()           # 시작 시간 기록
        self.__real_elapsed = 0.0             # 실제 실행 중에만 누적되는 리얼타임(초)
        self.__last_step_real_time = None     # 직전 스텝의 time.time() 값

        self.mini_calibrate_step_counter = StepCounter(20) # 20스텝마다 한 번씩 캘리브레이션 트리거

        self.color_filter_tuner = ColorFilterTuner(ColorFilter((0, 0, 29), (0, 0, 137)), TUNE_FILTER)


    def run(self):
        """메인 무한 루프: 매 프레임(Timestep)마다 실행되며 로봇의 모든 구성요소를 업데이트합니다."""
        
        while self.robot.do_loop():
            # 일시정지 구간은 제외하고 실제 실행 시간만 누적
            self._tick_real_elapsed()

            # 1. 하드웨어 업데이트 (센서값 읽어오기, 현재 위치 갱신)
            self.robot.update()

            # 2. 유틸리티 업데이트 (지연시간, 끼임 감지)
            self.delay_manager.update(self.robot.time)
            self.stuck_detector.update(self.robot.position,
                                       self.robot.previous_position,
                                       self.robot.drive_base.get_wheel_average_angular_velocity())
            
            # 카메라 색상 필터 튜닝
            self.color_filter_tuner.tune(self.robot.center_camera.image.image)

            # 3. 맵 그리기 (로봇이 안정적일 때 센서 데이터를 지도로 변환)
            self.do_mapping()
            
            # 4. 종료 시간 임박 시 지도 전송 체크
            self.check_map_sending()

            # 탐색 중이거나 조난자 보고 중일 때 끼임 감지 활성화
            if (self.state_machine.check_state("explore") or self.state_machine.check_state("report_fixture")) and self.stuck_detector.is_stuck():
                print(f"[끼임 감지:executor.run] 이동 중 정체 감지 -> stuck 상태 전환 "
                      f"(counter={self.stuck_detector.stuck_counter}, "
                      f"위치=({self.robot.position.x:.4f},{self.robot.position.y:.4f})m)")
                self.state_machine.change_state("stuck")
                self.sequencer.reset_sequence()

            # 5. ★ 로봇의 핵심 두뇌 (상태 머신) 실행 ★
            # 현재 상태에 맞춰 state_init, state_explore 등의 함수 중 하나를 실행합니다.
            self.state_machine.run()

            # 디버그용 느리게 보기 모드
            if DO_SLOW_DOWN:
                time.sleep(SLOW_DOWN_S)
            
    def do_mapping(self):
        """맵핑 기능이 켜져 있을 때(True) 지도를 업데이트합니다."""
        if self.mapping_enabled:
                if not self.robot.is_shaky(): # 로봇이 크게 흔들리지 않을 때 (정밀도 유지)
                    # 라이다와 카메라를 모두 사용하여 지도 업데이트
                    self.mapper.update(self.robot.get_point_cloud(), 
                                    self.robot.get_out_of_bounds_point_cloud(),
                                    self.robot.get_lidar_detections(),
                                    self.robot.get_camera_images(), 
                                    self.robot.position,
                                    self.robot.orientation,
                                    self.robot.time)
                else:
                    # 흔들릴 때는 라이다를 빼고 카메라 데이터와 로봇 위치만 대략적으로 업데이트
                    self.mapper.update(camera_images=self.robot.get_camera_images(), 
                                       robot_position= self.robot.position,
                                       robot_orientation=self.robot.orientation,
                                       time=self.robot.time)

    def check_swamp_proximity(self):
        """늪지대(Swamp)에 가까워졌는지 확인 (센서 오차 방지용)"""
        if self.mapper.is_close_to_swamp():
            self.robot.auto_decide_orientation_sensor = False
            self.robot.orientation_sensor = self.robot.GYROSCOPE # 늪지대에서는 GPS 에러가 커지므로 자이로만 사용
        else: 
            self.robot.auto_decide_orientation_sensor = True

    def _tick_real_elapsed(self):
        """robot.step()이 반환된 직후 호출: 일시정지 구간을 제외한 실제 실행 시간을 누적합니다."""
        now = time.time()
        if self.__last_step_real_time is not None:
            delta = now - self.__last_step_real_time
            # 한 스텝이 1초를 초과하면 일시정지로 간주하고 누적하지 않음
            if delta < 1.0:
                self.__real_elapsed += delta
        self.__last_step_real_time = now

    def check_map_sending(self):
        """대회 종료 2초 전이 되면, 상태를 강제로 'send_map'으로 바꿔서 지도를 제출하게 합니다.
        플레이타임(시뮬레이션)과 누적 리얼타임 중 더 많이 경과한 쪽을 기준으로 판단합니다.
        일시정지 구간은 리얼타임 누적에서 제외되므로 Pause 중에 오작동하지 않습니다."""
        sim_elapsed  = self.mapper.time
        real_elapsed = self.__real_elapsed
        elapsed = max(sim_elapsed, real_elapsed)
        if elapsed > self.max_time_in_run - 2 and not self.map_sent:
            print(f"[타임아웃:executor.check_map_sending] ⚠ 제한 시간 임박! 강제 지도 전송 트리거 "
                  f"(시뮬={sim_elapsed:.1f}s / 리얼={real_elapsed:.1f}s / 한계={self.max_time_in_run}s)")
            self.state_machine.change_state("send_map")
            self.sequencer.reset_sequence()

    # =====================================================================
    # 상태 (STATES) 함수들 : StateMachine에 의해 특정 상황에 맞게 호출됨
    # =====================================================================

    def state_init(self, change_state_function):
        """[초기화 상태] 프로그램 시작 시 센서 영점을 잡고 시작 위치를 등록합니다."""
        self.sequencer.start_sequence()
        self.seq_print("[초기화:executor.state_init] ▶ 로봇 초기화 시작: 캘리브레이션 및 초기 스캔")
        self.seq_delay_seconds(0.5)

        self.sequencer.simple_event(self.calibrate_position_offsets) # 시작점 오차 보정
        self.sequencer.simple_event(self.mapper.register_start, self.robot.position) # 매퍼에 시작점 알림

        self.seq_calibrate_robot_rotation() # GPS를 이용해 자이로스코프 각도 영점 조절

        # 맵핑 및 조난자 탐지 기능 활성화
        if self.sequencer.simple_event():
            print("[초기화:executor.state_init] 맵핑 및 조난자 탐지 활성화, 초기 360° 스캔 시작")
            self.mapping_enabled = True
            self.victim_reporting_enabled = True

        self.seq_delay_seconds(0.5)
        # 로봇을 제자리에서 왼쪽으로 90도, 180도 돌리며 주변 환경 스캔(초기 지도 확보)
        self.sequencer.complex_event(self.robot.rotate_to_angle, angle=Angle(90, Angle.DEGREES), direction=RotationCriteria.LEFT)
        self.sequencer.complex_event(self.robot.rotate_to_angle, angle=Angle(180, Angle.DEGREES), direction=RotationCriteria.LEFT)
        self.seq_delay_seconds(0.5)

        # 스캔이 끝나면 '탐색(explore)' 상태로 변경!
        if self.sequencer.simple_event():
            print("[초기화:executor.state_init] ✓ 초기화 완료 → explore 탐색 단계 시작")
            change_state_function("explore")
        self.sequencer.seq_reset_sequence()


    def state_explore(self, change_state_function):
        """[탐색 상태] 실제 미로를 돌아다니며 길을 찾고 미션을 수행하는 메인 모드입니다."""
        self.sequencer.start_sequence() 

        if self.sequencer.simple_event():
            self.mapping_enabled = True

        # --- 1. 길 찾기 ---
        # Agent에게 목표 위치를 다시 계산하도록 업데이트
        self.agent.update() 
        self.mini_calibrate()

        # Agent가 계산해준 다음 좌표(목표)를 향해 로봇 이동!
        self.seq_move_to_coords(self.agent.get_target_position())

        self.sequencer.seq_reset_sequence() # 명령을 다 내렸으면 시퀀서 리셋

        # 에이전트가 "탐색 다 끝났어!"라고 하면 종료 상태로 전환

        # --- 2. 조난자(Fixture) 탐색 로직 ---
        if self.agent.do_end():
            self.state_machine.change_state("end")

        cam_images = self.robot.get_camera_images()

        # 오탐 직후 쿨다운 카운트다운 (>0이면 감지 스킵)
        if self.fixture_detection_cooldown > 0:
            self.fixture_detection_cooldown -= 1

        # 아직 맵에 조난자가 안 찍혀 있고, 카메라가 켜져 있고, 쿨다운이 끝났으면 확인
        elif self.victim_reporting_enabled and cam_images is not None and not self.mapper.has_detected_victim_from_position():
            for cam_image in cam_images:
                self.robot.lidar.get_detections()
                # 카메라 이미지에서 구조물(글자) 감지
                fixtures = self.fixture_detector.find_fixtures(cam_image.image)

                if len(fixtures):
                    print(f"[?먯깋:executor.state_explore] ??fixture 媛먯?! ?앸퀎 ?湲? ?쒖옉, 諛⑺뼢={self.report_orientation.degrees:.1f}째, ?꾩튂=({self.robot.position.x:.3f},{self.robot.position.y:.3f})m ??report_fixture ?꾪솚")
                    # 심볼(Φ, Ψ, Ω) 판별 후 서버 보고용 글자로 변환
                    detected_symbol = self.fixture_detector.classify_fixture(fixtures[0])

                    # None = 이미 처리된(already_detected) fixture → 보고 불필요, 다음 카메라로
                    if detected_symbol is None:
                        continue

                    self.letter_to_report = self._to_report_letter(detected_symbol)
                    if self.letter_to_report is None:
                        continue
                    self.report_orientation = cam_image.data.horizontal_orientation

                    print(f"[탐색:executor.state_explore] ★ fixture 감지! 글자='{self.letter_to_report}', 방향={self.report_orientation.degrees:.1f}°, 위치=({self.robot.position.x:.3f},{self.robot.position.y:.3f})m → report_fixture 전환")
                    # 상태를 'report_fixture(조난자 보고 모드)'로 변경하고 이번 루프 탈출
                    change_state_function("report_fixture")
                    self.sequencer.reset_sequence()
                    break
                

    def mini_calibrate(self):
        """가끔씩 로봇이 미세하게 멈추며 자세를 교정하는 함수"""
        if self.mini_calibrate_step_counter.check():
            self.seq_move_wheels(1, 1)
            self.seq_delay_seconds(0.1)
        self.mini_calibrate_step_counter.increase()

    def align_with_fixture(self):
        """[비전 제어] 조난자를 발견했을 때 카메라 중앙에 오도록 로봇 바퀴를 미세 조정하는 함수"""
        center_image = self.robot.center_camera.image.image
        fixtures = self.fixture_detector.find_fixtures(center_image)
        
        if len(fixtures) == 0: return True
        
        fixture = fixtures[0]
        fixture_shape = Position2D(fixture["image"].shape)
        fixture_position = Position2D(fixture["position"])
        fixture_center = fixture_shape / 2 + fixture_position
        image_center = Position2D(center_image.shape) / 2

        # 화면 중앙과 글자 위치 간의 픽셀 차이(오차) 계산
        diff = image_center - fixture_center

        # Y축 오차가 크면 앞뒤로 이동
        if abs(diff.y) > 4:
            vel = diff.y * 0.1
            self.robot.move_wheels(vel, vel)
            return False # 아직 정렬 안 끝남

        # X축 오차가 크면 제자리 회전
        if abs(diff.x) > 6:
            sign = np.sign(diff.x)
            vel = 0.1
            self.robot.move_wheels(vel * sign, vel * -sign)
            return False # 아직 정렬 안 끝남

        return True # 정렬 완벽함!


    def state_end(self, change_state_function):
        """[미션 완료 상태] 완성된 지도를 추출하고 통신으로 본부에 보낸 후 끝냅니다."""
        final_matrix = self.final_matrix_creator.pixel_grid_to_final_grid(self.mapper.pixel_grid, self.mapper.start_position)
        self.robot.comunicator.send_map(final_matrix)
        self.robot.comunicator.send_end_of_play()

    def state_send_map(self, change_state_function):
        """[지도 강제 전송 상태] 시간이 다 되었을 때 지도를 보내고 다시 탐색으로 돌아갑니다."""
        final_matrix = self.final_matrix_creator.pixel_grid_to_final_grid(self.mapper.pixel_grid, self.mapper.start_position)
        self.robot.comunicator.send_map(final_matrix)
        self.map_sent = True
        change_state_function("explore")
        self.sequencer.seq_reset_sequence()

    def state_report_fixture(self, change_state_function):
        """[조난자 보고 상태] 조난자에게 다가가서 사진을 찍고 서버에 점수를 획득하는 모드"""
        # 5초 타임아웃 체크 (32ms step 기준 약 156 steps = 5s)
        self.report_timeout_counter += 1
        if self.report_timeout_counter > 160:
            print("[조난자 보고:executor.state_report_fixture] ⚠ 5초 경과(타임아웃) 또는 벽 충돌 의심 -> explore 강제 복귀")
            self.report_timeout_counter = 0
            self.mapping_enabled = True
            self.letter_to_report = None
            self.sequencer.reset_sequence()
            change_state_function("explore")
            return

        self.sequencer.start_sequence()
        self.seq_move_wheels(0, 0) # 일단 정지

        if self.letter_to_report is not None:
            # 보고 중에는 지도를 그리지 않음
            if self.sequencer.simple_event():
                print(f"[조난자 보고:executor.state_report_fixture] ▶ 보고 시퀀스 시작: 글자='{self.letter_to_report}', 방향={self.report_orientation.degrees:.1f}°")
                self.mapping_enabled = False

            # 조난자가 있는 방향으로 정확히 회전
            if self.sequencer.simple_event():
                self.report_orientation.normalize()
            self.seq_rotate_to_angle(self.report_orientation.degrees)

            # 카메라 정중앙에 오도록 미세 정렬
            self.seq_align_with_fixture()
            self.seq_move_wheels(0, 0)

                # 다시 한 번 카메라로 확실하게 심볼(Φ, Ψ, Ω) 식별
            if self.sequencer.simple_event():
                center_image = self.robot.center_camera.image.image
                fixtures = self.fixture_detector.find_fixtures(center_image)
            
                # 만약 놓쳤으면 쿨하게 다시 탐색으로 복귀
                if len(fixtures) == 0:
                    # 재확인 실패 = 오탐. 이 위치를 블랙리스트에 올리고
                    # 쿨다운 동안 감지를 억제하여 로봇이 해당 구역을 벗어날 시간을 줍니다.
                    self.mapper.fixture_mapper.map_detected_fixture(self.robot.position)
                    self.fixture_detection_cooldown = 150  # ~150프레임(≈4.8초) 동안 감지 스킵
                    print(f"[조난자 보고:executor.state_report_fixture] ✗ 재확인 중 fixture 소실 → 오탐 블랙리스트 등록 + {self.fixture_detection_cooldown}프레임 쿨다운 후 explore 복귀")
                    change_state_function("explore")
                    self.sequencer.reset_sequence()
                    self.mapping_enabled = True
                    return

                detected_symbol = self.fixture_detector.classify_fixture(fixtures[0])
                self.letter_to_report = self._to_report_letter(detected_symbol) if detected_symbol is not None else None
                if self.letter_to_report is None:
                    print("[조난자 보고:executor.state_report_fixture] ✗ 심볼 인식 실패(already_detected) 또는 변환 실패 → explore 복귀")
                    change_state_function("explore")
                    self.sequencer.reset_sequence()
                    self.mapping_enabled = True
                    return

            # 조난자를 향해 앞으로 살짝 다가감 (0.6 속도로 0.1초 전진)
            self.seq_move_wheels(0.6, 0.6)
            self.seq_delay_seconds(0.1)
            self.seq_move_wheels(0, 0)
            self.seq_delay_seconds(1.5) # 확실한 사진/보고를 위해 1.5초 대기

            # 서버 통신: "조난자 찾았습니다!"
            if self.sequencer.simple_event():
                print(f"[조난자 보고:executor.state_report_fixture] 보고글자='{self.letter_to_report}', 위치=({self.robot.raw_position.x:.4f}, {self.robot.raw_position.y:.4f})m, 시뮬 시간={self.robot.time:.1f}s")
                self.robot.comunicator.send_victim(self.robot.raw_position, self.letter_to_report)
                if self.letter_to_report in ('F', 'P', 'C', 'O'):
                    idx = self.mapper.pixel_grid.coordinates_to_array_index(self.robot.position)
                    rr, cc = skimage.draw.disk(idx, 4)
                    shape = self.mapper.pixel_grid.array_shape
                    valid = (rr >= 0) & (cc >= 0) & (rr < shape[0]) & (cc < shape[1])
                    self.mapper.pixel_grid.arrays["hazmats"][rr[valid], cc[valid]] = True
        
            self.seq_delay_seconds(0.1)
        
            # 지도에 조난자 마킹
            if self.sequencer.simple_event():
                self.mapper.fixture_mapper.map_detected_fixture(self.robot.position)

            # 다시 뒤로 물러나서 원래 위치 확보
            self.seq_move_wheels(-0.6, -0.6)
            self.seq_delay_seconds(0.1)
            self.seq_move_wheels(0, 0)

            if self.sequencer.simple_event():
                self.letter_to_report = None
            
        # 다 끝났으면 맵핑 다시 켜고 탐색으로 복귀
        if self.sequencer.simple_event():
            print("[조난자 보고:executor.state_report_fixture] ✓ 보고 완료, 맵핑 재개 → explore 복귀")
            self.report_timeout_counter = 0 # 카운터 초기화
            self.mapping_enabled = True
        self.sequencer.simple_event(change_state_function, "explore")
        self.sequencer.seq_reset_sequence()

    def _to_report_letter(self, symbol: str):
        """인식된 심볼을 서버 보고용 글자로 변환합니다."""
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

    def state_stuck(self, change_state_function):
        """[탈출 상태] 로봇이 벽에 끼였을 때 빠져나오기 위한 몸부림"""
        self.sequencer.start_sequence()
        if self.sequencer.simple_event():  # 탈출 시작 시 1회만 출력
            print(f"[끼임 감지:executor.state_stuck] ⚠ 로봇 끼임! 탈출 시도 시작: 위치=({self.robot.position.x:.4f},{self.robot.position.y:.4f})m, 방향={self.robot.orientation.degrees:.1f}°, 시뮬 시간={self.robot.time:.1f}s")
        # 뒤로 빼기 -> 대기 -> 오른쪽 바퀴만 뒤로 굴려 제자리 턴
        self.seq_move_wheels(-0.6, -0.6)
        self.seq_delay_seconds(0.2)
        self.seq_move_wheels(0.6, -0.6)
        self.seq_delay_seconds(1)

        # 탈출 시도 후 다시 탐색 모드로 전환
        if self.sequencer.simple_event():
            print(f"[끼임 감지:executor.state_stuck] ✓ 탈출 시도 완료 → explore 복귀, 현재 위치=({self.robot.position.x:.4f},{self.robot.position.y:.4f})m")
            change_state_function("explore")
        self.sequencer.seq_reset_sequence()

    def calibrate_position_offsets(self):
        """로봇이 처음에 정가운데 시작하지 않았을 때의 좌표 오차 계산"""
        self.robot.position_offsets = self.robot.position % (self.mapper.quarter_tile_size * 2)
        
    def seq_calibrate_robot_rotation(self):
        """[초기 캘리브레이션] 후진으로 GPS 방향을 안전하게 측정하고 자이로 영점을 맞추는 시퀀스.
        전방에 블랙홀이 있을 수 있으므로 후진(안전 방향)으로 5cm 이동하여 GPS heading을 취득,
        π를 더해 실제 전방 heading을 산출한 뒤 자이로에 세팅한다."""
        if self.sequencer.simple_event():
            self.robot.auto_decide_orientation_sensor = False

        # GPS baseline 초기화 → 이후 후진 구간만으로 방향 계산
        self.sequencer.simple_event(self.robot.gps.reset_orientation_baseline)

        # 후진 0.4s (~5cm) — 전방 블랙홀 반대 방향이므로 안전
        self.seq_move_wheels(-1, -1)
        self.seq_delay_seconds(0.4)
        self.seq_move_wheels(0, 0)

        # GPS heading 취득 (후진 방향) → π 더해서 전방 heading → 자이로에 세팅
        if self.sequencer.simple_event():
            gps_heading = self.robot.gps.get_orientation()
            if gps_heading is not None:
                true_forward = Angle(gps_heading.radians + np.pi)
                true_forward.normalize()
                self.robot.gyroscope.set_orientation(true_forward)

        # GPS baseline 재초기화 (복귀 이동이 heading 계산에 영향 안 주도록)
        self.sequencer.simple_event(self.robot.gps.reset_orientation_baseline)

        # 전진 0.4s로 원래 위치 복귀 (시작점, 안전)
        self.seq_move_wheels(1, 1)
        self.seq_delay_seconds(0.4)
        self.seq_move_wheels(0, 0)

        if self.sequencer.simple_event():
            self.robot.auto_decide_orientation_sensor = True
