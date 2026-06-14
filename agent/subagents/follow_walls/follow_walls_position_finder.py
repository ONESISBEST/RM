import numpy as np
import cv2 as cv
import math
from data_structures.vectors import Position2D

from algorithms.np_bool_array.bfs import NavigatingBFSAlgorithm

from agent.agent_interface import PositionFinderInterface
from mapping.mapper import Mapper


class PositionFinder(PositionFinderInterface):
    """
    fixture_distance_margin 레이어(벽 조난자 도달 가능 마진 영역)에서
    아직 방문하지 않은 최적 목표 위치를 BFS로 탐색하는 클래스입니다.

    동작 원리:
    1. fixture_distance_margin 레이어에서 후보 위치 추출
    2. 너무 고립된(주변 포인트 부족) 후보 제거
    3. 이미 로봇 중심이 지나간 곳 제거
    4. BFS로 현재 위치에서 가장 가까이 갈 수 있는 후보 선택
    """
    def __init__(self, mapper: Mapper) -> None:
        self.__mapper = mapper
        # BFS: 후보 위치 찾기 (found: 후보인 곳, traversable: 벽이 아닌 곳)
        self.__next_position_finder = NavigatingBFSAlgorithm(lambda x: x, lambda x: not x)
        self.__still_reachable_bfs = NavigatingBFSAlgorithm(lambda x: x, lambda x: not x)
        self.__target = None

        smoother_template_radious = int(0.03 * self.__mapper.pixel_grid.resolution)  # 약 3cm 반경
        smoother_template_diameter = math.ceil(smoother_template_radious * 2 + 1)

        self.min_number_to_be_valid = 10  # 주변 포인트가 이 수 미만이면 고립 포인트로 제거

        # 고립 포인트 감지용 도넛 모양 커널 (중심 제외 솔리드 디스크)
        self.smoother_template = np.zeros((smoother_template_diameter, smoother_template_diameter), dtype=np.int8)
        self.smoother_template = cv.circle(self.smoother_template,
                                            (smoother_template_radious, smoother_template_radious),
                                            smoother_template_radious, 1, -1)
        self.smoother_template[smoother_template_radious, smoother_template_radious] = 0  # 중심 제외

    def update(self, force_calculation=False) -> None:
        """
        목표 위치를 재계산합니다.
        - 목표가 없거나
        - 목표까지 경로가 막혔거나
        - 이미 로봇이 지나쳤거나
        - 강제 재계산 요청 시
        """
        if not self.target_position_exists() or \
           not self.__is_grid_index_still_reachable(self.__target) or \
           self.__already_passed_through_grid_index(self.__target) or \
           force_calculation:
            self.__calculate_position()

    def get_target_position(self) -> Position2D:
        if self.target_position_exists():
            return self.__mapper.pixel_grid.grid_index_to_coordinates(self.__target)

    def target_position_exists(self) -> bool:
        return self.__target is not None

    def __calculate_position(self):
        """
        fixture_distance_margin에서 유효한 후보 위치를 찾아 BFS로 가장 가까운 곳을 선택합니다.
        1. 고립 포인트 제거
        2. 이미 방문한 곳 제거
        3. 늪지대 제거
        4. BFS로 탐색 가능한 가장 가까운 위치 선택
        """
        possible_targets_array = self.__mapper.pixel_grid.arrays["fixture_distance_margin"]

        # 주변 포인트가 부족한 고립 후보 제거
        isolated = cv.filter2D(
            self.__mapper.pixel_grid.arrays["fixture_distance_margin"].astype(np.uint8),
            -1, self.smoother_template) < self.min_number_to_be_valid
        possible_targets_array[isolated] = False

        # 격자 패턴으로 후보 희소화 (연산 효율화)
        self.__dither_array(possible_targets_array, dither_interval=2)

        # 이미 지나간 중심 경로 위치 제거
        possible_targets_array[self.__mapper.pixel_grid.arrays["robot_center_traversed"]] = False
        # 통과 불가 영역 내 후보 제거
        self.__mapper.pixel_grid.arrays["fixture_distance_margin"][
            self.__mapper.pixel_grid.arrays["traversable"]] = False
        # 늪지대 위 후보 제거
        self.__mapper.pixel_grid.arrays["fixture_distance_margin"][
            self.__mapper.pixel_grid.arrays["swamps"]] = False

        if not np.any(possible_targets_array):
            print("[벽 추종:follow_walls_position_finder.__calculate_position] 후보 위치 없음: fixture_distance_margin에 유효한 미방문 위치가 없습니다")
            self.__target = None
            return

        robot_array_index = self.__mapper.pixel_grid.grid_index_to_array_index(
            self.__mapper.robot_grid_index)

        results = self.__next_position_finder.bfs(
            possible_targets_array,
            self.__mapper.pixel_grid.arrays["traversable"],
            robot_array_index)

        self.__target = self.__mapper.pixel_grid.array_index_to_grid_index(
            results[0]) if len(results) else None

    def __is_grid_index_still_reachable(self, grid_index) -> bool:
        """목표 위치가 여전히 통과 가능하고 로봇이 지나간 경로에서 접근 가능한지 확인합니다."""
        start_array_index = self.__mapper.pixel_grid.grid_index_to_array_index(grid_index)

        if self.__mapper.pixel_grid.arrays["traversable"][start_array_index[0], start_array_index[1]]:
            return False  # 목표가 통과 불가 영역이면 재계산 필요

        results = self.__still_reachable_bfs.bfs(
            self.__mapper.pixel_grid.arrays["traversed"],
            self.__mapper.pixel_grid.arrays["traversable"],
            start_array_index)

        return bool(len(results))

    def __already_passed_through_grid_index(self, grid_index):
        """로봇 중심이 이미 이 위치를 지나쳤으면 True 반환 (목표 재탐색 필요)."""
        array_index = self.__mapper.pixel_grid.grid_index_to_array_index(grid_index)
        return self.__mapper.pixel_grid.arrays["robot_center_traversed"][array_index[0], array_index[1]]

    def __dither_array(self, possible_targets_array: np.ndarray, dither_interval=2):
        """
        후보 배열을 격자 패턴으로 희소화합니다.
        dither_interval 픽셀마다 한 번만 후보를 남겨 연산량을 줄입니다.
        """
        mask = np.ones_like(possible_targets_array)
        mask[::dither_interval, ::dither_interval] = False
        mask[dither_interval//2::dither_interval, dither_interval//2::dither_interval] = False
        possible_targets_array[mask] = False
