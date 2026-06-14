import numpy as np
import cv2 as cv
from data_structures.vectors import Position2D

from algorithms.np_bool_array.bfs import NavigatingLimitedBFSAlgorithm, NavigatingBFSAlgorithm

from agent.agent_interface import PositionFinderInterface
from mapping.mapper import Mapper


class PositionFinder(PositionFinderInterface):
    """
    조난자(victim)가 감지된 위치 주변(zone of influence)에서
    아직 방문하지 않은 가장 가까운 목표 위치를 BFS로 탐색하는 클래스입니다.

    동작 원리:
    1. victims 레이어에 마킹된 조난자 위치에 원형 커널(zone of influence) 적용
    2. fixture_distance_margin과 AND 연산 → 유효한 도달 가능 후보 영역 도출
    3. 이미 지나간(robot_center_traversed) 위치 제거
    4. BFS로 현재 로봇 위치에서 가장 가까운 후보 선택 (탐색 한도: 1000)
    """
    def __init__(self, mapper: Mapper) -> None:
        self.__mapper = mapper
        # 탐색 한도 1000으로 제한된 BFS (후보: fixture_distance_margin & 조난자 영역, 통과: 벽 아닌 곳)
        self.__next_position_finder = NavigatingLimitedBFSAlgorithm(lambda x: x, lambda x: not x, limit=1000)
        # 목표가 아직 도달 가능한지 확인하는 BFS (traversed 레이어에서 연결 여부 판별)
        self.__still_reachable_bfs = NavigatingLimitedBFSAlgorithm(lambda x: x, lambda x: not x, limit=500)
        self.__target = None
        self.__target_age = 0
        self.__blacklisted_grid_indices: list = []

        # 조난자 영향 반경: 로봇 반경 + 여유 3픽셀
        circle_radius = round(self.__mapper.robot_diameter / 2 * self.__mapper.pixel_grid.resolution) + 3

        # 원형 커널 생성 - 조난자 주변 원 영역을 채워 zone of influence 계산에 사용
        self.circle_kernel = np.zeros((circle_radius * 2, circle_radius * 2), dtype=np.uint8)
        self.circle_kernel = cv.circle(self.circle_kernel, (circle_radius, circle_radius), circle_radius, 1, -1)
        self.__safety_kernel = np.ones((5, 5), dtype=np.uint8)
        self.__traversable_dilated_cache = None
        self.__traversable_shape_cache = None


    def update(self, force_calculation=False) -> None:
        """
        다음 조건 중 하나라도 해당하면 목표 위치를 재계산합니다:
        - 목표가 없음
        - 목표까지 경로가 막혔음
        - 로봇이 이미 그 위치를 지나쳤음
        - 강제 재계산 요청
        """
        needs_recalc = (
            not self.target_position_exists() or
            not self.__is_grid_index_still_reachable(self.__target) or
            self.__already_passed_through_grid_index(self.__target) or
            force_calculation
        )

        if needs_recalc:
            self.__target_age = 0
        else:
            self.__target_age += 1
            if self.__target_age >= 60:
                self.__blacklisted_grid_indices.append(self.__target)
                self.__target = None
                self.__target_age = 0
                needs_recalc = True

        if needs_recalc:
            self.__calculate_position()

    def get_target_position(self) -> Position2D:
        """현재 목표 위치의 실제 좌표(m)를 반환합니다."""
        if self.target_position_exists():
            return self.__mapper.pixel_grid.grid_index_to_coordinates(self.__target)

    def target_position_exists(self) -> bool:
        return self.__target is not None
    def __calculate_position(self):
        """
        조난자 주변 영역과 체크포인트 위치에서 후보를 계산하고,
        이미 지나간 위치를 제거한 후 BFS로 가장 가까운 위치를 선택합니다.
        """
        # 조난자 zone of influence와 fixture_distance_margin의 교집합
        possible_targets_array = self.__get_fixtures_zone_of_influence()
        # 체크포인트도 목표로 추가
        possible_targets_array += self.__mapper.pixel_grid.arrays["checkpoints"]
        # 이미 로봇 중심이 지나간 위치는 후보에서 제거
        possible_targets_array[self.__mapper.pixel_grid.arrays["robot_center_traversed"]] = False

        current_shape = self.__mapper.pixel_grid.arrays["traversable"].shape
        if self.__traversable_dilated_cache is None or current_shape != self.__traversable_shape_cache:
            self.__traversable_dilated_cache = cv.dilate(
                self.__mapper.pixel_grid.arrays["traversable"].astype(np.uint8),
                self.__safety_kernel
            ).astype(np.bool_)
            self.__traversable_shape_cache = current_shape
        possible_targets_array[self.__traversable_dilated_cache] = False

        for gi in self.__blacklisted_grid_indices:
            ai = self.__mapper.pixel_grid.grid_index_to_array_index(gi)
            r, c = int(ai[0]), int(ai[1])
            if 0 <= r < possible_targets_array.shape[0] and 0 <= c < possible_targets_array.shape[1]:
                possible_targets_array[r, c] = False

        if not np.any(possible_targets_array):
            self.__target = None
            return

        robot_array_index = self.__mapper.pixel_grid.grid_index_to_array_index(self.__mapper.robot_grid_index)

        # BFS로 탐색 가능한 가장 가까운 후보 선택 (최대 1000 루프 제한)
        results = self.__next_position_finder.bfs(possible_targets_array, self.__mapper.pixel_grid.arrays["traversable"], robot_array_index)

        self.__target = self.__mapper.pixel_grid.array_index_to_grid_index(results[0]) if len(results) else None
        self.__target_age = 0


    def __is_grid_index_still_reachable(self, grid_index) -> bool:
        """목표 위치가 traversable 영역에 있지 않고, traversed 경로와 연결되어 있는지 확인합니다."""
        start_array_index = self.__mapper.pixel_grid.grid_index_to_array_index(grid_index)

        # 목표가 통과 불가 영역이면 재계산 필요
        if self.__mapper.pixel_grid.arrays["traversable"][start_array_index[0], start_array_index[1]]:
             return False

        # traversed(지나간 경로) 레이어에서 목표까지 BFS로 연결 여부 확인
        results = self.__still_reachable_bfs.bfs(self.__mapper.pixel_grid.arrays["traversed"], self.__mapper.pixel_grid.arrays["traversable"], start_array_index)

        return bool(len(results))

    def __already_passed_through_grid_index(self, grid_index):
        """로봇 중심이 이미 이 위치를 지나쳤으면 True 반환 (목표 재탐색 필요)."""
        array_index = self.__mapper.pixel_grid.grid_index_to_array_index(grid_index)

        return self.__mapper.pixel_grid.arrays["robot_center_traversed"][array_index[0], array_index[1]]


    def __get_fixtures_zone_of_influence(self) -> np.ndarray:
        """
        victims 레이어에 원형 커널을 컨볼루션하여 각 조난자 주변의 원형 영역을 생성하고,
        fixture_distance_margin 영역과 AND 연산하여 최종 후보 영역을 반환합니다.
        """
        # victims 배열에 원형 커널 필터를 적용하여 zone of influence 계산
        zones = cv.filter2D(self.__mapper.pixel_grid.arrays["victims"].astype(np.uint8), -1, self.circle_kernel).astype(np.bool_)

        # 벽 근처 마진 영역과만 교집합 → 로봇이 실제로 도달 가능한 위치로 한정
        return np.bitwise_and(zones, self.__mapper.pixel_grid.arrays["fixture_distance_margin"])
