from mapping.mapper import Mapper
from data_structures.vectors import Position2D
import numpy as np
import cv2 as cv

from algorithms.np_bool_array.bfs import BFSAlgorithm, NavigatingBFSAlgorithm
from flags import SHOW_BEST_POSITION_FINDER_DEBUG

from agent.agent_interface import PositionFinderInterface

class PositionFinder(PositionFinderInterface):
    """
    아직 탐색되지 않은(discovered=False) 가장 가까운 위치를 BFS로 찾는 클래스입니다.

    discovered 레이어가 False인 픽셀 = 로봇 시야(전방 170도)에 한 번도 들어오지 않은 영역
    이 클래스는 그런 미탐색 영역 중 가장 가까이 접근 가능한 위치를 반환합니다.
    """
    def __init__(self, mapper: Mapper) -> None:
        self.mapper = mapper
        # 탐색: discovered=False 인 위치 탐색 (최대 1개), traversable=False 인 곳만 통과
        self.closest_unseen_finder = NavigatingBFSAlgorithm(
            found_function=lambda x: x == False,
            traversable_function=lambda x: x == False,
            max_result_number=1)

        # 로봇이 traversable 영역에 있을 때 가장 가까운 통과 가능 위치 탐색
        self.closest_free_point_finder = BFSAlgorithm(lambda x: x == 0)

        self.closest_unseen_grid_index = None

    def update(self, force_calculation=False):
        """
        목표 위치가 통과 불가 영역이 되었거나 강제 재계산 요청 시 새 미탐색 위치를 찾습니다.
        """
        if self.__is_objective_untraversable() or force_calculation:
            self.closest_unseen_grid_index = self.__get_closest_unseen_grid_index()

        if SHOW_BEST_POSITION_FINDER_DEBUG:
            debug_grid = self.mapper.pixel_grid.get_colored_grid()
            if self.target_position_exists():
                closest_unseen_array_index = self.mapper.pixel_grid.grid_index_to_array_index(
                    self.closest_unseen_grid_index)
                cv.circle(debug_grid,
                          (closest_unseen_array_index[1], closest_unseen_array_index[0]),
                          4, (0, 255, 100), -1)
                cv.imshow("closest_position_finder_debug", debug_grid)

    def get_target_position(self):
        """미탐색 목표 위치의 실제 좌표(m)를 반환합니다."""
        if self.target_position_exists():
            coords = self.mapper.pixel_grid.grid_index_to_coordinates(self.closest_unseen_grid_index)
            return Position2D(coords)

    def target_position_exists(self) -> bool:
        return self.closest_unseen_grid_index is not None

    def __is_objective_untraversable(self):
        """현재 목표 위치가 traversable(통과 불가) 영역이 되었는지 확인합니다."""
        if self.target_position_exists():
            closest_unseen_array_index = self.mapper.pixel_grid.grid_index_to_array_index(
                self.closest_unseen_grid_index)
            return self.mapper.pixel_grid.arrays["traversable"][
                closest_unseen_array_index[0], closest_unseen_array_index[1]]
        else:
            return False

    def __get_closest_unseen_grid_index(self):
        """
        현재 로봇 위치에서 BFS로 가장 가까운 미탐색(discovered=False) 위치를 탐색합니다.
        로봇이 traversable 영역에 있으면 먼저 통과 가능한 가장 가까운 위치로 출발점을 보정합니다.
        """
        robot_array_index = self.mapper.pixel_grid.coordinates_to_array_index(
            self.mapper.robot_position)
        start_node = self.__get_closest_traversable_array_index(robot_array_index)

        closest_unseen_array_indexes = self.closest_unseen_finder.bfs(
            found_array=self.mapper.pixel_grid.arrays["discovered"],
            traversable_array=self.mapper.pixel_grid.arrays["traversable"],
            start_node=start_node)

        if len(closest_unseen_array_indexes):
            grid_idx = self.mapper.pixel_grid.array_index_to_grid_index(closest_unseen_array_indexes[0])
            coords = self.mapper.pixel_grid.grid_index_to_coordinates(grid_idx)
            print(f"[미탐색:go_to_non_discovered_position_finder.__get_closest_unseen_grid_index] 미탐색 위치 발견: 배열인덱스={closest_unseen_array_indexes[0]}, 좌표=({coords[0]:.3f},{coords[1]:.3f})m")
            return grid_idx
        else:
            print("[미탐색:go_to_non_discovered_position_finder.__get_closest_unseen_grid_index] 미탐색 위치 없음: 전체 지도가 탐색 완료되었거나 BFS 도달 불가")
            return None

    def __get_closest_traversable_array_index(self, array_index):
        """로봇이 traversable 영역에 있으면 가장 가까운 통과 가능 위치로 이동하여 BFS를 시작합니다."""
        if self.mapper.pixel_grid.arrays["traversable"][array_index[0], array_index[1]]:
            return self.closest_free_point_finder.bfs(
                array=self.mapper.pixel_grid.arrays["traversable"],
                start_node=array_index)
        else:
            return array_index
