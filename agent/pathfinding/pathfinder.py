import numpy as np
import cv2 as cv

from data_structures.vectors import Position2D
from mapping.mapper import Mapper

from data_structures.compound_pixel_grid import CompoundExpandablePixelGrid
from algorithms.np_bool_array.efficient_a_star import aStarAlgorithm
from algorithms.np_bool_array.bfs import NavigatingBFSAlgorithm

from agent.pathfinding.path_smoothing import PathSmoother

from flags import SHOW_PATHFINDING_DEBUG, SHOW_GRANULAR_NAVIGATION_GRID

class PathFinder():
    """
    A* 알고리즘을 사용하여 현재 로봇 위치에서 목표 위치까지의 경로를 계산하는 클래스입니다.

    동작 흐름:
    1. update() 호출 시 목표 위치 변경 여부, 경로 완료, 경로 막힘 여부를 확인
    2. 필요하면 __calculate_path()로 A* 경로 재계산
    3. 경로를 dither(2칸마다 1개로 희소화)하고 PathSmoother로 부드럽게 처리
    4. get_next_position()이 현재 인덱스의 평활화된 경로 좌표를 반환
    5. 로봇이 현재 경로 노드에서 3픽셀 이내로 접근하면 다음 노드로 인덱스 전진
    """
    def __init__(self, mapper: Mapper):
        self.__a_star = aStarAlgorithm()
        # traversable=0인 위치를 찾는 BFS (시작/목표가 장애물 위에 있을 때 보정용)
        self.__closest_free_point_finder = NavigatingBFSAlgorithm(lambda x : x == 0, lambda x: True)

        # 경로 평활화 강도: 1 (이웃 노드 평균)
        self.__a_star_path_smoother = PathSmoother(1)

        self.__robot_grid_index = np.array([0, 0])
        self.__target_position = np.array([0, 0])

        self.__a_star_path = []         # A* 계산된 원본 경로 (grid index 목록)
        self.__smooth_astar_path = []   # 평활화된 경로
        self.__a_star_index = 0         # 현재 따라가는 경로 인덱스

        self.__mapper = mapper

        self.path_not_found = False     # 경로 탐색 실패 플래그
        self.__position_changed = True  # 목표 위치 변경 여부 플래그

    def update(self, target_position: np.ndarray = None, force_calculation=False) -> None:
        """
        경로를 갱신합니다.
        - 목표 위치가 변경되었거나
        - 경로가 완료되었거나
        - 경로가 장애물에 막혔거나
        - 강제 재계산 요청 시 A* 경로를 재계산합니다.
        """
        if target_position is not None:
            # 목표 위치가 실제로 변경되었는지 확인
            self.__position_changed = np.any(self.__target_position != target_position)
            self.__target_position = target_position

        # 로봇의 현재 그리드 인덱스 갱신 및 그리드 자동 확장
        self.__robot_grid_index = self.__mapper.pixel_grid.coordinates_to_grid_index(self.__mapper.robot_position)
        self.__mapper.pixel_grid.expand_to_grid_index(self.__robot_grid_index)

        if SHOW_PATHFINDING_DEBUG:
            if self.is_path_finished(): print(f"[경로탐색:pathfinder.update] 경로 완료 (인덱스={self.__a_star_index}/{len(self.__a_star_path)-1}) → 재계산 필요")
            if self.__is_path_obstructed(): print(f"[경로탐색:pathfinder.update] 경로 차단 감지 (노드 수={len(self.__a_star_path)}) → 재계산 필요")

        # 재계산이 필요한 조건: 경로 완료 | 막힘 | 목표 변경 | 강제
        if self.is_path_finished() or self.__is_path_obstructed() or self.__position_changed or force_calculation:
            self.__calculate_path()

        # 로봇 접근에 따른 경로 인덱스 전진
        self.__calculate_path_index()

        # 기존 디버그: 경로 노드를 지도 위에 파란 점으로 표시
        if SHOW_GRANULAR_NAVIGATION_GRID:
            debug_grid = self.__mapper.pixel_grid.get_colored_grid()
            for node in self.__a_star_path:
                n = np.array(self.__mapper.pixel_grid.grid_index_to_array_index(node))
                try:
                    debug_grid[n[0], n[1]] = [0, 0, 255]
                except IndexError:
                    pass
            cv.imshow("granular_grid", debug_grid)

        # 실시간 지도 시각화에 현재 경로 전달
        if self.__mapper.visualizer is not None:
            target_ai = self.__mapper.pixel_grid.coordinates_to_array_index(
                self.__target_position) if len(self.__a_star_path) else None
            self.__mapper.visualizer.set_path(self.__a_star_path)
            self.__mapper.visualizer.set_target(target_ai)


    def __calculate_path(self):
        """
        A* 알고리즘으로 현재 위치 → 목표 위치 경로를 계산합니다.
        시작점/목표점이 장애물 위에 있으면 가장 가까운 통과 가능 위치로 보정합니다.
        경로 계산 후 dither(희소화) + smooth(평활화) 처리를 합니다.
        """
        # 시작점: 로봇 현재 위치 (장애물 위면 가장 가까운 빈 공간으로 이동)
        start_array_index = self.__mapper.pixel_grid.coordinates_to_array_index(self.__mapper.robot_position)
        start_array_index = self.__get_closest_traversable_array_index(start_array_index)

        # 목표점: 그리드를 먼저 확장한 후 배열 인덱스로 변환
        target_grid_index = self.__mapper.pixel_grid.coordinates_to_grid_index(self.__target_position)
        self.__mapper.pixel_grid.expand_to_grid_index(target_grid_index)

        target_array_index = self.__mapper.pixel_grid.coordinates_to_array_index(self.__target_position)
        target_array_index = self.__get_closest_traversable_array_index(target_array_index)

        # A* 실행: traversable(장애물) 배열과 navigation_preference(선호도) 배열 사용
        best_path = self.__a_star.a_star(self.__mapper.pixel_grid.arrays["traversable"],
                                        start_array_index,
                                        target_array_index,
                                        self.__mapper.pixel_grid.arrays["navigation_preference"])

        # 경로 계산 성공: array index → grid index 변환 후 저장 (시작점 제외)
        if len(best_path) > 1:
            self.__a_star_path = []
            for array_index in best_path:
                self.__a_star_path.append(self.__mapper.pixel_grid.array_index_to_grid_index(array_index))

            self.__a_star_path = self.__a_star_path[1:]  # 현재 위치(시작점) 제거
            self.__a_star_index = 0
            self.path_not_found = False
        else:
            if SHOW_PATHFINDING_DEBUG: print(f"[경로탐색:pathfinder.__calculate_path] 경로 탐색 실패: 시작={tuple(start_array_index)}, 목표={tuple(target_array_index)}")
            print(f"[경로탐색:pathfinder.__calculate_path] 경로 없음: 시작({self.__mapper.robot_position.x:.3f},{self.__mapper.robot_position.y:.3f})m → 목표({self.__target_position[0]:.3f},{self.__target_position[1]:.3f})m")
            self.path_not_found = True

        # 경로 희소화(2칸에 1개) 후 평활화 처리
        self.__a_star_path = self.__dither_path(self.__a_star_path)
        self.__smooth_astar_path = self.__a_star_path_smoother.smooth(self.__a_star_path)

    def __calculate_path_index(self):
        """
        현재 로봇이 경로상 노드에 충분히 가까워지면(< 3픽셀) 다음 노드로 인덱스를 전진합니다.
        """
        self.__a_star_index = min(self.__a_star_index, len(self.__a_star_path) - 1)
        if len(self.__a_star_path) > 0:
            next_node = self.__a_star_path[self.__a_star_index]
            next_node = Position2D(next_node)

            current_grid_index = self.__mapper.pixel_grid.coordinates_to_grid_index(self.__mapper.robot_position)
            current_node = Position2D(current_grid_index[0], current_grid_index[1])

            # 3픽셀 이내로 접근 시 다음 경로 노드로 전진
            if abs(current_node.get_distance_to(next_node)) < 3:
                self.__a_star_index += 1

    def __dither_path(self, path):
        """
        경로를 2칸마다 1개의 노드로 희소화합니다.
        마지막 노드는 항상 유지하여 목표 위치가 보존되도록 합니다.
        """
        if not len(path):
            return path
        final_path = []
        dither_interval = 2
        for index, value in enumerate(path):
            if index % dither_interval == 0:
                final_path.append(value)
        # 마지막 노드(목표점)가 희소화로 제거되었으면 다시 추가
        if tuple(final_path[-1]) != tuple(path[-1]):
            final_path.append(path[-1])

        if len(final_path):
            return final_path
        else:
            return path

    def get_next_position(self) -> Position2D:
        """
        현재 경로 인덱스에 해당하는 평활화된 목표 좌표를 반환합니다.
        경로가 없으면 로봇의 현재 위치를 반환합니다.
        """
        self.__a_star_index = min(self.__a_star_index, len(self.__a_star_path) -1)
        if len(self.__smooth_astar_path):
            pos = self.__mapper.pixel_grid.grid_index_to_coordinates(np.array(self.__smooth_astar_path[self.__a_star_index]))
            pos = Position2D(pos[0], pos[1])
            return pos

        else:
            return self.__mapper.robot_position

    def __is_path_obstructed(self):
        """
        현재 A* 경로 위에 장애물이 생겼는지 확인합니다.
        경로의 어느 노드든 traversable=True이면 막힌 것으로 판단합니다.
        """
        array_index_path = []
        for n in self.__a_star_path:
            array_index_path.append(self.__mapper.pixel_grid.grid_index_to_array_index(n))

        for position in array_index_path:
            # 배열 범위 밖 인덱스는 건너뜀
            if position[0] >= self.__mapper.pixel_grid.arrays["traversable"].shape[0] or \
               position[1] >=  self.__mapper.pixel_grid.arrays["traversable"].shape[1]:
                continue

            if position[0] < 0 or position[1] < 0:
                continue

            if self.__mapper.pixel_grid.arrays["traversable"][position[0], position[1]]:
                return True

        return False

    def is_path_finished(self):
        """현재 경로 인덱스가 경로의 끝에 도달했는지 확인합니다."""
        return len(self.__a_star_path) - 1 <= self.__a_star_index


    def __get_closest_traversable_array_index(self, array_index):
        """
        주어진 배열 인덱스가 장애물(traversable=True) 위에 있으면
        BFS로 가장 가까운 빈 공간(traversable=False)의 인덱스를 반환합니다.
        """
        if self.__mapper.pixel_grid.arrays["traversable"][array_index[0], array_index[1]]:
            return  self.__closest_free_point_finder.bfs(found_array=self.__mapper.pixel_grid.arrays["traversable"],
                                                         traversable_array=self.__mapper.pixel_grid.arrays["traversable"],
                                                         start_node=array_index)[0]
        else:
            return array_index
