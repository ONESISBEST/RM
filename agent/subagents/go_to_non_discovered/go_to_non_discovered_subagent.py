import numpy as np
import cv2 as cv

from agent.agent_interface import SubagentInterface

from data_structures.vectors import Position2D
from mapping.mapper import Mapper

from agent.pathfinding.pathfinder import PathFinder
from agent.subagents.go_to_non_discovered.go_to_non_discovered_position_finder import PositionFinder

class GoToNonDiscoveredAgent(SubagentInterface):
    """
    아직 탐색하지 않은(discovered=False) 가장 가까운 위치로 이동하는 서브에이전트입니다.

    GoToFixturesAgent와 FollowWallsAgent가 모두 목표를 찾지 못할 때
    마지막 수단으로 사용되어 로봇이 완전히 탐색하지 않은 영역으로 향하게 합니다.

    동작:
    1. PositionFinder: discovered 레이어에서 False인 가장 가까운 위치를 BFS로 탐색
    2. PathFinder(A*): 해당 위치까지 장애물 회피 경로 계산
    """
    def __init__(self, mapper: Mapper):
        self.__path_finder = PathFinder(mapper)
        self.__position_finder = PositionFinder(mapper)

    def update(self, force_calculation=False) -> None:
        """경로가 완료되었거나 막혔을 때 새 목표 위치를 재계산합니다."""
        self.__position_finder.update(
            force_calculation=self.__do_force_position_finder() or force_calculation)

        if self.__position_finder.target_position_exists():
            target = self.__position_finder.get_target_position()
            self.__path_finder.update(
                target_position=np.array(target),
                force_calculation=force_calculation)

    def get_target_position(self) -> Position2D:
        """경로 상의 다음 이동 목표 좌표를 반환합니다."""
        return self.__path_finder.get_next_position()

    def __do_force_position_finder(self) -> bool:
        """경로가 완료되었거나 경로를 찾지 못했을 때 위치 재탐색을 강제합니다."""
        return self.__path_finder.is_path_finished() or self.__path_finder.path_not_found

    def target_position_exists(self) -> bool:
        """아직 탐색되지 않은 위치가 존재하는지 반환합니다."""
        return self.__position_finder.target_position_exists()
