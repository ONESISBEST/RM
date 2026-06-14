import numpy as np
from data_structures.vectors import Position2D

from agent.agent_interface import SubagentInterface
from mapping.mapper import Mapper

from agent.subagents.follow_walls.follow_walls_position_finder import PositionFinder
from agent.pathfinding.pathfinder import PathFinder

class FollowWallsAgent(SubagentInterface):
    """
    벽 가까이에 있는 위치(조난자 도달 가능 마진 영역)를 목표로 이동하는 서브에이전트입니다.

    로봇이 벽 근처를 따라 이동하면서 조난자(벽 마커)를 발견할 수 있도록 합니다.
    GoToFixturesAgent가 실패할 경우 다음 우선순위로 사용됩니다.

    동작:
    1. PositionFinder: fixture_distance_margin 레이어에서 아직 방문 안 한 벽 근처 위치 탐색
    2. PathFinder(A*): 해당 위치까지 장애물 회피 경로 계산
    """
    def __init__(self, mapper: Mapper) -> None:
        self.mapper = mapper
        self.__position_finder = PositionFinder(mapper)
        self.__pathfinder = PathFinder(mapper)

    def update(self, force_calculation=False):
        """목표 위치를 갱신하고 해당 위치까지의 경로를 계산합니다."""
        self.__position_finder.update(force_calculation=force_calculation)

        if self.__position_finder.target_position_exists():
            target = self.__position_finder.get_target_position()
            self.__pathfinder.update(np.array(target), force_calculation)

    def get_target_position(self) -> Position2D:
        """경로 상의 다음 이동 목표 좌표를 반환합니다."""
        return self.__pathfinder.get_next_position()

    def target_position_exists(self) -> bool:
        """도달 가능한 벽 근처 위치가 있는지 반환합니다."""
        return self.__position_finder.target_position_exists()
