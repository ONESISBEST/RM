import numpy as np
from data_structures.vectors import Position2D

from agent.agent_interface import SubagentInterface
from mapping.mapper import Mapper

from agent.subagents.go_to_fixtures.go_to_fixtures_position_finder import PositionFinder
from agent.pathfinding.pathfinder import PathFinder

class GoToFixturesAgent(SubagentInterface):
    """
    조난자(victim) 또는 체크포인트 근처로 이동하는 서브에이전트입니다.

    에이전트 우선순위에서 가장 높은 순위를 차지하며,
    victims 레이어에 감지된 조난자가 있을 때 해당 위치로 이동합니다.

    동작:
    1. PositionFinder: victims 레이어 + fixture_distance_margin에서 아직 방문 안 한 위치 탐색
    2. PathFinder(A*): 해당 위치까지 장애물 회피 경로 계산
    """
    def __init__(self, mapper: Mapper) -> None:
        self.mapper = mapper
        self.__position_finder = PositionFinder(mapper)
        self.__pathfinder = PathFinder(mapper)

    def update(self, force_calculation=False):
        """목표 위치와 경로를 갱신합니다."""
        self.__position_finder.update(force_calculation=force_calculation)

        if self.__position_finder.target_position_exists():
            target = self.__position_finder.get_target_position()
            self.__pathfinder.update(np.array(target), force_calculation)

    def get_target_position(self) -> Position2D:
        """경로 상의 다음 이동 목표 좌표를 반환합니다."""
        return self.__pathfinder.get_next_position()

    def target_position_exists(self) -> bool:
        """아직 방문하지 않은 조난자/체크포인트 위치가 있는지 반환합니다."""
        return self.__position_finder.target_position_exists()
    
