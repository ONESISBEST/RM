import numpy as np

from data_structures.vectors import Position2D

from agent.agent_interface import SubagentInterface
from mapping.mapper import Mapper

from agent.pathfinding.pathfinder import PathFinder


class ReturnToStartAgent(SubagentInterface):
    """
    미션 탐색 완료 후 로봇을 시작 위치로 복귀시키는 서브에이전트입니다.

    A* 경로 탐색(PathFinder)을 사용하여 현재 위치에서 start_position까지
    장애물을 피하는 최적 경로를 계산합니다.
    """
    def __init__(self, mapper: Mapper) -> None:
        self.__mapper = mapper
        self.__pathfinder = PathFinder(self.__mapper)

    def update(self, force_calculation) -> None:
        """시작 위치까지의 경로를 갱신합니다."""
        self.__pathfinder.update(
            np.array(self.__mapper.start_position),
            force_calculation=force_calculation)

    def get_target_position(self) -> Position2D:
        """경로 상의 다음 이동 목표 좌표를 반환합니다."""
        return self.__pathfinder.get_next_position()

    def target_position_exists(self) -> bool:
        """시작 위치가 등록되어 있으면 복귀 가능합니다."""
        return self.__mapper.start_position is not None
