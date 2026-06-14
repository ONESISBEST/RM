import random
from data_structures.vectors import Position2D
from abc import ABC, abstractmethod

class AgentInterface(ABC):
    """
    최상위 에이전트(Agent)의 추상 인터페이스입니다.
    Executor가 매 프레임 update()를 호출하고, get_target_position()으로 목표 좌표를 얻습니다.
    """
    def __init__(self, mapper) -> None:
        self.mapper = mapper

    @abstractmethod
    def update(self) -> None:
        """매 타임스텝 목표 위치를 계산합니다."""
        pass

    @abstractmethod
    def get_target_position(self) -> Position2D:
        """다음에 이동해야 할 목표 좌표를 반환합니다."""
        pass

    @abstractmethod
    def do_end(self) -> bool:
        """미션 종료 조건이 충족되었는지 반환합니다."""
        pass

class SubagentInterface(ABC):
    """
    전략별 서브에이전트의 추상 인터페이스입니다.
    Agent가 여러 SubagentInterface 구현체 중 하나를 선택하여 목표 위치를 얻습니다.
    예) FollowWallsAgent, GoToNonDiscoveredAgent, GoToFixturesAgent
    """
    def __init__(self, mapper) -> None:
        self.mapper = mapper

    @abstractmethod
    def update(self, force_calculation=False) -> None:
        """목표 위치를 재계산합니다. force_calculation=True이면 강제로 재계산합니다."""
        pass

    @abstractmethod
    def get_target_position(self) -> Position2D:
        """현재 계산된 목표 좌표를 반환합니다."""
        pass

    @abstractmethod
    def target_position_exists(self) -> bool:
        """유효한 목표 위치가 존재하는지 반환합니다."""
        pass

class PositionFinderInterface(ABC):
    """
    목표 위치를 탐색하는 알고리즘의 추상 인터페이스입니다.
    SubagentInterface 구현체 내부에서 사용됩니다.
    예) FollowWallsPositionFinder, GoToNonDiscoveredPositionFinder
    """
    @abstractmethod
    def __init__(self, mapper) -> None:
        pass

    @abstractmethod
    def update(self, force_calculation=False) -> None:
        pass

    @abstractmethod
    def get_target_position(self) -> Position2D:
        pass

    @abstractmethod
    def target_position_exists(self) -> bool:
        pass
