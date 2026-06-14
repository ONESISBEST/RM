from collections import namedtuple
import numpy as np
from data_structures.vectors import Position2D

from agent.agent_interface import AgentInterface, SubagentInterface
from mapping.mapper import Mapper

from agent.subagents.follow_walls.follow_walls_subagent import FollowWallsAgent
from agent.subagents.go_to_non_discovered.go_to_non_discovered_subagent import GoToNonDiscoveredAgent
from agent.subagents.return_to_start.return_to_start_subagent import ReturnToStartAgent
from agent.subagents.go_to_fixtures.go_to_fixtures_subagent import GoToFixturesAgent

from flow_control.state_machine import StateMachine
from flow_control.step_counter import StepCounter


class SubagentPriorityCombiner(SubagentInterface):
    """
    여러 서브에이전트를 우선순위 순서대로 시도하여 유효한 목표 위치를 제공하는 클래스입니다.

    에이전트 목록을 순서대로 시도하며, 첫 번째로 유효한 목표 위치를 반환하는 에이전트를 선택합니다.
    우선순위 (높은 것부터):
    1. GoToFixturesAgent  - 조난자(벽 마커)에 가까이 접근하는 것이 최우선
    2. FollowWallsAgent   - 벽 가까이 이동하여 조난자 마킹 가능성 높이기
    3. GoToNonDiscoveredAgent - 아직 탐색 못한 곳으로 이동
    """
    def __init__(self, agents: list) -> None:
        self.__agent_list = agents
        self.__current_agent_index = 0
        self.__previous_agent_index = 0

    def update(self, force_calculation=False) -> None:
        """에이전트를 우선순위대로 시도하여 유효한 목표가 나오는 첫 번째 에이전트를 선택합니다."""
        agent: SubagentInterface
        for index, agent in enumerate(self.__agent_list):
            agent.update(force_calculation=self.__agent_changed() or force_calculation)
            if agent.target_position_exists():
                prev = self.__current_agent_index
                self.__previous_agent_index = prev
                self.__current_agent_index = index
                if prev != index:
                    prev_name = type(self.__agent_list[prev]).__name__
                    curr_name = type(self.__agent_list[index]).__name__
                    print(f"[에이전트:agent.update] 서브에이전트 전환: {prev_name} → {curr_name}")
                break

    def get_target_position(self) -> Position2D:
        """현재 선택된 에이전트의 목표 좌표를 반환합니다."""
        return self.__agent_list[self.__current_agent_index].get_target_position()

    def target_position_exists(self) -> bool:
        return self.__agent_list[self.__current_agent_index].target_position_exists()
    def __agent_changed(self) -> bool:
        """선택된 에이전트가 이전 프레임과 다른지 확인합니다 (전환 시 강제 재계산)."""
        return self.__previous_agent_index != self.__current_agent_index


class Agent(AgentInterface):
    """
    Executor가 사용하는 최상위 에이전트 클래스입니다.

    두 가지 단계(Stage)로 동작합니다:
    1. explore: SubagentPriorityCombiner로 미로를 탐색
    2. return_to_start: 탐색이 완료되면 출발점으로 복귀

    복귀 완료 시 do_end()가 True를 반환하여 Executor가 미션 종료를 처리합니다.
    """
    def __init__(self, mapper: Mapper) -> None:
        self.__mapper = mapper

        # 탐색 에이전트: 조난자 → 벽 따라가기 → 미탐색 지역 순서로 시도
        self.__navigation_agent = SubagentPriorityCombiner([
            GoToFixturesAgent(self.__mapper),
            FollowWallsAgent(self.__mapper),
            GoToNonDiscoveredAgent(self.__mapper)])

        # 복귀 에이전트: 시작 위치로 돌아가는 경로 계산
        self.__return_to_start_agent = ReturnToStartAgent(self.__mapper)

        # 단계 상태 머신: explore ↔ return_to_start
        self.__stage_machine = StateMachine("explore", self.__set_force_calculation)
        self.__stage_machine.create_state(name="explore",
                                          function=self.__stage_explore,
                                          possible_changes={"return_to_start"})
        self.__stage_machine.create_state(name="return_to_start",
                                          function=self.__stage_return_to_start)

        self.do_force_calculation = False
        self.end_reached_distance_threshold = 0.04  # 시작점으로부터 이 거리 이내 = 복귀 완료
        self.max_time = 8 * 60  # 최대 미션 시간(초)

        self.__target_position = None
        self.__return_log_counter = StepCounter(200)  # 복귀 중 200스텝마다 위치 로그
        self.__no_target_consecutive = 0             # 탐색 목표 없음 연속 프레임 카운터
        self.__no_target_threshold = 120             # 이 프레임 수 연속으로 목표 없으면 복귀 전환

    def update(self) -> None:
        """매 타임스텝 호출: 현재 단계에 맞는 목표 위치를 계산합니다."""
        self.__stage_machine.run()

    def get_target_position(self) -> Position2D:
        """Executor가 로봇에게 이동 명령을 내릴 목표 좌표를 반환합니다."""
        return self.__target_position

    def do_end(self) -> bool:
        """복귀 단계이면서 시작점 가까이 도달했으면 True → Executor가 'end' 상태로 전환합니다."""
        return self.__stage_machine.state == "return_to_start" and \
               self.__mapper.robot_position.get_distance_to(self.__mapper.start_position) < self.end_reached_distance_threshold

    def __stage_explore(self, change_state_function):
        """탐색 단계: 탐색 에이전트에서 목표를 받아 설정합니다. 목표가 없으면 복귀 단계로 전환합니다."""
        self.__navigation_agent.update(force_calculation=self.do_force_calculation)
        self.do_force_calculation = False

        if not self.__navigation_agent.target_position_exists():
            self.__no_target_consecutive += 1
            if self.__no_target_consecutive >= self.__no_target_threshold:
                print(f"[에이전트:agent.__stage_explore] 탐색 완료: {self.__no_target_consecutive}프레임 연속 목표 없음 → 시작점 복귀 단계로 전환")
                change_state_function("return_to_start")
        else:
            self.__no_target_consecutive = 0
            self.__target_position = self.__navigation_agent.get_target_position()

    def __stage_return_to_start(self, _):
        """복귀 단계: 시작 위치로의 경로를 계산하고 목표를 설정합니다."""
        self.__return_to_start_agent.update(force_calculation=self.do_force_calculation)
        self.do_force_calculation = False

        if self.__return_to_start_agent.target_position_exists():
            self.__target_position = self.__return_to_start_agent.get_target_position()

        if self.__return_log_counter.check() and self.__mapper.start_position is not None:
            dist = self.__mapper.robot_position.get_distance_to(self.__mapper.start_position)
            print(f"[에이전트:agent.__stage_return_to_start] 출발점 복귀 중: 현재=({self.__mapper.robot_position.x:.3f},{self.__mapper.robot_position.y:.3f})m, 출발점까지 거리={dist:.3f}m")
        self.__return_log_counter.increase()

    def __set_force_calculation(self):
        """단계 전환 시 호출: 다음 프레임에 목표 위치를 강제 재계산하도록 플래그를 설정합니다."""
        self.do_force_calculation = True
