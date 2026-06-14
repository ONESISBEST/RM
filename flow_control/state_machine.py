from typing import Callable

class StateMachine:
    """
    단순한 유한 상태 머신(FSM) 구현체입니다.

    로봇의 동작 모드(상태)를 명확하게 분리하고, 각 상태에서 실행할 함수와
    이동 가능한 다음 상태들을 관리합니다.
    예) "init" → "explore" → "report_fixture" → "explore" → ... → "end"
    """
    def __init__(self, initial_state, function_on_change_state=lambda:None):
        self.state = initial_state                        # 현재 활성 상태 이름
        self.current_function = lambda:None               # 현재 상태에 대응하는 실행 함수

        self.change_state_function = function_on_change_state  # 상태 변경 시 추가로 호출할 콜백

        self.state_functions = {}         # {상태이름: 실행함수} 딕셔너리
        self.allowed_state_changes = {}   # {상태이름: {이동가능한상태들}} 딕셔너리
        self.possible_states = set()      # 등록된 모든 상태 이름의 집합

    def create_state(self, name: str, function: Callable, possible_changes = set()):
        """새 상태를 등록합니다. 이미 존재하는 이름은 등록 불가."""
        if name in self.possible_states:
            raise ValueError("Failed to create new state. State already exists.")
        self.possible_states.add(name)
        self.state_functions[name] = function
        self.allowed_state_changes[name] = possible_changes
        # 등록하는 상태가 초기 상태라면 바로 현재 함수로 설정
        if name == self.state:
            self.current_function = self.state_functions[self.state]

    def change_state(self, new_state):
        """현재 상태에서 허용된 상태로 전환합니다. 허용되지 않은 전환은 경고만 출력하고 무시."""
        if new_state not in self.possible_states:
            raise ValueError("Can't change state. New state doesn't exist.")

        if new_state in self.allowed_state_changes[self.state]:
            prev_state = self.state
            self.change_state_function()   # 상태 변경 시 콜백 호출
            self.state = new_state
            self.current_function = self.state_functions[self.state]
            print(f"[상태머신:state_machine.change_state] 상태 전환: '{prev_state}' → '{new_state}'")
        else:
            print(f"[상태머신:state_machine.change_state] 경고: '{self.state}'에서 '{new_state}'로 전환 불가 (허용: {self.allowed_state_changes[self.state]})")
        return True

    def check_state(self, state):
        """현재 상태가 지정한 상태와 같은지 확인합니다."""
        return self.state == state

    def run(self):
        """현재 상태에 해당하는 함수를 실행합니다. 함수에 change_state를 인자로 전달합니다."""
        return self.current_function(self.change_state)
