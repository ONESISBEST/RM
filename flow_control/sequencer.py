from flags import SHOW_DEBUG


class Sequencer:
    """
    매 타임스텝마다 계속 실행되어야 하는 코드를 중단하지 않으면서,
    여러 동작을 순서대로 실행할 수 있게 해주는 클래스입니다.

    동작 원리:
    - line_pointer: "지금 몇 번째 명령을 실행해야 하나?"를 기억하는 포인터
    - line_identifier: 매 run() 호출 시, 각 명령에 순서대로 부여되는 번호
    - check()를 통해 identifier와 pointer가 일치하는 명령만 실제 실행됩니다.

    이를 통해 멀티스레딩/프로세싱 없이도 순차 실행이 가능해집니다.
    예) "앞으로 이동 → 0.5초 대기 → 회전" 같은 연속 동작 처리
    """
    def __init__(self, reset_function=None):
        self.line_identifier = 0   # 현재 프레임에서 각 명령에 붙이는 번호 (매 start_sequence마다 초기화)
        self.line_pointer = 1      # 다음에 실행할 명령의 번호
        self.done = False
        self.reset_function = reset_function  # 시퀀스 리셋 시 함께 호출할 외부 함수 (예: delay 초기화)

    def reset_sequence(self):
        """시퀀스를 처음부터 다시 시작합니다."""
        if self.reset_function is not None:
            self.reset_function()
        self.line_pointer = 1
        if SHOW_DEBUG:
            print(f"[시퀀서:sequencer.reset_sequence] {'='*20}")
            print(f"[시퀀서:sequencer.reset_sequence] 시퀀스 리셋 (pointer={self.line_pointer} → 1 초기화)")
            print(f"[시퀀서:sequencer.reset_sequence] {'='*20}")

    def seq_reset_sequence(self):
        """시퀀스 내에서 순차적으로 reset을 실행하는 래퍼 함수입니다."""
        if self.check():
            self.reset_sequence()
            return True
        return False

    def start_sequence(self):
        """
        시퀀스의 매 프레임 호출 시작점에 반드시 호출해야 합니다.
        line_identifier를 0으로 초기화하여 각 명령에 번호를 다시 부여할 준비를 합니다.
        """
        self.line_identifier = 0
        self.done = False

    def check(self):
        """
        현재 호출이 실행되어야 하는 순서인지 확인합니다.
        identifier를 1 증가시킨 후, pointer와 일치하면 True를 반환합니다.
        모든 순차 명령의 끝에 반드시 포함되어야 합니다.
        """
        self.done = False
        self.line_identifier += 1
        return self.line_identifier == self.line_pointer

    def next_seq(self):
        """다음 명령으로 포인터를 이동시킵니다 (현재 명령이 완료되었을 때 호출)."""
        self.line_pointer += 1
        self.done = True

    def seq_done(self):
        """현재 시퀀스가 마지막 명령까지 완료되었는지 반환합니다."""
        return self.done

    def simple_event(self, function=None, *args, **kwargs):
        """
        단순히 한 번만 실행되는 순차 이벤트입니다.
        function이 없으면 if문 블록을 순차적으로 실행하는 게이트로 사용합니다.
        function이 있으면 해당 함수를 순차적으로 호출합니다.
        """
        if self.check():
            if function is not None:
                function(*args, **kwargs)
            self.next_seq()
            return True
        return False

    def complex_event(self, function, *args, **kwargs):
        """
        완료되기까지 여러 프레임이 걸리는 순차 이벤트입니다.
        function이 True를 반환하면 완료로 보고 다음 명령으로 넘어갑니다.
        예) 특정 좌표까지 이동, 특정 각도로 회전 등
        """
        if self.check():
            if function(*args, **kwargs):
                self.next_seq()
                return True
        return False

    def make_simple_event(self, function):
        """
        일반 함수를 simple_event로 래핑하여 반환합니다.
        반환된 함수를 self.seq_xxx = self.sequencer.make_simple_event(xxx) 형태로 사용합니다.
        """
        def event(*args, **kwargs):
            if self.check():
                function(*args, **kwargs)
                self.next_seq()
                return True
            return False
        return event

    def make_complex_event(self, function):
        """
        True를 반환할 때 완료되는 함수를 complex_event로 래핑하여 반환합니다.
        반환된 함수를 self.seq_xxx = self.sequencer.make_complex_event(xxx) 형태로 사용합니다.
        """
        def event(*args, **kwargs):
            if self.check():
                if function(*args, **kwargs):
                    self.next_seq()
                    return True
            return False
        return event
