import utilities
import struct

from robot.devices.sensor import Sensor

from flags import SHOW_MAP_AT_END

class Comunicator(Sensor):
    """
    Webots 에미터(Emitter)/리시버(Receiver)를 통해 대회 서버와 통신하는 클래스입니다.

    로봇이 서버에 보낼 수 있는 메시지 종류:
    - 조난자 발견 보고 (send_victim)
    - 최종 지도 전송 (send_map)
    - 진행 불능 신호 (send_lack_of_progress)
    - 미션 종료 신호 (send_end_of_play)

    서버로부터 수신:
    - 게임 점수 및 남은 시간 (game_score, remaining_time)
    - 진행 불능 페널티 알림 (lack_of_progress)
    """
    def __init__(self, emmiter, receiver, timeStep):
        self.receiver = receiver
        self.emmiter = emmiter
        self.receiver.enable(timeStep)
        self.lack_of_progress = False   # 진행 불능 페널티를 받았는지 여부
        self.do_get_world_info = True   # 게임 정보 요청 모드 (전송 후에는 비활성화)
        self.game_score = 0             # 서버로부터 받은 현재 점수
        self.remaining_time = 0         # 서버로부터 받은 남은 시간(초)

    def send_victim(self, position, victimtype):
        """조난자를 발견했을 때 위치(cm 단위)와 글자 코드를 서버에 전송합니다."""
        self.do_get_world_info = False
        letter = bytes(victimtype, "utf-8")
        position = utilities.multiplyLists(position, [100, 100])  # m → cm 변환
        position = [int(position[0]), int(position[1])]
        message = struct.pack("i i c", position[0], position[1], letter)
        self.emmiter.send(message)
        self.do_get_world_info = False

    def send_lack_of_progress(self):
        """로봇이 스스로 진행 불능임을 신고하는 메시지를 전송합니다."""
        self.do_get_world_info = False
        message = struct.pack('c', 'L'.encode())
        self.emmiter.send(message)
        self.do_get_world_info = False

    def send_end_of_play(self):
        """미션 종료를 서버에 알리는 메시지를 전송합니다."""
        self.do_get_world_info = False
        exit_mes = struct.pack('c', b'E')
        self.emmiter.send(exit_mes)
        print("[통신:comunicator.send_end_of_play] 미션 종료 신호(E) 서버 전송 완료 - 점수: {:.1f}점, 남은 시간: {}초".format(self.game_score, self.remaining_time))

    def send_map(self, np_array):
        """
        완성된 최종 지도(numpy 배열)를 바이너리로 직렬화하여 서버에 전송합니다.
        전송 형식: [행렬 크기(2 int)] + [','로 구분된 값 문자열]
        이후 서버에 지도 평가 요청('M') 메시지를 별도로 전송합니다.
        """
        print(f"[통신:comunicator.send_map] 지도 전송 시작: 행렬 크기={np_array.shape}, 총 셀={np_array.size}개")
        if SHOW_MAP_AT_END:
            print(np_array)
        s = np_array.shape
        s_bytes = struct.pack('2i', *s)
        flatMap = ','.join(np_array.flatten())
        sub_bytes = flatMap.encode('utf-8')
        a_bytes = s_bytes + sub_bytes
        self.emmiter.send(a_bytes)
        map_evaluate_request = struct.pack('c', b'M')
        self.emmiter.send(map_evaluate_request)
        self.do_get_world_info = False

    def request_game_data(self):
        """서버에 현재 게임 정보(점수, 남은 시간)를 요청합니다."""
        if self.do_get_world_info:
            message = struct.pack('c', 'G'.encode())
            self.emmiter.send(message)

    def update(self):
        """
        매 타임스텝 호출: 수신 큐를 처리합니다.
        - 게임 정보('G') 수신 시 점수와 시간을 업데이트합니다.
        - 진행 불능 페널티('L') 수신 시 lack_of_progress 플래그를 설정합니다.
        """
        if self.do_get_world_info:
            self.request_game_data()
            if self.receiver.getQueueLength() > 0:
                received_data = self.receiver.getBytes()
                if len(received_data) > 2:
                    tup = struct.unpack('c f i', received_data)
                    if tup[0].decode("utf-8") == 'G':
                        self.game_score = tup[1]
                        self.remaining_time = tup[2]
                        self.receiver.nextPacket()

            self.lack_of_progress = False
            if self.receiver.getQueueLength() > 0:
                received_data = self.receiver.getBytes()
                if len(received_data) < 2:
                    tup = struct.unpack('c', received_data)
                    if tup[0].decode("utf-8") == 'L':
                        print(f"[통신:comunicator.update] 진행 불능(Lack of Progress) 페널티 수신! 현재 점수: {self.game_score:.1f}점, 남은 시간: {self.remaining_time}초")
                        self.lack_of_progress = True
                    self.receiver.nextPacket()
        else:
            self.do_get_world_info = True
