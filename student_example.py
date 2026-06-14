"""
student_example.py — 학생용 로봇 제어 예시 코드

이 파일을 수정해서 여러분만의 로봇 프로그램을 만들어 보세요!
rescue_robot.py의 RescueRobot 클래스가 복잡한 부분을 대신 처리해 줍니다.

======================================================
방법 1: 완전 자율 주행 (가장 간단!)
======================================================
로봇이 스스로 모든 것을 합니다.
"""

from rescue_robot import RescueRobot


# ======================================================
# 방법 1: 완전 자율 주행 — 코드 한 줄!
# ======================================================
def example_autonomous():
    robot = RescueRobot()
    robot.run_autonomous()   # 초기화 → 탐색 → 조난자 보고 → 지도 전송까지 전부 자동


# ======================================================
# 방법 2: 자율 주행 + 내 코드 추가
# ======================================================
def example_autonomous_with_custom_code():
    robot = RescueRobot()

    while robot.is_running():
        robot.step()   # 자율 주행 두뇌 실행

        # 여기에 내가 원하는 코드를 추가하세요!
        print(f"현재 위치: x={robot.x:.2f}m, y={robot.y:.2f}m")
        print(f"바라보는 방향: {robot.direction:.1f}°")
        print(f"경과 시간: {robot.elapsed_time:.1f}초, 남은 시간: {robot.remaining_time:.1f}초")

        if robot.victim_visible:
            print(f"★ 조난자 발견! 글자: {robot.victim_letter}")

        if robot.is_time_almost_up:
            print("⚠ 시간이 얼마 남지 않았습니다!")


# ======================================================
# 방법 3: 완전 수동 제어
# ======================================================
def example_manual_control():
    robot = RescueRobot()

    while robot.is_running():
        # 조난자가 보이면 멈추고 보고
        if robot.victim_visible and not robot.already_reported:
            robot.stop()
            letter = robot.victim_letter
            if letter is not None:
                print(f"조난자 발견: {letter}")
                robot.report_victim(letter)

        # 시간이 얼마 없으면 출발점으로 복귀
        elif robot.is_time_almost_up:
            arrived = robot.go_to_start()
            if arrived:
                robot.finish_mission()
                break

        # 평소에는 에이전트가 추천하는 목표로 이동
        else:
            robot.go_to_next_target()


# ======================================================
# 방법 4: 탐색 완료 후 귀환하는 커스텀 루프
# ======================================================
def example_explore_then_return():
    robot = RescueRobot()

    while robot.is_running():

        # 탐색이 완료되었으면 출발점으로 복귀
        if robot.exploration_complete:
            print(f"탐색 완료! 출발점으로 복귀 중... 거리: {robot.distance_to_start:.2f}m")
            arrived = robot.go_to_start()
            if arrived or robot.is_at_start:
                print("출발점 도착! 미션 완료")
                robot.finish_mission()
                break

        # 조난자 발견 시 보고
        elif robot.victim_visible and not robot.already_reported:
            robot.stop()
            robot.report_victim()

        # 그 외에는 자율 탐색
        else:
            robot.go_to_next_target()


# ======================================================
# 실행할 예시를 선택하세요 (아래 한 줄만 남기고 나머지는 주석 처리)
# ======================================================
example_autonomous()           # 방법 1: 완전 자율 (추천)
# example_autonomous_with_custom_code()   # 방법 2: 자율 + 내 코드
# example_manual_control()               # 방법 3: 수동 제어
# example_explore_then_return()          # 방법 4: 탐색 후 귀환
