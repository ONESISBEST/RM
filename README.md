# RM - Rescue Maze Robot

Webots 기반 구조 로봇 시뮬레이션 프로젝트입니다. 로봇이 미로를 자율 주행하면서 벽과 바닥 정보를 지도화하고, 피해자 표식을 인식해 보고한 뒤, 최종 지도를 서버로 전송하는 것을 목표로 합니다.

## 주요 기능

- LIDAR 포인트 클라우드를 이용한 벽 지도 생성
- 카메라 기반 바닥 색상 분석 및 구멍/늪/체크포인트 탐지
- 피해자 표식 인식 및 중복 보고 방지
- A* 기반 경로 탐색과 벽 따라가기 탐색 전략
- GPS와 Gyroscope를 함께 사용하는 위치/방향 추정
- 탐색 완료 후 출발 지점 복귀 및 최종 지도 전송

## 기술 스택

- Python 3
- Webots controller API
- NumPy
- OpenCV
- scikit-image
- imutils

## 프로젝트 구조

```text
src/
├── main.py                         # 실행 예시 및 시작 코드
├── run.py                          # Webots controller 진입점
├── rescue_robot.py                 # 사용하기 쉬운 로봇 제어 API
├── flags.py                        # 디버그/기능 플래그
├── utilities.py                    # 수학, 시각화, 필터 보조 함수
├── map_visualizer.py               # OpenCV 기반 지도 시각화
├── mapping/                        # 지도 생성
│   ├── mapper.py                   # 전체 매핑 파이프라인 관리
│   ├── wall_mapper.py              # LIDAR 기반 벽 매핑
│   ├── floor_mapper.py             # 카메라 기반 바닥 매핑
│   ├── robot_mapper.py             # 로봇 이동/관측 영역 기록
│   ├── occupied_mapping.py         # 벽과 구멍을 장애물로 통합
│   └── fixture_mapper.py           # 피해자 위치/근접 영역 기록
├── agent/                          # 자율 탐색 에이전트
│   ├── agent.py                    # 하위 에이전트 우선순위 관리
│   ├── pathfinding/                # A*, 경로 보정
│   └── subagents/                  # 벽 따라가기, 미탐색 이동, 복귀 전략
├── executor/                       # 미션 실행 및 상태 머신
├── robot/                          # Webots 장치 추상화
├── fixture_detection/              # 피해자/표식 인식
├── final_matrix_creation/          # 최종 제출 지도 생성
├── data_structures/                # 좌표, 각도, 동적 픽셀 그리드
├── algorithms/                     # BFS, A* 알고리즘
└── flow_control/                   # StateMachine, Sequencer, Delay
```

## 실행 방법

Webots 프로젝트에서 컨트롤러가 `run.py`를 실행하도록 설정합니다.

가장 기본적인 자율 주행 실행은 `main.py`에서 다음 코드로 시작됩니다.

```python
from rescue_robot import RescueRobot

robot = RescueRobot()
robot.run_autonomous()
```

직접 제어 루프를 만들고 싶다면 아래처럼 사용할 수 있습니다.

```python
from rescue_robot import RescueRobot

robot = RescueRobot()

while robot.is_running():
    if robot.victim_visible and not robot.already_reported:
        robot.stop()
        robot.report_victim()
    elif robot.exploration_complete:
        if robot.go_to_start():
            robot.finish_mission()
            break
    else:
        robot.go_to_next_target()
```

## 동작 흐름

1. `run.py`가 Webots 컨트롤러로 실행됩니다.
2. `main.py`에서 `RescueRobot`을 생성합니다.
3. `Executor`가 센서 업데이트, 지도 생성, 상태 머신 실행을 반복합니다.
4. `Mapper`가 벽, 바닥, 로봇 이동 경로, 피해자 위치를 동적 그리드에 저장합니다.
5. `Agent`가 현재 지도 정보를 바탕으로 다음 이동 목표를 선택합니다.
6. 시간이 부족하거나 탐색이 끝나면 출발 지점으로 복귀합니다.
7. `FinalMatrixCreator`가 최종 지도를 만들고 서버로 전송합니다.

## 핵심 모듈

| 모듈 | 설명 |
| --- | --- |
| `rescue_robot.py` | 초보자가 쉽게 사용할 수 있는 고수준 로봇 API |
| `executor/executor.py` | 미션 전체 흐름, 상태 머신, 보고/전송 관리 |
| `mapping/wall_mapper.py` | LIDAR 데이터를 벽 픽셀로 변환하고 이동 가능 영역 생성 |
| `mapping/floor_mapper.py` | 카메라 이미지로 바닥 색상과 위험 지형 탐지 |
| `agent/agent.py` | 피해자 접근, 벽 따라가기, 미탐색 구역 이동 전략 선택 |
| `robot/pose_manager.py` | GPS와 Gyroscope를 이용한 위치/방향 추정 |
| `final_matrix_creation/final_matrix_creator.py` | 최종 제출용 지도 행렬 생성 |

## 매핑 방식

이 프로젝트는 `CompoundExpandablePixelGrid`라는 동적 픽셀 그리드를 중심으로 지도를 구성합니다. 로봇이 이동하며 관측 범위가 넓어지면 그리드가 자동으로 확장되고, 여러 레이어에 정보를 분리해 저장합니다.

- `walls`: LIDAR로 확인한 벽
- `detected_points`: LIDAR 누적 감지 횟수
- `traversed`: 로봇이 실제로 지나간 영역
- `seen_by_camera`: 카메라로 확인한 영역
- `seen_by_lidar`: LIDAR로 확인한 영역
- `occupied`: 벽 또는 구멍으로 인해 이동할 수 없는 영역
- `floor_color`: 바닥 색상 정보
- `victims`: 피해자 표식 위치

## 자율 주행 전략

탐색 에이전트는 여러 하위 전략을 우선순위로 조합합니다.

1. 피해자 후보 위치가 있으면 먼저 접근합니다.
2. 벽을 따라가며 미로 구조를 안정적으로 파악합니다.
3. 아직 발견하지 못한 영역이 있으면 해당 영역으로 이동합니다.
4. 탐색이 끝나면 출발 지점으로 돌아갑니다.

## 최종 지도 생성

미션 종료 시 픽셀 단위 지도는 제출 형식의 행렬로 변환됩니다.

- 벽 정보는 타일 단위 벽 노드로 변환됩니다.
- 바닥 색상은 일반 바닥, 구멍, 늪, 체크포인트 등으로 분류됩니다.
- 출발 지점과 장애물 정보가 함께 반영됩니다.
- 완성된 행렬은 Webots 통신 장치를 통해 서버로 전송됩니다.

## 참고

이 코드는 Webots 환경과 `controller` 모듈이 필요합니다. 일반 Python 환경에서 직접 실행하면 Webots 전용 모듈을 찾지 못할 수 있습니다.
