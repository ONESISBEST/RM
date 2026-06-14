
# ===================================================================
# 프로그램 전체 동작을 제어하는 디버그 및 기능 플래그 설정 파일
# 0 = 비활성화, 1 = 활성화
# ===================================================================

SHOW_FIXTURE_DEBUG = 0    # 조난자(Fixture) 탐지 과정을 시각화한 디버그 이미지 창 표시
SHOW_DEBUG = 0            # 일반 디버그 메시지 콘솔 출력

SHOW_GRANULAR_NAVIGATION_GRID = 0   # A* 경로가 그려진 픽셀 그리드 표시
SHOW_PATHFINDING_DEBUG = 0          # 경로 탐색(A*) 관련 디버그 메시지 출력
SHOW_BEST_POSITION_FINDER_DEBUG = 0 # 최적 목표 위치 탐색 과정 표시

SHOW_MAP_AT_END = 0  # 지도 전송 시 최종 행렬(맵) 내용을 콘솔에 출력

DO_WAIT_KEY = 0  # 매 프레임마다 OpenCV 윈도우에서 키 입력 대기 (디버그 영상 슬로우모션용)

DO_SLOW_DOWN = 0          # 메인 루프를 인위적으로 느리게 실행 (시뮬레이션 관찰용)
SLOW_DOWN_S = 0.032       # DO_SLOW_DOWN 활성 시 매 루프마다 쉬는 시간(초)

TUNE_FILTER = 0  # HSV 색상 필터 값을 트랙바 UI로 실시간 조정하는 모드 활성화

DO_SAVE_FIXTURE_DEBUG = 0  # 조난자 인식 과정 이미지를 파일로 저장
SAVE_FIXTURE_DEBUG_DIR = "/home/iitaadmin/simulated_rescue_maze/debug_imgs"  # 위 이미지 저장 경로

DO_SAVE_FINAL_MAP = 0  # 최종 지도 이미지를 파일로 저장
SAVE_FINAL_MAP_DIR = "/home/iitaadmin/simulated_rescue_maze/final_maps"  # 위 이미지 저장 경로

DO_SAVE_DEBUG_GRID = 0  # 디버그용 컬러 그리드 이미지를 파일로 저장
SAVE_DEBUG_GRID_DIR = "/home/iitaadmin/simulated_rescue_maze/final_maps"  # 위 이미지 저장 경로

# ===================================================================
# 실시간 지도 시각화 (map_visualizer.py)
# ===================================================================
# SHOW_LIVE_MAP = 1 로 설정하면 별도 OpenCV 창에 실시간 탐색 현황이 표시됩니다.
#
# 창에 표시되는 정보:
#   ■ 흰색   — 벽/장애물 (라이다로 감지됨)
#   ■ 파란색 — 로봇이 실제로 지나간 경로
#   ■ 시안색 — A* 계획 경로 (다음 목표까지)
#   ■ 녹색   — 미탐색 목표 또는 벽-조난자 접근 후보 위치
#   ■ 주황색 — 발견된 조난자(Victim) 위치 [흰 테두리 원]
#   ● 빨간색 — 로봇 현재 위치 + 방향 화살표
#   ● 노란색 — 현재 이동 목표 위치
#   ✦ 마젠타 — 출발점(Start)
#   배경 밝기: 밝은 회색 = 탐색 완료, 어두운 회색 = 미탐색
# ===================================================================
SHOW_LIVE_MAP = 1  # 실시간 탐색 지도 창 표시 (0=비활성화, 1=활성화)
