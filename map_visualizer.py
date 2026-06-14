"""
map_visualizer.py — 실시간 탐색 지도 시각화 모듈

flags.py에서 SHOW_LIVE_MAP = 1 로 설정하면 활성화됩니다.

MapVisualizer 창에 표시되는 정보:
  ■ 흰색   — 벽/장애물 (라이다 감지)
  ■ 파란색 — 로봇이 실제로 지나간 경로
  ■ 시안색 — A* 계획 경로 (다음 목표까지)
  ■ 녹색   — 탐색 후보 위치 (fixture_distance_margin)
  ■ 주황색 — 발견된 조난자(Victim) [흰 테두리 원]
  ● 빨간색 — 로봇 현재 위치 + 방향 화살표
  ● 노란색 — 현재 이동 목표 위치
  ✦ 마젠타 — 출발점(Start)
  배경: 밝은 회색=탐색 완료, 어두운 회색=미탐색
"""

import numpy as np
import cv2 as cv
import math

from mapping.mapper import Mapper
from flow_control.step_counter import StepCounter


# 각 레이어의 BGR 색상
_COLORS = {
    "discovered_bg":    (55,  55,  55),   # 탐색 완료 배경 (어두운 회색)
    "undiscovered_bg":  (25,  25,  25),   # 미탐색 배경 (매우 어두운 회색)
    "wall":             (220, 220, 220),  # 벽 (밝은 흰색)
    "traversed":        (180,  80,  20),  # 로봇이 지나간 경로 (파란색 계열)
    "path":             (255, 220,   0),  # A* 계획 경로 (시안)
    "candidate":        (  0, 200,   0),  # 탐색 후보 위치 (순수 녹색)
    "victim":           (  0, 165, 255),  # 조난자 (주황색 — 후보와 명확히 구분)
    "swamp":            ( 30, 100, 140),  # 늪지대 (갈색)
    "hole":             (  0,   0, 180),  # 구멍 (빨간색)
    "robot":            (  0,   0, 255),  # 로봇 현재 위치 (빨간색)
    "target":           (  0, 220, 255),  # 현재 목표 (노란색)
    "start":            (255,   0, 255),  # 출발점 (마젠타)
    "path_node":        (200, 200,   0),  # 경로 노드 점
}

_DISPLAY_SIZE = 600   # 창 표시 크기 (정사각형 픽셀)
_WINDOW_NAME  = "Live Map — 탐색 현황"


class MapVisualizer:
    """
    Mapper의 픽셀 그리드 레이어를 실시간으로 시각화하는 OpenCV 창을 관리합니다.

    사용법:
        visualizer = MapVisualizer(mapper)
        # 매 프레임 호출
        visualizer.update(path=current_astar_path, target=current_target)
    """

    def __init__(self, mapper: Mapper):
        self._mapper = mapper
        self._path: list = []        # A* 경로 (grid index 목록)
        self._target = None          # 현재 목표 위치 (array index np.array)
        self.__render_counter = StepCounter(10)

    def set_path(self, path: list):
        """PathFinder에서 계산된 A* 경로를 설정합니다 (grid index 목록)."""
        self._path = path if path is not None else []

    def set_target(self, target_array_index):
        """현재 이동 목표를 배열 인덱스로 설정합니다."""
        self._target = target_array_index

    def update(self, path: list = None, target=None):
        """
        시각화 창을 갱신합니다. 매 프레임 호출하세요.

        Args:
            path:   A* 경로 노드 목록 (grid index np.array 리스트). None이면 이전 값 유지.
            target: 현재 목표 배열 인덱스 np.array. None이면 이전 값 유지.
        """
        if path is not None:
            self._path = path
        if target is not None:
            self._target = target

        if self.__render_counter.check():
            if self._mapper.robot_position is not None:
                image = self._render()
                display = cv.resize(image, (_DISPLAY_SIZE, _DISPLAY_SIZE),
                                    interpolation=cv.INTER_NEAREST)
                cv.imshow(_WINDOW_NAME, display)
                cv.waitKey(1)
        self.__render_counter.increase()

    def _render(self) -> np.ndarray:
        """모든 레이어를 합성한 이미지를 반환합니다."""
        grid = self._mapper.pixel_grid
        arrays = grid.arrays
        h, w = grid.array_shape

        # ── 1. 배경: 탐색 여부에 따라 밝기 구분 ──────────────────────
        image = np.full((h, w, 3), _COLORS["undiscovered_bg"], dtype=np.uint8)
        image[arrays["discovered"]] = _COLORS["discovered_bg"]

        # ── 2. 로봇이 지나간 경로 (파란색 계열) ──────────────────────
        image[arrays["traversed"]] = _COLORS["traversed"]

        # ── 3. 탐색 후보 위치 (fixture_distance_margin) ──────────────
        image[arrays["fixture_distance_margin"]] = _COLORS["candidate"]

        # ── 4. 특수 지형 ──────────────────────────────────────────────
        image[arrays["swamps"]] = _COLORS["swamp"]
        image[arrays["holes"]]  = _COLORS["hole"]

        # ── 5. 벽/장애물 (occupied) ───────────────────────────────────
        image[arrays["occupied"]] = _COLORS["wall"]

        # ── 6. 조난자 위치 (작은 원) ──────────────────────────────────
        victim_positions = np.argwhere(arrays["victims"])
        for pos in victim_positions:
            cv.circle(image, (pos[1], pos[0]), 5, _COLORS["victim"], -1)
            cv.circle(image, (pos[1], pos[0]), 5, (255, 255, 255), 1)  # 흰색 테두리로 더 선명하게

        # ── 7. A* 계획 경로 선 ────────────────────────────────────────
        if len(self._path) >= 2:
            pts = []
            for node in self._path:
                arr_idx = grid.grid_index_to_array_index(np.array(node))
                pts.append((int(arr_idx[1]), int(arr_idx[0])))
            for i in range(len(pts) - 1):
                cv.line(image, pts[i], pts[i + 1], _COLORS["path"], 1)
            # 경로 노드 점
            for pt in pts:
                cv.circle(image, pt, 1, _COLORS["path_node"], -1)

        # ── 8. 출발점 표시 (마젠타 십자) ─────────────────────────────
        if self._mapper.start_position is not None:
            sp_ai = grid.coordinates_to_array_index(self._mapper.start_position)
            sp = (int(sp_ai[1]), int(sp_ai[0]))
            cv.drawMarker(image, sp, _COLORS["start"],
                          cv.MARKER_STAR, 7, 1, cv.LINE_AA)

        # ── 9. 현재 목표 위치 (노란색 원) ────────────────────────────
        if self._target is not None:
            tgt = (int(self._target[1]), int(self._target[0]))
            cv.circle(image, tgt, 5, _COLORS["target"], 1)
            cv.circle(image, tgt, 2, _COLORS["target"], -1)

        # ── 10. 로봇 위치 + 방향 화살표 (빨간색) ─────────────────────
        robot_ai = grid.coordinates_to_array_index(self._mapper.robot_position)
        robot_pt = (int(robot_ai[1]), int(robot_ai[0]))
        cv.circle(image, robot_pt, 4, _COLORS["robot"], -1)

        if self._mapper.robot_orientation is not None:
            angle_rad = math.radians(self._mapper.robot_orientation.degrees)
            arrow_len = 8
            tip = (
                int(robot_pt[0] + math.sin(angle_rad) * arrow_len),
                int(robot_pt[1] + math.cos(angle_rad) * arrow_len),
            )
            cv.arrowedLine(image, robot_pt, tip, _COLORS["robot"], 1,
                           tipLength=0.4, line_type=cv.LINE_AA)

        return image

    def close(self):
        """OpenCV 창을 닫습니다."""
        cv.destroyWindow(_WINDOW_NAME)
