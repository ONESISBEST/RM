
import numpy as np
import cv2 as cv
import copy
from data_structures.vectors import Position2D, Vector2D
from data_structures.angle import Angle
import math
from flow_control.step_counter import StepCounter


class CompoundExpandablePixelGrid:
    """
    여러 종류의 정보를 레이어(채널)별로 저장하는 동적 확장 가능 픽셀 그리드입니다.

    로봇이 탐색하면서 필요에 따라 배열 크기가 자동으로 커집니다.
    좌표계(m) ↔ 그리드 인덱스 ↔ 배열 인덱스 변환을 제공합니다.

    저장되는 레이어(arrays 딕셔너리):
    - "walls"                  : 라이다가 감지한 벽/장애물 여부
    - "detected_points"        : 라이다 감지 횟수 (일정 횟수 초과 시 wall로 확정)
    - "occupied"               : 벽 OR 구멍 (실제 이동 불가 영역)
    - "traversable"            : 로봇이 통과할 수 없는 영역 (벽에서 로봇 반경만큼 팽창)
    - "navigation_preference"  : 경로 탐색 시 피해야 할 정도 (벽 근처일수록 높음)
    - "traversed"              : 로봇이 실제로 지나간 영역
    - "robot_center_traversed" : 로봇 중심이 지나간 좁은 영역
    - "seen_by_camera"         : 카메라 시야에 포착된 영역
    - "seen_by_lidar"          : 라이다 빔이 지나간 영역
    - "discovered"             : 로봇 탐색 시야 내에 들어온 영역
    - "floor_color"            : 카메라로 촬영된 바닥 색상 (BGR)
    - "floor_color_detection_distance" : 바닥 색상 감지 정확도 (가까울수록 높음)
    - "holes"                  : 구멍(떨어지면 패널티) 영역
    - "swamps"                 : 늪지대(GPS 오차 큰 영역)
    - "checkpoints"            : 체크포인트 위치
    - "victims"                : 조난자(H/S/U 마커) 위치
    - "fixture_distance_margin": 벽 조난자 도달 가능 위치 (벽 주변 마진 영역)
    - "robot_detected_fixture_from": 이미 조난자를 보고한 위치 (중복 보고 방지)
    """
    def __init__(self, initial_shape, pixel_per_m, robot_radius_m):
        self.array_shape = np.array(initial_shape, dtype=int)
        self.offsets = self.array_shape // 2
        self.resolution = pixel_per_m  # 픽셀/미터 비율

        # 모든 정보 레이어를 하나의 딕셔너리로 관리
        self.arrays = {
            "detected_points":              np.zeros(self.array_shape, np.uint16),
            "walls":                        np.zeros(self.array_shape, np.bool_),
            "occupied":                     np.zeros(self.array_shape, np.bool_),
            "traversable":                  np.zeros(self.array_shape, np.bool_),
            "navigation_preference":        np.zeros(self.array_shape, np.float32),
            "traversed":                    np.zeros(self.array_shape, np.bool_),
            "seen_by_camera":               np.zeros(self.array_shape, np.bool_),
            "seen_by_lidar":                np.zeros(self.array_shape, np.bool_),
            "walls_seen_by_camera":         np.zeros(self.array_shape, np.bool_),
            "walls_not_seen_by_camera":     np.zeros(self.array_shape, np.bool_),
            "discovered":                   np.zeros(self.array_shape, np.bool_),
            "floor_color":                  np.zeros((self.array_shape[0], self.array_shape[1], 3), np.uint8),
            "floor_color_detection_distance": np.zeros(self.array_shape, np.uint8),
            "average_floor_color":          np.zeros((self.array_shape[0], self.array_shape[1], 3), np.uint8),
            "holes":                        np.zeros(self.array_shape, np.bool_),
            "hole_detections":              np.zeros(self.array_shape, np.uint16),
            "swamps":                       np.zeros(self.array_shape, np.bool_),
            "victims":                      np.zeros(self.array_shape, np.bool_),
            "checkpoints":                  np.zeros(self.array_shape, np.bool_),
            "victim_angles":                np.zeros(self.array_shape, np.float32),
            "fixture_detection":            np.zeros(self.array_shape, np.bool_),
            "fixture_detection_zone":       np.zeros(self.array_shape, np.bool_),
            "fixture_distance_margin":      np.zeros(self.array_shape, np.bool_),
            "robot_detected_fixture_from":  np.zeros(self.array_shape, np.bool_),
            "robot_center_traversed":       np.zeros(self.array_shape, np.bool_),
            "hazmats":                      np.zeros(self.array_shape, np.bool_),
        }

    @property
    def grid_index_max(self):
        """그리드 인덱스의 최댓값 (배열 오른쪽/아래쪽 끝)"""
        return self.array_shape - self.offsets

    @property
    def grid_index_min(self):
        """그리드 인덱스의 최솟값 (배열 왼쪽/위쪽 끝)"""
        return self.offsets * -1

    # -------- 좌표 변환 메서드 --------

    def coordinates_to_grid_index(self, coordinates: np.ndarray) -> np.ndarray:
        """실제 좌표(m) → 그리드 인덱스 변환 (로봇 시작점 기준 상대 인덱스)"""
        coords = np.round(coordinates * self.resolution).astype(int)
        return np.flip(coords)  # (x, y) → (row, col) 변환

    def grid_index_to_coordinates(self, grid_index: np.ndarray) -> np.ndarray:
        """그리드 인덱스 → 실제 좌표(m) 변환"""
        index = (grid_index.astype(float) / self.resolution)
        return np.flip(index)

    def array_index_to_grid_index(self, array_index: np.ndarray) -> np.ndarray:
        """배열 인덱스 → 그리드 인덱스 변환 (오프셋 제거)"""
        return array_index - self.offsets

    def grid_index_to_array_index(self, grid_index: np.ndarray) -> np.ndarray:
        """그리드 인덱스 → 배열 인덱스 변환 (오프셋 추가)"""
        return grid_index + self.offsets

    def array_index_to_coordinates(self, array_index) -> np.ndarray:
        """배열 인덱스 → 실제 좌표(m) 변환"""
        grid_index = self.array_index_to_grid_index(array_index)
        return self.grid_index_to_coordinates(grid_index)

    def coordinates_to_array_index(self, coordinates) -> np.ndarray:
        """실제 좌표(m) → 배열 인덱스 변환"""
        grid_index = self.coordinates_to_grid_index(coordinates)
        return self.grid_index_to_array_index(grid_index)

    # -------- 그리드 동적 확장 --------

    def expand_to_grid_index(self, grid_index: np.ndarray):
        """
        지정된 그리드 인덱스가 배열 범위를 벗어나면 모든 레이어 배열을 자동으로 확장합니다.
        확장 후 배열 인덱스가 변경되므로 이 호출 이후에는 배열 인덱스를 다시 계산해야 합니다.
        """
        array_index = self.grid_index_to_array_index(grid_index)
        if array_index[0] + 1 > self.array_shape[0]:
            self.add_end_row(array_index[0] - self.array_shape[0] + 1)
        if array_index[1] + 1 > self.array_shape[1]:
            self.add_end_column(array_index[1] - self.array_shape[1] + 1)
        if array_index[0] < 0:
            self.add_begining_row(array_index[0] * -1)
        if array_index[1] < 0:
            self.add_begining_column(array_index[1] * -1)

    def add_end_row(self, size):
        """배열 아래쪽에 빈 행을 추가합니다."""
        self.array_shape = np.array([self.array_shape[0] + size, self.array_shape[1]])
        for key in self.arrays:
            self.arrays[key] = self.__add_end_row_to_grid(self.arrays[key], size)

    def add_begining_row(self, size):
        """배열 위쪽에 빈 행을 추가하고 오프셋을 조정합니다."""
        self.offsets[0] += size
        self.array_shape = np.array([self.array_shape[0] + size, self.array_shape[1]])
        for key in self.arrays:
            self.arrays[key] = self.__add_begining_row_to_grid(self.arrays[key], size)

    def add_end_column(self, size):
        """배열 오른쪽에 빈 열을 추가합니다."""
        self.array_shape = np.array([self.array_shape[0], self.array_shape[1] + size])
        for key in self.arrays:
            self.arrays[key] = self.__add_end_column_to_grid(self.arrays[key], size)

    def add_begining_column(self, size):
        """배열 왼쪽에 빈 열을 추가하고 오프셋을 조정합니다."""
        self.offsets[1] += size
        self.array_shape = np.array([self.array_shape[0], self.array_shape[1] + size])
        for key in self.arrays:
            self.arrays[key] = self.__add_begining_column_to_grid(self.arrays[key], size)

    def __add_end_row_to_grid(self, grid, size):
        shape = np.array(grid.shape)
        shape[0] = size
        shape[1] = self.array_shape[1]
        grid = np.vstack((grid, np.zeros(shape, dtype=grid.dtype)))
        return grid

    def __add_begining_row_to_grid(self, grid, size):
        shape = np.array(grid.shape)
        shape[0] = size
        shape[1] = self.array_shape[1]
        grid = np.vstack((np.zeros(shape, dtype=grid.dtype), grid))
        return grid

    def __add_end_column_to_grid(self, grid, size):
        shape = np.array(grid.shape)
        shape[0] = self.array_shape[0]
        shape[1] = size
        grid = np.hstack((grid, np.zeros(shape, dtype=grid.dtype)))
        return grid

    def __add_begining_column_to_grid(self, grid, size):
        shape = np.array(grid.shape)
        shape[0] = self.array_shape[0]
        shape[1] = size
        grid = np.hstack((np.zeros(shape, dtype=grid.dtype), grid))
        return grid

    def get_colored_grid(self):
        """
        디버그용 컬러 이미지를 생성합니다.
        - 파란색: 조난자 도달 가능 마진 영역
        - 흰색(어둡게): 벽/장애물 영역
        - 초록색: 조난자(victim) 위치
        - 빨간색: 위험물질(hazmat) 위치
        - 노란색: 보고 완료된 fixture
        """
        color_grid = np.zeros((self.array_shape[0], self.array_shape[1], 3), dtype=np.float32)

        color_grid[self.arrays["fixture_distance_margin"]] = (0, 0, 1)
        color_grid[self.arrays["occupied"]] = (1, 1, 1)
        color_grid *= 0.3
        color_grid[self.arrays["victims"]] = (0, 1, 0)
        color_grid[self.arrays["hazmats"]] = (1, 0, 0)
        color_grid[self.arrays["fixture_detection"]] = (0, 1, 1)

        return color_grid
