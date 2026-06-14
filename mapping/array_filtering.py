from flow_control.step_counter import StepCounter

import numpy as np
import cv2 as cv


class ArrayFilterer:
    """
    픽셀 그리드의 노이즈를 주기적으로 제거하는 필터 클래스입니다.

    고립 포인트 제거(isolated point removal):
    - 주변에 다른 점이 없는 단독 점은 라이다 노이즈로 간주하고 삭제합니다.
    - 커널: 중앙 1점 vs 주변 8점. 주변보다 중앙만 높으면 고립 포인트.

    엣지 스무딩(edge smoothing, 현재 미사용):
    - 들쭉날쭉한 벽 경계를 매끄럽게 만들어 경로 탐색을 돕습니다.
    """
    def __init__(self) -> None:
        # 고립 포인트 감지 커널: 중앙이 1이고 주변 8개가 -2 → 중앙만 있으면 양수
        self.isolated_point_remover_kernel = np.array([[-2, -2, -2],
                                                       [-2,  1, -2],
                                                       [-2, -2, -2]])

        # 들쭉날쭉한 엣지 감지 커널
        self.jagged_edge_remover_kernel = np.array([[ 0, -1,  0],
                                                    [-1,  3, -1],
                                                    [ 0, -1,  0]])

        # 빠진 점 채우기 커널
        self.missing_point_filler_kernel = np.array([[0, 1, 0],
                                                     [1, 0, 1],
                                                     [0, 1, 0]])

        # 20 스텝마다 한 번씩 실행 (통로 근처 오탐 벽을 빠르게 제거)
        self.isolated_point_step_counter = StepCounter(20)
        self.jagged_edge_step_counter = StepCounter(100)

    def remove_isolated_points(self, pixel_grid) -> np.ndarray:
        """100 스텝마다 한 번씩 고립된 노이즈 점을 벽/구멍/점유 레이어에서 제거합니다."""
        if self.isolated_point_step_counter.check():
            isolated_points_mask = cv.filter2D(
                pixel_grid.arrays["occupied"].astype(np.uint8), -1,
                self.isolated_point_remover_kernel) > 0
            pixel_grid.arrays["occupied"][isolated_points_mask] = False
            pixel_grid.arrays["walls"][isolated_points_mask] = False
            pixel_grid.arrays["holes"][isolated_points_mask] = False
            pixel_grid.arrays["detected_points"][isolated_points_mask] = 0
        self.isolated_point_step_counter.increase()

    def smooth_edges(self, array: np.ndarray) -> np.ndarray:
        """엣지 스무딩 (현재 주석 처리된 상태로 사용하지 않음)."""
        if self.jagged_edge_step_counter.check():
            int_array = array.astype(np.uint8)
            missing_point_filler_mask = cv.filter2D(int_array, -1, self.missing_point_filler_kernel) >= 3
            array[missing_point_filler_mask] = True
        self.jagged_edge_step_counter.increase()
