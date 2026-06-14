from data_structures.compound_pixel_grid import CompoundExpandablePixelGrid
import numpy as np

class OccupiedMapper:
    """
    '점유됨(occupied)' 레이어를 계산하는 클래스입니다.

    점유 = 벽(walls) 또는 구멍(holes) 중 하나라도 True인 영역
    단, 로봇이 실제로 지나간 영역(traversed)은 점유 취소 (오탐 제거)
    이 레이어는 경로 탐색에서 이동 불가 영역을 나타냅니다.
    """
    def __init__(self, grid: CompoundExpandablePixelGrid) -> None:
        self.__grid = grid

    def map_occupied(self):
        """벽과 구멍을 합산하여 점유 레이어를 갱신합니다."""
        # 벽 OR 구멍인 영역을 점유로 표시
        self.__grid.arrays["occupied"] = np.bitwise_or(
            self.__grid.arrays["walls"],
            self.__grid.arrays["holes"])

        # 로봇이 실제로 지나간 곳은 점유 취소 (센서 오탐으로 잘못 표시된 벽 제거)
        self.__grid.arrays["occupied"][self.__grid.arrays["traversed"]] = False
