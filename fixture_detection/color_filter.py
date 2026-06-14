import cv2 as cv
import numpy as np

class ColorFilter:
    """
    HSV 색 공간에서 지정된 범위의 색상을 추출하는 마스크 필터입니다.

    Webots 시뮬레이터의 카메라 이미지(BGR 형식)를 HSV로 변환하여
    lower_hsv ~ upper_hsv 범위에 해당하는 픽셀의 이진 마스크를 반환합니다.

    사용 예:
    - 조난자 이미지에서 검정/흰색/노란색/빨간색 픽셀 추출 (fixture 분류)
    - 바닥 타일에서 구멍/늪지대/체크포인트 색상 추출
    - 벽 색상 추출 (fixture 후보에서 벽 제거)
    """
    def __init__(self, lower_hsv, upper_hsv):
        # NumPy 배열로 변환하여 cv.inRange()에 직접 사용
        self.lower = np.array(lower_hsv)
        self.upper = np.array(upper_hsv)

    def filter(self, img):
        """
        입력 이미지를 HSV로 변환 후 lower~upper 범위의 이진 마스크를 반환합니다.
        반환값: 해당 색상 범위 픽셀은 255, 그 외는 0인 uint8 배열
        """
        # BGR → HSV 변환 (Webots 카메라는 BGR 형식)
        hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # HSV 범위 내 픽셀만 255로 표시하는 마스크 생성
        mask = cv.inRange(hsv_image, self.lower, self.upper)
        return mask


WALL_COLOR_FILTER = ColorFilter((90, 44, 0), (95, 213, 158))


def get_wall_mask(image: np.ndarray) -> np.ndarray:
    """벽 색상 영역을 채워진 마스크로 반환합니다."""
    margin = 1
    raw_wall = WALL_COLOR_FILTER.filter(image)
    wall = np.ones(shape=(raw_wall.shape[0], raw_wall.shape[1] + margin * 2), dtype=np.uint8) * 255
    wall[:, margin:-margin] = raw_wall
    conts, _ = cv.findContours(wall, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filled_wall = np.zeros_like(wall, dtype=np.bool_)
    for c in conts:
        this_cont = np.zeros_like(wall, dtype=np.uint8)
        cv.fillPoly(this_cont, [c,], 255)
        filled_wall += this_cont > 0
    filled_wall = filled_wall[:, margin:-margin]
    return filled_wall
