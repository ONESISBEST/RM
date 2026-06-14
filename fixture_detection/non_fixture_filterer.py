import numpy as np
from fixture_detection.color_filter import ColorFilter

class NonFixtureFilter:
    """
    fixture(조난자/위험물 표지)가 아닌 배경 색상을 감지하는 필터입니다.

    카메라 이미지에서 벽, 바닥, 장애물, 구멍 등 fixture가 아닌
    알려진 배경 색상들을 마스킹하여, fixture 분류 시
    배경 픽셀이 fixture 색상으로 오인식되는 것을 방지합니다.

    filter() 메서드는 배경 색상이 있는 픽셀을 True로 표시하며,
    fixture_clasification.py에서 이 마스크를 반전하여 non-fixture 영역을 제거합니다.
    """
    def __init__(self) -> None:
        # 각 배경 색상에 대한 HSV 필터 목록
        self.color_filters = (
            ColorFilter((94, 112,  32), (95, 143, 139)),  # 벽 (청록 계열)
            ColorFilter((0, 0, 192,), (0, 0, 192)),       # 일반 바닥 (흰색 계열)
            ColorFilter((0, 0, 29), (0, 0, 138)),         # 장애물 (회색 계열)
            ColorFilter((0, 0, 10), (0, 0, 30)),          # 외부 구멍 (매우 어두운 검정)
            ColorFilter((0, 0, 106), (0, 0, 106)),        # 내부 구멍 (중간 회색)
            ColorFilter((0, 205, 233), (0, 205, 233)),    # 빨간 타일
            ColorFilter((107, 0, 72), (116, 90, 211))     # 체크포인트 (보라/파란색)
        )

    def filter(self, image):
        """
        이미지에서 배경 색상(non-fixture) 픽셀을 True로 표시한 불리언 마스크를 반환합니다.
        모든 배경 색상 필터를 OR 합산하여 최종 마스크를 생성합니다.
        """
        base = np.zeros(image.shape[:2], np.bool_)
        for f in self.color_filters:
            filtered = f.filter(image).astype(np.bool_)
            # 배경 색상 마스크들을 OR로 합산
            base += filtered

        return base
