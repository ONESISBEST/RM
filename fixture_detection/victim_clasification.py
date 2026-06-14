from flags import SHOW_FIXTURE_DEBUG
import cv2 as cv
import numpy as np

from fixture_detection.color_filter import ColorFilter


class VictimClassifier:
    """
    조난자 심볼(Φ/Ψ/Ω)을 이미지 형태로 분류하는 클래스입니다.

    내부 인식 결과는 실제 보이는 심볼 기준으로 반환합니다.
    - Φ (Phi)
    - Ψ (Psi)
    - Ω (Omega)

    서버 보고용 문자(H/S/U)는 다른 계층에서 매핑합니다.

    분류 방법:
    1. 심볼 이진화 및 최대 blob 추출
    2. 컨투어 계층 분석 → 닫힌 내부 공간(구멍) 유무 판별
       - 구멍 있음 → Φ
    3. 구멍 없을 때 하단 중앙 / 양쪽 하단 픽셀 밀도로 Ψ/Ω 구분
       - 하단 중앙 밀도 높음 → Ψ
       - 하단 중앙 밀도 낮음 → Ω
    """

    PHI = "Φ"
    PSI = "Ψ"
    OMEGA = "Ω"

    def __init__(self):
        self.victim_letter_filter = ColorFilter(lower_hsv=(0, 0, 0), upper_hsv=(0, 0, 130))
        self.min_hole_area = 80

    def isolate_victim(self, image):
        binary = self.victim_letter_filter.filter(image)
        letter = self.get_biggest_blob(binary)
        if SHOW_FIXTURE_DEBUG:
            cv.imshow("thresh", binary)
        return letter

    def get_biggest_blob(self, binary_image: np.ndarray) -> np.ndarray:
        contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        max_size = 0
        biggest_blob = None
        for c in contours:
            x, y, w, h = cv.boundingRect(c)
            if w * h > max_size:
                biggest_blob = binary_image[y:y + h, x:x + w]
                max_size = w * h
        return biggest_blob

    def classify_victim(self, victim):
        letter = self.isolate_victim(victim["image"])
        if letter is None or letter.size == 0:
            return None

        letter = cv.resize(letter, (100, 100), interpolation=cv.INTER_AREA)

        if SHOW_FIXTURE_DEBUG:
            cv.imshow("victim_symbol", letter)

        # 1) 닫힌 내부 공간(구멍) 검출 — Φ 판별
        hole_count = self._count_holes(letter)

        if hole_count >= 1:
            if SHOW_FIXTURE_DEBUG:
                print(f"[조난자 분류] Φ 감지: 내부 구멍 {hole_count}개")
            return self.PHI

        # 2) 하단 중앙 픽셀 밀도 — Ψ vs Ω 판별
        bottom_center = letter[70:95, 35:65]
        bottom_left = letter[70:95, 10:35]
        bottom_right = letter[70:95, 65:90]

        if bottom_center.size > 0:
            center_density = np.count_nonzero(bottom_center) / bottom_center.size
        else:
            center_density = 0

        if bottom_left.size > 0:
            left_density = np.count_nonzero(bottom_left) / bottom_left.size
        else:
            left_density = 0

        if bottom_right.size > 0:
            right_density = np.count_nonzero(bottom_right) / bottom_right.size
        else:
            right_density = 0

        side_density = max(left_density, right_density)

        if SHOW_FIXTURE_DEBUG:
            print(
                "[조난자 분류] 밀도: "
                f"center={center_density:.3f}, left={left_density:.3f}, right={right_density:.3f}"
            )

        if center_density >= 0.22 or center_density > side_density + 0.05:
            if SHOW_FIXTURE_DEBUG:
                print(f"[조난자 분류] Ψ 감지: center={center_density:.3f} → '{self.PSI}'")
            return self.PSI

        if SHOW_FIXTURE_DEBUG:
            print(f"[조난자 분류] Ω 감지: center={center_density:.3f} → '{self.OMEGA}'")
        return self.OMEGA

    def _count_holes(self, binary_100x100):
        """RETR_CCOMP 계층으로 닫힌 내부 공간(구멍) 개수를 반환합니다."""
        contours, hierarchy = cv.findContours(
            binary_100x100, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

        if hierarchy is None:
            return 0

        count = 0
        for i, h in enumerate(hierarchy[0]):
            if h[3] != -1:  # parent 있음 = 내부 컨투어(구멍)
                area = cv.contourArea(contours[i])
                if area >= self.min_hole_area:
                    count += 1
        return count
