import math
import numpy as np
import cv2 as cv

from fixture_detection.victim_clasification import VictimClassifier
from fixture_detection.color_filter import ColorFilter, get_wall_mask
from fixture_detection.non_fixture_filterer import NonFixtureFilter

from flags import SHOW_FIXTURE_DEBUG


class FixtureType:
    """색상별 픽셀 수 범위(ranges)로 fixture 여부를 판별하는 데이터 클래스"""
    def __init__(self, fixture_type, default_letter, ranges=None):
        self.fixture_type = fixture_type
        self.default_letter = default_letter
        self.ranges = ranges

    def is_fixture(self, colour_counts: dict):
        for color in self.ranges:
            if not self.ranges[color][0] <= colour_counts[color] <= self.ranges[color][1]:
                return False
        return True


class FixtureClasiffier:
    """
    v26 fixture 분류기.

    분류 체계:
    - Victim (Φ/Ψ/Ω): 흑백 심볼 → Φ/Ψ/Ω 인식 후 보고 시 H/S/U로 변환
    - HazardMap/CognitiveTarget (UN 플래카드): 색상 패턴 → F/P/C/O 보고
    - Fake: 벽에서 돌출된 3D 심볼 → 보고 안 함

    분류 순서:
    1. already_detected 체크
    2. cognitive target처럼 보이면 우선 판별
    3. 빨강/노랑 존재 → hazmat 분류 (F/P/C/O)
    4. 흑백만 → victim 분류 (Φ/Ψ/Ω)
    5. 매칭 없음 → None (보고 안 함, 랜덤 추측 제거)
    """
    def __init__(self):
        self.victim_classifier = VictimClassifier()

        self.color_filters = {
            "black":    ColorFilter(lower_hsv=(0, 0, 0),     upper_hsv=(0, 0, 160)),
            "white":    ColorFilter(lower_hsv=(0, 0, 170),   upper_hsv=(255, 110, 208)),
            "yellow":   ColorFilter(lower_hsv=(25, 170, 82), upper_hsv=(30, 255, 255)),
        }
        self.extra_color_filters = {
            "red_low":  ColorFilter(lower_hsv=(0, 80, 80),   upper_hsv=(10, 255, 255)),
            "red_high": ColorFilter(lower_hsv=(160, 80, 80), upper_hsv=(179, 255, 255)),
            "green":    ColorFilter(lower_hsv=(40, 80, 80),  upper_hsv=(85, 255, 255)),
            "blue":     ColorFilter(lower_hsv=(100, 80, 80), upper_hsv=(130, 255, 255)),
        }
        self.cognitive_color_values = {
            "K": -2,
            "R": -1,
            "Y": 0,
            "G": 1,
            "B": 2,
        }
        self.cognitive_sum_to_letter = {
            0: "F",
            1: "P",
            2: "C",
            3: "O",
        }
        self.min_fixture_height = 16
        self.min_fixture_width_factor = 0.8

        # hazmat 분류용 색상 범위 (100x100 이미지 기준, 우선순위 순)
        self.hazmat_types = (
            FixtureType("organic_peroxide", "O", {"red":   (500, math.inf),
                                                   "yellow":(500, math.inf)}),
            FixtureType("flammable", "F",        {"white": (500, math.inf),
                                                   "red":   (500, math.inf)}),
            FixtureType("corrosive", "C",        {"white": (700,  4500),
                                                   "black": (900, 3000)}),
            FixtureType("poison",    "P",        {"white": (2000, 5000),
                                                   "black": (100,  1000)}),
        )

        # already_detected 판별 규칙
        self.already_detected_types = (
            FixtureType("already_detected", "", {"white": (5000, math.inf),
                                                  "black": (0, 0),
                                                  "red":   (0, 0),
                                                  "yellow":(0, 0)}),
            FixtureType("already_detected", "", {"white": (0, 2000),
                                                  "black": (0, 0),
                                                  "red":   (0, 0),
                                                  "yellow":(0, 0)}),
        )

        self.non_fixture_filter = NonFixtureFilter()

    def sum_images(self, images):
        final_img = np.zeros_like(images[0])
        for image in images:
            final_img += image
        final_img[final_img > 255] = 255
        return final_img

    def filter_fixtures(self, victims) -> list:
        final_victims = []
        for vic in victims:
            if vic["image"].shape[0] > self.min_fixture_height and vic["image"].shape[1] > self.min_fixture_height * self.min_fixture_width_factor:
                final_victims.append(vic)
        return final_victims

    def find_fixtures(self, image) -> list:
        image = np.rot90(image, k=3)
        if SHOW_FIXTURE_DEBUG:
            cv.imshow("image", image)

        binary_filters = list(self.color_filters.values()) + list(self.extra_color_filters.values())
        binary_images = [f.filter(image) for f in binary_filters]
        binary_image = self.sum_images(binary_images)

        walls_mask = get_wall_mask(image)
        non_fixture_by_color = self.non_fixture_filter.filter(image)
        binary_image *= (walls_mask + (non_fixture_by_color == 0))

        if SHOW_FIXTURE_DEBUG:
            cv.imshow("binaryImage", binary_image)

        contours, _ = cv.findContours(binary_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        final_victims = []

        for c in contours:
            x, y, w, h = cv.boundingRect(c)
            final_victims.append({"image":image[y:y + h, x:x + w], "position":(x, y)})

        return self.filter_fixtures(final_victims)

    def count_colors(self, image) -> dict:
        color_point_counts = {}
        for name, f in self.color_filters.items():
            color_point_counts[name] = np.count_nonzero(f.filter(image))
        for name, f in self.extra_color_filters.items():
            color_point_counts[name] = np.count_nonzero(f.filter(image))
        color_point_counts["red"] = color_point_counts.pop("red_low", 0) + color_point_counts.pop("red_high", 0)
        return color_point_counts

    def classify_fixture(self, fixture) -> str:
        """
        fixture를 분류하여 보고용 심볼/문자를 반환합니다.

        v26 분류 순서:
        1. already_detected → None
        2. cognitive target처럼 보이면 우선 분류
        3. 빨강/노랑 있음 → hazmat 플래카드 (F/P/C/O)
        4. 흑백만 → victim (Φ/Ψ/Ω)
        5. 매칭 없음 → None (보고 안 함)
        """
        image = cv.resize(fixture["image"], (100, 100), interpolation=cv.INTER_AREA)
        color_counts = self.count_colors(image)

        if SHOW_FIXTURE_DEBUG:
            print(f"[Fixture] 색상: black={color_counts['black']}, white={color_counts['white']}, "
                  f"yellow={color_counts['yellow']}, red={color_counts['red']}, "
                  f"green={color_counts.get('green',0)}, blue={color_counts.get('blue',0)}")

        # 1) already_detected 체크
        for ad in self.already_detected_types:
            if ad.is_fixture(color_counts):
                if SHOW_FIXTURE_DEBUG:
                    print("[Fixture] already_detected → None")
                return None

        # 2) cognitive target 우선 판별
        if self._looks_like_cognitive_target(image, color_counts):
            symbol = self._classify_cognitive_target(fixture)
            if symbol is not None:
                return symbol

        # 3) 빨강/노랑 존재 → hazmat 플래카드 (F/P/C/O)
        if color_counts["red"] > 300 or color_counts["yellow"] > 300:
            return self._classify_hazmat(color_counts)

        # 4) 흑백만 → victim (Φ/Ψ/Ω)
        if color_counts["black"] > 300 and color_counts["white"] > 300:
            symbol = self.victim_classifier.classify_victim(fixture)
            if SHOW_FIXTURE_DEBUG:
                print(f"[Fixture] victim 분류 결과: '{symbol}'")
            return symbol

        # 5) 매칭 없음
        if SHOW_FIXTURE_DEBUG:
            print("[Fixture] 매칭 없음 → None")
        return None

    def _looks_like_cognitive_target(self, image, color_counts) -> bool:
        """카메라에 보이는 객체가 cognitive target 후보인지 빠르게 판단합니다."""
        if color_counts.get("green", 0) > 20 or color_counts.get("blue", 0) > 20:
            return True

        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (9, 9), 2)

        circles = cv.HoughCircles(
            gray,
            cv.HOUGH_GRADIENT,
            dp=1.2,
            minDist=18,
            param1=100,
            param2=18,
            minRadius=18,
            maxRadius=45,
        )
        return circles is not None and len(circles[0]) > 0

    def _classify_hazmat(self, color_counts) -> str:
        """색상 픽셀 수로 hazmat 종류(F/P/C/O)를 판별합니다."""
        for ht in self.hazmat_types:
            if ht.is_fixture(color_counts):
                if SHOW_FIXTURE_DEBUG:
                    print(f"[Fixture] hazmat: {ht.fixture_type} → '{ht.default_letter}'")
                return ht.default_letter

        # hazmat 색상은 있지만 정확한 매칭 없음 → 확신 없으면 보고 안 함
        return None

    def _classify_cognitive_target(self, fixture) -> str:
        """5-ring 동심원 CognitiveTarget 분류.
        원형 외곽을 먼저 안정적으로 찾고, 그 반지름을 5등분하여 각 링을 독립적으로 판별합니다.
        링이 서로 같은 색이어도 병합하지 않으며, 각 링은 마스크 내 색 비율로 다수결 판정을 합니다.
        합계 0~3 → F/P/C/O, 그 외 → None(fake)."""
        image = cv.resize(fixture["image"], (100, 100), interpolation=cv.INTER_AREA)
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        cx, cy, outer_r = self._estimate_cognitive_geometry(image)
        outer_r = float(np.clip(outer_r, 24.0, 45.0))
        ring_step = outer_r / 5.0
        ring_bounds = [
            (outer_r - ring_step * 1, outer_r),
            (outer_r - ring_step * 2, outer_r - ring_step * 1),
            (outer_r - ring_step * 3, outer_r - ring_step * 2),
            (outer_r - ring_step * 4, outer_r - ring_step * 3),
            (0.0, outer_r - ring_step * 4),
        ]
        score_sum = 0
        ring_colors = []
        for inner_bound, outer_bound in ring_bounds:
            color = self._classify_ring_color(hsv, cx, cy, inner_bound, outer_bound)
            ring_colors.append(color)
            score_sum += self.cognitive_color_values[color]

        if SHOW_FIXTURE_DEBUG:
            print(f"[CognitiveTarget] 중심=({cx:.1f},{cy:.1f}), 바깥반지름={outer_r:.1f}, 링={ring_colors}, 합계={score_sum}")

        letter = self.cognitive_sum_to_letter.get(score_sum)
        if letter is None:
            if SHOW_FIXTURE_DEBUG:
                print(f"[CognitiveTarget] 합계 {score_sum} → 범위 밖 → None (fake)")
            return None

        if SHOW_FIXTURE_DEBUG:
            print(f"[CognitiveTarget] 합계 {score_sum} → '{letter}'")
        return letter

    def _estimate_cognitive_geometry(self, image):
        """HoughCircles 우선, 실패 시 외곽 컨투어로 중심과 반지름을 추정합니다."""
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (9, 9), 2)

        circles = cv.HoughCircles(
            gray,
            cv.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=100,
            param2=18,
            minRadius=18,
            maxRadius=45,
        )

        if circles is not None and len(circles[0]) > 0:
            circles = np.round(circles[0]).astype(int)
            cx, cy, outer_r = max(circles, key=lambda c: c[2])
            return float(cx), float(cy), float(outer_r)

        hue = hsv_image[:, :, 0]
        sat = hsv_image[:, :, 1]
        val = hsv_image[:, :, 2]

        foreground = (
            (((hue <= 10) | (hue >= 170)) & (sat >= 45) & (val >= 40)) |
            ((hue >= 20) & (hue <= 35) & (sat >= 45) & (val >= 40)) |
            ((hue >= 40) & (hue <= 85) & (sat >= 45) & (val >= 40)) |
            ((hue >= 100) & (hue <= 130) & (sat >= 45) & (val >= 40)) |
            ((val < 70) & (sat < 80))
        ).astype(np.uint8) * 255

        kernel = np.ones((3, 3), np.uint8)
        foreground = cv.morphologyEx(foreground, cv.MORPH_OPEN, kernel)
        foreground = cv.morphologyEx(foreground, cv.MORPH_CLOSE, kernel)

        contours, _ = cv.findContours(foreground, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 50.0, 50.0, 45.0

        biggest = max(contours, key=cv.contourArea)
        (cx, cy), outer_r = cv.minEnclosingCircle(biggest)
        return float(cx), float(cy), float(outer_r)

    def _classify_ring_color(self, hsv_image, cx, cy, inner_r, outer_r) -> str:
        """링 구간 안에서 전체 픽셀을 집계해 색을 판별합니다."""
        yy, xx = np.ogrid[:hsv_image.shape[0], :hsv_image.shape[1]]
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        ring_mask = (dist2 >= inner_r ** 2) & (dist2 < outer_r ** 2)

        ring_area = np.count_nonzero(ring_mask)
        if ring_area == 0:
            return "K"

        hue = hsv_image[:, :, 0]
        sat = hsv_image[:, :, 1]
        val = hsv_image[:, :, 2]

        color_masks = {
            "K": ring_mask & ((val < 55) | (sat < 45)),
            "R": ring_mask & (((hue <= 10) | (hue >= 170)) & (sat >= 45) & (val >= 45)),
            "Y": ring_mask & ((hue >= 20) & (hue <= 35) & (sat >= 45) & (val >= 45)),
            "G": ring_mask & ((hue >= 40) & (hue <= 85) & (sat >= 45) & (val >= 45)),
            "B": ring_mask & ((hue >= 100) & (hue <= 130) & (sat >= 45) & (val >= 45)),
        }

        counts = {code: int(np.count_nonzero(mask)) for code, mask in color_masks.items()}
        best_code = max(counts, key=counts.get)
        best_ratio = counts[best_code] / float(ring_area)

        if SHOW_FIXTURE_DEBUG:
            print(f"[CognitiveTarget] ring({inner_r:.1f},{outer_r:.1f}) counts={counts}, best={best_code}, ratio={best_ratio:.2f}")

        if best_ratio < 0.20:
            return "K"
        return best_code
