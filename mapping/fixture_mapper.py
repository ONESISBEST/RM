import math

import numpy as np
import cv2 as cv

from data_structures.compound_pixel_grid import CompoundExpandablePixelGrid
from flow_control.step_counter import StepCounter


class FixtureMapper:
    """
    조난자(Fixture) 관련 지도 정보를 관리하는 클래스입니다.

    주요 역할:
    1. generate_detection_zone(): 벽 주변에 조난자가 붙어 있을 수 있는 '도달 가능 마진 영역' 생성
       - 벽으로부터 약 5cm 반경의 고리 모양 템플릿으로 벽 바깥 영역 표시
    2. clean_up_fixtures(): 점유 영역(벽 등) 위에 잘못 표시된 조난자 마커 제거
    3. map_detected_fixture(): 조난자를 보고한 로봇 위치를 기록하여 중복 보고 방지
    """
    def __init__(self, pixel_grid: CompoundExpandablePixelGrid, tile_size: float) -> None:
        self.tile_size = tile_size
        self.grid = pixel_grid

        template_radious = int(0.05 * self.grid.resolution)   # 5cm에 해당하는 픽셀 반경
        template_diameter = math.ceil(template_radious * 2 + 1)

        # 도달 가능 마진 템플릿: 내부는 -50(감소), 외곽 테두리만 +1(증가)
        # 이 템플릿으로 벽과 합성하면 벽 바로 바깥쪽 영역이 양수가 됩니다.
        self.fixture_distance_margin_template = np.zeros((template_diameter, template_diameter), dtype=np.int8)
        self.fixture_distance_margin_template = cv.circle(self.fixture_distance_margin_template,
                                                          (template_radious, template_radious),
                                                          template_radious, -50, -1)  # 내부 채우기 (-50)
        self.fixture_distance_margin_template = cv.circle(self.fixture_distance_margin_template,
                                                          (template_radious, template_radious),
                                                          template_radious, 1, 1)     # 테두리만 (+1)

        self.detected_from_radius = round(0.10 * self.grid.resolution)  # 보고 기록 반경(픽셀) — 10cm: 같은 fixture 중복 보고 방지
        self.__detection_zone_counter = StepCounter(15)

    def generate_detection_zone(self):
        """벽 픽셀 근처에 조난자 도달 가능 마진 영역(fixture_distance_margin)을 생성합니다."""
        if self.__detection_zone_counter.check():
            occupied_as_int = self.grid.arrays["walls"].astype(np.int8)
            self.grid.arrays["fixture_distance_margin"] = cv.filter2D(
                occupied_as_int, -1, self.fixture_distance_margin_template) > 0
        self.__detection_zone_counter.increase()

    def clean_up_fixtures(self):
        """점유된 영역(occupied=True) 위에 잘못 표시된 조난자/위험물 마커를 제거합니다."""
        self.grid.arrays["victims"][self.grid.arrays["occupied"]] = False
        self.grid.arrays["hazmats"][self.grid.arrays["occupied"]] = False

    def map_detected_fixture(self, robot_position):
        """
        조난자를 보고한 로봇 위치(반경 2cm 원)를 robot_detected_fixture_from 레이어에 기록합니다.
        이 레이어를 확인하여 같은 위치에서 중복 보고하지 않도록 합니다.
        """
        robot_array_index = self.grid.coordinates_to_array_index(robot_position)
        template = self.__get_circle_template_indexes(self.detected_from_radius, robot_array_index)
        self.grid.arrays["robot_detected_fixture_from"][template[0], template[1]] = True

    def __get_circle_template_indexes(self, radius, offsets=(0, 0)):
        """지정 반경의 원 안에 포함되는 배열 인덱스 목록을 오프셋과 함께 반환합니다."""
        diameter = int(radius * 2 + 1)
        diameter_template = np.zeros((diameter, diameter), dtype=np.uint8)
        diameter_template = cv.circle(diameter_template, (radius, radius), radius, 255, -1)
        diameter_template = diameter_template.astype(np.bool_)
        return self.__get_indexes_from_template(diameter_template, (-radius + offsets[0], -radius + offsets[1]))

    def __get_indexes_from_template(self, template: np.ndarray, offsets=(0, 0)):
        """템플릿(bool 배열)에서 True인 위치의 인덱스를 오프셋 적용 후 반환합니다."""
        indexes = template.nonzero()
        indexes = np.array(indexes)
        offsets = np.array(offsets)
        indexes[0] += offsets[0]
        indexes[1] += offsets[1]
        return indexes
