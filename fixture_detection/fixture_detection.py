from data_structures.vectors import Position2D, Vector2D
from data_structures.angle import Angle
from typing import List
from robot.devices.camera import CameraImage
from fixture_detection.color_filter import ColorFilter, get_wall_mask
import skimage

import copy

import math

import numpy as np
import cv2 as cv

from data_structures.compound_pixel_grid import CompoundExpandablePixelGrid
from flags import SHOW_FIXTURE_DEBUG

class FixtureDetector:
    """
    카메라 이미지에서 fixture 위치를 픽셀 그리드에 매핑하는 클래스입니다.

    v26 변경:
    - 최소 컨투어 면적 필터 추가 (오탐 감소)
    - 가짜 피해자 판별: 벽에서 돌출 거리 체크 (fake = 벽 밖으로 많이 돌출)
    - fixture 위치가 벽 픽셀 근처(2px 이내)에 있어야 유효
    """
    def __init__(self, pixel_grid: CompoundExpandablePixelGrid) -> None:
        self.pixel_grid = pixel_grid

        self.color_filters = {
            "black":    ColorFilter(lower_hsv=(0, 0, 0),     upper_hsv=(0, 0, 9)),
            "white":    ColorFilter(lower_hsv=(0, 0, 193),   upper_hsv=(255, 110, 208)),
            "yellow":   ColorFilter(lower_hsv=(25, 170, 82), upper_hsv=(30, 255, 255)),
            "red_low":  ColorFilter(lower_hsv=(0, 80, 80),   upper_hsv=(10, 255, 255)),
            "red_high": ColorFilter(lower_hsv=(160, 80, 80), upper_hsv=(179, 255, 255)),
        }

        self.max_detection_distance = 0.12 * 5

        self.min_contour_area = 30
        self.fake_protrusion_threshold = 3
        

    def get_fixture_positions_and_angles(self, robot_position: Position2D, camera_image: CameraImage) -> list:
        """
        카메라 이미지 내 fixture 위치를 실제 공간 좌표와 각도로 변환합니다.
        벽에서 돌출된 가짜 피해자를 필터링합니다.
        """
        positions_in_image = self.get_fixture_positions_in_image(np.flip(camera_image.image, axis=1))

        fixture_positions = []
        fixture_angles = []
        for position in positions_in_image:
            relative_horizontal_angle = Angle(position[1] * (camera_image.data.horizontal_fov.radians / camera_image.data.width))
            fixture_horizontal_angle = (relative_horizontal_angle - camera_image.data.horizontal_fov / 2) + camera_image.data.horizontal_orientation
            fixture_horizontal_angle.normalize()

            camera_vector = Vector2D(camera_image.data.horizontal_orientation, camera_image.data.distance_from_center)
            camera_pos = camera_vector.to_position()
            camera_pos += robot_position

            detection_vector = Vector2D(fixture_horizontal_angle, self.max_detection_distance)
            detection_pos = detection_vector.to_position()
            detection_pos += camera_pos

            camera_array_index = self.pixel_grid.coordinates_to_array_index(camera_pos)
            detection_array_index = self.pixel_grid.coordinates_to_array_index(detection_pos)

            line_xx, line_yy = skimage.draw.line(camera_array_index[0], camera_array_index[1], detection_array_index[0], detection_array_index[1])

            index = 0
            for x, y in zip(line_xx, line_yy):
                if x >= 0 and y >= 0 and x < self.pixel_grid.array_shape[0] and y < self.pixel_grid.array_shape[1]:
                    back_index = index - 2
                    back_index = max(back_index, 0)
                    if self.pixel_grid.arrays["walls"][x, y]:
                        x1 = line_xx[back_index]
                        y1 = line_yy[back_index]
                        fixture_pos = self.pixel_grid.array_index_to_coordinates(np.array([x1, y1]))

                        # 가짜 피해자 필터: 벽에서 너무 멀리 돌출되어 있으면 무시
                        if not self._is_near_wall(x1, y1):
                            if SHOW_FIXTURE_DEBUG:
                                print(f"[FixtureDetector] 벽에서 먼 fixture 무시 (가짜 의심): ({x1},{y1})")
                            break

                        fixture_positions.append(fixture_pos)
                        fixture_angles.append(copy.deepcopy(fixture_horizontal_angle))
                        break
                index += 1

        return fixture_positions, fixture_angles

    def _is_near_wall(self, x, y):
        """fixture 위치가 벽 픽셀 근처(±threshold)에 있는지 확인합니다.
        가짜 피해자는 벽에서 돌출되어 벽으로부터 멀리 감지됩니다."""
        t = self.fake_protrusion_threshold
        x_min = max(0, x - t)
        x_max = min(self.pixel_grid.array_shape[0], x + t + 1)
        y_min = max(0, y - t)
        y_max = min(self.pixel_grid.array_shape[1], y + t + 1)
        region = self.pixel_grid.arrays["walls"][x_min:x_max, y_min:y_max]
        return np.any(region)

    def get_fixture_positions_in_image(self, image: np.ndarray) -> List[Position2D]:
        """
        이미지에서 fixture 후보의 이미지 내 위치를 반환합니다.
        최소 컨투어 면적 필터 적용으로 오탐을 줄입니다.
        """
        image_sum = np.zeros(image.shape[:2], dtype=np.bool_)
        for filter in self.color_filters.values():
            image_sum += filter.filter(image) > 0

        image_sum = image_sum.astype(np.uint8) * 255

        wall_mask = get_wall_mask(image)
        image_sum *= wall_mask

        if SHOW_FIXTURE_DEBUG:
            cv.imshow("fixtures", image_sum)

        contours, _ = cv.findContours(image_sum, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        final_victims = []
        for c in contours:
            # 최소 면적 필터: 노이즈 제거
            if cv.contourArea(c) < self.min_contour_area:
                continue

            x, y, w, h = cv.boundingRect(c)

            # 종횡비 필터: 너무 길쭉하면 fixture 아님
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > 4.0:
                continue

            final_victims.append(Position2D((x + x + w) / 2, (y + y + h) / 2))

        if SHOW_FIXTURE_DEBUG:
            debug = copy.deepcopy(image)
            for f in final_victims:
                debug = cv.circle(debug, np.array(f, dtype=int), 3, (255, 0, 0), -1)
            cv.imshow("victim_pos_debug", debug)

        return final_victims

    def map_fixtures(self, camera_images, robot_position):
        """
        카메라 이미지에서 fixture를 감지하여 victims 레이어에 기록합니다.
        벽 근처에 있고 최소 크기 이상인 fixture만 기록합니다.
        같은 위치에 여러 토큰이 중복되지 않도록 필터링합니다.
        """
        for i in camera_images:
            positions, angles = self.get_fixture_positions_and_angles(robot_position, i)
            for pos, angle in zip(positions, angles):
                index = self.pixel_grid.coordinates_to_array_index(pos)
                if 0 <= index[0] < self.pixel_grid.array_shape[0] and \
                   0 <= index[1] < self.pixel_grid.array_shape[1]:
                    self.pixel_grid.arrays["victims"][index[0], index[1]] = True
                    self.pixel_grid.arrays["victim_angles"][index[0], index[1]] = angle.radians

    

    def mark_reported_fixture(self, robot_position, fixture_position):
        fixture_array_index = self.pixel_grid.coordinates_to_array_index(fixture_position)
        rr, cc = skimage.draw.disk(fixture_array_index, 4)
        self.pixel_grid.arrays["fixture_detection"][rr, cc] = True
