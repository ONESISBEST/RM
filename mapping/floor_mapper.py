import numpy as np
import cv2 as cv
from data_structures.compound_pixel_grid import CompoundExpandablePixelGrid
from data_structures.angle import Angle
import imutils
from copy import copy, deepcopy
from robot.devices.camera import CameraImage
from typing import List

class ColorFilter:
    """HSV 색 공간에서 지정된 범위의 색을 추출하는 필터입니다."""
    def __init__(self, lower_hsv, upper_hsv):
        self.lower = np.array(lower_hsv)
        self.upper = np.array(upper_hsv)

    def filter(self, img):
        """입력 이미지를 HSV로 변환 후 lower~upper 범위의 마스크를 반환합니다."""
        hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_image, self.lower, self.upper)
        return mask

class FloorMapper:
    """
    카메라 이미지를 역투시 변환(IPM)하여 바닥 색상을 픽셀 그리드에 매핑하는 클래스입니다.

    동작 흐름:
    1. flatten_camera_pov(): 카메라 영상을 IPM(역투시변환)으로 탑뷰로 변환
    2. set_in_background(): 큰 배경 배열 중앙에 배치
    3. rotate_image_to_angle(): 카메라 방향에 맞게 회전
    4. get_unified_povs(): 3개 카메라 뷰를 하나로 합성
    5. load_povs_to_grid(): 픽셀 그리드의 바닥 색상 레이어에 기록
    6. detect_holes/swamps/checkpoints(): 색상 필터로 바닥 타일 종류 판별
    """
    def __init__(self, pixel_grid: CompoundExpandablePixelGrid, tile_resolution, tile_size, camera_distance_from_center) -> None:
        self.pixel_grid = pixel_grid
        self.tile_resolution = tile_resolution
        self.tile_size = tile_size
        self.pixel_per_m = tile_resolution / tile_size
        # 카메라 POV 이미지의 로봇 중심 오프셋 (단위: 픽셀)
        self.pov_distance_from_center = round(0.079 * self.pixel_per_m)
        # 바닥 타일 색상 분류용 HSV 필터
        self.hole_color_filter = ColorFilter((0, 0, 10), (0, 0, 27))          # 구멍 (검은색)
        self.swamp_color_filter = ColorFilter((19, 112, 32), (19, 141, 166))   # 늪지대 (갈색)
        self.checkpoint_color_filter = ColorFilter((95, 0, 65), (128, 122, 198)) # 체크포인트 (보라/파란색)
        self.wall_color_filter = ColorFilter((90, 61,  0), (100, 150, 255))   # 벽 색상 (제거용)

        # IPM 변환 대상 영역 설정 (타일 단위)
        tiles_up = 0
        tiles_down = 1
        tiles_sides = 1

        # 변환 후 이미지에서 중심 타일의 영역 좌표
        min_x = self.tile_resolution * tiles_sides
        max_x = self.tile_resolution * (tiles_sides + 1)
        min_y = self.tile_resolution * tiles_down
        max_y = self.tile_resolution * (tiles_down + 1)

        # IPM 변환의 출력 영역 꼭짓점 (탑뷰 이미지에서의 위치)
        self.center_tile_points_in_final_image = np.array(((min_x, min_y),
                                                           (max_x, min_y),
                                                           (max_x, max_y),
                                                           (min_x, max_y),), dtype=np.float32)

        # IPM 변환의 입력 영역 꼭짓점 (원본 카메라 이미지에서의 위치)
        self.center_tile_points_in_input_image = np.array(([0, 24],  [39, 24], [32, 16], [7, 16]), dtype=np.float32)

        # IPM 변환 결과 이미지 크기
        self.flattened_image_shape = (self.tile_resolution * (tiles_sides * 2 + 1),
                                      self.tile_resolution * (tiles_up + tiles_down + 1))

        # 최종 POV 이미지 크기와 거리 기반 가중치 그라디언트
        self.final_povs_shape = (120, 120)
        # 중심에서 가까울수록 높은 값 → 최근 가까운 관측이 우선되도록 함
        self.distance_to_center_gradient = self.__get_distance_to_center_gradient(self.final_povs_shape)

    def flatten_camera_pov(self, camera_pov: np.ndarray):
        """
        역투시 변환(IPM)으로 카메라 뷰를 탑뷰(조감도)로 변환합니다.
        로봇 앞방향에 공백을 추가하여 로봇 중심 기준으로 정렬합니다.
        """
        ipm_matrix = cv.getPerspectiveTransform(self.center_tile_points_in_input_image,
                                                self.center_tile_points_in_final_image,
                                                solveMethod=cv.DECOMP_SVD)

        ipm = cv.warpPerspective(camera_pov, ipm_matrix, self.flattened_image_shape, flags=cv.INTER_NEAREST)

        ipm = cv.resize(ipm, self.flattened_image_shape, interpolation=cv.INTER_CUBIC)

        # 카메라와 로봇 중심 사이의 거리만큼 위에 공백 추가
        blank_space = np.zeros((self.pov_distance_from_center, self.flattened_image_shape[0], 4), dtype=np.uint8)
        ipm = np.vstack((blank_space, ipm))

        return ipm

    def set_in_background(self, pov: np.ndarray, background=None):
        """
        POV 이미지를 큰 정사각형 배경의 중앙에 배치합니다.
        회전 시 이미지가 잘리지 않도록 충분한 여백을 확보합니다.
        """
        max_dim = max(pov.shape)
        if background is None: background = np.zeros((max_dim * 2, max_dim * 2, 4), dtype=np.uint8)

        start = (max_dim, max_dim - round(pov.shape[1] / 2))
        end =  (start[0] + pov.shape[0], start[1] + pov.shape[1])

        background[start[0]:end[0], start[1]:end[1], :] = pov[:,:,:]

        return background


    def get_global_camera_orientations(self, robot_orientation: Angle):
        """각 카메라의 로컬 방향에 로봇의 현재 방향을 더해 전역 방향을 계산합니다."""
        global_camera_orientations = []
        for camera_orientation in self.pixel_grid.camera_orientations:
            o = camera_orientation + robot_orientation
            o.normalize()
            global_camera_orientations.append(o)

        return global_camera_orientations

    def rotate_image_to_angle(self, image: np.ndarray, angle: Angle):
        """이미지를 중심을 기준으로 지정된 각도만큼 회전합니다."""
        return imutils.rotate(image, angle.degrees, (image.shape[0] // 2, image.shape[1] // 2))


    def get_unified_povs(self, camera_images: List[CameraImage]):
        """
        3개 카메라 이미지를 각각 IPM 변환, 배경 배치, 방향 회전하여
        하나의 통합된 탑뷰 이미지로 합산합니다.
        """
        povs_list = []
        for camera_image in camera_images:
            # 카메라 이미지를 90도 회전(세로→가로 변환) 후 좌우 반전
            pov = self.flatten_camera_pov(np.rot90(copy(camera_image.image), k=3))
            pov = np.flip(pov, 1)
            pov = self.set_in_background(pov)
            # 카메라의 수평 방향으로 회전하여 전역 좌표계에 정렬
            pov = self.rotate_image_to_angle(pov, camera_image.data.horizontal_orientation)
            povs_list.append(pov)

        # 3개 POV를 합산하여 통합 탑뷰 생성
        return sum(povs_list)

    def map_floor(self, camera_images, robot_grid_index):
        """통합 POV를 생성하여 픽셀 그리드에 바닥 색상을 매핑합니다."""
        povs = self.get_unified_povs(camera_images)

        self.load_povs_to_grid(robot_grid_index, povs)

    def load_povs_to_grid(self, robot_grid_index, povs):
        """
        통합 POV 이미지를 픽셀 그리드의 바닥 색상 레이어에 기록합니다.

        거리 그라디언트를 활용하여 로봇에 가까운(더 확실한) 관측이
        먼 곳의 오래된 관측을 덮어쓸 수 있도록 합니다.
        카메라가 이미 본 영역(seen_by_camera)에서만, 벽이 아닌 픽셀만 업데이트합니다.
        """
        # 그리드를 POV 크기만큼 확장
        start = np.array((robot_grid_index[0] - (povs.shape[0] // 2), robot_grid_index[1] - (povs.shape[1] // 2)))
        end = np.array((robot_grid_index[0] + (povs.shape[0] // 2), robot_grid_index[1] + (povs.shape[1] // 2)))

        self.pixel_grid.expand_to_grid_index(start)
        self.pixel_grid.expand_to_grid_index(end)

        start = self.pixel_grid.grid_index_to_array_index(start)
        end = self.pixel_grid.grid_index_to_array_index(end)

        # POV 알파 채널이 254 초과인 픽셀만 유효한 관측으로 처리
        mask = povs[:,:,3] > 254

        # 거리 그라디언트: 유효 픽셀만 남기고 나머지는 0
        povs_gradient = np.zeros_like(self.distance_to_center_gradient)
        povs_gradient[mask] = self.distance_to_center_gradient[mask]

        # 현재 저장된 거리값보다 가까운(그라디언트 높은) 관측만 업데이트
        detection_distance_mask = self.pixel_grid.arrays["floor_color_detection_distance"][start[0]:end[0], start[1]:end[1]] < povs_gradient

        # 카메라가 본 영역만 업데이트
        seen_by_camera_mask = self.pixel_grid.arrays["seen_by_camera"][start[0]:end[0], start[1]:end[1]]

        # 벽 색상 픽셀은 제외 (벽 색상이 바닥 색상으로 잘못 기록되지 않도록)
        not_walls_mask = self.wall_color_filter.filter(povs) == False

        # 세 마스크 모두 True인 픽셀만 업데이트
        final_mask = seen_by_camera_mask * detection_distance_mask * not_walls_mask

        self.pixel_grid.arrays["floor_color_detection_distance"][start[0]:end[0], start[1]:end[1]][final_mask] = povs_gradient[final_mask]

        self.pixel_grid.arrays["floor_color"][start[0]:end[0], start[1]:end[1]][final_mask] = povs[:,:,:3][final_mask]

        # 바닥 색상 업데이트 후 각 타일 종류 감지
        self.detect_holes()
        self.detect_swamps()
        self.detect_checkpoints()


    def __get_distance_to_center_gradient(self, shape):
        """
        이미지 중심에서 거리의 역수를 0~255 범위로 정규화한 그라디언트를 생성합니다.
        중심에 가까울수록 높은 값 → 가까운 관측이 우선순위를 가짐.
        """
        gradient = np.zeros(shape, dtype=np.float32)
        for x in range(shape[0]):
            for y in range(shape[1]):
                gradient[x, y] = (x - shape[0] // 2) ** 2 + (y - shape[1] // 2) ** 2

        # 거리 제곱의 역수로 변환 후 0~255 범위로 정규화
        gradient = 1 - gradient / gradient.max()

        return (gradient * 255).astype(np.uint8)

    def __get_offsets(self, tile_size):
        """픽셀 그리드의 오프셋을 타일 크기로 모듈러 계산하여 타일 정렬 오프셋을 반환합니다."""
        x_offset = int(self.pixel_grid.offsets[0] % tile_size + tile_size / 2)
        y_offset = int(self.pixel_grid.offsets[1] % tile_size + tile_size / 2)

        return (x_offset, y_offset)

    def offset_array(self, array, offsets):
        """배열을 오프셋만큼 잘라 타일 정렬된 부분 배열을 반환합니다."""
        return array[offsets[0]:, offsets[1]:]

    def get_color_average_kernel(self):
        """타일 크기의 80% 사각형 커널 (타일 내 평균 색상 계산용)을 반환합니다."""
        tile_size = round(self.tile_size * self.pixel_per_m)
        square_propotion = 0.8
        square_size = round(tile_size * square_propotion)

        kernel = np.ones((square_size, square_size), dtype=np.float32)

        kernel = kernel / kernel.sum()

        return kernel

    def detect_swamps(self):
        """
        바닥 색상에서 늪지대 색을 필터링하여 타일 단위로 swamps 레이어에 기록합니다.
        detection_proportion=0.3 (타일의 30% 이상이 늪 색이면 늪지대로 판정).
        """
        swamp_array = self.swamp_color_filter.filter(self.pixel_grid.arrays["floor_color"]).astype(np.bool_)

        self.pixel_grid.arrays["swamps"] = self.get_squares_from_raw_array(swamp_array, self.pixel_grid.offsets - self.tile_resolution // 2, self.tile_resolution, margin=3, detection_proportion=0.3)

    def detect_holes(self):
        """
        바닥 색상에서 구멍 색을 필터링하여 hole_detections에 누적하고
        타일 단위로 holes 레이어에 기록합니다.
        detection_proportion=0.2 (타일의 20% 이상이 구멍 색이면 구멍으로 판정).
        """
        valid_floor_mask = self.pixel_grid.arrays["floor_color_detection_distance"] > 0
        current_holes = self.hole_color_filter.filter(self.pixel_grid.arrays["floor_color"]).astype(np.bool_)
        current_holes &= valid_floor_mask
        current_holes &= ~self.pixel_grid.arrays["walls"]
        current_holes &= ~self.pixel_grid.arrays["traversed"]

        self.pixel_grid.arrays["hole_detections"] += current_holes
        self.pixel_grid.arrays["hole_detections"][~valid_floor_mask] = 0
        self.pixel_grid.arrays["hole_detections"][self.pixel_grid.arrays["walls"]] = 0
        self.pixel_grid.arrays["hole_detections"][self.pixel_grid.arrays["traversed"]] = 0

        confirmed_holes = self.pixel_grid.arrays["hole_detections"] >= 2
        self.pixel_grid.arrays["holes"] = self.get_squares_from_raw_array(
            confirmed_holes,
            self.pixel_grid.offsets - self.tile_resolution // 2,
            self.tile_resolution,
            detection_proportion=0.45)

    def detect_checkpoints(self):
        """
        바닥 색상에서 체크포인트 색을 필터링하여
        타일 중심점을 checkpoints 레이어에 기록합니다.
        """
        checkpoint_array = self.checkpoint_color_filter.filter(self.pixel_grid.arrays["floor_color"]).astype(np.bool_)

        self.pixel_grid.arrays["checkpoints"] = self.get_tile_centers_from_raw_array(checkpoint_array, self.pixel_grid.offsets - self.tile_resolution // 2, self.tile_resolution)


    def get_squares_from_raw_array(self, hole_array: np.ndarray, raw_offsets: np.ndarray, square_size, margin=0, detection_proportion=0.5) -> np.ndarray:
        """
        원시 픽셀 배열을 타일 단위로 분할하여,
        각 타일에서 True 픽셀 비율이 detection_proportion을 초과하면
        해당 타일 전체(margin 포함)를 True로 설정한 배열을 반환합니다.
        구멍/늪지대 감지에 사용됩니다.
        """
        offsets = np.round(raw_offsets % square_size).astype(int)

        final_hole_array = np.zeros_like(hole_array)

        for x in range(offsets[0], hole_array.shape[0] - square_size, square_size):
            for y in range(offsets[1], hole_array.shape[1] - square_size, square_size):
                min_x = x
                min_y = y
                max_x = x + square_size
                max_y = y + square_size

                square = hole_array[min_x:max_x, min_y:max_y]

                count = np.count_nonzero(square)

                # 타일 최대 차원의 제곱 기준으로 비율 계산
                if count / (np.max(square.shape) ** 2) > detection_proportion:
                    # margin만큼 확장하여 경계도 포함
                    final_hole_array[min_x - margin:max_x + margin, min_y -margin:max_y+margin] = True


        return final_hole_array

    def get_tile_centers_from_raw_array(self, raw_array: np.ndarray, raw_offsets: np.ndarray, square_size) -> np.ndarray:
        """
        원시 픽셀 배열을 타일 단위로 분할하여,
        각 타일에서 True 픽셀 비율이 0.3을 초과하면
        해당 타일의 중심점만 True로 설정한 배열을 반환합니다.
        체크포인트 감지에 사용됩니다 (점 하나로 위치 표현).
        """
        offsets = np.round(raw_offsets % square_size).astype(int)

        final_array = np.zeros_like(raw_array)

        for x in range(offsets[0], raw_array.shape[0] - square_size, square_size):
            for y in range(offsets[1], raw_array.shape[1] - square_size, square_size):
                min_x = x
                min_y = y
                max_x = x + square_size
                max_y = y + square_size

                square = raw_array[min_x:max_x, min_y:max_y]

                count = np.count_nonzero(square)

                if count / (np.max(square.shape) ** 2) > 0.3:
                    # 타일 전체가 아닌 중심점 하나만 True로 설정
                    final_array[min_x + square_size // 2, min_y + square_size // 2] = True


        return final_array


    def load_average_tile_color(self):
        """
        타일 크기의 평균 커널로 바닥 색상을 평활화하여
        average_floor_color 레이어에 타일 단위 평균 색상을 기록합니다.
        최종 맵 제출 시 색상 매트릭스 생성에 활용됩니다.
        """
        tile_size = self.tile_size * self.pixel_per_m
        offsets = self.__get_offsets(tile_size)
        floor_color = deepcopy(self.pixel_grid.arrays["floor_color"])

        kernel = self.get_color_average_kernel()

        # 타일 크기의 평균 커널로 컨볼루션 (각 타일의 평균 색상 계산)
        floor_color = cv.filter2D(floor_color, -1, kernel)

        image = []

        # 타일 중심점들의 색상만 추출
        for x in range(round(offsets[0] + tile_size / 2), floor_color.shape[0], round(tile_size)):
            row = []
            for y in range(round(offsets[1] + tile_size / 2), floor_color.shape[1], round(tile_size)):
                row.append(floor_color[x, y, :])
            image.append(row)

        image = np.array(image, dtype=np.uint8)

        # 타일 크기로 다시 확대하여 픽셀 그리드에 정렬
        image = cv.resize(image, (0, 0), fx=tile_size, fy=tile_size, interpolation=cv.INTER_NEAREST)

        final_x = image.shape[0] if image.shape[0] + offsets[0] < self.pixel_grid.array_shape[0] else self.pixel_grid.array_shape[0] - offsets[0]
        final_y = image.shape[1] if image.shape[1] + offsets[1] < self.pixel_grid.array_shape[1] else self.pixel_grid.array_shape[1] - offsets[1]

        self.pixel_grid.arrays["average_floor_color"][offsets[0]:offsets[0] + final_x:, offsets[1]:offsets[1] + final_y, :] = image[:final_x,:final_y, :]
