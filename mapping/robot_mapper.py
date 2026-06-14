import numpy as np
import cv2 as cv
from data_structures.compound_pixel_grid import CompoundExpandablePixelGrid
from data_structures.angle import Angle
from data_structures.vectors import Position2D, Vector2D
import math

class RobotMapper:
    """
    로봇이 지나간 경로, 카메라 시야, 탐색 범위를 픽셀 그리드에 기록하는 클래스입니다.

    3가지 레이어를 업데이트합니다:
    1. traversed: 로봇 몸체(원형)가 지나간 영역
    2. seen_by_camera: 3개 카메라의 시야 범위(원뿔 모양)가 포착한 영역
    3. discovered: 로봇 전방 넓은 시야 범위에서 탐색된 영역

    seen_by_camera와 discovered는 seen_by_lidar(라이다가 빔을 쐈던 영역)와
    AND 연산하여 실제로 센서가 닿은 곳만 기록합니다.
    """
    def __init__(self, pixel_grid: CompoundExpandablePixelGrid, robot_diameter, pixels_per_m) -> None:
        self.pixel_grid = pixel_grid
        self.robot_radius = round(robot_diameter / 2 * pixels_per_m)
        self.robot_center_radius = round(0.02 * pixels_per_m)  # 중심 반경(약 2cm)

        # 로봇 몸체 전체와 중심부의 원형 인덱스 템플릿 (오프셋 기준)
        self.__robot_center_indexes = self.__get_circle_template_indexes(self.robot_center_radius)
        self.__robot_diameter_indexes = self.__get_circle_template_indexes(self.robot_radius)

        # 카메라 시야 파라미터
        self.__camera_pov_amplitude = Angle(25, Angle.DEGREES)   # 카메라 수평 시야각 절반
        self.__camera_pov_lenght = int(0.12 * 2 * pixels_per_m) # 카메라 유효 감지 거리(픽셀)
        self.__camera_orientations = (
            Angle(0, Angle.DEGREES),    # 전방 카메라
            Angle(270, Angle.DEGREES),  # 우측 카메라
            Angle(90, Angle.DEGREES))   # 좌측 카메라

        # 탐색 시야 파라미터 (로봇 전방 170도 넓은 범위)
        self.__discovery_pov_amplitude = Angle(170, Angle.DEGREES)
        self.__discovery_pov_lenght = self.__camera_pov_lenght
        self.__discovery_pov_orientation = Angle(0, Angle.DEGREES)

    def map_traversed_by_robot(self, robot_grid_index):
        """로봇 몸체 크기의 원형 영역을 traversed 레이어에 기록합니다."""
        circle = np.zeros_like(self.__robot_diameter_indexes)
        circle[0] = self.__robot_diameter_indexes[0] + np.array(robot_grid_index)[0]
        circle[1] = self.__robot_diameter_indexes[1] + np.array(robot_grid_index)[1]

        self.pixel_grid.expand_to_grid_index((np.max(circle[0]), np.max(circle[1])))
        self.pixel_grid.expand_to_grid_index((np.min(circle[0]), np.min(circle[1])))

        robot_array_index = self.pixel_grid.grid_index_to_array_index(robot_grid_index)[:]

        circle[0] = self.__robot_diameter_indexes[0] + robot_array_index[0]
        circle[1] = self.__robot_diameter_indexes[1] + robot_array_index[1]

        self.pixel_grid.arrays["traversed"][circle[0], circle[1]] = True
        self.map_traversed_by_center_of_robot(robot_grid_index)

    def map_traversed_by_center_of_robot(self, robot_grid_index):
        """로봇 중심부(약 2cm 반경) 영역을 robot_center_traversed 레이어에 기록합니다."""
        circle = np.zeros_like(self.__robot_center_indexes)
        circle[0] = self.__robot_center_indexes[0] + np.array(robot_grid_index)[0]
        circle[1] = self.__robot_center_indexes[1] + np.array(robot_grid_index)[1]

        self.pixel_grid.expand_to_grid_index((np.max(circle[0]), np.max(circle[1])))
        self.pixel_grid.expand_to_grid_index((np.min(circle[0]), np.min(circle[1])))

        robot_array_index = self.pixel_grid.grid_index_to_array_index(robot_grid_index)[:]

        circle[0] = self.__robot_center_indexes[0] + robot_array_index[0]
        circle[1] = self.__robot_center_indexes[1] + robot_array_index[1]

        self.pixel_grid.arrays["robot_center_traversed"][circle[0], circle[1]] = True

    def map_seen_by_camera(self, robot_grid_index, robot_rotation: Angle):
        """
        3개 카메라 각각의 원뿔 시야 영역을 계산하고,
        라이다가 닿은 곳(seen_by_lidar)과 교차하여 seen_by_camera 레이어를 업데이트합니다.
        """
        global_camera_orientations = []
        for o in self.__camera_orientations:
            o1 = o + robot_rotation
            o1.normalize()
            global_camera_orientations.append(o1)

        camera_povs = self.__get_camera_povs_template_indexes(global_camera_orientations, robot_grid_index)

        self.pixel_grid.expand_to_grid_index(np.array((np.max(camera_povs[0]), np.max(camera_povs[1]))))
        self.pixel_grid.expand_to_grid_index(np.array((np.min(camera_povs[0]), np.min(camera_povs[1]))))

        camera_povs[0] += self.pixel_grid.offsets[0]
        camera_povs[1] += self.pixel_grid.offsets[1]

        # 카메라 시야 안이면서 라이다 빔이 닿은 곳만 seen_by_camera에 누적
        self.pixel_grid.arrays["seen_by_camera"][camera_povs[0], camera_povs[1]] += \
            self.pixel_grid.arrays["seen_by_lidar"][camera_povs[0], camera_povs[1]]

    def map_discovered_by_robot(self, robot_grid_index, robot_rotation: Angle):
        """
        로봇 전방 170도, 카메라 거리만큼의 원뿔 영역을 discovered 레이어에 업데이트합니다.
        GoToNonDiscoveredAgent가 이 레이어를 보고 아직 탐색 안 된 곳을 찾습니다.
        """
        global_discovered_orientation = self.__discovery_pov_orientation + robot_rotation
        global_discovered_orientation.normalize()

        discovered_template = self.__get_cone_template(
            self.__discovery_pov_lenght,
            global_discovered_orientation,
            self.__discovery_pov_amplitude)

        disc_povs = self.__get_indexes_from_template(
            discovered_template,
            robot_grid_index - np.array((self.__discovery_pov_lenght, self.__discovery_pov_lenght)))

        self.pixel_grid.expand_to_grid_index(np.array((np.max(disc_povs[0]), np.max(disc_povs[1]))))
        self.pixel_grid.expand_to_grid_index(np.array((np.min(disc_povs[0]), np.min(disc_povs[1]))))

        disc_povs[0] += self.pixel_grid.offsets[0]
        disc_povs[1] += self.pixel_grid.offsets[1]

        self.pixel_grid.arrays["discovered"][disc_povs[0], disc_povs[1]] += \
            self.pixel_grid.arrays["seen_by_lidar"][disc_povs[0], disc_povs[1]]

    def __get_cone_template(self, lenght, orientation: Angle, amplitude: Angle):
        """
        원뿔(부채꼴) 모양의 이진 마스크를 생성합니다.
        원뿔 = 원(반지름=lenght) AND 삼각형(정점=로봇, 각도=amplitude)
        """
        matrix_size = math.ceil(lenght) * 2
        int_lenght = math.ceil(lenght)
        matrix = np.zeros((matrix_size + 1, matrix_size + 1), np.uint8)

        # 원 마스크
        circle_matrix = cv.circle(np.zeros_like(matrix), (int_lenght, int_lenght), int_lenght, 1, -1)

        center_position = Position2D(int_lenght, int_lenght)

        # 부채꼴의 시작/중간/끝 방향 벡터 계산
        start_angle = orientation - (amplitude / 2)
        start_angle.normalize()
        start_vector = Vector2D(start_angle, lenght * 2)
        start_position = start_vector.to_position()
        start_position += center_position
        start_position = (math.ceil(start_position.x), math.ceil(start_position.y))

        center_angle = orientation
        center_angle.normalize()
        center_vector = Vector2D(center_angle, lenght * 2)
        center_up_position = center_vector.to_position()
        center_up_position += center_position
        center_up_position = center_up_position.astype(int)

        end_angle = orientation + (amplitude / 2)
        end_angle.normalize()
        end_vector = Vector2D(end_angle, lenght * 2)
        end_position = end_vector.to_position()
        end_position += center_position
        end_position = (math.ceil(end_position.x), math.ceil(end_position.y))

        # 삼각형 마스크 (원뿔 방향 범위)
        triangle_matrix = cv.fillPoly(np.zeros_like(matrix),
                                      [np.array([start_position, center_up_position,
                                                  end_position, np.array(center_position)])],
                                      1)

        # 원 AND 삼각형 = 부채꼴 (원뿔 단면)
        final_matrix = triangle_matrix * circle_matrix
        return final_matrix

    def __get_camera_povs_template_indexes(self, camera_orientations, robot_index):
        """3개 카메라의 시야 원뿔을 합산하여 전체 카메라 시야 인덱스를 반환합니다."""
        final_template = None
        for orientation in camera_orientations:
            cone_template = self.__get_cone_template(
                self.__camera_pov_lenght, orientation, self.__camera_pov_amplitude)
            if final_template is None:
                final_template = cone_template
            else:
                final_template += cone_template

        povs_indexes = self.__get_indexes_from_template(
            final_template,
            (-self.__camera_pov_lenght + robot_index[0], -self.__camera_pov_lenght + robot_index[1]))

        return povs_indexes

    def __get_circle_template_indexes(self, radius):
        """지정 반경의 원 내부 픽셀 인덱스를 중심 오프셋 기준으로 반환합니다."""
        diameter = int(radius * 2 + 1)
        diameter_template = np.zeros((diameter, diameter), dtype=np.uint8)
        diameter_template = cv.circle(diameter_template, (radius, radius), radius, 255, -1)
        diameter_template = diameter_template.astype(np.bool_)
        return self.__get_indexes_from_template(diameter_template, (-radius, -radius))

    def __get_indexes_from_template(self, template: np.ndarray, offsets=(0, 0)):
        """bool 배열에서 True인 위치의 인덱스를 오프셋 적용 후 반환합니다."""
        indexes = template.nonzero()
        indexes = np.array(indexes)
        offsets = np.array(offsets)
        indexes[0] += offsets[0]
        indexes[1] += offsets[1]
        return indexes
