import math
import cv2 as cv
import numpy as np
import os
from functools import wraps
from fixture_detection.color_filter import ColorFilter

# 현재 스크립트 디렉토리 기준으로 이미지 저장 경로 설정
script_dir = os.path.dirname(__file__)
image_dir = os.path.join(script_dir, "images")

def save_image(image, filename):
    """이미지를 지정된 파일명으로 images/ 디렉토리에 저장합니다."""
    cv.imwrite(os.path.join(image_dir, filename), image)


def normalizeRads(rad):
    """라디안 값을 0 ~ 2π 범위로 정규화합니다."""
    rad %= 2 * math.pi
    if rad < 0:
        rad += 2 + math.pi
    return rad

# 도(degree)를 라디안으로 변환
def degsToRads(deg):
    return deg * math.pi / 180

# 라디안을 도(degree)로 변환
def radsToDegs(rad):
    return rad * 180 / math.pi

# 값을 한 범위에서 다른 범위로 선형 매핑합니다 (예: 센서 값 → 모터 출력)
def mapVals(val, in_min, in_max, out_min, out_max):
    return (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# 라디안 각도와 거리로부터 (x, y) 좌표를 계산합니다
# cos은 앞방향(y), sin은 측면방향(x)에 대응
def getCoordsFromRads(rad, distance):
    y = float(distance * math.cos(rad))
    x = float(distance * math.sin(rad))
    return (x, y)

# 도(degree) 각도와 거리로부터 (x, y) 좌표를 계산합니다
def getCoordsFromDegs(deg, distance):
    rad = degsToRads(deg)
    y = float(distance * math.cos(rad))
    x = float(distance * math.sin(rad))
    return (x, y)


def multiplyLists(list1, list2):
    """두 리스트를 원소별로 곱한 새 리스트를 반환합니다."""
    finalList = []
    for item1, item2 in zip(list1, list2):
        finalList.append(item1 * item2)
    return finalList

def sumLists(list1, list2):
    """두 리스트를 원소별로 더한 새 리스트를 반환합니다."""
    finalList = []
    for item1, item2 in zip(list1, list2):
        finalList.append(item1 + item2)
    return finalList

def substractLists(list1, list2):
    """list1에서 list2를 원소별로 뺀 새 리스트를 반환합니다."""
    finalList = []
    for item1, item2 in zip(list1, list2):
        finalList.append(item1 - item2)
    return finalList

def divideLists(list1, list2):
    """두 리스트를 원소별로 나눈 새 리스트를 반환합니다."""
    finalList = []
    for item1, item2 in zip(list1, list2):
        finalList.append(item1 / item2)
    return finalList


def draw_grid(image, square_size, offset = [0,0], color=255):
    """이미지에 격자선을 그립니다. 디버그용 시각화에 활용됩니다."""
    for y, row in enumerate(image):
        for x, pixel in enumerate(row):
            if (y + offset[1]) % square_size == 0 or (x + offset[0]) % square_size == 0:
                if len(image.shape) == 3:
                    image[y][x][:] = color
                else:
                    image[y][x] = color

def draw_poses(image, poses, color=255, back_image=None, xx_yy_format=False):
    """
    포즈(위치) 목록을 이미지에 점으로 표시합니다.
    xx_yy_format=True이면 (x배열, y배열) 형식, False이면 (N,2) 배열 형식을 받습니다.
    back_image가 지정되면 해당 위치에 back_image 픽셀 값을 복사합니다.
    """
    if xx_yy_format:
        if back_image is not None:
            in_bounds_x = (poses[0] < min(image.shape[0], back_image.shape[0]) - 1) & (poses[0] > 0)
            in_bounds_y = (poses[1] < min(image.shape[1], back_image.shape[1]) - 1) & (poses[1] > 0)
        else:
            in_bounds_x = (poses[0] < image.shape[0] - 1) & (poses[0] > 0)
            in_bounds_y = (poses[1] < image.shape[1] - 1) & (poses[1] > 0)

        poses = (poses[0][in_bounds_x & in_bounds_y], poses[1][in_bounds_x & in_bounds_y])

        if back_image is None:
            image[poses[1], poses[0], :] = color
        else:
            image[poses[1], poses[0], :] = back_image[poses[1], poses[0], :]

    else:
        # 배열 경계 밖의 포즈는 제거
        in_bounds = (poses[:, 0] >= 0) & (poses[:, 0] < image.shape[1]) & (poses[:, 1] >= 0) & (poses[:, 1] < image.shape[0])
        poses = poses[in_bounds]

        if back_image is None:
            image[poses[:, 1], poses[:, 0], :] = color
        else:
            image[poses[:, 1], poses[:, 0], :] = back_image[poses[:, 1], poses[:, 0], :]


def draw_squares_where_not_zero(image, square_size, offsets, color=(255, 255, 255)):
    """비어있지 않은(0이 아닌 값이 있는) 타일 구역에 사각형 테두리를 그립니다."""
    ref_image = image.copy()
    for y in range(image.shape[0] // square_size):
        for x in range(image.shape[1] // square_size):
            square_points = [
                (y * square_size)        + (square_size - offsets[1]),
                ((y + 1) * square_size)  + (square_size - offsets[1]),
                (x * square_size)        + (square_size - offsets[0]),
                ((x + 1) * square_size)  + (square_size - offsets[0])]
            square = ref_image[square_points[0]:square_points[1], square_points[2]:square_points[3]]
            non_zero_count = np.count_nonzero(square)
            if non_zero_count > 0:
                cv.rectangle(image, (square_points[2], square_points[0]), (square_points[3], square_points[1]), color, 3)

def get_squares(image, square_size, offsets):
    """
    이미지를 square_size 크기의 격자로 분할하여
    각 타일의 [min_x, max_x, min_y, max_y] 좌표 목록을 2D 배열로 반환합니다.
    """
    grid = []
    for y in range(image.shape[0] // square_size):
        row = []
        for x in range(image.shape[1] // square_size):
            square_points = [
                (y * square_size)        + (square_size - offsets[1]),
                ((y + 1) * square_size)  + (square_size - offsets[1]),
                (x * square_size)        + (square_size - offsets[0]),
                ((x + 1) * square_size)  + (square_size - offsets[0])]
            row.append(square_points)
        grid.append(row)
    return grid

def resize_image_to_fixed_size(image, size):
    """
    이미지를 지정된 최대 크기(size)에 맞게 비율 유지하며 리사이즈합니다.
    디버그 창 표시에 적합한 크기로 축소할 때 사용합니다.
    """
    if image.shape[0] > size[0]:
        ratio = size[0] / image.shape[0]

        width = round(image.shape[1] * ratio)
        final_image = cv.resize(image.astype(np.uint8), dsize=(width, size[0]))

    elif image.shape[1] > size[1]:
        ratio = size[1] / image.shape[1]

        height = round(image.shape[0] * ratio)
        final_image = cv.resize(image.astype(np.uint8), dsize=(size[1], height))

    elif image.shape[1] >= image.shape[0]:
        ratio = size[1] / image.shape[1]

        height = round(image.shape[0] * ratio)
        final_image = cv.resize(image.astype(np.uint8), dsize=(size[1], height), interpolation=cv.INTER_NEAREST)

    elif image.shape[0] >= image.shape[1]:
        ratio = size[0] / image.shape[0]

        width = round(image.shape[1] * ratio)
        final_image = cv.resize(image.astype(np.uint8), dsize=(width, size[0]), interpolation=cv.INTER_NEAREST)

    return final_image


def divide_into_chunks(lst, n):
    """리스트를 n개 크기의 청크(부분 리스트)로 나눠서 순차적으로 반환합니다."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class ColorFilterTuner:
    """
    OpenCV 트랙바 슬라이더를 사용하여 HSV 색상 필터의 범위를
    실시간으로 조정할 수 있는 디버그 도구입니다.
    activate=True일 때만 창이 열립니다.
    """
    def __init__(self, color_filter: ColorFilter, activate=False) -> None:
        self.filter_for_tuning = color_filter

        self.activate = activate

        if self.activate:
            # HSV 각 채널(H, S, V)의 최솟값/최댓값을 조정하는 트랙바 생성
            cv.namedWindow("filter_tuner")

            cv.createTrackbar("min_h", "filter_tuner", self.filter_for_tuning.lower[0], 255, lambda x: None)
            cv.createTrackbar("max_h", "filter_tuner", self.filter_for_tuning.upper[0], 255, lambda x: None)

            cv.createTrackbar("min_s", "filter_tuner", self.filter_for_tuning.lower[1], 255, lambda x: None)
            cv.createTrackbar("max_s", "filter_tuner", self.filter_for_tuning.upper[1], 255, lambda x: None)

            cv.createTrackbar("min_v", "filter_tuner", self.filter_for_tuning.lower[2], 255, lambda x: None)
            cv.createTrackbar("max_v", "filter_tuner", self.filter_for_tuning.upper[2], 255, lambda x: None)

    def tune(self, image):
        """트랙바 현재 값을 읽어 필터를 갱신하고, 적용된 마스크를 창에 표시합니다."""
        if self.activate and image is not None:
            min_h = cv.getTrackbarPos("min_h", "filter_tuner")
            max_h = cv.getTrackbarPos("max_h", "filter_tuner")
            min_s = cv.getTrackbarPos("min_s", "filter_tuner")
            max_s = cv.getTrackbarPos("max_s", "filter_tuner")
            min_v = cv.getTrackbarPos("min_v", "filter_tuner")
            max_v = cv.getTrackbarPos("max_v", "filter_tuner")
            self.filter_for_tuning = ColorFilter((min_h, min_s, min_v), (max_h, max_s, max_v))
            print(tuple(self.filter_for_tuning.lower), tuple(self.filter_for_tuning.upper))
            cv.imshow("filter_tuner", self.filter_for_tuning.filter(image))
