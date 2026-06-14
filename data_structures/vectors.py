import math
import numpy as np

from data_structures.angle import Angle

class Position2D:
    """
    2D 좌표(x, y)를 나타내는 자료형 클래스입니다.

    numpy 배열, 튜플, 리스트, 또는 두 개의 별도 인수로 생성할 수 있습니다.
    사칙연산(+, -, *, /)과 비교 연산을 지원하며,
    다른 좌표까지의 거리(get_distance_to)와 방향 각도(get_angle_to)를 계산합니다.
    """
    def __init__(self, *args, **kwargs):
        """
        생성 방법:
        - Position2D()         → x=None, y=None
        - Position2D(iterable) → x=iterable[0], y=iterable[1]
        - Position2D(x, y)     → x, y 직접 지정
        """
        if len(args) == 0:
            self.x = None
            self.y = None
        elif len(args) == 1:
            self.x = args[0][0]
            self.y = args[0][1]
        elif len(args) == 2:
            self.x = args[0]
            self.y = args[1]
        else:
            raise TypeError()

    def __iter__(self):
        yield self.x
        yield self.y

    def __array__(self, *args, **kwargs):
        """numpy 배열로 변환할 때 [x, y] 형태로 반환합니다."""
        return np.array([self.x, self.y], *args, **kwargs)

    def __repr__(self):
        return f"Position2D({self.x}, {self.y})"

    def __eq__(self, other):
        if isinstance(other, Position2D):
            return self.x == other.x and self.y == other.y
        else:
            return False

    # 산술 연산자
    def __add__(self, other):
        if isinstance(other, Position2D):
            return Position2D(self.x + other.x, self.y + other.y)
        else:
            return Position2D(self.x + other, self.y + other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, Position2D):
            return Position2D(self.x - other.x, self.y - other.y)
        else:
            return Position2D(self.x - other, self.y - other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if isinstance(other, Position2D):
            return Position2D(self.x * other.x, self.y * other.y)
        else:
            return Position2D(self.x * other, self.y * other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, Position2D):
            return Position2D(self.x / other.x, self.y / other.y)
        else:
            return Position2D(self.x / other, self.y / other)

    def __rtruediv__(self, other):
        return Position2D(other / self.x, other / self.y)

    def __floordiv__(self, other):
        if isinstance(other, Position2D):
            return Position2D(self.x // other.x, self.y // other.y)
        return Position2D(self.x // other, self.y // other)

    def __rfloordiv__(self, other):
        return self.__floordiv__(other)

    def __mod__(self, other):
        if isinstance(other, Position2D):
            return Position2D(self.x % other.x, self.y % other.y)
        else:
            return Position2D(self.x % other, self.y % other)

    def __rmod__(self, other):
        return self.__mod__(other)

    def __divmod__(self, other):
        return self.__floordiv__(other), self.__mod__(other)

    def __rdivmod__(self, other):
        return self.__divmod__(other)

    def __pow__(self, other):
        if isinstance(other, Position2D):
            return Position2D(self.x ** other.x, self.y ** other.y)
        else:
            return Position2D(self.x ** other, self.y ** other)

    def __rpow__(self, other):
        return self.__pow__(other)

    def __neg__(self):
        return Position2D(-self.x, -self.y)

    def __pos__(self):
        return Position2D(self.x, self.y)

    def __abs__(self):
        """원점에서의 거리(유클리드 크기)를 반환합니다."""
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            raise IndexError("Vector index out of range")

    def __setitem__(self, index, value):
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        else:
            raise IndexError("Vector index out of range")

    def astype(self, dtype: type):
        """x, y 각각에 dtype 함수를 적용한 새 Position2D를 반환합니다."""
        return self.apply_to_all(dtype)

    def apply_to_all(self, function):
        """x, y 각각에 함수를 적용한 새 Position2D를 반환합니다."""
        return Position2D(function(self.x), function(self.y))

    def get_distance_to(self, other):
        """다른 좌표까지의 유클리드 거리를 반환합니다."""
        return abs(self - other)

    def get_angle_to(self, other):
        """
        이 좌표에서 other 좌표까지의 방향 각도를 반환합니다.
        결과는 0 ~ 2π 범위로 정규화됩니다.
        """
        delta = self - other
        result = Angle(math.atan2(delta.x, delta.y)) + Angle(180, Angle.DEGREES)
        result.normalize()
        return result

    def to_vector(self):
        """이 좌표를 원점 기준 Vector2D(방향 + 크기)로 변환합니다."""
        m = Position2D(0, 0).get_distance_to(self)
        a = Position2D(0, 0).get_angle_to(self)
        return Vector2D(a, m)


class Vector2D:
    """
    방향(Angle)과 크기(magnitude)로 표현하는 2D 벡터 클래스입니다.
    라이다 감지 결과(방향 + 거리)를 표현하거나 카메라 위치 계산에 사용합니다.
    """
    def __init__(self, direction: Angle = None, magnitude=None):
        self.direction = direction    # 방향 (Angle 객체)
        self.magnitude = magnitude   # 크기 (거리)

    def __repr__(self):
        return f"Vector2D(direction={self.direction}, magnitude={self.magnitude})"

    def __eq__(self, other):
        if isinstance(other, Vector2D):
            return self.direction == other.direction and self.magnitude == other.magnitude
        else:
            return False

    def __add__(self, other):
        if isinstance(other, Vector2D):
            return Vector2D(self.direction + other.direction, self.magnitude + other.magnitude)
        else:
            raise TypeError("Argument must be of type Vector2D")

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, Vector2D):
            return Vector2D(self.direction - other.direction, self.magnitude - other.magnitude)
        else:
            raise TypeError("Argument must be of type Vector2D")

    def __rsub__(self, other):
        return -self + other

    def __neg__(self):
        return Vector2D(-self.direction, -self.magnitude)

    def __pos__(self):
        return Vector2D(self.direction, self.magnitude)

    def to_position(self):
        """벡터를 직교 좌표(x, y)의 Position2D로 변환합니다."""
        y = float(self.magnitude * math.cos(self.direction.radians))
        x = float(self.magnitude * math.sin(self.direction.radians))
        return Position2D(x, y)
