import numpy as np

import math

class BFSAlgorithm:
    """
    단순 BFS 탐색 알고리즘입니다.
    traversable 제약 없이 found_function이 True를 반환하는 첫 번째 노드를 탐색합니다.
    주로 가장 가까운 빈 공간(traversable=0) 찾기에 사용됩니다.

    주의: closed set이 없어 큰 배열에서는 느릴 수 있습니다.
    """
    def __init__(self, found_function) -> None:
        self.found_function = found_function
        # 4방향 이웃 (상하좌우)
        self.adjacents = [[0, 1], [0, -1], [-1, 0], [1, 0], ]

    def get_neighbours(self, node):
        """현재 노드의 4방향 이웃 노드를 생성합니다."""
        for a in self.adjacents:
            yield [node[0] + a[0], node[1] + a[1]]

    def bfs(self, array, start_node):
        """
        start_node에서 BFS를 시작하여 found_function이 True인 첫 번째 노드를 반환합니다.
        배열 경계 체크가 없으므로 사용 시 범위에 주의해야 합니다.
        """
        open_list = []
        open_list.append(start_node)

        while len(open_list) > 0:
            node = open_list.pop(0)

            value = array[node[0], node[1]]

            if self.found_function(value):
                return node

            for n in self.get_neighbours(node):
                if not n in open_list:
                    open_list.append(n)


class NavigatingBFSAlgorithm:
    """
    탐색 가능 영역(traversable_function)만 통과하면서
    found_function이 True인 위치를 BFS로 탐색하는 알고리즘입니다.

    closed set(방문 집합)을 사용하여 중복 방문을 방지하며,
    max_result_number개의 결과를 찾으면 조기 종료합니다.

    사용 예:
    - discovered=False인 미탐색 위치 찾기
    - fixture_distance_margin 후보 탐색
    - traversed 경로에서 목표까지 연결 여부 확인
    """
    def __init__(self, found_function, traversable_function, max_result_number=1) -> None:
        self.found_function = found_function
        self.traversable_function = traversable_function
        # 4방향 이웃만 사용 (대각선 제외)
        self.adjacents = ((0, 1), (0, -1), (-1, 0), (1, 0))
        self.max_result_number = max_result_number

    def get_neighbours(self, node):
        """현재 노드의 4방향 이웃 노드를 생성합니다."""
        for a in self.adjacents:
            yield (node[0] + a[0], node[1] + a[1])

    def bfs(self, found_array, traversable_array, start_node):
        """
        start_node에서 BFS를 시작합니다.
        - traversable_array로 통과 가능 여부 판단
        - found_array로 목표 노드 판단
        - max_result_number개의 결과를 찾으면 조기 반환
        배열 경계 밖의 노드는 자동으로 건너뜁니다.
        """
        open_list = []
        open_list.append(tuple(start_node))

        closed_set = set()
        closed_set.add(tuple(start_node))

        results = []

        while len(open_list) > 0:
            node = open_list.pop(0)

            # 배열 경계 밖이면 건너뜀
            if node[0] < 0 or node[1] < 0 or node[0] >= traversable_array.shape[0] or node[1] >= traversable_array.shape[1]:
                continue

            # 통과 불가 영역이면 탐색 중단 (이 방향으로는 더 이상 확장하지 않음)
            if not self.traversable_function(traversable_array[node[0], node[1]]):
                continue

            value = found_array[node[0], node[1]]

            # 목표 조건을 만족하면 결과에 추가
            if self.found_function(value):
                results.append(node)
                if len(results) >= self.max_result_number:
                    return results

            # 이웃 노드를 closed_set에 없으면 open_list에 추가
            for n in self.get_neighbours(node):
                if n not in closed_set:
                    open_list.append(n)
                    closed_set.add(n)

        return results

class NavigatingLimitedBFSAlgorithm:
    """
    NavigatingBFSAlgorithm과 동일하지만 최대 루프 횟수(limit)로 탐색을 제한합니다.
    탐색 범위가 매우 넓을 때 성능 보호를 위해 사용합니다.

    limit을 초과하면 그 시점까지 찾은 결과만 반환합니다.
    GoToFixturesAgent에서 limit=1000으로 사용됩니다.
    """
    def __init__(self, found_function, traversable_function, max_result_number=1, limit=math.inf) -> None:
        self.limit = limit
        self.found_function = found_function
        self.traversable_function = traversable_function
        # 4방향 이웃만 사용 (대각선 제외)
        self.adjacents = ((0, 1), (0, -1), (-1, 0), (1, 0))
        self.max_result_number = max_result_number

    def get_neighbours(self, node):
        """현재 노드의 4방향 이웃 노드를 생성합니다."""
        for a in self.adjacents:
            yield (node[0] + a[0], node[1] + a[1])

    def bfs(self, found_array, traversable_array, start_node):
        """
        start_node에서 제한된 BFS를 시작합니다.
        루프 횟수가 limit을 초과하면 조기 종료하고 현재까지의 결과를 반환합니다.
        """
        self.loops = 0
        open_list = []
        open_list.append(tuple(start_node))

        closed_set = set()
        closed_set.add(tuple(start_node))

        results = []

        while len(open_list) > 0:
            self.loops += 1
            # 루프 횟수 제한 초과 시 조기 종료
            if self.loops > self.limit:
                break
            node = open_list.pop(0)

            # 배열 경계 밖이면 건너뜀
            if node[0] < 0 or node[1] < 0 or node[0] >= traversable_array.shape[0] or node[1] >= traversable_array.shape[1]:
                continue

            # 통과 불가 영역이면 해당 방향으로 확장하지 않음
            if not self.traversable_function(traversable_array[node[0], node[1]]):
                continue

            value = found_array[node[0], node[1]]

            # 목표 조건 만족 시 결과에 추가
            if self.found_function(value):
                results.append(node)
                if len(results) >= self.max_result_number:
                    return results

            # 미방문 이웃 노드를 큐에 추가
            for n in self.get_neighbours(node):
                if n not in closed_set:
                    open_list.append(n)
                    closed_set.add(n)


        return results
