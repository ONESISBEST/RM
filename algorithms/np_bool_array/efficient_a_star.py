import numpy as np
import cv2 as cv
from heapq import heappop, heappush
import math


class aStarNode:
    """
    힙 기반 A* 알고리즘의 노드 클래스입니다.
    heapq에서 사용하기 위해 __gt__ 비교 연산자를 구현합니다.
    g=inf로 초기화하여 첫 방문 시 어떤 비용이든 갱신되도록 합니다.
    """
    def __init__(self, location):
        self.location = location
        self.parent = None
        self.g = float('inf')  # 시작점에서 이 노드까지의 실제 비용 (초기값 무한대)
        self.p = 0             # 선호도 패널티
        self.f = 0             # f = g + h + p (총 비용, 힙 정렬 기준)

    def __gt__(self, other):
        """heapq 비교를 위해 f 값으로 대소 비교합니다."""
        return self.f > other.f

    def __repr__(self):
        return str(self.location)


class aStarAlgorithm:
    """
    힙(priority queue) 기반의 효율적인 A* 경로 탐색 알고리즘입니다.
    a_star.py의 O(n²) 리스트 기반 구현 대비 훨씬 빠릅니다.

    특징:
    - heapq를 사용한 O(log n) 노드 선택
    - best_cost_for_node_lookup 딕셔너리로 중복 노드 처리 최적화
    - closed set으로 이미 처리된 노드 재방문 방지
    - 4방향 이동 (상하좌우)
    - preference_weight=2 (a_star.py의 50보다 낮아 선호도 영향 줄임)
    - search_limit 파라미터로 최대 탐색 루프 수 제한 가능
    - 옥탄 휴리스틱 사용: 대각선 거리 추정으로 더 정확한 h값 계산
    """
    def __init__(self):
        # 4방향 인접 이동 벡터 (대각선 제외)
        self.adjacents = [[0, 1], [0, -1], [-1, 0], [1, 0], ]#[1, 1], [1, -1], [-1, -1], [-1, 1]]
        self.preference_weight = 2  # 벽 근처 경로 회피 패널티 가중치

    @staticmethod
    def reconstructpath(node):
        """목표 노드에서 부모를 따라 역추적하여 시작점→목표 순서의 경로를 반환합니다."""
        path = []
        while node is not None:
            path.append(node.location)
            node = node.parent
        path.reverse()
        return path

    @staticmethod
    def heuristic(start, target):
        """
        옥탄(Octile) 거리 휴리스틱: 4방향 이동 시의 낙관적 비용 추정.
        수평/수직 이동 비용을 10으로, 대각선 이동 비용을 15로 가정합니다.
        (실제로는 대각선 이동이 없지만 더 정확한 h 추정을 제공합니다)
        """
        dy = abs(start[0] - target[0])
        dx = abs(start[1] - target[1])
        return min(dx, dy) * 15 + abs(dx - dy) * 10

    @staticmethod
    def get_preference(preference_grid, position):
        """주어진 위치의 선호도 값을 반환합니다. 범위 밖이면 0을 반환합니다."""
        if preference_grid is None:
            return 0
        elif not (position[0] >= preference_grid.shape[0] or position[1] >= preference_grid.shape[1] or position[0] < 0 or position[1] < 0):
            return int(preference_grid[position[0], position[1]])
        else:
            return 0

    @staticmethod
    def is_traversable(grid, position):
        """
        주어진 위치가 배열 범위 내이고 통과 가능한지 확인합니다.
        배열 범위 밖은 통과 가능(True)으로 처리합니다.
        """
        if not (position[0] >= grid.shape[0] or position[1] >= grid.shape[1] or position[0] < 0 or position[1] < 0):
            return not grid[position[0], position[1]]
        else:
            return True


    # 주어진 grid에서 start에서 end까지의 경로 노드 목록을 반환합니다
    def a_star(self, grid: np.ndarray, start, end, preference_grid=None, search_limit=float('inf')):
        debug_grid = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)

        # 시작 노드 초기화 (g=0)
        start_node = aStarNode(tuple(start))
        start_node.g = 0

        if not self.is_traversable(grid, start):
            print(f"[A*:efficient_a_star.a_star] 경고: 시작점 {tuple(start)}이 통과 불가(traversable) 영역입니다")

        # 목표가 장애물 위에 있으면 빈 경로 반환
        end_node = aStarNode(tuple(end))

        if not self.is_traversable(grid, end):
            print(f"[A*:efficient_a_star.a_star] 경고: 목표점 {tuple(end)}이 통과 불가 영역 → 빈 경로 반환")
            return []

        end_node.g = end_node.h = end_node.f = 0
        # open list: 힙 기반 우선순위 큐 (f값 최소 노드가 맨 앞)
        openList = [start_node]
        # 각 위치의 최적 비용을 추적하는 딕셔너리 (중복 노드 처리에 활용)
        best_cost_for_node_lookup = {tuple(start_node.location): start_node.g}
        # 이미 처리된 노드 집합 (재방문 방지)
        closed = set()

        loop_n = 0
        # 목표 도달 또는 open list가 빌 때까지 반복
        while openList:
            # 힙에서 f값이 가장 낮은 노드 꺼냄 (O(log n))
            node = heappop(openList)
            # 이미 처리된 노드(지연 삭제된 중복)는 건너뜀
            if node.location in closed:
                continue

            closed.add(node.location)
            # 목표 노드에 도달 시 경로 역추적
            if node.location == end_node.location:
                return self.reconstructpath(node)

            # 4방향 자식 노드 탐색
            for adj in self.adjacents:
                child_location = (node.location[0] + adj[0], node.location[1] + adj[1])
                # 통과 불가 위치는 건너뜀
                if not self.is_traversable(grid, child_location):
                    continue
                # 자식 노드의 비용 계산
                new_child = aStarNode(child_location)
                new_child.parent = node

                new_child.g = node.g + 1
                new_child.h = self.heuristic(new_child.location, end_node.location)

                # 선호도 패널티 적용 (벽 근처 위치에 추가 비용)
                new_child.p = self.get_preference(preference_grid, new_child.location) * self.preference_weight

                new_child.f = new_child.g + new_child.h + new_child.p

                # 이 위치를 이전에 더 낮은 비용으로 방문했으면 건너뜀
                # 더 낮은 비용이면 힙에 추가 (이전 항목은 closed set에서 걸러짐)
                if child_location in best_cost_for_node_lookup.keys():
                    if new_child.g + new_child.p < best_cost_for_node_lookup[child_location]:
                        best_cost_for_node_lookup[child_location] = new_child.g + new_child.p
                        heappush(openList, new_child)

                else:
                    best_cost_for_node_lookup[child_location] = new_child.g + new_child.p
                    heappush(openList, new_child)

            loop_n += 1
            # 탐색 루프 한계 초과 시 조기 종료
            if loop_n > search_limit:
                break

        return []
