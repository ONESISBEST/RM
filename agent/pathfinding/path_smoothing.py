class PathSmoother:
    """
    경로 노드를 인접 노드의 가중 평균으로 부드럽게 만드는 클래스입니다.

    각 노드의 좌표를 이전/다음 노드와 strength 가중치로 평균 내어
    로봇이 날카로운 꺾임 없이 부드러운 곡선으로 이동할 수 있게 합니다.

    strength=0이면 평활화 없음, 클수록 더 부드럽게 됩니다.
    """
    def __init__(self, strenght) -> None:
        self.strenght = strenght

    def smooth(self, path):
        """
        경로의 각 노드를 이전/다음 노드와 가중 평균하여 평활화된 경로를 반환합니다.

        공식: avg = (node + prior * strength + next * strength) / (1 + strength * 2)
        첫 번째와 마지막 노드는 자기 자신이 이전/다음 노드로 사용되므로 변화가 적습니다.
        """
        new_path = []
        for index, node in enumerate(path):
            prior = path[max(index - 1, 0)]
            next = path[min(index + 1, len(path) - 1)]

            avg_x = (node[0] + prior[0] * self.strenght + next[0] * self.strenght) / (1 + self.strenght * 2)
            avg_y = (node[1] + prior[1] * self.strenght + next[1] * self.strenght) / (1 + self.strenght * 2)

            new_path.append([avg_x, avg_y])

        return new_path
