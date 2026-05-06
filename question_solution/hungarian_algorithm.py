"""
Hungarian Algorithm for assignment problems.

This module supports maximum-weight assignment and can handle rectangular
matrices by padding with virtual rows/columns.
"""
from typing import List, Tuple, Optional


class HungarianAlgorithm:
    """Solve assignment problems with a robust O(n^3) implementation."""

    def __init__(self, cost_matrix: List[List[int]]):
        self.original_n = len(cost_matrix)
        self.original_m = len(cost_matrix[0]) if cost_matrix else 0
        self.original_matrix = [row[:] for row in cost_matrix]

        self.cost_matrix = [row[:] for row in cost_matrix]
        self.n = self.original_n
        self.m = self.original_m
        if self.n != self.m:
            self._pad_matrix()

    def _pad_matrix(self):
        """Pad to square with zeros so unassigned rows/cols are possible."""
        max_size = max(self.n, self.m)
        padded = [[0] * max_size for _ in range(max_size)]
        for i in range(self.n):
            for j in range(self.m):
                padded[i][j] = self.cost_matrix[i][j]
        self.cost_matrix = padded
        self.n = max_size
        self.m = max_size

    @staticmethod
    def _hungarian_min(square_cost: List[List[float]]) -> Tuple[List[int], float]:
        """
        Standard Hungarian algorithm for minimum-cost matching on square matrix.
        Returns row->col assignment and minimum total cost.
        """
        n = len(square_cost)
        if n == 0:
            return [], 0.0

        # Potentials for rows (u) and columns (v).
        u = [0.0] * (n + 1)
        v = [0.0] * (n + 1)
        p = [0] * (n + 1)
        way = [0] * (n + 1)

        for i in range(1, n + 1):
            p[0] = i
            minv = [float("inf")] * (n + 1)
            used = [False] * (n + 1)
            j0 = 0

            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float("inf")
                j1 = 0

                for j in range(1, n + 1):
                    if used[j]:
                        continue
                    cur = square_cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

                for j in range(n + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta

                j0 = j1
                if p[j0] == 0:
                    break

            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        assignment = [-1] * n
        for j in range(1, n + 1):
            if p[j] > 0:
                assignment[p[j] - 1] = j - 1

        total_cost = sum(square_cost[i][assignment[i]] for i in range(n))
        return assignment, total_cost

    def _to_original_assignment(self, assignment_full: List[int]) -> List[int]:
        """
        Keep only original rows. If matched to a padded virtual column, return -1.
        """
        result = []
        for i in range(self.original_n):
            j = assignment_full[i] if i < len(assignment_full) else -1
            if j < 0 or j >= self.original_m:
                result.append(-1)
            else:
                result.append(j)
        return result

    def solve_minimum_assignment(
        self, cost_matrix: Optional[List[List[int]]] = None
    ) -> Tuple[List[int], int]:
        """
        Solve minimum-cost assignment for the provided matrix.
        Returns assignments for original rows only.
        """
        if self.original_n == 0 or self.original_m == 0:
            return [], 0

        square = [row[:] for row in (cost_matrix if cost_matrix is not None else self.cost_matrix)]
        if len(square) != len(square[0]):
            raise ValueError("Minimum assignment expects a square matrix.")

        assignment_full, _ = self._hungarian_min(square)
        assignment = self._to_original_assignment(assignment_full)

        total_cost = 0
        for i, j in enumerate(assignment):
            if j != -1:
                total_cost += self.original_matrix[i][j]
        return assignment, total_cost

    def solve_maximum_assignment(self) -> Tuple[List[int], int]:
        """Solve maximum-weight assignment and return original-row assignments."""
        if self.original_n == 0 or self.original_m == 0:
            return [], 0

        max_val = max(max(row) for row in self.cost_matrix)
        min_cost_matrix = [[max_val - val for val in row] for row in self.cost_matrix]
        assignment_full, _ = self._hungarian_min(min_cost_matrix)
        assignment = self._to_original_assignment(assignment_full)

        total_score = 0
        for i, j in enumerate(assignment):
            if j != -1:
                total_score += self.original_matrix[i][j]
        return assignment, total_score


def hungarian_maximum_assignment(cost_matrix: List[List[int]]) -> Tuple[List[int], int]:
    """Convenience wrapper for maximum assignment."""
    solver = HungarianAlgorithm(cost_matrix)
    return solver.solve_maximum_assignment()
