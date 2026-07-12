"""Deterministic critical-path list scheduler.

The placement is a hint for persistent workers, never a claim that logical CTA
``i`` runs on physical SM ``i``. A future device runtime may steal ready work.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Sequence

from .errors import CompilationError
from .task import Task, TaskDependencies


@dataclass(frozen=True)
class ScheduleEntry:
    task_id: int
    worker_hint: int
    start: float
    finish: float
    critical_path: float


@dataclass(frozen=True)
class ExecutionPlan:
    entries: tuple[ScheduleEntry, ...]
    dispatch_order: tuple[int, ...]
    num_workers: int
    makespan: float

    def entry(self, task_id: int) -> ScheduleEntry:
        return self.entries[task_id]


def schedule_tasks(
    tasks: Sequence[Task], dependencies: TaskDependencies, *, num_workers: int
) -> ExecutionPlan:
    if num_workers <= 0:
        raise CompilationError("num_workers must be positive")
    if len(tasks) != len(dependencies.predecessors):
        raise CompilationError("task/dependency sizes differ")

    critical = [0.0] * len(tasks)
    for task_id in range(len(tasks) - 1, -1, -1):
        child_cost = max(
            (critical[child] for child in dependencies.successors[task_id]), default=0.0
        )
        critical[task_id] = tasks[task_id].estimated_cost + child_cost

    indegree = [len(items) for items in dependencies.predecessors]
    ready: list[tuple[float, int]] = []
    for task_id, count in enumerate(indegree):
        if count == 0:
            heapq.heappush(ready, (-critical[task_id], task_id))

    worker_available = [0.0] * num_workers
    finish_times = [0.0] * len(tasks)
    entries: list[ScheduleEntry | None] = [None] * len(tasks)
    dispatch_order: list[int] = []
    while ready:
        _, task_id = heapq.heappop(ready)
        dependency_ready = max(
            (finish_times[parent] for parent in dependencies.predecessors[task_id]),
            default=0.0,
        )
        worker = min(
            range(num_workers),
            key=lambda item: (max(worker_available[item], dependency_ready), item),
        )
        start = max(worker_available[worker], dependency_ready)
        finish = start + tasks[task_id].estimated_cost
        worker_available[worker] = finish
        finish_times[task_id] = finish
        entries[task_id] = ScheduleEntry(
            task_id, worker, start, finish, critical[task_id]
        )
        dispatch_order.append(task_id)
        for child in dependencies.successors[task_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, (-critical[child], child))

    if len(dispatch_order) != len(tasks):
        raise CompilationError("task graph is cyclic")
    return ExecutionPlan(
        tuple(entry for entry in entries if entry is not None),
        tuple(dispatch_order),
        num_workers,
        max(worker_available, default=0.0),
    )
