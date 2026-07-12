import unittest
import random

from megakernel.errors import CompilationError

from megakernel.task import (
    Access,
    AccessMode,
    Region,
    ResourceRef,
    Task,
    TaskDraft,
    derive_dependencies,
)


def make_task(task_id, accesses, *, node_id=None, cost=1.0):
    return Task(
        task_id,
        task_id if node_id is None else node_id,
        1,
        "test",
        (task_id, 0, 0),
        tuple(),
        tuple(accesses),
        b"",
        cost,
    )


class DependencyTests(unittest.TestCase):
    @staticmethod
    def _closure(edges, task_count):
        reachable = [set() for _ in range(task_count)]
        for parent, child in edges:
            reachable[parent].add(child)
        for parent in range(task_count - 1, -1, -1):
            for child in tuple(reachable[parent]):
                reachable[parent].update(reachable[child])
        return tuple(frozenset(items) for items in reachable)

    def test_task_draft_rejects_non_integer_coordinates_and_non_finite_cost(self):
        for coordinates in ((True, 0, 0), (1.5, 0, 0), (float("nan"), 0, 0)):
            with self.assertRaises(CompilationError):
                TaskDraft(0, coordinates, (), (), {})
        for cost in (float("nan"), float("inf"), 0.0):
            with self.assertRaises(CompilationError):
                TaskDraft(0, (0, 0, 0), (), (), {}, estimated_cost=cost)

    def test_region_overlap_enables_streaming(self):
        buffer = ResourceRef.buffer(0)
        tasks = (
            make_task(0, [Access(buffer, AccessMode.WRITE, Region(((0, 2), (0, 4))))]),
            make_task(1, [Access(buffer, AccessMode.WRITE, Region(((2, 4), (0, 4))))]),
            make_task(2, [Access(buffer, AccessMode.READ, Region(((0, 2), (0, 4))))]),
            make_task(3, [Access(buffer, AccessMode.READ, Region(((2, 4), (0, 4))))]),
        )
        dependencies = derive_dependencies(tasks)
        self.assertEqual(dependencies.predecessors[2], (0,))
        self.assertEqual(dependencies.predecessors[3], (1,))
        self.assertEqual(dependencies.edge_count, 2)

    def test_read_write_and_write_after_read_hazards(self):
        buffer = ResourceRef.buffer(0)
        tasks = (
            make_task(0, [Access(buffer, AccessMode.READ)]),
            make_task(1, [Access(buffer, AccessMode.WRITE)]),
            make_task(2, [Access(buffer, AccessMode.READ_WRITE)]),
        )
        dependencies = derive_dependencies(tasks)
        self.assertEqual(dependencies.predecessors[1], (0,))
        self.assertEqual(dependencies.predecessors[2], (1,))

    def test_transitive_edges_are_removed(self):
        a = ResourceRef.buffer(0)
        b = ResourceRef.buffer(1)
        tasks = (
            make_task(0, [Access(a, AccessMode.WRITE)]),
            make_task(1, [Access(a, AccessMode.READ), Access(b, AccessMode.WRITE)]),
            make_task(2, [Access(a, AccessMode.READ), Access(b, AccessMode.READ)]),
        )
        dependencies = derive_dependencies(tasks, transitive_reduction=True)
        self.assertEqual(dependencies.successors[0], (1,))
        self.assertEqual(dependencies.successors[1], (2,))

    def test_atomic_reductions_can_run_independently(self):
        output = ResourceRef.buffer(0)
        tasks = (
            make_task(
                0,
                [Access(output, AccessMode.REDUCE, reduction_op="add", atomic=True)],
            ),
            make_task(
                1,
                [Access(output, AccessMode.REDUCE, reduction_op="add", atomic=True)],
            ),
        )
        self.assertEqual(derive_dependencies(tasks).edge_count, 0)

    def test_accesses_inside_one_task_do_not_create_self_edges(self):
        buffer = ResourceRef.buffer(0)
        task = make_task(
            0,
            [Access(buffer, AccessMode.READ), Access(buffer, AccessMode.WRITE)],
        )
        self.assertEqual(derive_dependencies((task,)).edge_count, 0)

    def test_long_whole_buffer_write_chain_stays_sparse(self):
        buffer = ResourceRef.buffer(0)
        tasks = tuple(
            make_task(task_id, [Access(buffer, AccessMode.WRITE)])
            for task_id in range(2000)
        )
        dependencies = derive_dependencies(tasks)
        self.assertEqual(dependencies.edge_count, len(tasks) - 1)
        self.assertEqual(dependencies.successors[0], (1,))

    def test_whole_resource_frontier_matches_naive_conflict_reachability(self):
        rng = random.Random(17)
        buffer = ResourceRef.buffer(0)
        for _ in range(100):
            accesses = []
            for _task_id in range(12):
                mode = rng.choice(
                    (
                        AccessMode.READ,
                        AccessMode.WRITE,
                        AccessMode.READ_WRITE,
                        AccessMode.REDUCE,
                    )
                )
                if mode == AccessMode.REDUCE:
                    accesses.append(
                        Access(
                            buffer,
                            mode,
                            reduction_op=rng.choice(("add", "max")),
                            atomic=rng.choice((True, False)),
                        )
                    )
                else:
                    accesses.append(Access(buffer, mode))
            tasks = tuple(
                make_task(task_id, [access]) for task_id, access in enumerate(accesses)
            )
            actual = derive_dependencies(tasks)
            actual_edges = {
                (parent, child)
                for parent, children in enumerate(actual.successors)
                for child in children
            }
            naive_edges = {
                (parent, child)
                for child, access in enumerate(accesses)
                for parent, previous in enumerate(accesses[:child])
                if access.conflicts(previous)
            }
            self.assertEqual(
                self._closure(actual_edges, len(tasks)),
                self._closure(naive_edges, len(tasks)),
            )


if __name__ == "__main__":
    unittest.main()
