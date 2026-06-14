"""
TaskRegistry — task registry for the RL environment.

Stores registered tasks, allows loading them by id,
from a directory, or programmatically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from automl_eval.domain.task import Task


class TaskRegistry:
    """Collection of tasks accessible by task_id."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def register(self, task: Task) -> None:
        self._tasks[task.task_id] = task

    def get(self, task_id: str) -> Task:
        if task_id not in self._tasks:
            raise KeyError(
                f"Task '{task_id}' not found. Available: {list(self._tasks)}"
            )
        return self._tasks[task_id]

    def list_ids(self) -> list[str]:
        return list(self._tasks.keys())

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self) -> Iterator[Task]:
        return iter(self._tasks.values())

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._tasks

    def load_directory(self, directory: str | Path) -> int:
        """Load all .json tasks from a directory. Returns the number loaded."""
        directory = Path(directory)
        count = 0
        for path in sorted(directory.glob("*.json")):
            try:
                task = Task.from_json(path)
            except Exception as exc:
                logging.warning("Skipping non-task json '%s': %s", path, exc)
                continue
            self.register(task)
            count += 1
        return count
