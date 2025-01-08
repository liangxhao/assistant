from typing import Dict, Type

from ..base.task import TaskBase
from .singleton import SingletonType, make_singleton


class TaskTable(metaclass=SingletonType):
    _table: Dict[str, Type[TaskBase]] = {}

    def add_task(self, name, task: Type[TaskBase]):
        self._table[name] = task

    def get_task(self, name) -> Type[TaskBase]:
        return self._table[name]

    def list_tasks(self) -> Dict[str, Type[TaskBase]]:
        return self._table


task_table: TaskTable = make_singleton('task_table', TaskTable())
