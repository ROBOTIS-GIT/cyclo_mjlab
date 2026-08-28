"""Script to list mjlab environments."""

import tyro
from prettytable import PrettyTable

import mjlab
import mjlab.tasks  # noqa: F401
import source.tasks
from mjlab.tasks.registry import list_tasks


def list_environments(keyword: str | None = None):
  """List all registered environments.

  Args:
    keyword: Optional filter to only show environments containing this keyword.
  """
  table = PrettyTable(["#", "Task ID"])
  table.title = "Available Environments in mjlab"
  table.align["Task ID"] = "l"

  k1_task_prefixes = ("Cyclo-Velocity-", "Cyclo-Mimic-K1-Rev1-")
  all_tasks = [
    task_id for task_id in list_tasks() if task_id.startswith(k1_task_prefixes)
  ]
  idx = 0
  for task_id in all_tasks:
    try:
      # Optionally filter by keyword.
      if keyword and keyword.lower() not in task_id.lower():
        continue

      table.add_row([idx + 1, task_id])
      idx += 1
    except Exception:
      continue

  print(table)
  if idx == 0:
    msg = "[INFO] No tasks matched"
    if keyword:
      msg += f" keyword '{keyword}'"
    print(msg)
  return idx


def main():
  return tyro.cli(list_environments, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
  main()
