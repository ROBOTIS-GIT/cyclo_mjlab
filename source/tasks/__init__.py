"""Task registration scoped to the installed native mjlab runtime.

Keep the historical import-source.tasks registration on mjlab 1.2. Research
mjlab uses source.tasks.supermimic's explicit registration; importing its parent
must not initialize incompatible legacy assets or PPO task configurations.
"""

from importlib.metadata import version

if version("mjlab") == "1.2.0":
    from mjlab.utils.lab_api.tasks.importer import import_packages

    import_packages(__name__, ["utils", ".mdp", ".supermimic"])
