"""Route registered task families to their train/play entry points."""


def task_family(task):
    if task.startswith("Cyclo-XMimic-"):
        return "xmimic"
    if task.startswith("Cyclo-BlinkMimic-"):
        return "blinkmimic"
    if task.startswith("Cyclo-SuperMimic-"):
        return "supermimic"
    if task.startswith("Cyclo-BeyondMimic-"):
        return "beyondmimic"
    return None


def dispatch(action, argv):
    """Run a specialized backend, or leave BeyondMimic/Velocity to the caller."""
    task = next((arg for arg in argv if arg.startswith("Cyclo-")), "")
    family = task_family(task)
    if family == "xmimic":
        from source.tasks.xmimic.cli import main

        main(action, argv)
        return True
    if family == "blinkmimic":
        from source.tasks.blinkmimic.cli import main

        main(action, argv)
        return True
    if family == "supermimic":
        from source.tasks.supermimic.cli import main

        main([action, *argv])
        return True
    return False
