"""Register native Cyclo tasks; SuperMimic is selected explicitly by its CLI."""

from mjlab.utils.lab_api.tasks.importer import import_packages

import_packages(__name__, ["utils", ".mdp", ".supermimic"])
