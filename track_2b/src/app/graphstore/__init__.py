"""The application's own storage, kept separate from the Thalabus platform.

``neo4j`` is the access layer: the only module in the submission that imports the
database driver, and the only place connection settings are read. It is a package
member rather than ``app/neo4j.py`` on purpose — see its module docstring.
"""
