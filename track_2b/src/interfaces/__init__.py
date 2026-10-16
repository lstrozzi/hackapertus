# Data interfaces module

from .dossiers_interface import DossiersInterface
from .documents_interface import DocumentsInterface
from .hints_interface import HintsInterface
from .memory_interface import MemoryInterface
from .cache_entry_interface import CacheEntryInterface
from .ontology_interface import OntologyInterface

__all__ = ["DossiersInterface", "DocumentsInterface", "HintsInterface", "MemoryInterface", "CacheEntryInterface", "OntologyInterface"]