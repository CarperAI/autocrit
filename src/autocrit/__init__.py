from importlib.metadata import version as importlib_version

from autocrit.pipeline import (
    AgreementFilter,
    AgreementFilterChain,
    PersonaStatement,
    PersonaStatementChain,
)

__version__ = importlib_version("autocrit")

__all__ = [
    "AgreementFilter",
    "PersonaStatement",
    "PersonaStatementChain",
    "AgreementFilterChain",
]
