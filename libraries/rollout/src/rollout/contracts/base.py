"""Shared base for contract types."""

from collections.abc import Sequence
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict

type FrozenSequence[T] = Annotated[Sequence[T], AfterValidator(tuple)]
"""A sequence field that accepts any sequence and is stored as a tuple, so contract values stay immutable."""


class ContractModel(BaseModel):
    """Base for every contract type: immutable, and unknown fields are kept.

    Keeping unknown fields lets a component read and re-write a record written by newer code without dropping
    what it does not understand.
    """

    model_config = ConfigDict(frozen=True, extra="allow", use_attribute_docstrings=True)
