from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class PipelineState:
    name: str
    step: int
    batch_mask: Optional[List[bool]] = None

NestedPipelineState = List[PipelineState]

@dataclass
class ComposerStoreElement:
    id: int
    state: NestedPipelineState
    data: Dict[str, Any]