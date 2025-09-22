from typing import Protocol, runtime_checkable

from rag_lab.contracts.generation import GenerationRequest, GenerationResponse


@runtime_checkable
class IGenerator(Protocol):
    name: str
    model_id: str

    def generate(self, request: GenerationRequest) -> GenerationResponse: ...
