from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)
    role: Role
    content: str


class Prompt(BaseModel):
    model_config = ConfigDict(frozen=True)
    messages: List[Message]
    version: str = "v1"


class GenerationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    prompt: Prompt
    context: List[str] = Field(default_factory=list)
    max_tokens: int = 512
    temperature: float = 0.2


class GenerationResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    text: str
    tokens_in: int
    tokens_out: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
