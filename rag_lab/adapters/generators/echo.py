from rag_lab.contracts.generation import GenerationRequest, GenerationResponse


class EchoGenerator:
    name = "echo"
    model_id = "echo/0"

    def __init__(self, max_tokens: int = 128, temperature: float = 0.0):
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        ctx = "\n".join(request.context[:3])
        text = f"[ECHO] {request.prompt.messages[-1].content}\n\n[CONTEXT]\n{ctx[:800]}"
        return GenerationResponse(
            text=text, tokens_in=len(ctx.split()), tokens_out=min(self.max_tokens, 64), metadata={}
        )
