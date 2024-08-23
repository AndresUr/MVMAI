from pydantic import BaseModel

class PromptRequest(BaseModel):
    promptText: str
    