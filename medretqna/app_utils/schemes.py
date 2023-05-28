from pydantic import BaseModel


class UserQuery(BaseModel):
    text: str
    topk: int = None
    threshold: float = None
    return_ner: bool = None
