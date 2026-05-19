from pydantic import BaseModel


class TextInput(BaseModel):
    texte: str