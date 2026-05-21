from pydantic import BaseModel


class TextInput(BaseModel):
    rapport_collecte: str