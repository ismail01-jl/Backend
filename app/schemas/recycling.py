from pydantic import BaseModel

class RecyclingLot(BaseModel):
    Poids: float
    Volume: float
    Conductivite: float
    Opacite: float
    Rigidite: float
    Source: str