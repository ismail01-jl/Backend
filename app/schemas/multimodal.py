from pydantic import BaseModel


class MultimodalInput(BaseModel):
    texte:        str
    Poids:        float
    Volume:       float
    Conductivite: float
    Opacite:      float
    Rigidite:     float
    Source:       str
