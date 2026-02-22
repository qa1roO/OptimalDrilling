from dataclasses import dataclass


@dataclass(frozen=True)
class RockLayer:
    name: str
    thickness: float
    color_hex: str


def generate_rock_layers_stub() -> list[RockLayer]:
    # Stub generator. Replace with real geology function later.
    return [
        RockLayer(name="TopSoil", thickness=32, color_hex="#c8b08e"),
        RockLayer(name="Sandstone", thickness=58, color_hex="#d9b37a"),
        RockLayer(name="Limestone", thickness=64, color_hex="#b6c2cf"),
        RockLayer(name="Shale", thickness=50, color_hex="#7f8a94"),
        RockLayer(name="Granite", thickness=86, color_hex="#8e8f9a"),
    ]
