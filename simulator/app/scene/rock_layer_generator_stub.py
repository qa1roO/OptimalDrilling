import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RockLayer:
    name: str
    thickness: float
    color_hex: str
    cluster_id: int


ROCK_COLORS = {
    "TopSoil": "#c8b08e",
    "Clay": "#a88f6a",
    "Sandstone": "#d9b37a",
    "Shale": "#7f8a94",
    "Limestone": "#b6c2cf",
    "Granite": "#8e8f9a",
    "Basalt": "#3c3c44",
}


REGIONS = {
    "Europe Basin": ["TopSoil", "Clay", "Sandstone", "Limestone", "Shale", "Granite"],
    "Far East Volcanic": ["TopSoil", "Clay", "Sandstone", "Basalt", "Granite"],
    "Brazil Shield": ["TopSoil", "Sandstone", "Shale", "Granite"],
}

CLUSTER_IDS = tuple(range(7))


def generate_rock_layers():

    region = random.choice(list(REGIONS.keys()))
    stratigraphy = REGIONS[region]

    layers = []

    for rock in stratigraphy:

        thickness = random.uniform(30, 150)

        layers.append(
            RockLayer(
                name=rock,
                thickness=thickness,
                color_hex=ROCK_COLORS[rock],
                cluster_id=random.choice(CLUSTER_IDS),
            )
        )

    return region, layers
