import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RockLayer:
    name: str
    thickness: float
    color_hex: str
    energy_type: str


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

ROCK_ENERGY_TYPES = {
    "TopSoil": "soft_low_energy",
    "Clay": "soft_low_energy",
    "Sandstone": "medium_low_energy",
    "Shale": "medium_high_energy",
    "Limestone": "medium_high_energy",
    "Granite": "hard_high_energy",
    "Basalt": "hard_high_energy",
}

ENERGY_TYPE_TO_ID = {
    "soft_low_energy": 0,
    "medium_low_energy": 1,
    "medium_high_energy": 2,
    "hard_high_energy": 3,
}


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
                energy_type=ROCK_ENERGY_TYPES[rock],
            )
        )

    return region, layers
