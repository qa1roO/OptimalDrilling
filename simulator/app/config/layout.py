from dataclasses import dataclass


@dataclass(frozen=True)
class PanelLayout:
    rig_image_path: str = "assets/rig_2d.png"
    rig_x: int = 20
    rig_y: int = 20
    rig_width: int = 320
    rig_height: int = 300

    section_x: int = 360
    section_y: int = 20
    section_width: int = 380
    section_height: int = 300
