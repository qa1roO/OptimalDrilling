from dataclasses import dataclass


@dataclass(frozen=True)
class PerformancePoint:
    pressure: float
    rpm: float
    rop: float


def get_model_feed_placeholder() -> list[PerformancePoint]:
    return [
        PerformancePoint(pressure=20.0, rpm=80.0, rop=2.5),
        PerformancePoint(pressure=32.0, rpm=100.0, rop=3.2),
        PerformancePoint(pressure=45.0, rpm=120.0, rop=4.0),
        PerformancePoint(pressure=58.0, rpm=140.0, rop=4.8),
        PerformancePoint(pressure=52.0, rpm=130.0, rop=4.4),
    ]
