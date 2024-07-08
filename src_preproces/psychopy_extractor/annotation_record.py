
from enum import Enum
from pathlib import Path
from typing import List

import pandas as pd

class Quadrant(Enum):
    Q1 = 1
    Q2 = 2
    Q3 = 3
    Q4 = 4
    Q5 = 5
    Q6 = 6
    Q7 = 7
    Q8 = 8
    Q9 = 9
    UNKNOWN = 10

class AnnotationRecord:
    def __init__(self,
                 video_name: str,
                 skip_video: bool,
                 arousal: int,
                 valence: int,
                 expected_quadrant: Quadrant) -> None:
        self.video_name = video_name
        self.skip_video = skip_video
        self.arousal = arousal
        self.valence = valence
        self.expected_quadrant = expected_quadrant

class AnnotationRecordSaver:
    @staticmethod
    def save(annotations: List[AnnotationRecord],
             output_filename: Path):
        data = {}
        for i, annotation in enumerate(annotations):
            data[i] = {
                "video_name": annotation.video_name,
                "skip_video": annotation.skip_video,
                "valence": annotation.valence,
                "arousal": annotation.arousal,
                "expected_quadrant": annotation.expected_quadrant.name
            }


        df = pd.DataFrame.from_dict(data, orient="index")

        df.to_csv(output_filename, index=False)
