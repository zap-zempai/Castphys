import cv2
import numpy as np
from pathlib import Path

from video_trimming.bbox import BBox

### CLASS ---------------------------------------------------------------------
class LockImage:
    def __init__(self, bbox: BBox, threshold: int, imgs_path: Path) -> None:
        self.bb = bbox
        self.th = threshold
        self.imgs_path = imgs_path
    
    def crop_compute_mean(self, frame: np.array):
        return np.mean(frame[self.bb.y0:self.bb.y1, self.bb.x0:self.bb.x1,1])
    
    def img_path_id(self, id: int) -> Path:
        return self.imgs_path / f"{id}.png"
    
    def has_light(self, id: int) -> bool:
        return self.crop_compute_mean(cv2.imread(str(self.img_path_id(id)))) > self.th
    
    def probe_zone(self, frame_start, frame_end, old_green) -> int:
        for frame in range(frame_start+1,frame_end):
            has_green = self.has_light(frame)
            if old_green^has_green:
                return frame
        return frame_end
