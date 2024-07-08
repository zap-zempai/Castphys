from pathlib import Path
import cv2
import pandas as pd

from video_trimming.feature_detection.SIFT import search_box

### CLASS ---------------------------------------------------------------------
class BBox:
    """
    Convenient way to define Bounding Box in the following way:
     x0: left
     y0: upper
     x1: right
     y1: lower pixel
    """
    def __init__(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
    def __init__(self, list_bbox: list) -> None:
        self.x0 = list_bbox[0]
        self.y0 = list_bbox[1]
        self.x1 = list_bbox[2]
        self.y1 = list_bbox[3]

### FUNCIONS ------------------------------------------------------------------
def serch_bbox(id:int, info_path, df_path, mark_path, path_img, look_bbox:bool=False) -> BBox:
    #raise Exception(f"Error: {id} in bbox")
    print("Search BBox")
    x0,y0,x1,y1 = search_box(path_img, info_path/mark_path, look_box=look_bbox)
    # Guarda la BBox
    try:
        f = open(str(info_path / df_path), "a")
        f.write(f"\n{id},{x0},{y0},{x1},{y1}")
    except FileNotFoundError:
        print("Error: File not found")
    finally:
        f.close()
    return BBox(x0,y0,x1,y1)


def create_bbox(id:int, info_path, df_path, mark_path, path_img, look_bbox:bool=False) -> BBox:
    df = pd.read_csv(info_path / df_path)
    df = df[df.patient == id]
    if len(df) != 1:
        return serch_bbox(id, info_path, df_path, mark_path, path_img, look_bbox=look_bbox)
    
    # look images
    if look_bbox:
        bbox = BBox(df.drop(["patient"], axis=1).values[0])
        img = cv2.imread(str(path_img))
        cv2.rectangle(img,(bbox.x0,bbox.y0),(bbox.x1,bbox.y1),(0,255,0),1)
        cv2.imshow('BBox',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return bbox

    return BBox(df.drop(["patient"], axis=1).values[0])


def main():
    path_init = Path("C:/Users/Xavi/Desktop/Dataset_init")
    path_imgs = path_init / "Imgs_prova"
    path_csv = path_init / "id_bbox.csv"
    id = 85
    path_img = path_imgs / f"patient_{id}.png"
    bbox = create_bbox(path_csv, id)

    img = cv2.imread(str(path_img))
    cv2.rectangle(img,(bbox.x0,bbox.y0),(bbox.x1,bbox.y1),(0,255,0),1)
    cv2.imshow('imagen',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()