from dataclasses import dataclass

@dataclass
class JointMap:
    use_hands: bool = True
    use_face: bool = True
    use_face_contour: bool = False
    openpose_format: str = 'coco25'

@dataclass
class JointMapper:
    use: bool = False
    type: str = 'smpl_to_openpose'

    # list joint maps here
    smpl_to_openpose: JointMap = JointMap()