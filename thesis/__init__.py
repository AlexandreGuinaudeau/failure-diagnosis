from .base_generator import BaseGenerator
from .base_model import BaseModel
from .binary_segmentation_model import BinarySegmentationModel
from .hmm_model import HMMModel, BinaryHMM

ALL_COLORS = ["#1F4B99", "#2B5E9C", "#38709E", "#48819F", "#5B92A1", "#71A3A2", "#8AB3A2",
              "#A7C3A2", "#C7D1A1", "#EBDDA0", "#FCD993", "#F5C57D", "#EDB269", "#E49F57",
              "#DA8C46", "#CF7937", "#C4662A", "#B8541E", "#AB4015", "#9E2B0E"]

def get_all_colors(n):
    return [ALL_COLORS[int(i*len(ALL_COLORS)/n)] for i in range(n)]