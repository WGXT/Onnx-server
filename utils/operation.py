from io import BytesIO

import onnxruntime
import numpy as np
from PIL import Image

from utils.orientation import non_max_suppression, tag_images
