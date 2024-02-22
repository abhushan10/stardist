from stardist.models import StarDist2D 
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
from skimage import io
# Load the saved model
model_path = "model"
model = StarDist2D(None, name='stardist', basedir=model_path)

# Load the test image
image_path = "3.tif"
image = io.imread(image_path)

# Normalize the image to a float type
img = image

labels, _ = model.predict_instances(normalize(img))

plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("input image")

plt.subplot(1,2,2)
plt.imshow(render_label(labels, img=img))
plt.axis("off")
plt.title("prediction + input overlay")
plt.show()
