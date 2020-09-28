
# First import libraries.
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

path_content = "data/content_img.jpg"
# Open your file image using the path
img = Image.open(path_content)

# Since plt knows how to handle instance of the Image class, just input your loaded image to imshow method
plt.imshow(img)

print(matplotlib.get_backend())

import PyQt4
import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
plt.plot([1,2,3])
plt.show()
