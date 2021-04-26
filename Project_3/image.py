import numpy as np
from PIL import Image

def load_image(path):
    image = Image.open(path)

    arr = np.asarray(image, dtype='int64')
    
    """
    print(arr.shape)

    reloaded = Image.fromarray(arr)
    print(reloaded.format)
    print(reloaded.size)
    print(reloaded.mode)
    reloaded.show()
    """
    return arr

if __name__ == "__main__":
    a = load_image("training_images/86016/Test image.jpg")
    print(a[0,0])