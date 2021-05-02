import numpy as np
from PIL import Image
#from utils import arr_idx_to_genotype_idx

def load_image(path):
    image = Image.open(path)

    arr = np.asarray(image, dtype='int64')
    
    return arr

def check_picture_border(row, col, num_rows, num_cols):
    if row == 0 or col == 0:
        return True
    
    elif row == num_rows-1 or col == num_cols-1:
        return True
    
    else:
        return False

def save_black_white(path, arr, segments, edge_fn):
    arr = arr.copy()
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            pixel = (r, c)
            
            if (check_picture_border(r, c, arr.shape[0], arr.shape[1]) 
                or edge_fn(segments, arr.shape[0], arr.shape[1], pixel)):
                arr[r,c,0] = 0
                arr[r,c,1] = 0
                arr[r,c,2] = 0
        
            else:
                arr[r,c,0] = 255
                arr[r,c,1] = 255
                arr[r,c,2] = 255
            
    arr = arr.astype(np.int8)
    im = Image.fromarray(arr, "RGB")
    im.save(path+"black_white.jpg")

def save_segmented(path, arr, segments, edge_fn):
    arr = arr.copy()
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            pixel = (r, c)
            
            if (check_picture_border(r, c, arr.shape[0], arr.shape[1]) 
                or edge_fn(segments, arr.shape[0], arr.shape[1], pixel)):
                arr[r,c,0] = 0
                arr[r,c,1] = 255
                arr[r,c,2] = 0
            
    arr = arr.astype(np.int8)
    im = Image.fromarray(arr, "RGB")
    im.save(path+"green.jpg")



if __name__ == "__main__":
    a = load_image("training_images/118035/Test image.jpg")
    print(np.max(a))