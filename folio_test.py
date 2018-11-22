import os
import numpy as np
from PIL import Image


def load_folio_in_right_format(size=(32, 32)):
    path = "./datasets/Folio Leaf Dataset/Folio/"
    classes = os.listdir(path)
    all_data = []
    for idx, class_name in enumerate(classes):
        print("({}/32) Loading {} images...".format(idx + 1, class_name))
        path_to_class_dir = os.path.join(path, class_name)
        image_filenames = os.listdir(path_to_class_dir)
        class_data = []
        for filename in image_filenames:
            image_path = os.path.join(path_to_class_dir, filename)
            image = Image.open(image_path)
            image = image.resize((32, 32), Image.ANTIALIAS)
            np_image = np.array(image)
            class_data.append(np_image)
        all_data.append(np.array(class_data))
    return np.array(all_data)


data = load_folio_in_right_format()
print(data[0][0][0][0])
data = data / 255.0
print(data[0][0][0][0])

print(data.shape)
for class_data in data:
    print(class_data.shape)
