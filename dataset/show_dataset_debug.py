# tutorial https://medium.com/voxel51/loading-open-images-v6-and-custom-datasets-with-fiftyone-18b5334851c3
import fiftyone as fo
import os
from PIL import Image, ImageDraw
import numpy as np


dataset = fo.Dataset("open_images_v7")
dataset.persistent = False

name = "Europa"
dataname = 'Ghiacciai'
dataset_dir = "data/europa/128x128/"


for image_name in os.listdir(os.path.join(dataset_dir, 'imgs')):
    if image_name.endswith('.png'):
        # load image
        image_path = os.path.join(dataset_dir, 'imgs', image_name)
        sample = fo.Sample(filepath=image_path)

        # programmatically read
        label_path = os.path.join(dataset_dir, 'labels',image_name[:-4] + '_label.npz')
        with np.load(label_path) as npz:
            lpos = npz["lpos"][:, :, :2]
            lpre = lpos[:, :, :2]

        image = Image.open(image_path)
        label_img = Image.new('1', (image.width, image.height))
        draw = ImageDraw.Draw(label_img)
        for d in lpre:
          draw.line(
                (
                    d[1][1], d[1][0],
                    d[0][1], d[0][0]
                ),
                fill=2000
          )

        loaded_mask = np.asarray(label_img)

        sample["ground-truth"] = fo.Segmentation(
            mask=loaded_mask,
        )

        # load check mask
        label_path = os.path.join(dataset_dir, 'labels', image_name + '_truth_check.jpg')
        label_img = Image.open(label_path).convert('1')
        loaded_mask = np.asarray(label_img)
        sample["labels-check"] = fo.Segmentation(
            mask=loaded_mask,
        )

        # load masks
        label_path = os.path.join(dataset_dir, 'masks', image_name)
        label_img = Image.open(label_path)
        loaded_mask = np.asarray(label_img)
        sample["mask"] = fo.Segmentation(
            mask=loaded_mask,
        )

        dataset.add_sample(sample)

if __name__ == "__main__":
    session = fo.launch_app(dataset)
    session.wait()
