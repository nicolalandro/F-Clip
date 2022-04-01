# tutorial https://medium.com/voxel51/loading-open-images-v6-and-custom-datasets-with-fiftyone-18b5334851c3
import fiftyone as fo
import os
from PIL import Image, ImageDraw
import numpy as np


dataset = fo.Dataset("open_images_v7")
dataset.persistent = False

name = "Wireframe"
dataname = 'shanghaiTech'
dataset_dir = "/home/supreme/datasets-nas/line_detection/wireframe1_datarota_3w/valid"


for image_name in os.listdir(os.path.join(dataset_dir)):
    if image_name.endswith('.png'):
        image_path = os.path.join(dataset_dir, image_name)
        sample = fo.Sample(filepath=image_path)

        label_path = os.path.join(dataset_dir, image_name[:-4] + '_label.npz')

        detections = []
        with np.load(label_path) as npz:
            lpos = npz["lpos"][:, :, :2]
            lpre = lpos[:, :, :2]

        create_img = Image.open(image_path)
        width, height = create_img.size
        width_factor = width/128
        height_factor = height/128
        matrix = (
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0
        )
        create_img = create_img.convert("L", matrix)
        draw = ImageDraw.Draw(create_img)
        for d in lpre:
            draw.line(
                (
                    d[1][1] * width_factor, d[1][0] * height_factor,
                    d[0][1] * width_factor, d[0][0] * height_factor
                ),
                fill=2000
            )

        loaded_mask = np.asarray(create_img)

        sample["segmentations"] = fo.Segmentation(
            mask=loaded_mask,
        )

        # sample["prediction"] = fo.Segmentation(
        #     mask=loaded_mask,
        #     confidence=0.7
        # )
        # line_path = os.path.join(dataset_dir, image_name[:-4] + '_line.npz')

        dataset.add_sample(sample)

if __name__ == "__main__":
    session = fo.launch_app(dataset)
    session.wait()
