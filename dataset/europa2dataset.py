import os
from PIL import Image, ImageDraw
from patchify import patchify
import numpy as np
from tqdm import tqdm
import json
from shapely.geometry import LineString, Point

# to read large images
Image.MAX_IMAGE_PIXELS = 192697896
PATCH_DIM = 128
patch_shape = (PATCH_DIM, PATCH_DIM)
STRIDE_DIM = 64
LABELS_PATH = '/home/studente/alessandro/letr/data/lineaments/'
DATASET_PATH = 'data/europa/'
# DATASET_PATH = '/home/super/datasets-nas/line_detection/europa/'

DESTINATION_DIR = os.path.join(DATASET_PATH, f'{PATCH_DIM}x{PATCH_DIM}/labels/')
DESTINATION_IMG = os.path.join(DATASET_PATH,f'{PATCH_DIM}x{PATCH_DIM}/imgs/')
DESTINATION_MASK = os.path.join(DATASET_PATH,f'{PATCH_DIM}x{PATCH_DIM}/masks/')

for folder in [DESTINATION_DIR, DESTINATION_IMG, DESTINATION_MASK]:
  if not os.path.exists(folder):
    os.makedirs(folder)


# original image
image = Image.open('/home/studente/alessandro/letr/figures/max_mosaic_glocal_Galileo_eq.png')
# image.save('figures/max_mosaic_glocal_Galileo_eq.png')
# exit()
shape = (image.width, image.height)

# labels
mask_image = Image.new('1', shape)
draw = ImageDraw.Draw(mask_image)
draw.rectangle([(0, 0), (shape)], fill="#000000")
files_list = os.listdir(LABELS_PATH)
lineaments = []

for file in files_list:
    lineament = open(os.path.join(LABELS_PATH, file))
    lineament = [[float(coords) for coords in lin.split(',')] for lin in lineament.read().split()]
    for i in range(len(lineament) - 1):
        l = (lineament[i][0], lineament[i][1], lineament[i + 1][0], lineament[i + 1][1])
        lineaments.append([(int(l[0]), int(l[1])), (int(l[2]), int(l[3]))])
        draw.line(l, fill=1, width=1)

mask_image.convert('L').save('figures/total_europa_mask.png')
img_patch = patchify(np.asarray(image.convert('L')), patch_shape, step=STRIDE_DIM)
mask_patch = patchify(np.asarray(mask_image.convert('L')), patch_shape, step=STRIDE_DIM)


def check(line, square):
  minxl, minyl = min((line[0][0], line[1][0])), min((line[0][1], line[1][1]))
  maxxl, maxyl = max((line[0][0], line[1][0])), max((line[0][1], line[1][1]))

  minxs, minys = square[0][0], square[0][1]
  maxxs, maxys = square[1][0], square[1][1]

  if (minxl < minxs and maxxl < minxs) or (minxl > maxxs and maxxl > maxxs) or (minyl < minys and maxyl < minys) or ( minyl > minys and maxyl > maxys):
    return False

  if (minxs <= minxl <= maxxs and minys <= minyl <= maxys) or (minxs <= maxxl <= maxxs and minys <= maxyl <= maxys):
    return True
  
  s_lines = [
      [square[0], (minxs, maxys)],
      [square[0], (maxxs, minys)],
      [(minxs, maxys), square[1]],
      [(maxxs, minys), square[1]]
  ]
  # for s_line in s_lines:
  #   inter_p = get_intersect(s_line, line)
  line1 = LineString(line)
  for s_line in s_lines:
    line2 = LineString(s_line)
    inter_p = line1.intersection(line2)
    if str(inter_p) != 'LINESTRING EMPTY':
      inter_p = (inter_p.x, inter_p.y)
      if (minxs <= inter_p[0] <= maxxs and minys <= inter_p[1] <= maxys):
        return True

  return False


def transposed_points(line, square):
  minxs, minys = square[0][0], square[0][1]
  maxxs, maxys = square[1][0], square[1][1]

  points = []

  if (minxs <= line[0][0] <= maxxs and minys <= line[0][1] <= maxys):
    points.append((line[0][0], line[0][1]))
  if (minxs <= line[1][0] <= maxxs and minys <= line[1][1] <= maxys):
    points.append((line[1][0], line[1][1]))
  
  s_lines = [
      [square[0], (minxs, maxys)],
      [square[0], (maxxs, minys)],
      [(minxs, maxys), square[1]],
      [(maxxs, minys), square[1]]
  ]

  line1 = LineString(line)
  for s_line in s_lines:
    line2 = LineString(s_line)
    inter_p = line1.intersection(line2)
    # print(inter_p)
    if str(inter_p) == 'LINESTRING EMPTY':
      continue
    elif type(inter_p) == Point:
        inter_p = (inter_p.x, inter_p.y)
        points.append(inter_p)
    else:
        # print(type(inter_p))
        # print(inter_p.coords[0])
        inter_p = (inter_p.coords[0][0], inter_p.coords[0][1])
        points.append(inter_p)
    if len(points) == 2:
      break

  return points


def debug_plot(line, square, image_size, scale, points=None):
  mask_image = Image.new('RGB', [x * scale for x in image_size])
  draw = ImageDraw.Draw(mask_image)
  draw.rectangle([(s[0]*scale, s[1]*scale) for s in square], fill=(255, 0, 0))
  draw.line([(s[0]*scale, s[1]*scale) for s in line], fill=(0, 255, 0))
  if points:
    for p in points:
      draw.rectangle([(p[0] * scale - 2, p[1] * scale - 2), (p[0] * scale + 2, p[1] * scale + 2)], fill=(0, 0, 255))
  return mask_image

for x, (imgs, masks) in tqdm(enumerate(zip(img_patch, mask_patch)), total=len(img_patch)):
    for y, (img, mask) in enumerate(zip(imgs, masks)):
        if mask.sum() > 255*1000: # scarta quelle che non hanno almento 1000 pixel 
            patch_number = x+y*img_patch.shape[0]
            destination_path = os.path.join(DESTINATION_DIR, f'{patch_number:08d}' + '.png')
            destination_path_json = os.path.join(DESTINATION_DIR, f'{patch_number:08d}' + '.json')
            destination_path_npz = os.path.join(DESTINATION_DIR, f'{patch_number:08d}' + '_label.npz')
            destination_img_path = os.path.join(DESTINATION_IMG, f'{patch_number:08d}' + '.png')
            destination_mask_path = os.path.join(DESTINATION_MASK, f'{patch_number:08d}' + '.png')

            a = (y, x)
            a = (a[0]*STRIDE_DIM, a[1]*STRIDE_DIM)
            b = (a[0] + PATCH_DIM, a[1] + PATCH_DIM)
            # check
            # image.crop((a[0], a[1], b[0], b[1])).save(destination_path + '_truth_check.jpg')
            image.crop((a[0], a[1], b[0], b[1])).convert('RGB').save(destination_path)
            
            label_img = Image.new('1', patch_shape)
            draw = ImageDraw.Draw(label_img)

            lines = []
            square = [a, b]
            for l in lineaments:
              if l[0] != l[1]:
                if check(l, square):
                    points = transposed_points(l, square)
                    # print(points)
                    if len(points) == 2:
                      line = [
                          (int(points[0][0]) - a[0], int(points[0][1]) - a[1]),
                          (int(points[1][0]) - a[0], int(points[1][1]) - a[1])
                      ]
                      # print(l, square, shape)
                      draw.line(line, fill=1, width=1)
                      lines.append(line)
                      #debug_plot(l,square, shape, 1, points).save('test.png')
                      #exit()

            label_img.save(destination_path + '_truth_check.jpg')
            with open(destination_path_json, 'w') as f:
                json.dump(lines, f)
            lines_with_class = [ [ [p[1], p[0], 0] for p in l] for l in lines]
            np.savez_compressed(
              destination_path_npz, 
              aspect_ratio=1.0,
              lpos=lines_with_class,
            )

            # print(img.shape)
            Image.fromarray(img).save(destination_img_path)
            Image.fromarray(mask).save(destination_mask_path)
            # exit()
