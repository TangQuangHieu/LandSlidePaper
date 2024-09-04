import numpy as np
import cv2

def addNormalize(img):
    """
    add RGB channels to image using Min-Max scaling
    """
    if not isinstance(img, np.ndarray): # convert tf.tensor to np.array
        img = img.numpy()

    # img = multispectral_img[:,:,0:3] # img
    # img = img.astype(np.float32)
    h,w,c = img.shape 
    if c==3:
      img[:,:,2]   = (img[:,:,2]-np.min(img[:,:,2])) / (np.max(img[:,:,2])-np.min(img[:,:,2])+1e-6)
      img[:,:,1] = (img[:,:,1]-np.min(img[:,:,1])) / (np.max(img[:,:,1])-np.min(img[:,:,1])+1e-6)
      img[:,:,0]  = (img[:,:,0]-np.min(img[:,:,0])) / (np.max(img[:,:,0])-np.min(img[:,:,0])+1e-6)
    elif c==1:
       img = (img - np.min(img))/(np.max(img)-np.min(img)+1e-6)
    return img

def addGray(img):
  """
    gray = (B+G+R)/3 ( band 3)
  """
  if not isinstance(img, np.ndarray): # convert tf.tensor to np.array
    img = img.numpy()
  # img = img.astype(np.float32)
  b    = img[:,:,0] # B
  g    = img[:,:,1] # G
  r    = img[:,:,2] # R
  gray = (b + g + r) / 3
  gray = np.expand_dims(gray, axis=2)

  img = np.concatenate((img, gray), axis=-1)

  return img

def addEdge(img):
  """
    Canny Edge Detection, for gray image of img ( band 3)
  """
  if not isinstance(img, np.ndarray): # convert tf.tensor to np.array
    img = img.numpy()

  gray  = img[:,:,3] #
  gray  = (gray-np.min(gray)) / (np.max(gray)-np.min(gray)+1e-6)
  gray *= 255
  gray  = gray.astype(np.uint8)

  edge  = cv2.Canny(gray,150,227)
  edge  = edge.astype(np.float32)
  edge /= 255.
  edge = np.expand_dims(edge, axis=2)

  img = np.concatenate((img, edge), axis=-1)

  return img

def addBlur(img):
  """
    Gaussian and Median Blurring (band 3)
  """
  if not isinstance(img, np.ndarray): # convert tf.tensor to np.array
    img = img.numpy()

  gray  = img[:,:,3] #
  gray  = (gray-np.min(gray)) / (np.max(gray)-np.min(gray))
  gray *= 255.
  gray  = gray.astype(np.uint8)

  blur  = cv2.blur(gray,(10,10))
  blur  = blur.astype(np.float32)
  blur /= 255.
  blur  = np.expand_dims(blur, axis=2)
  img = np.concatenate((img, blur), axis=-1)

  blur  = cv2.medianBlur(gray,15)
  blur  = blur.astype(np.float32)
  blur /= 255.
  blur  = np.expand_dims(blur, axis=2)
  img = np.concatenate((img, blur), axis=-1)

  return img

def addGradient(img):
  """
    Gradient along x-axis and y-axis (band 3)
  """
  if not isinstance(img, np.ndarray): # convert tf.tensor to np.array
    img = img.numpy()

  norm_factor = 27.32001
  gray  = img[:,:,3] #
  gray  = (gray-np.min(gray)) / (np.max(gray)-np.min(gray))

  sobel_x  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
  sobel_x  = sobel_x.astype(np.float32)
  sobel_x /= norm_factor
  sobel_x  = np.expand_dims(sobel_x, axis=2)
  img = np.concatenate((img, sobel_x), axis=-1)

  sobel_y  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
  sobel_y  = sobel_y.astype(np.float32)
  sobel_y /= norm_factor
  sobel_y  = np.expand_dims(sobel_y, axis=2)
  img = np.concatenate((img, sobel_y), axis=-1)

  return img

def draw_image(image:np.ndarray,path:str):
  """
  draw_image(image:np.ndarray,path:str): Draw image with given path
  ### Arguments:
    image(np.ndarray): image to be write (color or gray)
    path(str):Name of image
  """
  if image is None:return 
  cv2.imwrite(path,image)
  cv2.waitKey(10)

#  import numpy as np
# import matplotlib.pyplot as plt
# from skimage import color
# from skimage import io
from skimage.feature import hog
# from skimage import exposure
def hog_feature(image:np.ndarray):
  # Load the image
  # image = io.imread('path_to_your_image.jpg')

  # Convert the image to grayscale
  # gray_image = img.mean(axis=2)
  # Convert the image from BGR (OpenCV default) to RGB
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Convert the image from RGB to HSV
  image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
  # gray = (image_hsv[...,0])
  # Compute HOG features
  hog_features, hog_image = hog(image_hsv[...,2], orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), block_norm='L2-Hys',
                                visualize=True)
  return hog_features,hog_image

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

class AdjustBrightness:
    def __init__(self, brightness_factor):
        self.brightness_factor = brightness_factor

    def __call__(self, img):
        return F.adjust_brightness(img, self.brightness_factor)

class AdjustSaturation:
    def __init__(self, saturation_factor):
        self.saturation_factor = saturation_factor

    def __call__(self, img):
        img = np.array(img) / 255.0
        hsl = self.rgb_to_hsl(img)
        hsl[..., 1] *= self.saturation_factor
        hsl[..., 1] = np.clip(hsl[..., 1], 0, 1)
        img = self.hsl_to_rgb(hsl)
        img = (img * 255).astype(np.uint8)
        return Image.fromarray(img)

    def rgb_to_hsl(self, rgb):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb, axis=-1)
        minc = np.min(rgb, axis=-1)
        l = (maxc + minc) / 2
        s = np.zeros_like(l)
        delta = maxc - minc
        s[l < 0.5] = delta[l < 0.5] / (maxc[l < 0.5] + minc[l < 0.5])
        s[l >= 0.5] = delta[l >= 0.5] / (2.0 - maxc[l >= 0.5] - minc[l >= 0.5])
        s[maxc == minc] = 0
        h = np.zeros_like(r)
        mask = maxc == r
        h[mask] = (g[mask] - b[mask]) / delta[mask]
        mask = maxc == g
        h[mask] = 2.0 + (b[mask] - r[mask]) / delta[mask]
        mask = maxc == b
        h[mask] = 4.0 + (r[mask] - g[mask]) / delta[mask]
        h[minc == maxc] = 0
        h = (h / 6.0) % 1.0
        return np.stack([h, s, l], axis=-1)

    def hsl_to_rgb(self, hsl):
        h, s, l = hsl[..., 0], hsl[..., 1], hsl[..., 2]
        c = (1 - np.abs(2 * l - 1)) * s
        x = c * (1 - np.abs((h * 6) % 2 - 1))
        m = l - c / 2
        r, g, b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)
        mask = (0 <= h) & (h < 1/6)
        r[mask], g[mask], b[mask] = c[mask], x[mask], 0
        mask = (1/6 <= h) & (h < 1/3)
        r[mask], g[mask], b[mask] = x[mask], c[mask], 0
        mask = (1/3 <= h) & (h < 1/2)
        r[mask], g[mask], b[mask] = 0, c[mask], x[mask]
        mask = (1/2 <= h) & (h < 2/3)
        r[mask], g[mask], b[mask] = 0, x[mask], c[mask]
        mask = (2/3 <= h) & (h < 5/6)
        r[mask], g[mask], b[mask] = x[mask], 0, c[mask]
        mask = (5/6 <= h) & (h < 1)
        r[mask], g[mask], b[mask] = c[mask], 0, x[mask]
        r, g, b = r + m, g + m, b + m
        return np.stack([r, g, b], axis=-1)

class AdjustSharpness:
    def __init__(self, sharpness_factor):
        self.sharpness_factor = sharpness_factor

    def __call__(self, img):
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(self.sharpness_factor)

class AddNoise:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = np.array(img) / 255.0
        noise = np.random.normal(self.mean, self.std, img.shape)
        img = img + noise
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        return Image.fromarray(img)