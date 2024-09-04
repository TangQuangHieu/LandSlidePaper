import numpy as np
import cv2
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image, ImageEnhance
import random
import torch 

def draw_segment_on_image(image:np.ndarray,label:np.ndarray)->np.ndarray:
    """
    Draw label on image
    ### Arguments:
        image: image to be written 
        label: sementation image 
    ### Returns:
        image that is written by label ( landslide will be black)
    """
    label = np.expand_dims(np.clip(label,0,1),axis=-1)
    return image*(1-label)


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

class AdjustSharpness:
    def __init__(self, sharpness_factor):
        self.sharpness_factor = sharpness_factor

    def __call__(self, img):
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(self.sharpness_factor)
    
class Numpy2Tensor:
    def __init__(self):
        pass

    def __call__(self, img):
        # Convert image (WxHxC) to (CxWxH)
        # Change to tensor type 
        if len(img.shape)<3:
            # Gray image has shape as WxH, add one more channel for this image  
            img = img[...,np.newaxis]
        img = np.transpose(img,(2,0,1))
        tensor = torch.tensor(img,dtype=torch.float32)
        return tensor




def remove_small_region_by_contour(label:np.ndarray,threshold=5):
    """
    Capture all contours in label and set small contor regions to zero to reduce 
    false positive pixesl in the final prediction 
    ### Arugments: 
        label(np.ndarray): WxH, label map need to be reduced number of false positive 
        threshold(int): smallest perimeter that one contour need to achive to be retained as true positive
    ### Regturns:
        label with small areas deleted  
    """
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by perimeter
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= threshold]

    # Create an empty image to draw the contours
    contour_image = np.zeros_like(label)

    # Draw the filtered contours (the first parameter is the destination image)
    cv2.drawContours(contour_image, filtered_contours, -1, (1), thickness=cv2.FILLED)  # 2 is the thickness of the contour lines 
    contour_image = np.clip(contour_image,a_min=0,a_max=1)   
    return contour_image 
