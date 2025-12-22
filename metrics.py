from skimage.metrics import structural_similarity as ssim
import torch
import lpips
import numpy as np
import cv2

class Metrics:
    def __init__(self, image1, image2, device='cpu', lpips_model=None):
        self.image1 = image1
        self.image2 = image2
        self.image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        self.image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        self.mse = self.get_mse()
        self.inv_ssim = self.get_inv_ssim()
        self.lpips = self.get_lpips(device, lpips_model)
        self.difference = self.get_difference()

    def get_mse(self):
        return np.mean((self.image1_gray.astype("float") - self.image2_gray.astype("float")) ** 2)
    
    def get_inv_ssim(self):
        ssim_val, _ = ssim(self.image1_gray, self.image2_gray, full=True, data_range=255)
        return 1 - ssim_val
    
    def lpips_preprocess(self, img, device):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = (tensor * 2) - 1
        return tensor.to(device)

    def get_lpips(self, device, lpips_model):
        tensor1 = self.lpips_preprocess(self.image1, device)
        tensor2 = self.lpips_preprocess(self.image2, device)
        with torch.no_grad():
            lpips_val = lpips_model(tensor1, tensor2).item()
        return lpips_val

    def get_difference(self, alpha=0.5, beta=0.3, gamma=0.2):
        return (alpha * self.mse) + (beta * self.inv_ssim) + (gamma * self.lpips)