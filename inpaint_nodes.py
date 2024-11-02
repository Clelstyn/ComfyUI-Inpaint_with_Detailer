import numpy as np
import torch
from PIL import Image, ImageFilter
import cv2

class MaskedResizeImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
                "max_resolution": (["1024x1024", "1280x1280", "1536x1536"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("Image", "Mask")
    FUNCTION = "crop_and_resize"
    CATEGORY = "Image Processing"

    def crop_and_resize(self, image, mask, max_resolution):
        if max_resolution == "1024x1024":
            max_pixels = 1024 * 1024
        elif max_resolution == "1280x1280":
            max_pixels = 1280 * 1280
        elif max_resolution == "1536x1536":
            max_pixels = 1536 * 1536
        else:
            raise ValueError("Unsupported max resolution value.")

        img_np = image.cpu().numpy()[0]
        mask_np = mask.cpu().numpy()[0]

        non_zero_coords = np.argwhere(mask_np > 0)
        top_left = non_zero_coords.min(axis=0)
        bottom_right = non_zero_coords.max(axis=0) + 1

        cropped_img = img_np[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

        cropped_img_pil = Image.fromarray(np.clip(cropped_img * 255, 0, 255).astype(np.uint8))
        aspect_ratio = cropped_img_pil.width / cropped_img_pil.height

        new_width = int((max_pixels * aspect_ratio) ** 0.5)
        new_height = int(new_width / aspect_ratio)

        new_width = (new_width // 64) * 64
        new_height = (new_height // 64) * 64

        resized_img_pil = cropped_img_pil.resize((new_width, new_height), Image.LANCZOS)

        resized_img_np = np.array(resized_img_pil).astype(np.float32) / 255.0
        resized_img_tensor = torch.from_numpy(resized_img_np).unsqueeze(0)

        mask_tensor = mask.clone()  # Ensure mask is returned in its original form

        return (resized_img_tensor, mask_tensor)


class PasteMaskedImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "mask": ("IMAGE",),
                "modified_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paste_modified"
    CATEGORY = "Image Processing"

    def paste_modified(self, original_image, mask, modified_image):
        original_np = original_image.cpu().numpy()[0]
        mask_np = mask.cpu().numpy()[0]
        modified_np = modified_image.cpu().numpy()[0]

        non_zero_coords = np.argwhere(mask_np > 0)
        top_left = non_zero_coords.min(axis=0)
        bottom_right = non_zero_coords.max(axis=0) + 1

        modified_pil = Image.fromarray(np.clip(modified_np * 255, 0, 255).astype(np.uint8))

        if modified_pil.mode == 'RGBA':
            modified_pil = modified_pil.convert('RGB')

        new_width = bottom_right[1] - top_left[1]
        new_height = bottom_right[0] - top_left[0]
        resized_modified_pil = modified_pil.resize((new_width, new_height), Image.LANCZOS)

        resized_modified_np = np.array(resized_modified_pil).astype(np.float32) / 255.0

        if mask_np.shape[-1] == 4:
            mask_np = mask_np[:, :, :3]

        updated_image_np = original_np.copy()
        updated_image_np[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = (
            resized_modified_np * mask_np[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
            + original_np[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] * (1 - mask_np[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]])
        )

        updated_image_tensor = torch.from_numpy(updated_image_np).unsqueeze(0)

        return (updated_image_tensor,)
        
class FilterAndBlurMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "blur_radius": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 50.0}),
                "blur_expansion_radius": ("INT", {"default": 1, "min": 0, "max": 20}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_mask_with_expanded_blur"
    CATEGORY = "Image Processing"

    def apply_mask_with_expanded_blur(self, mask, blur_radius, blur_expansion_radius):
        min_mask_area = 10

        mask_np = mask.cpu().numpy()[0]
        mask_np = np.clip(mask_np * 255, 0, 255).astype(np.uint8)

        _, mask_binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blur_mask = np.zeros_like(mask_binary)
        keep_mask = np.zeros_like(mask_binary)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area >= min_mask_area:
                cv2.drawContours(blur_mask, [contour], -1, 255, thickness=cv2.FILLED)
                if blur_expansion_radius > 0:
                    kernel_size = blur_expansion_radius * 2 + 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    blur_mask = cv2.dilate(blur_mask, kernel, iterations=1)
            else:
                cv2.drawContours(keep_mask, [contour], -1, 255, thickness=cv2.FILLED)

        blur_mask_pil = Image.fromarray(blur_mask)
        blur_mask_blurred = blur_mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        blur_mask_np = np.array(blur_mask_blurred).astype(np.float32) / 255.0
        keep_mask_np = keep_mask.astype(np.float32) / 255.0
        final_mask_np = blur_mask_np + keep_mask_np

        final_mask_tensor = torch.from_numpy(final_mask_np).unsqueeze(0)

        tensor = final_mask_tensor
        tensor = tensor.unsqueeze(-1)
        tensor_rgb = torch.cat([tensor] * 3, dim=-1)

        return (tensor_rgb,)
