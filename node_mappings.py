from .inpaint_nodes import MaskedResizeImage, PasteMaskedImage, FilterAndBlurMask

NODE_CLASS_MAPPINGS = {
    "MaskedResizeImage": MaskedResizeImage,
    "PasteMaskedImage": PasteMaskedImage,
    "FilterAndBlurMask": FilterAndBlurMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskedResizeImage": "Masked Resize Image",
    "PasteMaskedImage": "Paste Masked Image",
    "FilterAndBlurMask": "Filter And Blur Mask",
}

