import torch

class TemporalNetPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "execute"

    CATEGORY = "Temporal ControlNet/test"

    def execute(self, images):
        processed_tensors = []
        for i in range(images.shape[0]):
            if i == 0:
                concat_item = torch.cat((images[i], images[i]), dim=2)
            else:
                concat_item = torch.cat((images[i], images[i-1]), dim=2)
            processed_tensors.append(concat_item)
        concat_images = torch.stack(processed_tensors, dim=0)
        return (concat_images,)