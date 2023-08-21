import torch

from torch.nn.functional import mse_loss


def region_loss(
    input: torch.Tensor, output: torch.Tensor, annotations: list[list[dict]], invert: bool
) -> torch.Tensor:
    N, _, H, W = input.shape
    batch_loss = 0

    for tensor_idx, item in enumerate(annotations):
        if len(item) == 0:
            continue

        tensor_loss = 0

        for detection in item:
            x_tl, y_tl, x_br, y_br = detection["xyxy"]
            region_output, region_input = (
                output[tensor_idx, :, y_tl:y_br, x_tl:x_br].unsqueeze(0),
                input[tensor_idx, :, y_tl:y_br, x_tl:x_br].unsqueeze(0),
            )
            instance_loss = mse_loss(region_output, region_input)

            if invert:
                instance_loss = 1 - instance_loss

            tensor_loss = tensor_loss + instance_loss

        batch_loss = batch_loss + tensor_loss / len(item)

    return batch_loss / N
