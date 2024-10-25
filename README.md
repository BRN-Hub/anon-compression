# Region of Interest Loss for Anonymizing Image Compression

**[[IEEEXplore](https://ieeexplore.ieee.org/document/10711721)] [[arXiv](https://arxiv.org/abs/2406.05726#)]**

This is the public repository of the corresponding paper published in the conference proceedings IEEE CASE 2024.
The original code has been reduced to a minimal wokring example for read- and usability, that achieves the same results.
If you find it to be useful for your work, please cite it as follows:

```bibtex
@inproceedings{liebender2024region,
  author    = {Liebender, Christoph and Bezerra, Ranulfo and Ohno, Kazunori and Tadokoro, Satoshi},
  booktitle = {2024 IEEE 20th International Conference on Automation Science and Engineering (CASE)}, 
  title     = {{Region of Interest Loss for Anonymizing Learned Image Compression}}, 
  year      = {2024},
  pages     = {3569--3576},
  doi       = {10.1109/CASE59546.2024.10711721}
}
```

> [!NOTE]
> The `main` branch marks the state of the code as the paper was submitted.
> Updated dependencies are used on `devel`; Without guaranteed bit-by-bit reproducibility compared to `main`.

> [!IMPORTANT]
> Scripts are expected to be executed from the root-directory of this repository.

https://github.com/user-attachments/assets/a2bc4a8d-5b0b-439b-bf4c-b57d0498f6d3

## Quick start

0. Prerequesites: `git`, `python3`, [`gum`](https://github.com/charmbracelet/gum)

1. Clone this repo and prepare environment:
```console
$ git clone https://github.com/BRN-Hub/anon-compression
$ cd anon-compression
$ python3 -m venv .venv
$ source .venv/bin/activate
$ python3 -m pip install -r requirements.txt
```
2. Download weights: (See paper for configuration reference)
```console
$ tools/download_weights.sh
```
3. Use weights for coder / decoder:
```python
import torch
from coders import Encoder, Decoder

weight_path = "weights/128-2.pt"

encoder = Encoder(weight_path)
decoder = Decoder(weight_path)

image = torch.rand(1, 3, 512, 512) # N, C, H, W format, RGB normalized to [0,1]

bytes = encoder(image)

# ... save to file, transmit over websocket, etc.

anon_decompressed = decoder(bytes)

```

## Training
1. Prepare training data. When prompted, answer with yes:
```console
$ tools/prepare_data.sh
```
2. Adjust hyperparameters in `train.py`. (optional)
3. Start training:
```console
$ ./train.py
```
This will dump logs and intermediate checkpoints into `lightning_logs`.
Upon finishing, the last checkpoint with model parameters required for quickstart is saved seperately.

## Benchmarking
1. Prepare data. Skip this if you already ran step 1 of ##Training. When prompted, answer with no:
```console
$ tools/prepare_data.sh
```
2. Download weights / Train model.
3. Adjust paths for weights to use in benchmark files.
4. Run benchmarks:
```console
$ ./yolo_rate_precision.py  # runs both compression rate and yolo precision measurements
$ ./mtcnn_precision.py      # runs mtcnn face precision measurements
$ ./latency.py              # runs latency benchmarks, optionally in comparison with AV1 and JPEG
```

## Using the loss function
The loss function introduced by the paper can be found in [`region_loss.py`](/region_loss.py).
It can be used seperately with arbitrary bounding boxes.
It expects a parameter `annotations` of type list of list of dict, with the following structure:
```python
[
  # batch index 0
  [
    # annotation index 0
    { "xyxy": (x_top_left, y_top_left, x_bottom_right, y_bottom_right) }, # absolute values
    ...
  ],
  ...
]
```
Where each sub-list contains all annotations for the image at the given batch index in the top list.
The loss can be inverted by setting the parameter `invert = True`.
Note that for inversion to work properly, input tensors have to be normalized to [0,1].

## Example capture/client-Application

### Capture-side
```console
$ examples/capture.py -h
$ examples/capture.py weights/256-2.pt --dimension 256
```
You can also use [`tools/configure_service.sh`](/tools/configure_service.sh) to create a systemd-unit file that enables the capture program to be executed at every startup.
The script will guide you through the process of creating this unit file.
#### 🤯 Advanced systemd tips 🔥
- If you end up not using a virtual environment, you need to set the `User=` directive under `[Service]` to the user that has access to the required pip packages.
- Since the parameter `--captures` supports glob syntax, you can just specify `--captures /dev/video*` to stream every video capture available to the system, without needing to modify the systemd-unit file every time. In that case, it would suffice to power off the system, (un-)plug cameras and restart. (Note that this might not work depending on the video capture type, as some serve as loopbacks.)
### Client-side
```console
$ examples/client.py -h
$ examples/client.py weights/256-2.pt --detect --yoloweights yolov8n.pt
```

