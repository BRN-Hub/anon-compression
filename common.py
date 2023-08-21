import os, math, torch, pandas

from PIL import Image
from io import BytesIO
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import resize
from model import AutoEncoder


def ensure_path_exists(path):
    path = os.path.dirname(path)
    if not os.path.isdir(path):
        os.makedirs(path)


def partition(list, n):
    part_size = math.ceil(len(list) / n)
    for i in range(n - 1):
        yield list[part_size * i : part_size * (i + 1)]
    yield list[part_size * (i + 1) : len(list) - 1]


def compress(image_path, quality, codec_ext, input_shape) -> Image:
    with Image.open(image_path) as image, BytesIO() as buffer:
        W, H = image.size
        image = resize(image, input_shape, antialias=True)
        image.save(buffer, codec_ext, quality=quality)
        bpp = buffer.getvalue().__len__() * 8 / image.size[0] / image.size[1]

        image = Image.open(buffer)
        image.load()
        return image, bpp, (H, W)


def compress_nn(image_path, nn, device, input_shape) -> torch.Tensor:
    tensor = read_image(image_path, ImageReadMode.RGB).unsqueeze(0).to(device) / 255.0
    resized = resize(tensor, input_shape, antialias=True)

    y = nn._encoder(resized)
    compressed = nn._bottleneck.compress(y)
    y_ = nn._bottleneck.decompress(compressed, y.shape[2:])
    x_ = nn._decoder(y_)

    _, _, H, W = x_.shape
    assert H == input_shape[0] and W == input_shape[1]
    bpp = compressed[0].__len__() * 8 / W / H

    return x_, bpp, tensor.shape[2:]


def load_nn(device_id, checkpoint_path):
    ae = AutoEncoder()
    ae.to(device_id)
    state_dict = {}
    for k, v in torch.load(checkpoint_path, map_location=device_id)[
        "state_dict"
    ].items():
        k = ".".join(k.split(".")[1:])
        state_dict[k] = v
    ae.load_state_dict(state_dict)
    ae.eval()
    ae._bottleneck.update()

    return ae


def get_annotations(annotations_root, images_root, image_path, input_shape, orig_shape):
    annotation_path = (
        os.path.splitext(annotations_root + image_path.replace(images_root, ""))[0]
        + ".txt"
    )
    if not os.path.exists(annotation_path):
        print("Missing annotations:", annotation_path)

    H_, W_ = orig_shape
    H, W = input_shape

    scale_x, scale_y = W / W_, H / H_

    try:
        frame = pandas.read_csv(annotation_path, delim_whitespace=True, header=None)
    except (pandas.errors.EmptyDataError, FileNotFoundError):
        frame = pandas.DataFrame()

    for _, x, y, w, h in frame.itertuples(index=False, name=None):
        x = x * scale_x
        y = y * scale_y
        w = w * scale_x
        h = h * scale_y

        x = max(0, x)
        y = max(0, y)
        if x + w > W:
            w = W - x
        if y + h > H:
            h = H - y

        yield x, y, w, h
