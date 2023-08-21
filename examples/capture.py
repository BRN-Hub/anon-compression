#!/usr/bin/env python3

import sys, os
sys.path.append(os.getcwd())

import argparse
import cv2 as cv
import coders
import torch
import socket
import asyncio

from torchvision.transforms.functional import resize
from websockets.server import serve
from websockets.exceptions import ConnectionClosedOK
from glob import glob

parser = argparse.ArgumentParser(prog="examples/capture.py", description="anon-compression example capture server")
parser.add_argument("weights_path", help="path to weights")
parser.add_argument("--dimension", default=512, type=int, help="rescale image for larger dimension to have this size (default: %(default)s)")
parser.add_argument("--host", default="localhost")
parser.add_argument("--port", default=8888, type=int)
parser.add_argument("--captures", default=[0], nargs="*", help="capture indices or paths (default: %(default)s)")
args = parser.parse_args()

capture_paths = []
for cap_idx in args.captures:
    if isinstance(cap_idx, int) or cap_idx.isdigit():    # 0
        capture_paths.append(int(cap_idx))
    elif '*' in cap_idx: # /dev/video*
        capture_paths.extend(glob(cap_idx))
    else:                # /dev/video0
        capture_paths.append(cap_idx)

captures = []
max_frame_dimension = -1
for idx in capture_paths:
    cap = cv.VideoCapture(idx)
    if not cap.isOpened():
        print(f"Opening camera {idx} failed")
        exit(1)
    ret, frame = cap.read()
    if not ret:
        print(f"Reading camera {idx} failed")
        exit(1)
    captures.append(cap)
    max_frame_dimension = max(max_frame_dimension, *frame.shape[:2])

scale_factor = args.dimension / max_frame_dimension
input_shape = (len(captures), 3, int(frame.shape[0] * scale_factor), int(frame.shape[1] * scale_factor))
print("Input tensor shape:", input_shape)

coder = coders.Encoder(args.weights_path)

async def forward(socket, _):
    peer_addr = socket.remote_address
    print(peer_addr, "connected")
    while True:
        frames = []
        for i, cap in enumerate(captures):
            ret, frame = cap.read()
            if not ret:
                print(f"Reading camera at index {i} failed")
                continue
            frames.append(
                resize(
                    torch.from_numpy(frame[:, :, [2, 1, 0]])
                         .permute(2, 0, 1)
                         .unsqueeze(0)
                         / 255.,
                    input_shape[2:],
                    antialias=True,
                ).squeeze()
            )
        

        bytes = coder(torch.stack(frames))
        
        try:
            await socket.recv()
            await socket.send(bytes)
        except ConnectionClosedOK:
            print(peer_addr, "closed")
            return
            

async def main():
    print("Listening on:", f"{args.host}:{args.port}")
    async with serve(forward, args.host, args.port, ping_timeout=None):
        await asyncio.Future()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())


