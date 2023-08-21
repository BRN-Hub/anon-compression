#!/usr/bin/env python3

import sys, os
sys.path.append(os.getcwd())

import asyncio
import coders
import cv2 as cv
import argparse

from websockets.client import connect

parser = argparse.ArgumentParser(prog="examples/client.py", description="anon-compression example client")
parser.add_argument("weights_path")
parser.add_argument("--host", default="localhost")
parser.add_argument("--port", default=8888, type=int)
parser.add_argument("--detect", nargs='?', const=True, default=False, help="Enable detection with YOLO, optionally with specified weights (default: yolov8x.pt)")
parser.add_argument("--conf", default=0.25, type=float, help="Set YOLO detection threshold (default: %(default)s)")
args = parser.parse_args()

capture_host = f"ws://{args.host}:{args.port}"
enable_yolo_detection = True if args.detect else False
yoloweights = 'yolov8x.pt' if args.detect == True else args.detect


def show_stream(idx, frame):
    winname = f"Stream #{idx}"
    cv.namedWindow(winname, cv.WINDOW_KEEPRATIO)
    cv.imshow(winname, frame)


decoder = coders.Decoder(args.weights_path)
if enable_yolo_detection:
    from ultralytics import YOLO
    yolo = YOLO(yoloweights)
    yolo.to(decoder._device)

async def decode():
    async with connect(capture_host, ping_timeout=None) as socket:
        while True:
            await socket.send('')
            bytes = await socket.recv()
            tensor = decoder(bytes)

            reconstructions = [
                yolo(single.unsqueeze(0), classes=0, verbose=False, conf=args.conf)[0].plot()[:, :, [2, 1, 0]]
                if enable_yolo_detection
                else
                (single * 255).byte().permute(1, 2, 0)[:, :, [2, 1, 0]].cpu().numpy()
                for single in tensor
            ]
 
            for idx, rec in enumerate(reconstructions):
                show_stream(idx, rec)
            cv.pollKey()

if __name__ == "__main__":
    asyncio.run(decode())
