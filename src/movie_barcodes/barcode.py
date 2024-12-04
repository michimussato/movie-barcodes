# USAGE: python barcode.py -video VIDEO [-u]

import os
import sys
import time
import argparse
import logging
import pathlib
import cv2
import numpy as np

__author__ = "Michael Mussato"
__copyright__ = "Michael Mussato"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


# ---- Python API ----


def generate_barcode(args):
    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # sample at most 8*OUT_WIDTH frames
    nth_frame = max(1, int(total_frames / args.width / 8))
    print("Sampling every {} frame(s); {} total frames"
        .format(nth_frame, total_frames))

    counter, avg_cols = 0, []
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if counter % nth_frame == 0:
            if not args.uniform:
                avg_cols.append(
                    cv2.resize(
                        frame,
                        (1, args.sample_height)
                    )
                )
            else:
                avg_cols.append(np.array([[np.mean(frame, axis=(0, 1))]]))
            print(
                "{}%".format(np.round(counter/total_frames * 100, 2)),
                end="\r", flush=True
            )
        counter += 1

    cap.release()
    concatenated = np.concatenate(avg_cols, axis=1)
    print("Resizing {} frames to {}".format(concatenated.shape[1], args.width))
    barcode = cv2.resize(concatenated, (args.width, args.height))

    out_png = args.out_dir / "barcode.png"
    cv2.imwrite(out_png, barcode)

    return out_png


# ---- CLI ----


def parse_args(args):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )

    parser.add_argument(
        "--video",
        "-v",
        dest="video",
        metavar="VIDEO",
        default=None,
        type=pathlib.Path,
        required=True,
        help=f"Video file.",
    )

    parser.add_argument(
        "--uniform",
        "-u",
        dest="uniform",
        metavar="UNIFORM_COLS",
        default=False,
        action="store_true",
        required=False,
        help="Use uniform color columns.",
    )

    parser.add_argument(
        "--out-dir",
        "-d",
        dest="out_dir",
        metavar="OUT_DIR",
        required=True,
        default=False,
        type=pathlib.Path,
        help="Where to save the output file.",
    )

    parser.add_argument(
        "--width",
        "-w",
        dest="width",
        metavar="WIDTH",
        required=False,
        default=2560,
        type=int,
        help="Width of the barcoded image.",
    )

    parser.add_argument(
        "--height",
        "-h",
        dest="height",
        metavar="HEIGHT",
        required=False,
        default=1280,
        type=int,
        help="Height of the barcoded image.",
    )

    parser.add_argument(
        "--sample-height",
        "-s",
        dest="sample_height",
        metavar="SAMPLE_HEIGHT",
        required=False,
        default=8,
        type=int,
        help="Sample Height of the barcoded image. "
             "In compressed mode, each frame is resized into a 1xSAMPLE_HEIGHT vector. "
             "SAMPLE_HEIGHT should be at most the input height and at least 1 (which "
             "is equivalent to uniform mode). Smaller values yield smoother results.",
    )

    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    args = parse_args(args)
    setup_logging(args.loglevel)

    start_time = time.time()

    generate_barcode(args)

    elapsed_time = time.time() - start_time
    print("Time elapsed: {}"
        .format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))


def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
