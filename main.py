#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from workers import calibrate, ssl, linear, slimmable, distill

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Olympics Research Platform")
    subparsers = parser.add_subparsers(help="sub-command help")

     # a worker for linear evaluation
    parser_linear = subparsers.add_parser(
        "linear", help="the entrance of Linear Evaluation"
    )
    parser_linear.set_defaults(func=linear)
    parser_linear.add_argument(
        "--config", type=str, required=True, help="the path of config file"
    )
    parser_linear.add_argument(
        "-j", "--num_workers", type=int, required=False, default=4
    )
    parser_linear.add_argument(
        "--print-freq",
        "-p",
        default=50,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser_linear.add_argument(
        "--output", type=str, default="./train_log", help="a folder save training log"
    )

    # a worker for training slimmable network
    parser_slimmable = subparsers.add_parser(
        "slimmable", help="the entrance of Slimmable network training"
    )
    parser_slimmable.set_defaults(func=slimmable)
    parser_slimmable.add_argument(
        "--config", type=str, required=True, help="the path of config file"
    )
    parser_slimmable.add_argument(
        "-j", "--num_workers", type=int, required=False, default=4
    )
    parser_slimmable.add_argument(
        "--print-freq",
        "-p",
        default=50,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser_slimmable.add_argument(
        "--output", type=str, default="./train_log", help="a folder save training log"
    )


    # Distill training
    parser_distill = subparsers.add_parser(
        "distill", help="the entrance of distill Training"
    )
    parser_distill.set_defaults(func=distill)
    parser_distill.add_argument(
        "--config", type=str, required=True, help="the path of config file"
    )
    parser_distill.add_argument("-j", "--num_workers", type=int, required=False, default=4)
    parser_distill.add_argument(
        "--print-freq",
        "-p",
        default=50,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser_distill.add_argument(
        "--output", type=str, default="./train_log", help="a folder save training log"
    )


    # quantization aware SSL training
    parser_ssl = subparsers.add_parser(
        "ssl", help="the entrance of SSL Training"
    )
    parser_ssl.set_defaults(func=ssl)
    parser_ssl.add_argument(
        "--config", type=str, required=True, help="the path of config file"
    )
    parser_ssl.add_argument("-j", "--num_workers", type=int, required=False, default=4)
    parser_ssl.add_argument(
        "--print-freq",
        "-p",
        default=50,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser_ssl.add_argument(
        "--output", type=str, default="./train_log", help="a folder save training log"
    )

    # calibration SSL training
    parser_calibrate = subparsers.add_parser(
        "calibrate", help="the entrance of SSL Calibration BN"
    )
    parser_calibrate.set_defaults(func=calibrate)
    parser_calibrate.add_argument(
        "--config", type=str, required=True, help="the path of config file"
    )
    parser_calibrate.add_argument("-j", "--num_workers", type=int, required=False, default=4)
    parser_calibrate.add_argument(
        "--print-freq",
        "-p",
        default=50,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser_calibrate.add_argument(
        "--output", type=str, default="./train_log", help="a folder save training log"
    )

    args = parser.parse_args()
    args.func(args)