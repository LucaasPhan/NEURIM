#!/usr/bin/env python3
"""Run the GPU-side NEURIM diffusion supervisor."""

import argparse
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args(argv)
    uvicorn.run("src.server.diffusion.supervisor:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
