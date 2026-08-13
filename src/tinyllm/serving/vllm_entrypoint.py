"""Pinned vLLM 0.8.5 entrypoint with explicit HTTP parser limits."""

from __future__ import annotations

import uvloop  # type: ignore[import-not-found]
from vllm.entrypoints.openai.api_server import run_server  # type: ignore[import-not-found]
from vllm.entrypoints.openai.cli_args import (  # type: ignore[import-not-found]
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.entrypoints.utils import cli_env_setup  # type: ignore[import-not-found]
from vllm.utils import FlexibleArgumentParser  # type: ignore[import-not-found]


def main() -> None:
    """Run the reviewed legacy server while bounding headers before ASGI parsing."""

    cli_env_setup()
    parser = FlexibleArgumentParser(description="TinyLLM guarded vLLM OpenAI server")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(
        run_server(
            args,
            http="h11",
            h11_max_incomplete_event_size=16_384,
            proxy_headers=False,
            server_header=False,
            date_header=False,
        )
    )


if __name__ == "__main__":
    main()
