#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "steady-burn"


def escape_make_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("$", "$$").replace("#", r"\#")


def prompt(prompt_text: str) -> str:
    return input(prompt_text).strip().lstrip("\ufeff")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prompt for active Steady Burn metadata and write a make include file.")
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--default-date", required=True)
    args = parser.parse_args()

    title = ""
    while not title:
        title = prompt("Title: ")
        if not title:
            print("Title is required.")

    date_value = prompt(f"Date [{args.default_date}]: ") or args.default_date
    slug = slugify(title)
    target_dir = f"{args.target_root.rstrip('/')}/{date_value}-{slug}"

    state_file = Path(args.state_file)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(
        "\n".join(
            [
                f"BURN_STATE_TITLE := {escape_make_value(title)}",
                f"BURN_STATE_DATE := {escape_make_value(date_value)}",
                f"BURN_STATE_SLUG := {escape_make_value(slug)}",
                f"BURN_STATE_DIR := {escape_make_value(target_dir)}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Active burn: {title}")
    print(f"Date: {date_value}")
    print(f"Slug: {slug}")
    print(f"Directory: {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
