from __future__ import annotations

import argparse
import json
from pathlib import Path

from .controller import promote, read_state, repair, run_operator, verify
from .domain import QualityPolicy
from .intake import load_intake, write_intake


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="gutenberg-ralph", description="Prepare, run, verify, repair, and promote Ralph editorial packages.")
    sub = root.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="create a Ralph package from editorial inputs")
    init.add_argument("package", type=Path)
    init.add_argument("--primary", required=True, type=Path)
    init.add_argument("--supplement", action="append", type=Path, default=[])
    init.add_argument("--role", action="append", default=[])
    init.add_argument("--focus", default="")
    init.add_argument("--track", action="append", default=[])
    init.add_argument("--title", default="SteadyBurn weekly package")
    init.add_argument("--upstream-run", help="run identifier for prior-artifact provenance")
    run = sub.add_parser("run", help="submit the package to the research operator")
    run.add_argument("package", type=Path)
    run.add_argument("--operator-command")
    run.add_argument("--dry-run", action="store_true")
    for name, help_text in (("resume", "resume the durable operator run"), ("link", "ask the operator to generate and link the graph")):
        alias = sub.add_parser(name, help=help_text)
        alias.add_argument("package", type=Path)
        alias.add_argument("--operator-command")
        alias.add_argument("--dry-run", action="store_true")
    for name in ("verify", "status"):
        sub.add_parser(name).add_argument("package", type=Path)
    fix = sub.add_parser("repair", help="record a score-gated repair brief")
    fix.add_argument("package", type=Path)
    fix.add_argument("--brief", default="")
    publish = sub.add_parser("promote", help="promote only a passing staged package")
    publish.add_argument("package", type=Path)
    publish.add_argument("--destination", required=True, type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "init":
        roles = args.role or ["evidence"] * len(args.supplement)
        if len(roles) != len(args.supplement):
            raise SystemExit("provide one --role for each --supplement")
        intake = load_intake(args.primary, supplements=list(zip(args.supplement, roles)), focus=args.focus, specification={"title": args.title}, tracks=args.track, quality=QualityPolicy(), upstream_run=args.upstream_run)
        write_intake(args.package.resolve(), intake)
        print(json.dumps({"package": str(args.package.resolve()), "state": "draft"}, indent=2))
        return 0
    package = args.package.resolve()
    if args.command in {"run", "resume", "link"}:
        print(json.dumps(run_operator(package, args.operator_command, dry_run=args.dry_run), indent=2))
    elif args.command == "verify":
        report = verify(package)
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1
    elif args.command == "status":
        print(json.dumps(read_state(package), indent=2))
    elif args.command == "repair":
        print(json.dumps(repair(package, args.brief), indent=2))
    elif args.command == "promote":
        print(json.dumps(promote(package, args.destination), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
