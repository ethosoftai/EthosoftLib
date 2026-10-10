"""Mercan Plugin SDK command-line helpers; validate and pack never load code."""
from __future__ import annotations
import argparse
import json

from . import package_plugin, read_manifest, scaffold


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m ethosoftlib.plugins")
    commands = parser.add_subparsers(dest="command", required=True)
    new = commands.add_parser("scaffold")
    new.add_argument("name")
    new.add_argument("directory")
    check = commands.add_parser("validate")
    check.add_argument("manifest")
    pack = commands.add_parser("pack")
    pack.add_argument("manifest")
    pack.add_argument("destination")
    args = parser.parse_args()
    if args.command == "scaffold":
        print(scaffold(args.name, args.directory))
    elif args.command == "validate":
        manifest = read_manifest(args.manifest)
        print(json.dumps({"name": manifest["name"], "version": manifest["version"],
                          "abi_version": manifest["abi_version"],
                          "platforms": sorted(manifest["binaries"])}))
    else:
        print(package_plugin(args.manifest, args.destination))


if __name__ == "__main__":
    main()
