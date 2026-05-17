import argparse
import importlib

DEMOS = {
    "basic": "lab3.demos.basic",
    "b1": "lab3.demos.b1",
}


def main():
    parser = argparse.ArgumentParser(description="Lab 3: FEM Soft Body Simulation")
    parser.add_argument("--demo", choices=DEMOS.keys(), default="basic", help="Demo to run (default: basic)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    mod = importlib.import_module(DEMOS[args.demo])
    mod.run(debug=args.debug)


if __name__ == "__main__":
    main()
