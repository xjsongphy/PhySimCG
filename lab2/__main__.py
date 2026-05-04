import argparse
import importlib


DEMOS = {
    "basic": "lab2.demos.basic",
    "b1": "lab2.demos.basic",
    "b2": "lab2.demos.b2",
    "b3": "lab2.demos.b3",
    "b4": "lab2.demos.b4",
    "b5": "lab2.demos.b5",
}


def main():
    parser = argparse.ArgumentParser(description="Lab 2: FLIP Fluid Simulation")
    parser.add_argument("--demo", choices=DEMOS.keys(), default="basic",
                        help="Demo to run (default: basic)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode (profiling, timing overlay)")
    args = parser.parse_args()

    mod = importlib.import_module(DEMOS[args.demo])
    mod.run(debug=args.debug)


if __name__ == "__main__":
    main()
