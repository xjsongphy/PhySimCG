import argparse
import importlib


DEMOS = {
    "basic": "lab2.demos.basic",
    "b1": "lab2.demos.b1",
    "b3": "lab2.demos.b3",
    "b4": "lab2.demos.b4",
}


def main():
    parser = argparse.ArgumentParser(description="Lab 2: FLIP Fluid Simulation")
    parser.add_argument("--demo", choices=DEMOS.keys(), default="basic",
                        help="Demo to run (default: basic)")
    args = parser.parse_args()

    mod = importlib.import_module(DEMOS[args.demo])
    mod.run()


if __name__ == "__main__":
    main()
