import argparse
import json

from vasagent import hello
from vasagent.ktv import predict_ktv


def main() -> None:
    parser = argparse.ArgumentParser(description="Process a lab report.")
    parser.add_argument("report", nargs="?", help="Path to JSON report")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Predict Kt/V if BUN and Creatinine are present in the report",
    )
    args = parser.parse_args()

    print(hello())

    if args.report:
        with open(args.report, "r", encoding="utf-8") as f:
            report = json.load(f)
        if args.predict:
            if "BUN" in report and "Creatinine" in report:
                result = predict_ktv(report["BUN"], report["Creatinine"])
                print(f"Predicted Kt/V: {result}")
            else:
                print("Prediction requested but BUN and Creatinine not found in report.")
    else:
        if args.predict:
            print("No report provided.")


if __name__ == "__main__":
    main()
