import argparse
import great_expectations as ge
from great_expectations.data_context import DataContext


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Data with Great Expectations")
    parser.add_argument('--suite', type=str, help='Expectation suite name', required=True)
    parser.add_argument('--data_path', type=str, help='Path to the data file', required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize DataContext
    data_context = DataContext()

    # Create a DataFrame using Great Expectations
    df = ge.read_csv(args.data_path)

    # Run validation
    results = df.validate(expectation_suite=args.suite, result_format="SUMMARY")

    # Check results
    if results['success']:
        print(f"Data at {args.data_path} passed all expectations.")
    else:
        print(f"Data at {args.data_path} failed some expectations.")
        print(results['results'])


if __name__ == "__main__":
    main()