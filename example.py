import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_filepath", type=str, required=True, help="Input data file (.csv)")
parser.add_argument("-o",
    "--output",
    type=str,
    required=True,
    help="File where the output will be appended (.csv)",
)
parser.add_argument("-m", "--model", type=str, required=True, help="Model")
parser.add_argument("-l", "--lookback", type=int, default=1, help="Lookback window")
parser.add_argument("-f", "--forecast", type=int, default=1, help="Forecast window")
parser.add_argument(
    "-r",
    "--dimensionality_reduction",
    type=str,
    default=None,
    help="Dimensionality reduction method",
)
args = parser.parse_args()