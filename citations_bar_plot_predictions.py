import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pathlib
import argparse
import sys


def csv_path(path, required_columns={}):
    """
    Define the csv_path type for use with argparse. Checks
    that the given path string is a path to a csv file and that the
    header of the csv file contains the required columns.
    """
    p = pathlib.Path(path)
    required_columns = set(required_columns)
    if p.is_file():
        try:  # only read the csv header
            expected_columns_exist = required_columns.issubset(
                set(pd.read_csv(path, nrows=0).columns.tolist())
            )
            if expected_columns_exist:
                return p
            else:
                raise argparse.ArgumentTypeError(
                    f"Invalid argument ({path}), does not contain all expected columns."
                )
        except UnicodeDecodeError:
            raise argparse.ArgumentTypeError(
                f"Invalid argument ({path}), not a csv file."
            )
    else:
        raise argparse.ArgumentTypeError(f"Invalid argument ({path}), not a file.")


def positive_int(i):
    res = int(i)
    if res > 0:
        return res
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid argument ({i}), expected value > 0 ."
        )


def non_negative_int(i):
    res = int(i)
    if res >= 0:
        return res
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid argument ({i}), expected value >= 0 ."
        )


class PolynomialExponentialModelTimeSeries:
    """ """

    def __init__(self):
        self._model = None
        self._exponential_model = False

    def _make_superscript(self, normal_str):
        normal = ".+-x0123456789"
        super_s = "‧⁺⁻ˣ⁰¹²³⁴⁵⁶⁷⁸⁹"
        return normal_str.translate("".maketrans(normal, super_s))

    def __str__(self):
        """
        String representation of the model.
        """
        # The model coefficients are in the scaled domain (domain is scaled
        # for numerical stability). To obtain the coefficients in the data
        # domain we need to convert() the polynomial.
        # For more details, see discussion on GitHub:
        # https://github.com/numpy/numpy/issues/9533
        if self._model and not self._exponential_model:
            return str(self._model.convert())
        else:
            return "e" + self._make_superscript(str(self._model.convert()))

    def fit(self, x, y, use_weights=False, last_k_test=None):
        """
        Fit a polynomial or exponential function to the data. This is an optimal model in the least squares
        error sense, assuming normally distributed noise. Try all polynomials, y = a^T[1,x, x^2,...]+eps, with
        degrees in [1,n-1] where n is the number of points. Also try the exponential function y = ae^{bx}.
        The final model is the one with minimal AICc value or the minimal RMSE for the last_k_test points if
        that parameter is given (assumes that the time series is ascending and uniformly spaced, i.e.
        x=[1, 6, 11...]).
        For descriptions of underlying mathematical formulation see:
        https://mathworld.wolfram.com/LeastSquaresFittingPolynomial.html
        https://mathworld.wolfram.com/LeastSquaresFittingExponential.html
        """
        if last_k_test:
            if len(x) - last_k_test < 2:
                raise ValueError(
                    "Specified last_k_test is too large, not enough data to estimate model."
                )
            x_test = x[-last_k_test:]
            y_test = y[-last_k_test:]
            x = x[0 : len(x) - last_k_test]  # noqa E203
            y = y[0 : len(y) - last_k_test]  # noqa E203

        weights = None
        if use_weights:  # softmax weights, e^(1/k) / sum(e^(1/n)...e^1)
            weights = np.exp(1 / np.array(range(len(x), 0, -1)))
            weights /= sum(weights)
        deg = 1
        lsq_poly_fit = Polynomial.fit(x, y, deg=deg, w=weights)
        self._model = lsq_poly_fit
        if last_k_test:
            best_value = np.sqrt(np.mean((lsq_poly_fit(x_test) - y_test) ** 2))
        else:
            # number of parameters used in AIC computation includes the error term, so polynomial degree + 2
            best_value = self._aic_c(
                estimated_values=lsq_poly_fit(x),
                actual_values=y,
                num_parameters=deg + 2,
            )

        for deg in range(2, len(x)):
            lsq_poly_fit = Polynomial.fit(x, y, deg, w=weights)
            if last_k_test:
                value = np.sqrt(np.mean((lsq_poly_fit(x_test) - y_test) ** 2))
            else:
                value = self._aic_c(
                    estimated_values=lsq_poly_fit(x),
                    actual_values=y,
                    num_parameters=deg + 2,
                )
            if value < best_value:
                best_value = value
                self._model = lsq_poly_fit
        lsq_exp_fit = Polynomial.fit(x, np.log(y), deg=1, w=np.sqrt(y))
        if last_k_test:
            value = np.sqrt(np.mean((np.exp(lsq_exp_fit(x_test)) - y_test) ** 2))
        else:
            value = self._aic_c(
                estimated_values=np.exp(lsq_exp_fit(x)),
                actual_values=y,
                num_parameters=3,
            )
        if value < best_value:
            self._model = lsq_exp_fit
            self._exponential_model = True

    def predict(self, x):
        y = self._model(x)
        if self._exponential_model:
            y = np.exp(y)
        return y

    def _aic(self, estimated_values, actual_values, num_parameters):
        """
        The Akaike Information Criterion (AIC) for model selection.
        """
        n = len(actual_values)
        squared_residuals = (estimated_values - actual_values) ** 2
        return n * np.log(sum(squared_residuals) / n) + 2 * num_parameters

    def _aic_c(self, estimated_values, actual_values, num_parameters):
        """
        AICc is a modification of the AIC criterion for small sample sizes.
        """
        denom = len(actual_values) - num_parameters - 1
        if denom > 0:
            return (
                self._aic(estimated_values, actual_values, num_parameters)
                + (2 * num_parameters * (num_parameters + 1)) / denom
            )
        return np.inf  # more parameters than observations, ouch.


def main(argv=None):
    parser = argparse.ArgumentParser(description="Argparse usage example")
    # required input csv file which has columns titled year and citations
    parser.add_argument(
        "csv_file_name",
        type=lambda x: csv_path(
            x,
            ["year", "citations"],
        ),
        help="csv file containing columns titled 'year' and 'citations' corresponding"
        + " to the year and number of paper citations.",
    )
    parser.add_argument(
        "pred_years",
        type=non_negative_int,
        help="Number of years into the future which are predicted by the optimal model.",
    )
    parser.add_argument(
        "--use_weights",
        default=False,
        action="store_true",
        help="Use softmax weights inversely proportional to years from end.",
    )
    parser.add_argument(
        "--last_k_test",
        type=positive_int,
        help="Last k entries will be used to evaluate model (best has minimal RMSE on these entries).",
    )
    parser.add_argument(
        "--output_file_name",
        help="Output file name. File extension defines the output file type. If not provided, "
        + "output is written to the input directory using input file name and the extension pdf.",
    )
    args = parser.parse_args(argv)
    output_file_path = (
        args.output_file_name
        if args.output_file_name
        else args.csv_file_name.with_suffix(".pdf")
    )

    df = pd.read_csv(args.csv_file_name)
    years = np.array(range(df["year"].min(), df["year"].max() + args.pred_years + 1))
    known_citations = df["citations"].to_numpy()

    if args.pred_years != 0:
        lsq_model = PolynomialExponentialModelTimeSeries()
        # Treat the approximation task as continuous with an unbounded range.
        # It isn't, so we need to project the results onto the valid solution space.
        lsq_model.fit(
            x=df["year"],
            y=df["citations"],
            use_weights=args.use_weights,
            last_k_test=args.last_k_test,
        )
        # Project the predicted citation numbers onto the valid discrete non-negative
        # solution space.
        predicted_citations = np.rint(lsq_model.predict(years))
        predicted_citations = np.clip(predicted_citations, 0, predicted_citations.max())

        # plotting the known and predicted bars side by side, interesting but
        # hard to read (keeping the code for future reference):
        #
        # known_citations = np.concatenate([df['citations'].to_numpy(), [0]*args.pred_years])
        # df = pd.DataFrame(data = {"known citations": known_citations, "predicted citations": predicted_citations}, index= years)  # noqa E501
        # ax = df.plot.bar(rot=0, xlabel = "year", ylabel="number of citations")
        # # Add the number of citations at the top of each bar. Because they
        # # are so close we use a smaller font size, rotate the text 90 degrees
        # # and move it slightly above the bar (padding).
        # for container in ax.containers:
        #     ax.bar_label(container, fontsize=8, rotation=90, padding=3)

    colors = ["blue", "orange"]
    fig, ax = plt.subplots()
    plt.bar(
        years[0 : len(known_citations)],  # noqa E203
        known_citations,
        label="known",
        color=colors[0],
    )
    if args.pred_years != 0:
        plt.bar(
            years[-args.pred_years :],  # noqa E203
            predicted_citations[-args.pred_years :],  # noqa E203
            label="predicted",
            color=colors[1],
        )
        plt.plot(years, predicted_citations, color=colors[1])
        plt.scatter(years, predicted_citations, color=colors[1])
        # Add text to the figure using axes coordinates (range of [0,1]).
        model_str = f"prediction model: {str(lsq_model)}\n" if str(lsq_model) else ""
        ax.text(
            0.01,
            0.75,
            model_str
            + "median absolute error: "
            + f"{int(np.median(np.abs(predicted_citations[0:len(known_citations)] - known_citations)))}",
            # noqa E501
            transform=ax.transAxes,
            color="green",
            fontsize=8,
        )

    # place the number of citations at the top of each bar.
    for container, color in zip(ax.containers, colors):
        ax.bar_label(container, color=color)
    ax.set_xlabel("year")
    ax.set_ylabel("citation count")
    ax.set_xticks(years)
    np.set_printoptions(formatter={"float_kind": "{:.2f}".format})

    plt.legend()
    plt.savefig(output_file_path, dpi=150)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
