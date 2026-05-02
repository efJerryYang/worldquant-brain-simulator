import numpy as np

from wqsim.fast_expr import (
    correlation,
    covariance,
    delay,
    delta,
    rank,
    scale,
    ts_argmax,
    ts_argmin,
    ts_mean,
    ts_rank,
    ts_sum,
)


def test_rank_normalizes_rows_and_ignores_nans():
    values = np.array([[3.0, 1.0, 1.0, np.nan], [5.0, np.nan, 10.0, 0.0]])

    actual = rank(values)

    np.testing.assert_allclose(actual[0, :3], [1.0, 0.25, 0.25])
    assert np.isnan(actual[0, 3])
    np.testing.assert_allclose(actual[1, [0, 2, 3]], [0.5, 1.0, 0.0])


def test_delay_and_delta_shift_time_axis():
    values = np.array([[1.0, 10.0], [3.0, 11.0], [6.0, 15.0]])

    np.testing.assert_allclose(delay(values, 1)[1:], values[:-1])
    np.testing.assert_allclose(delta(values, 1)[1:], [[2.0, 1.0], [3.0, 4.0]])
    assert np.isnan(delay(values, 1)[0, 0])


def test_time_series_reductions_use_full_window():
    values = np.array([[1.0, 10.0], [2.0, 20.0], [4.0, 30.0]])

    summed = ts_sum(values, 2)
    mean = ts_mean(values, 2)

    assert np.isnan(summed[0, 0])
    np.testing.assert_allclose(summed[1:], [[3.0, 30.0], [6.0, 50.0]])
    np.testing.assert_allclose(mean[1:], [[1.5, 15.0], [3.0, 25.0]])


def test_correlation_and_scale():
    x = np.array([[1.0, 1.0], [2.0, 3.0], [3.0, 9.0]])
    y = np.array([[2.0, 9.0], [4.0, 3.0], [6.0, 1.0]])

    corr = correlation(x, y, 3)
    np.testing.assert_allclose(corr[2], [1.0, -0.8461538461538459])

    weights = scale(np.array([[1.0, -3.0, np.nan], [0.0, 0.0, np.nan]]))
    np.testing.assert_allclose(weights[0, :2], [0.25, -0.75])
    np.testing.assert_allclose(weights[1, :2], [0.0, 0.0])


def test_time_series_rank_and_arg_operators_are_hand_calculated():
    values = np.array(
        [
            [1.0, 4.0],
            [3.0, 3.0],
            [2.0, 2.0],
            [5.0, 1.0],
        ]
    )

    ranked = ts_rank(values, 3)
    argmax = ts_argmax(values, 3)
    argmin = ts_argmin(values, 3)

    assert np.isnan(ranked[1, 0])
    np.testing.assert_allclose(ranked[2], [0.5, 0.0])
    np.testing.assert_allclose(ranked[3], [1.0, 0.0])
    np.testing.assert_allclose(argmax[2], [2.0, 1.0])
    np.testing.assert_allclose(argmin[2], [1.0, 3.0])


def test_covariance_uses_finite_pairs_only():
    x = np.array([[1.0, 1.0], [2.0, np.nan], [3.0, 4.0]])
    y = np.array([[2.0, 1.0], [4.0, 2.0], [6.0, 5.0]])

    actual = covariance(x, y, 3)

    np.testing.assert_allclose(actual[2, 0], 2.0)
    np.testing.assert_allclose(actual[2, 1], 6.0)
