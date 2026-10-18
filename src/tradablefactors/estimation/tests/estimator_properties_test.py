"""Factor covariance identities migrated from OP to the pipeline owner."""
from functools import lru_cache
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import qis
from factorlasso import LassoModel, LassoModelType, DependenceMeasure, DistanceTransform
from tradablefactors import FactorCovarEstimator

RETURNS_FREQ = 'ME'
SPAN = 24
N_FACTORS = 4
PSD_TOL = -1e-12


@lru_cache(maxsize=1)
def _prices() -> pd.DataFrame:
    """Synthetic factor/asset panel independent of OP fixtures."""
    rng = np.random.default_rng(8703)
    x = rng.normal(0.002, 0.03, (144, 4))
    beta = rng.normal(0.25, 0.2, (4, 8))
    y = x @ beta + rng.normal(0.001, 0.015, (144, 8))
    returns = np.column_stack([x, y])
    return pd.DataFrame(100 * np.exp(returns.cumsum(axis=0)),
                        index=pd.date_range('2000-01-31', periods=144, freq='ME'),
                        columns=[f'Synthetic{i}' for i in range(12)])


@lru_cache(maxsize=2)
def _factor_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    split the fixture into risk factors and assets.

    The first four instruments stand in as the factor block and the remaining fifteen as the
    assets; the split is arbitrary but fixed, and the identity being checked does not depend on
    which columns play which role.
    """
    prices = _prices()
    factors, assets = prices.iloc[:, :N_FACTORS], prices.iloc[:, N_FACTORS:]
    asset_returns = {RETURNS_FREQ: qis.to_returns(assets, freq=RETURNS_FREQ,
                                                  is_log_returns=True, drop_first=True)}
    return factors, assets, asset_returns


@lru_cache(maxsize=2)
def _current_factor_covars(n_clusters: int = None):
    """fit the factor model once; ``n_clusters`` selects the clustering model type."""
    factors, assets, asset_returns = _factor_inputs()
    if n_clusters is None:
        lasso_model = LassoModel(model_type=LassoModelType.LASSO, reg_lambda=1e-5, span=36)
    else:
        lasso_model = LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                                 reg_lambda=1e-5, span=36, n_clusters=n_clusters)
    estimator = FactorCovarEstimator(lasso_model=lasso_model, factor_returns_freq=RETURNS_FREQ,
                                     factor_covar_span=SPAN)
    return estimator.fit_current(risk_factor_prices=factors,
                                 asset_returns_dict=asset_returns,
                                 assets=assets.columns,
                                 cutoff=factors.index[-1]).data


def test_factor_covariance_reconstructs_from_its_parts() -> None:
    """
    Σ_y = β Σ_x β' + diag(residual variance).

    The identity the factor model exists to impose. If it fails, the assembled covariance is not
    the model's covariance and the risk decomposition reported beside it describes a different
    matrix from the one the optimiser used.
    """
    data = _current_factor_covars()
    betas = data.y_betas.to_numpy()
    factor_covar = data.x_covar.to_numpy()
    residual_var = data.y_variances['residual_var'].to_numpy()
    reconstructed = betas @ factor_covar @ betas.T + np.diag(residual_var)
    np.testing.assert_allclose(data.get_y_covar().to_numpy(), reconstructed, atol=1e-12)


def test_factor_covariance_off_diagonal_is_purely_systematic() -> None:
    """
    residual risk is diagonal by construction, so every off-diagonal entry is β Σ_x β'.

    Sharper than the identity above: it isolates the modelling assumption from the diagonal, where
    a residual term could hide a reconstruction error.
    """
    data = _current_factor_covars()
    betas = data.y_betas.to_numpy()
    systematic = betas @ data.x_covar.to_numpy() @ betas.T
    y_covar = data.get_y_covar().to_numpy()
    off_diagonal = ~np.eye(len(y_covar), dtype=bool)
    np.testing.assert_allclose(y_covar[off_diagonal], systematic[off_diagonal], atol=1e-12)


def test_factor_and_asset_covariances_are_symmetric_and_psd() -> None:
    """both the factor block and the assembled asset covariance are usable by an optimiser."""
    data = _current_factor_covars()
    for name, matrix in (('x_covar', data.x_covar.to_numpy()),
                         ('y_covar', data.get_y_covar().to_numpy())):
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12,
                                   err_msg=f'{name} is not symmetric')
        assert np.linalg.eigvalsh(matrix).min() >= PSD_TOL, f'{name} is not PSD'


def test_n_clusters_above_the_universe_size_clamps_instead_of_raising() -> None:
    """
    the defect that set this package's factorlasso floor at 0.10.1.

    factorlasso 0.10.0 raised when ``n_clusters`` exceeded the number of assets. A count
    calibrated on the full sample therefore failed at earlier dates holding fewer instruments,
    which is precisely a rolling estimation over a growing universe. 0.10.1 clamps to the universe
    size, which is what scipy's 'maxclust' criterion does natively. This test fails against a
    floor that slips back below 0.10.1.
    """
    _, assets, _ = _factor_inputs()
    n_assets = len(assets.columns)
    data = _current_factor_covars(n_clusters=n_assets + 10)
    clusters = data.clusters
    labels = set(clusters.values()) if isinstance(clusters, dict) else set(np.ravel(clusters))
    assert 0 < len(labels) <= n_assets, (
        f'asked for {n_assets + 10} clusters over {n_assets} assets and got {len(labels)}')


def test_clustering_defaults_are_the_documented_ones() -> None:
    """
    the defaults the 6.3.0 and 6.4.0 changelog entries promise, and nothing checked.

    Both entries state that the new ``distance_transform`` and ``dependence_measure`` parameters
    default to the pre-existing behaviour, so results computed before them are unchanged. That
    promise is only kept while the defaults stay put; changing either silently re-clusters every
    factor covariance in the stack.
    """
    model = LassoModel(model_type=LassoModelType.LASSO, reg_lambda=1e-5, span=36)
    assert model.distance_transform == DistanceTransform.ONE_MINUS_RHO
    assert model.dependence_measure == DependenceMeasure.PEARSON
    assert model.n_clusters is None, (
        'n_clusters now defaults to a value, so the fractional-distance cut is no longer the '
        'default clustering and the 6.4.0 changelog claim no longer holds')
