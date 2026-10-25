"""Purpose: Compute labelled opportunity and funding-neutral alpha dispersion.
Inputs: Completed FM moments, fixed alpha and optional compatible factor premia.
Outputs: Dispersion tables, access diagnostics and prepared optimisation inputs.
Units: Annual native risk coordinates and alpha in matching declared return units.
Conditioning: Loadings and raw diagonal residual D remain fixed in GLS diagnostics.
Ownership: FM owns labelled FAJ accounting; QIS owns risk reporting and interchange.

The array-level residual_opportunity primitive removes explicit controls. This
labelled path preserves its funding-plus-factor arithmetic and reported unwhitened
rank metadata; an independent capacity/noise-floor test reconciles the two routes.

Statistical alpha describes longer-horizon residual means, not tactical forecasts.
The caller owns estimation, currency, membership and source provenance. Invalid
assets must be resolved before construction; nothing is floored or normalized.
Risk reporting delegates to QIS. Statistical outputs and prepared optimisation
arrays are complete FM results; portfolio solving belongs to downstream consumers.
"""
from __future__ import annotations
from numpy.typing import ArrayLike


from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import qis

from factormodels.analytics.models import FactorModelSnapshot
from factormodels.gls_diagnostic import sharpe_accounting as sa
from factormodels.gls_diagnostic.geometry import gls_projector, gls_split
from factormodels.gls_diagnostic._validation import _ALIGN_ERROR, _align, _array, _vector


__all__ = ['FactorOpportunityModel', 'alpha_dispersion', 'factor_access_metrics', 'candidate_access_gain', 'portfolio_opportunity_metrics']


@dataclass(frozen=True)
class FactorOpportunityModel:
    """Owned labelled inputs and a canonical risk model; treat frames as immutable.

    Residual variances must be strictly positive. A joint asset axis may include
    reference-only instruments; an investability mask controls candidate holdings.
    All moments and alpha use annualized decimal units in one reference currency.
    Supply completed residual/asset covariance and the unit-weight raw-diagonal
    reference, or use from_snapshot. This consumer never assembles covariance.
    Dispersion and GLS retain the raw diagonal-D reference geometry.
    """

    betas: pd.DataFrame
    residual_variances: pd.Series
    factor_covariance: pd.DataFrame
    alpha: pd.Series
    date: pd.Timestamp
    residual_covariance: pd.DataFrame = field(kw_only=True)
    asset_covariance: pd.DataFrame = field(kw_only=True)
    raw_diagonal_reference_covariance: pd.DataFrame = field(kw_only=True)
    provenance: dict = field(default_factory=dict, kw_only=True)
    factor_return_types: dict | None = field(default=None, kw_only=True)
    factor_groups: dict | None = field(default=None, kw_only=True)
    risk_model: object = field(init=False, repr=False, compare=False)
    factor_risk_model: object = field(init=False, repr=False, compare=False)
    residual_risk_model: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Reject invalid axes/moments, copy inputs, and build canonical risk."""
        for name, kind in [('betas', pd.DataFrame), ('residual_variances', pd.Series),
                           ('factor_covariance', pd.DataFrame), ('alpha', pd.Series),
                           ('residual_covariance', pd.DataFrame), ('asset_covariance', pd.DataFrame),
                           ('raw_diagonal_reference_covariance', pd.DataFrame)]:
            value = getattr(self, name)
            if not isinstance(value, kind):
                raise ValueError(f'{name} must be a labelled {kind.__name__}')
            object.__setattr__(self, name, value.copy(deep=True))
        b, d, f = self.betas, self.residual_variances, self.factor_covariance
        sa._model(pd.Series(0., index=b.columns), f, b, d)
        _vector(self.alpha, b, 'alpha', label_error=_ALIGN_ERROR)
        date = pd.Timestamp(self.date)
        if pd.isna(date):
            raise ValueError('date must be finite')
        object.__setattr__(self, 'date', date)
        from copy import deepcopy
        object.__setattr__(self, 'provenance', deepcopy(self.provenance))
        object.__setattr__(self, 'factor_return_types', deepcopy(self.factor_return_types))
        object.__setattr__(self, 'factor_groups', deepcopy(self.factor_groups))
        for name in ('residual_covariance', 'asset_covariance', 'raw_diagonal_reference_covariance'):
            matrix = getattr(self, name)
            if not matrix.index.equals(b.index) or not matrix.columns.equals(b.index):
                raise ValueError(f'{name} must match the exact asset order')
            values = matrix.to_numpy(dtype=float)
            if not np.isfinite(values).all() or not np.allclose(values, values.T, rtol=1e-10, atol=1e-12):
                raise ValueError(f'{name} must be finite and symmetric')
        diagonal = qis.RiskModel(covar={date: self.raw_diagonal_reference_covariance.copy()},
            factor_loadings={date: b.copy()}, factor_covar={date: f.copy()},
            residual_vars={date: pd.Series(d.to_numpy(), index=d.index)},
            factor_return_types=(self.factor_return_types if self.factor_return_types is not None
                                 else dict.fromkeys(b.columns, qis.ReturnTypes.LOG)),
            factor_groups=self.factor_groups)
        object.__setattr__(self, 'factor_risk_model', diagonal)
        residual = self.residual_covariance
        raw_diagonal = np.diag(d.to_numpy())
        # Check supplied views against each other, without duplicating FL's B F B'
        # assembly or changing any supplied matrix/rounding.
        if not np.allclose(self.asset_covariance - self.raw_diagonal_reference_covariance,
                           residual.to_numpy() - raw_diagonal, rtol=1e-10, atol=1e-12):
            raise ValueError('selected and raw-reference covariance views disagree with residual risk')
        selected_differs = not np.array_equal(residual.to_numpy(), raw_diagonal)
        selected = qis.RiskModel(
            covar={date: self.asset_covariance.copy()},
            factor_loadings=None if selected_differs else diagonal.factor_loadings,
            factor_covar=None if selected_differs else diagonal.factor_covar,
            residual_vars=None if selected_differs else diagonal.residual_vars,
            factor_return_types=None if selected_differs else diagonal.factor_return_types,
            factor_groups=None if selected_differs else diagonal.factor_groups,
        )
        object.__setattr__(self, 'risk_model', selected)
        object.__setattr__(self, 'residual_risk_model',
            qis.RiskModel(covar={date: residual.copy()}) if selected_differs else None)

    @classmethod
    def from_snapshot(cls, snapshot: FactorModelSnapshot, alpha: pd.Series) -> FactorOpportunityModel:
        """Consume completed selected risk and raw-D reference from one cutoff."""
        if not isinstance(snapshot, FactorModelSnapshot):
            raise TypeError('snapshot must be a completed FactorModelSnapshot')
        return cls(snapshot.loadings, snapshot.fit_statistics['residual_var'],
                   snapshot.factor_covariance, alpha, date=snapshot.cutoff,
                   residual_covariance=snapshot.residual_covariance,
                   asset_covariance=snapshot.asset_covariance,
                   raw_diagonal_reference_covariance=snapshot.raw_diagonal_reference_covariance,
                   provenance=snapshot.provenance, factor_return_types=snapshot.factor_return_types,
                   factor_groups=snapshot.factor_groups)


    def optimisation_inputs(self) -> dict:
        """Return copied labelled arrays for a downstream constrained optimiser.

        Full selected asset and residual covariance retain their risk policy.
        Orthogonal alpha uses the explicitly diagonal-D funding/factor geometry;
        consumers must not re-estimate it or infer it from covariance.
        """
        joint = pd.DataFrame(np.column_stack([np.ones(len(self.betas)), self.betas]),
                             index=self.betas.index)
        _, orthogonal = gls_split(self.alpha, joint, self.residual_variances)
        return dict(covariance=self.asset_covariance.copy(),
                    alpha=self.alpha.copy(), factor_loadings=self.betas.copy(),
                    factor_covariance=self.factor_covariance.copy(),
                    residual_covariance=self.residual_covariance.copy(),
                    orthogonal_alpha=pd.Series(orthogonal, index=self.alpha.index),
                    date=self.date)

    def prepare_alpha_covariance(self, covariance: pd.DataFrame, *, orthogonal: bool = False) -> pd.DataFrame:
        """Validate and optionally project supplied alpha-estimation covariance.

        Calibration and GLS projection belong upstream of optimisation. This is
        estimation covariance for the selected objective, not return covariance.
        No confidence level or ellipsoid radius is inferred.
        """
        from factorlasso import gaussian_quadratic_summary
        if (not isinstance(covariance, pd.DataFrame)
                or not covariance.index.equals(self.alpha.index)
                or not covariance.columns.equals(self.alpha.index)):
            raise ValueError('alpha covariance must match the exact asset order')
        gaussian_quadratic_summary(self.alpha, covariance, np.eye(len(self.alpha)))
        if not orthogonal:
            return covariance.copy()
        joint = pd.DataFrame(np.column_stack([np.ones(len(self.betas)), self.betas]),
                             index=self.betas.index)
        projector = gls_projector(joint, self.residual_variances)
        projected = projector @ covariance.to_numpy() @ projector.T
        return pd.DataFrame((projected+projected.T)/2,
                            index=self.alpha.index, columns=self.alpha.index)


def _alpha_dispersion_inputs(b: pd.DataFrame, d: pd.Series, a: pd.Series) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate canonical dispersion without constructing unused portfolio risk models."""
    _, h = gls_split(a, b, d)
    c = pd.DataFrame(np.column_stack([np.ones(len(b)), b]), index=b.index)
    _, hc = gls_split(a, c, d)
    centre = float((a / d).sum() / (1. / d).sum())
    positive = a.clip(lower=0.)
    detail = pd.DataFrame({'alpha': a, 'h_factor': h, 'h_joint': hc,
                           'residual_vol': np.sqrt(d), 'A_contribution': a * a / d,
                           'J_contribution': (a - centre)**2 / d,
                           'K_contribution': hc * hc / d})
    rows = dict(N=len(b), rank_B=int(np.linalg.matrix_rank(b)),
                rank_funding_B=int(np.linalg.matrix_rank(c)),
                A=float((a * a / d).sum()), J=float(((a - centre)**2 / d).sum()),
                S_h2=float((h * h / d).sum()), K=float((hc * hc / d).sum()),
                A_plus=float((positive**2 / d).sum()), alpha_common_level=centre,
                alpha_mean=float(a.mean()), alpha_median=float(a.median()),
                alpha_q25=float(a.quantile(.25)), alpha_q75=float(a.quantile(.75)),
                positive_count=int((a > 0).sum()), negative_count=int((a < 0).sum()))
    for key in ('A', 'J', 'K'):
        rows[key + '_per_asset'] = rows[key] / len(b)
        rows[key + '_rms'] = np.sqrt(rows[key] / len(b))
    rows['J_weighted_alpha_std'] = np.sqrt(rows['J'] / (1. / d).sum())
    if positive.max() > 0:
        weights = positive / d
        detail['residual_reference_weight'] = weights / weights.sum()
        rows['residual_reference_signed_sharpe'] = np.sqrt(rows['A_plus'])
    else:
        weights = pd.Series(0., index=b.index)
        weights.loc[(a / np.sqrt(d)).idxmax()] = 1.
        detail['residual_reference_weight'] = weights
        rows['residual_reference_signed_sharpe'] = float((a / np.sqrt(d)).max())
    detail.attrs['gls_geometry'] = 'diagonal_residual_reference'
    return rows, detail


def alpha_dispersion(model: FactorOpportunityModel) -> tuple[dict[str, float], pd.DataFrame]:
    """Return D1-D6 capacities and asset-level GLS residuals on the supplied axis.

    K removes the joint span of funding and loadings, using canonical GLS.
    A+ is a positive-part residual-risk reference, not a full-model frontier.
    """
    return _alpha_dispersion_inputs(model.betas, model.residual_variances, model.alpha)


def factor_access_metrics(model: FactorOpportunityModel, premia: pd.Series | None = None) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Return premium-free access modes and optional canonical F1-F4/F3 results."""
    b, d, f = model.betas, model.residual_variances, model.factor_covariance
    gram = b.to_numpy().T @ (b.to_numpy() / d.to_numpy()[:, None])
    root = np.linalg.cholesky(f)
    modes = np.linalg.eigvalsh(root.T @ gram @ root)
    spectrum = pd.DataFrame({'mode': np.arange(1, len(modes) + 1),
                             'g': modes, 'access_fraction': modes / (1. + modes)})
    summary, factors = {}, pd.DataFrame(index=b.columns)
    if premia is not None:
        ceiling = sa.factor_ceiling(premia, f)
        systematic = sa.achievable_sharpe2(premia, f, b, d)
        hurdle, conditional, gain = sa.hurdle_premia(premia, f)
        z = np.linalg.solve(f, premia)
        summary = dict(factor_ceiling2=ceiling, systematic_sharpe2=systematic,
                       access_loss=ceiling - systematic,
                       FPIR=systematic / ceiling if ceiling > 0 else np.nan)
        factors = pd.DataFrame({'premium': premia, 'tangency_direction': z,
                                'premium_sensitivity': 2 * z, 'hurdle': hurdle,
                                'conditional_variance': conditional, 'drop_one_gain': gain})
    return summary, factors, spectrum


def candidate_access_gain(model: FactorOpportunityModel, premia: ArrayLike, candidate_beta: ArrayLike, candidate_variance: float | np.integer | np.floating) -> float:
    """Fixed-model systematic capacity gain for one independent-residual addition.

    Existing FM achievable_sharpe2 computes total capacity, but does not expose
    the Sherman-Morrison marginal-addition identity used here. No refit occurs.
    """
    sa._model(premia, model.factor_covariance, model.betas, model.residual_variances)
    _align(candidate_beta, model.betas.columns)
    beta = _array(candidate_beta, 'candidate_beta', 1)
    if len(beta) != len(premia) or not np.isfinite(candidate_variance) or candidate_variance <= 0:
        raise ValueError('candidate requires aligned loadings and positive residual variance')
    precision = np.linalg.solve(model.factor_covariance, np.eye(len(premia)))
    b, d = model.betas.to_numpy(), model.residual_variances.to_numpy()
    h = precision + b.T @ (b / d[:, None])
    hb = np.linalg.solve(h, beta)
    return float((hb @ precision @ premia)**2 / (candidate_variance + beta @ hb))


def _tracking_error(risk: qis.RiskModel, reference: pd.Series, weights: pd.Series, date: pd.Timestamp) -> float:
    """Delegate risk to QIS, using canonical coordinates for cancellation at a PSD null.

    QIS's covariance quadratic may become slightly negative through cancellation
    for a signed, exactly zero-risk book. The fitted covariance is validated upstream; an eigendecomposition preserves
    null directions and rejects material indefiniteness.
    QIS then evaluates the same book in uncorrelated risk coordinates.
    """
    with np.errstate(invalid='ignore'):
        value = risk.compute_tre_at_date(reference, weights, date)
    if np.isfinite(value):
        return value
    covariance = risk.covar[date]
    # QIS has no stable null-direction factorization. This fallback only removes
    # eigensolver roundoff after FactorLasso has validated the fitted covariance.
    values, vectors = np.linalg.eigh(covariance)
    if values.min() < -1e-10 * max(1., float(np.abs(values).max())):
        raise ValueError('covariance is materially indefinite')
    root = vectors * np.sqrt(np.maximum(values, 0.))
    positions = pd.Series(root.T @ (weights-reference).to_numpy(), index=covariance.index)
    coordinates = qis.RiskModel(covar={date: pd.DataFrame(
        np.eye(len(positions)), index=positions.index, columns=positions.index)})
    return coordinates.compute_tre_at_date(positions*0., positions, date)


def portfolio_opportunity_metrics(model: FactorOpportunityModel, weights: pd.Series, reference: pd.Series | None=None) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Report actual funding, exposures, risk and statistical-alpha attribution.

    Actual saved books need not be unit funded. Their funding is reported exactly;
    only the optimizer requires a long-only unit reference. No holdings are scaled.
    """
    _vector(weights, model.betas, 'weights', label_error=_ALIGN_ERROR)
    if reference is None:
        reference = weights * 0.
    _vector(reference, model.betas, 'reference', label_error=_ALIGN_ERROR)
    risk, date = model.risk_model, model.date
    delta = weights - reference
    factor_model = model.factor_risk_model
    f = factor_model.compute_exposures_at_date(weights, date)
    df = factor_model.compute_exposures_at_date(delta, date)
    active = factor_model.compute_tre_decomposition_at_date(reference, weights, date)
    absolute = factor_model.compute_tre_decomposition_at_date(weights * 0., weights, date)
    residual_te, residual_vol = active.residual_te, absolute.residual_te
    if model.residual_risk_model is not None:
        residual_te = _tracking_error(model.residual_risk_model, reference, weights, date)
        residual_vol = _tracking_error(model.residual_risk_model, weights * 0., weights, date)
    wf, wh, _ = sa.holdings_decomposition(weights, model.betas, model.residual_variances)
    _, h = gls_split(model.alpha, model.betas, model.residual_variances)
    joint = pd.DataFrame(np.column_stack([np.ones(len(weights)), model.betas]),
                         index=model.betas.index)
    joint_coefficients, hc = gls_split(model.alpha, joint, model.residual_variances)
    fm = qis.RiskModel(covar={date: model.factor_covariance})
    factor_risk = fm.compute_marginal_tre_at_date(f * 0., f, date)
    active_risk = fm.compute_marginal_tre_at_date(df * 0., df, date)
    rows = dict(net_exposure=float(weights.sum()), reference_net_exposure=float(reference.sum()),
                min_weight=float(weights.min()), max_weight=float(weights.max()),
                concentration=float(weights @ weights), alpha_level=float(model.alpha @ weights),
                active_l1=float(delta.abs().sum()),
                held_assets=int((weights.abs() > 1e-7).sum()),
                alpha_increment=float(model.alpha @ delta),
                orthogonal_alpha_increment=float(h @ delta),
                joint_orthogonal_alpha_increment=float(hc @ delta),
                joint_common_level_increment=float(joint_coefficients[0] * delta.sum()),
                joint_factor_aligned_alpha_increment=float(joint_coefficients[1:] @ df),
                factor_aligned_alpha_increment=float((model.alpha.to_numpy() - h) @ delta),
                total_te=float(_tracking_error(risk, reference, weights, date)),
                factor_te=float(active.factor_te), residual_te=float(residual_te),
                portfolio_vol=float(_tracking_error(risk, weights * 0., weights, date)),
                factor_variance=float(absolute.factor_te)**2,
                residual_variance=float(residual_vol)**2,
                gls_factor_book_vol=float(_tracking_error(
                    risk, weights * 0., pd.Series(wf, index=weights.index), date)),
                gls_neutral_book_vol=float(_tracking_error(
                    risk, weights * 0., pd.Series(wh, index=weights.index), date)))
    exposures = pd.DataFrame({'exposure': f, 'active_exposure': df,
                               'variance_contribution': factor_risk.mcte * absolute.factor_te,
                               'active_variance_contribution': active_risk.mcte * active.factor_te})
    holdings = pd.DataFrame({'weight': weights, 'reference': reference, 'active_weight': delta,
                             'w_factor': wf, 'w_neutral': wh})
    holdings.attrs['gls_geometry'] = 'diagonal_residual_reference'
    exposures.attrs['decomposition'] = 'diagonal_residual_reference'
    exposures.attrs['total_covariance'] = 'full_selected_covariance'
    return rows, exposures, holdings
