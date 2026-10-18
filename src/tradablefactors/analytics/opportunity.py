"""Long-only opportunity diagnostics on a frozen, annualized factor model.

Statistical alpha describes longer-horizon residual means, not tactical forecasts.
The caller owns estimation, currency, membership and source provenance. Invalid
assets must be resolved before construction; nothing is floored or normalized.
Risk reporting delegates to QIS. Statistical outputs and prepared optimisation
arrays are complete TF results; portfolio solving belongs to downstream consumers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from factorlasso import CurrentFactorCovarData, ResidualCorrelationData, ResidualType

from tradablefactors.risk import build_risk_model
from tradablefactors.premia import sharpe_accounting as sa


@dataclass(frozen=True)
class FactorOpportunityModel:
    """Owned labelled inputs and a canonical risk model; treat frames as immutable.

    Residual variances must be strictly positive. A joint asset axis may include
    reference-only instruments; an investability mask controls candidate holdings.
    All moments and alpha use annualized decimal units in one reference currency.
    Optional prepared residual correlation changes portfolio risk only; dispersion
    and GLS attribution retain the explicitly diagonal-D reference geometry.
    """

    betas: pd.DataFrame
    residual_variances: pd.Series
    factor_covariance: pd.DataFrame
    alpha: pd.Series
    date: pd.Timestamp = pd.Timestamp('2000-01-01')
    residual_correlation: ResidualCorrelationData | None = None
    residual_corr_weight: float = 1.
    risk_model: object = field(init=False, repr=False, compare=False)
    factor_risk_model: object = field(init=False, repr=False, compare=False)
    residual_risk_model: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Reject invalid axes/moments, copy inputs, and build canonical risk."""
        for name, kind in [('betas', pd.DataFrame), ('residual_variances', pd.Series),
                           ('factor_covariance', pd.DataFrame), ('alpha', pd.Series)]:
            value = getattr(self, name)
            if not isinstance(value, kind):
                raise ValueError(f'{name} must be a labelled {kind.__name__}')
            object.__setattr__(self, name, value.copy(deep=True))
        b, d, f = self.betas, self.residual_variances, self.factor_covariance
        sa._model(pd.Series(0., index=b.columns), f, b, d)
        sa._vector(self.alpha, b, 'alpha')
        date = pd.Timestamp(self.date)
        if pd.isna(date):
            raise ValueError('date must be finite')
        object.__setattr__(self, 'date', date)
        current = CurrentFactorCovarData(x_covar=f, y_betas=b,
            y_variances=pd.DataFrame({'residual_var': d}),
            residual_correlation=self.residual_correlation)
        diagonal = build_risk_model({date: current}, residual_type='orthogonal')
        object.__setattr__(self, 'factor_risk_model', diagonal)
        kind = (ResidualType.EMPIRICAL if self.residual_correlation is not None
                else ResidualType.ORTHOGONAL)
        options = dict(residual_type=kind, residual_corr_weight=self.residual_corr_weight)
        residual = current.get_residual_covar(**options)
        if self.residual_correlation is not None:
            self.residual_correlation.get_corr(date=date, assets=b.index)
        correlated = self.residual_correlation is not None and self.residual_corr_weight > 0
        object.__setattr__(self, 'risk_model',
            build_risk_model({date: current.get_y_covar(**options)}) if correlated else diagonal)
        object.__setattr__(self, 'residual_risk_model',
            build_risk_model({date: residual}) if correlated else None)


    def optimisation_inputs(self):
        """Return copied labelled arrays for a downstream constrained optimiser.

        Full selected asset and residual covariance retain their risk policy.
        Orthogonal alpha uses the explicitly diagonal-D funding/factor geometry;
        consumers must not re-estimate it or infer it from covariance.
        """
        joint = pd.DataFrame(np.column_stack([np.ones(len(self.betas)), self.betas]),
                             index=self.betas.index)
        _, orthogonal = sa.gls_split(self.alpha, joint, self.residual_variances)
        residual = (self.residual_risk_model.covar[self.date]
                    if self.residual_risk_model is not None else
                    pd.DataFrame(np.diag(self.residual_variances),
                                 index=self.betas.index, columns=self.betas.index))
        return dict(covariance=self.risk_model.covar[self.date].copy(),
                    alpha=self.alpha.copy(), factor_loadings=self.betas.copy(),
                    factor_covariance=self.factor_covariance.copy(),
                    residual_covariance=residual.copy(),
                    orthogonal_alpha=pd.Series(orthogonal, index=self.alpha.index),
                    date=self.date)

    def prepare_alpha_covariance(self, covariance, *, orthogonal=False):
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
        projector = sa.gls_projector(joint, self.residual_variances)
        projected = projector @ covariance.to_numpy() @ projector.T
        return pd.DataFrame((projected+projected.T)/2,
                            index=self.alpha.index, columns=self.alpha.index)


def _alpha_dispersion_inputs(b, d, a):
    """Evaluate canonical dispersion without constructing unused portfolio risk models."""
    _, h = sa.gls_split(a, b, d)
    c = pd.DataFrame(np.column_stack([np.ones(len(b)), b]), index=b.index)
    _, hc = sa.gls_split(a, c, d)
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


def alpha_dispersion(model: FactorOpportunityModel):
    """Return D1-D6 capacities and asset-level GLS residuals on the supplied axis.

    K removes the joint span of funding and loadings, using canonical GLS.
    A+ is a positive-part residual-risk reference, not a full-model frontier.
    """
    return _alpha_dispersion_inputs(model.betas, model.residual_variances, model.alpha)


def factor_access_metrics(model: FactorOpportunityModel, premia: pd.Series | None = None):
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


def candidate_access_gain(model, premia, candidate_beta, candidate_variance):
    """Fixed-model systematic capacity gain for one independent-residual addition.

    Existing TF achievable_sharpe2 computes total capacity, but does not expose
    the Sherman-Morrison marginal-addition identity used here. No refit occurs.
    """
    sa._model(premia, model.factor_covariance, model.betas, model.residual_variances)
    sa._align(candidate_beta, model.betas.columns)
    beta = sa._array(candidate_beta, 'candidate_beta', 1)
    if len(beta) != len(premia) or not np.isfinite(candidate_variance) or candidate_variance <= 0:
        raise ValueError('candidate requires aligned loadings and positive residual variance')
    precision = np.linalg.solve(model.factor_covariance, np.eye(len(premia)))
    b, d = model.betas.to_numpy(), model.residual_variances.to_numpy()
    h = precision + b.T @ (b / d[:, None])
    hb = np.linalg.solve(h, beta)
    return float((hb @ precision @ premia)**2 / (candidate_variance + beta @ hb))


def _tracking_error(risk, reference, weights, date):
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
    coordinates = build_risk_model({date: pd.DataFrame(
        np.eye(len(positions)), index=positions.index, columns=positions.index)})
    return coordinates.compute_tre_at_date(positions*0., positions, date)


def portfolio_opportunity_metrics(model, weights, reference=None):
    """Report actual funding, exposures, risk and statistical-alpha attribution.

    Actual saved books need not be unit funded. Their funding is reported exactly;
    only the optimizer requires a long-only unit reference. No holdings are scaled.
    """
    sa._vector(weights, model.betas, 'weights')
    if reference is None:
        reference = weights * 0.
    sa._vector(reference, model.betas, 'reference')
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
    _, h = sa.gls_split(model.alpha, model.betas, model.residual_variances)
    joint = pd.DataFrame(np.column_stack([np.ones(len(weights)), model.betas]),
                         index=model.betas.index)
    joint_coefficients, hc = sa.gls_split(model.alpha, joint, model.residual_variances)
    fm = build_risk_model({date: model.factor_covariance})
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
