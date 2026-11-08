"""Independent terminal cash flows and correlated-block references for SIMPLE equity risk."""
import numpy as np
import pandas as pd
import pytest
import qis

from factormodels.analytics.models import FactorModelSnapshot
from factormodels.estimation.equity_payoff import (
    EquityPayoffRiskSpec, assemble_equity_payoff_risk, build_funded_currency_exposure,
    compute_payoff_factor_returns,
)


def fixture():
    """Create changing cash and hedges with a deliberately shrunk, nonorthogonal fit."""
    rng=np.random.default_rng(20261108)
    dates=pd.date_range('2018-01-31',periods=80,freq='ME')
    assets=pd.Index(['EU','BR','US','CH'])
    ccys=pd.Series(['EUR','BRL','USD','EUR'],index=assets)
    spot=pd.DataFrame(rng.normal(0,.06,(80,2)),index=dates,columns=['EUR','BRL'])
    cash=pd.DataFrame({'EUR':np.linspace(-.002,.01,80),'BRL':np.linspace(.005,.03,80),
                       'USD':np.linspace(.001,.005,80)},index=dates)
    local=pd.DataFrame(rng.normal(.01,.05,(80,4)),index=dates,columns=assets)
    local['EU']+=.5*spot.EUR
    local['BR']+=.8*spot.BRL
    phi=spot+1-(1+cash.USD).to_numpy()[:,None]/(1+cash[['EUR','BRL']])
    p=pd.DataFrame({a:(local[a]-cash[ccys[a]])*(1+spot[ccys[a]] if ccys[a]!='USD' else 1)
                    for a in assets})
    weights={name:pd.DataFrame([w]*80,index=dates,columns=assets)
             for name,w in {'Market':[.25]*4,'Style':[.5,-.5,0,0]}.items()}
    x=compute_payoff_factor_returns(p,weights)
    b=pd.DataFrame(.7*np.linalg.lstsq(x.to_numpy(),p.to_numpy(),rcond=None)[0].T,
                   index=assets,columns=x.columns)
    residual=p-x@b.T
    f=x.cov()*12;r=residual.cov()*12
    snapshot=FactorModelSnapshot(b,f,r,b@f@b.T+r,b@f@b.T+r,
        pd.DataFrame({'residual_var':np.diag(r)},index=assets),cutoff=dates[-1],
        residual_type='empirical',factor_return_types={c:qis.ReturnTypes.DIFFERENCE for c in x},
        provenance={'factor_returns_freq':'ME','asset_return_type':'simple_payoff'})
    hs={};direct=[]
    for i,date in enumerate(dates):
        h=pd.Series([0.,.5,1.,1+cash.loc[date,'EUR']],index=assets)
        local_cash=pd.Series({a:cash.loc[date,ccys[a]] for a in assets})
        hs[date]=build_funded_currency_exposure(ccys,h,local_cash,spot.columns,
                                               reference_ccy='USD',cutoff=date)
        row=[]
        for a in assets:
            c=ccys[a];s=1+spot.loc[date,c] if c!='USD' else 1.
            forward=(1+cash.loc[date,'USD'])/(1+cash.loc[date,c])
            # Terminal asset plus short opening-principal forward, less reference loan.
            row.append((1+local.loc[date,a])*s+h[a]*(forward-s)-(1+cash.loc[date,'USD']))
        direct.append(row)
    return dict(payoff_model=snapshot,factor_payoffs=x,translated_payoffs=p,
                fx_forward_payoffs=phi,forecast_exposures=hs[dates[-1]],historical_exposures=hs),pd.DataFrame(direct,index=dates,columns=assets)


def assemble(parts=None, spec=None, **options):
    """Apply explicit cutoff, cash horizon, coordinates and common-sample settings."""
    parts=fixture()[0] if parts is None else parts
    cutoff=parts['payoff_model'].cutoff
    return assemble_equity_payoff_risk(**parts,
        spec=spec or EquityPayoffRiskSpec('ME',window=80,min_observations=30),
        reference_ccy='USD',forecast_end=cutoff+pd.offsets.MonthEnd(),
        metadata_available_date=options.get('metadata_available_date',cutoff),
        data_available_date=options.get('data_available_date',cutoff))


def test_full_covariance_matches_independent_simple_cashflows_with_dated_exposures():
    """Both projections retain every cross block; the dated one matches actual USD cash flows."""
    parts,direct=fixture();result=assemble(parts)
    np.testing.assert_allclose(result.reconstructed_returns,direct,atol=1e-14)
    np.testing.assert_allclose(result.historical_covariance,np.cov(direct,rowvar=False)*12,atol=1e-14)
    fixed=parts['translated_payoffs']+parts['fx_forward_payoffs']@parts['forecast_exposures'].T
    np.testing.assert_allclose(result.covariance,np.cov(fixed,rowvar=False)*12,atol=1e-14)
    assert result.diagnostics['fixed_vs_dated_covariance_max_abs']>1e-5
    assert result.diagnostics['factor_residual_effect_frobenius']>1e-3
    assert result.diagnostics['residual_fx_effect_frobenius']>1e-3
    assert result.provenance['response_coordinate']=='translated_local_excess_simple'
    assert result.provenance['research_only']
    pd.testing.assert_frame_equal(result.to_risk_model().covar[result.cutoff],result.covariance)
    assert result.to_risk_model().factor_covar is None


def test_cash_aware_exposure_reference_zero_and_debt_maturity_cancels_currency_leg():
    """Principal hedging leaves local cash exposure; hedging the debt maturity removes it."""
    ccys=pd.Series({'A':'EUR','B':'USD'})
    cash=pd.Series({'A':.02,'B':.01})
    h=build_funded_currency_exposure(ccys,1.,cash,['EUR'],reference_ccy='USD',cutoff='2020-01-31')
    np.testing.assert_allclose(h.to_numpy(),[[.02],[0.]])
    debt=build_funded_currency_exposure(ccys,1+cash,cash,['EUR'],reference_ccy='USD',cutoff='2020-01-31')
    np.testing.assert_allclose(debt,0.,atol=1e-16)
    with pytest.raises(ValueError,match='positive gross'):
        build_funded_currency_exposure(ccys,0.,cash*0-1,['EUR'],reference_ccy='USD',cutoff='2020-01-31')


def test_ewma_projection_matches_independent_recursion():
    """The joint projection commutes with the same linear uncentered EWMA recursion."""
    parts,direct=fixture()
    spec=EquityPayoffRiskSpec('ME',80,30,'ewma',20,'none')
    result=assemble(parts,spec)
    expected=np.zeros((4,4));decay=1-2/21
    for row in direct.to_numpy():
        expected=decay*expected+(1-decay)*np.outer(row,row)
    np.testing.assert_allclose(result.historical_covariance,expected*12,atol=1e-14)


def test_one_complete_case_mask_and_owned_results():
    """Missing returns remove a whole joint row and copies cannot alter the result."""
    parts,direct=fixture();parts['translated_payoffs'].iloc[8,0]=np.nan
    result=assemble(parts)
    expected=direct.drop(index=direct.index[8])
    np.testing.assert_allclose(result.historical_covariance,expected.cov()*12,atol=1e-14)
    assert result.diagnostics['complete_rows']==79
    original=result.covariance
    copied=result.covariance
    copied.iloc[:]=0.
    result.provenance['coordinate']='LOG'
    pd.testing.assert_frame_equal(result.covariance,original)
    assert result.provenance['coordinate']=='reference_simple_excess'


def test_future_histories_do_not_change_the_completed_snapshot():
    """Future rows are sliced before common-sample selection and moments."""
    parts,_=fixture();before=assemble(parts)
    future=pd.date_range(before.cutoff+pd.offsets.MonthEnd(),periods=2,freq='ME')
    for name in ('translated_payoffs','factor_payoffs','fx_forward_payoffs'):
        frame=parts[name]
        parts[name]=pd.concat([frame,pd.DataFrame(100.,index=future,columns=frame.columns)])
    after=assemble(parts)
    pd.testing.assert_frame_equal(before.covariance,after.covariance,check_exact=True)


@pytest.mark.parametrize('defect',['log','order','grid','missing_exposure','infinite','availability'])
def test_incompatible_inputs_fail(defect):
    """Reject coordinate, timing, availability and labelling mistakes explicitly."""
    parts,_=fixture();options={}
    if defect=='log': parts['payoff_model']._fields['factor_return_types']['Market']=qis.ReturnTypes.LOG
    elif defect=='order': parts['forecast_exposures']=parts['forecast_exposures'].iloc[::-1]
    elif defect=='grid': parts['factor_payoffs']=parts['factor_payoffs'].iloc[1:]
    elif defect=='missing_exposure': parts['historical_exposures'].pop(parts['factor_payoffs'].index[3])
    elif defect=='infinite': parts['fx_forward_payoffs'].iloc[4,0]=np.inf
    else: options['metadata_available_date']=pd.Timestamp('2040-01-01')
    with pytest.raises(ValueError): assemble(parts,**options)


def test_arithmetic_baskets_reject_misaligned_weights_and_preserve_missing_payoffs():
    """Basket construction uses declared opening notionals without logs or renormalisation."""
    parts,_=fixture();p=parts['translated_payoffs']
    w=pd.DataFrame(.25,index=p.index,columns=p.columns)
    x=compute_payoff_factor_returns(p,{'Market':w})
    np.testing.assert_allclose(x.Market,p.sum(axis=1)/4)
    p.iloc[7,0]=np.nan
    assert np.isnan(compute_payoff_factor_returns(p,{'Market':w}).iloc[7,0])
    with pytest.raises(ValueError,match='exact date/asset'):
        compute_payoff_factor_returns(p,{'Market':w.iloc[::-1].sort_index().iloc[1:]})


def test_payoff_fit_coordinate_is_explicit_and_mixed_cadences_are_rejected():
    """A log-labelled fit cannot be reused as a SIMPLE-payoff fit by relabelling the output."""
    from factorlasso import LassoModel, LassoModelType
    from factormodels.estimation.estimator import FactorCovarEstimator
    parts,_=fixture()
    parts['payoff_model']._fields['provenance']['asset_return_type']='log'
    with pytest.raises(ValueError,match='declare asset_return_type'):
        assemble(parts)
    estimator=FactorCovarEstimator(lasso_model=LassoModel(model_type=LassoModelType.LASSO),
                                   factor_returns_freq='ME',asset_return_type='simple_payoff')
    with pytest.raises(ValueError,match='one common'):
        estimator.fit_current(parts['factor_payoffs'].cumsum(),
            {'QE':parts['translated_payoffs']},cutoff=parts['payoff_model'].cutoff)
    with pytest.raises(ValueError,match='asset_return_type'):
        FactorCovarEstimator(lasso_model=LassoModel(),asset_return_type='implicit')
