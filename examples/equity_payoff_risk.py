"""Offline SIMPLE equity payoff covariance; no provider, NAV simulation or production activation."""
import numpy as np
import pandas as pd
import qis
from factorlasso import LassoModel, LassoModelType
from factormodels import (
    FactorCovarEstimator, EquityPayoffRiskSpec, build_funded_currency_exposure,
    compute_payoff_factor_returns, assemble_equity_payoff_risk,
)


def run_local_test(case=1):
    """Fit arithmetic basket loadings and check full covariance against independent USD cash flows."""
    if case != 1:
        raise ValueError('Only case 1 is defined')
    rng=np.random.default_rng(20261108)
    dates=pd.date_range('2018-01-31',periods=81,freq='ME');grid=dates[1:]
    assets=pd.Index(['EU','BR','US','CH'])
    ccys=pd.Series(['EUR','BRL','USD','CHF'],index=assets)
    cash=pd.Series([.002,.01,.003,-.001],index=assets)
    local=pd.DataFrame(rng.normal(.01,.05,(80,4)),index=grid,columns=assets)
    spot=pd.DataFrame(rng.normal(0.,.04,(80,3)),index=grid,columns=['EUR','BRL','CHF'])
    spot['USD']=0.
    phi=pd.DataFrame({c:1+spot[c]-(1+cash['US'])/(1+cash[a])
                      for a,c in ccys.items() if c!='USD'})
    p=pd.DataFrame({a:(local[a]-cash[a])*(1+spot[c]) for a,c in ccys.items()})
    weights={'Market':pd.DataFrame(.25,index=grid,columns=assets),
             'Style':pd.DataFrame([[.5,-.5,0.,0.]]*80,index=grid,columns=assets)}
    x=compute_payoff_factor_returns(p,weights)
    levels=pd.concat([pd.DataFrame(0.,index=[dates[0]],columns=x.columns),x.cumsum()])
    fit=FactorCovarEstimator(lasso_model=LassoModel(model_type=LassoModelType.LASSO,
        reg_lambda=1e-5,span=36,demean=True,solver='CLARABEL'),factor_returns_freq='ME',
        factor_covar_span=36,residual_type='empirical',residual_covar_freq='ME',
        factor_return_types={c:qis.ReturnTypes.DIFFERENCE for c in x},
        asset_return_type='simple_payoff').fit_current(levels,{'ME':p},cutoff=grid[-1])
    h=build_funded_currency_exposure(ccys,.5,cash,phi.columns,reference_ccy='USD',cutoff=grid[-1])
    result=assemble_equity_payoff_risk(fit,x,p,phi,h,{d:h for d in grid},
        spec=EquityPayoffRiskSpec('ME',window=80,min_observations=60),reference_ccy='USD',
        forecast_end=grid[-1]+pd.offsets.MonthEnd(),metadata_available_date=grid[-1],data_available_date=grid[-1])
    direct=pd.DataFrame({a:(1+local[a])*(1+spot[c])+.5*((1+cash['US'])/(1+cash[a])-1-spot[c])
                         -(1+cash['US']) for a,c in ccys.items()})
    np.testing.assert_allclose(result.historical_covariance,direct.cov()*12,atol=1e-13)
    np.testing.assert_allclose(result.reconstructed_returns,direct,atol=1e-14)
    assert fit.provenance['asset_return_type']=='simple_payoff'
    assert result.provenance['research_only']
    print('SIMPLE payoff covariance reconciled:',result.covariance.shape)
    return result


if __name__=='__main__':
    run_local_test(case=1)
