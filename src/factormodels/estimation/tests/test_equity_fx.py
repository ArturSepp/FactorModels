"""Independent covariance, timing and selected-residual references for equity FX augmentation."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import qis

from factormodels.analytics.models import FactorModelSnapshot
from factormodels.estimation.equity_fx import (
    EquityFxRiskSpec, augment_equity_fx_model, build_currency_exposure,
)


def fixture(residual_scale=1.0):
    """Construct correlated residuals orthogonal to mixed native factors on one common sample."""
    rng = np.random.default_rng(4417)
    dates = pd.date_range('2018-01-31', periods=80, freq='ME')
    x = rng.normal(0, .025, (80, 2))
    e = rng.normal(0, .02, (80, 4))
    basis = np.column_stack([np.ones(80), x])
    e -= basis @ np.linalg.lstsq(basis, e, rcond=None)[0]
    e[:, 1] += .5 * e[:, 0]
    b = pd.DataFrame([[1.,.2],[.8,-.3],[1.3,.1],[.6,.4]],
                      index=['EU','BR','US','CH'],columns=['Market','Style'])
    y = pd.DataFrame(x @ b.to_numpy().T + e,index=dates,columns=b.index)
    f = pd.DataFrame(np.cov(x,rowvar=False)*12,index=b.columns,columns=b.columns)
    r = pd.DataFrame(np.cov(e,rowvar=False)*12*residual_scale,index=b.index,columns=b.index)
    covariance = b @ f @ b.T+r
    snapshot = FactorModelSnapshot(b,f,r,covariance,
        b@f@b.T+pd.DataFrame(np.diag(np.diag(r)),index=b.index,columns=b.index),
        pd.DataFrame({'residual_var':np.diag(r)},index=b.index),cutoff=dates[-1],
        residual_type='empirical',factor_return_types={'Market':qis.ReturnTypes.LOG,'Style':qis.ReturnTypes.DIFFERENCE},
        provenance={'factor_returns_freq':'ME','source':'synthetic independent covariance reference'})
    phi = pd.DataFrame({'EUR': .4*x[:,0]+.6*e[:,0]+rng.normal(0,.01,80),
                        'BRL': -.2*x[:,1]+1.4*e[:,1]+rng.normal(0,.02,80)},index=dates)
    h = build_currency_exposure(pd.Series({'EU':'EUR','BR':'BRL','US':'USD','CH':'EUR'}),
                                 pd.Series({'EU':0.,'BR':0.,'US':0.,'CH':.5}),
                                 ['EUR','BRL'],reference_ccy='USD',cutoff=dates[-1])
    return snapshot,pd.DataFrame(x,index=dates,columns=b.columns),y,phi,h


def augment(parts=None, policy='preserve_local', **kwargs):
    """Use explicitly declared units and availability in the reference fixture."""
    snapshot,x,y,phi,h = fixture() if parts is None else parts
    spec=kwargs.pop('spec',EquityFxRiskSpec('ME',policy,window=80,min_observations=30))
    return augment_equity_fx_model(snapshot,x,y,phi,h,spec=spec,
        factor_return_types=kwargs.pop('factor_return_types',snapshot.factor_return_types),
        reference_ccy='USD',metadata_available_date=kwargs.pop('metadata_available_date',snapshot.cutoff),
        data_available_date=kwargs.pop('data_available_date',snapshot.cutoff),**kwargs)


def test_exact_common_sample_reference_preserves_full_selected_residual():
    """With coherent marginals, augmented covariance equals a separately formed return covariance."""
    parts=fixture();snapshot,x,y,phi,h=parts
    result=augment(parts)
    independently_translated=y.to_numpy()+phi.to_numpy()@h.to_numpy().T
    np.testing.assert_allclose(result.covariance,np.cov(independently_translated,rowvar=False)*12,rtol=1e-12,atol=1e-14)
    pd.testing.assert_frame_equal(result.residual_covariance,snapshot.residual_covariance)
    np.testing.assert_allclose(result.local_covariance,snapshot.asset_covariance,rtol=1e-13,atol=1e-15)
    assert result.residual_covariance.iloc[0,1] != 0
    assert result.diagnostics['cross_covariance_scale']==1.
    assert result.provenance['annualization_factor']==12
    assert result.sample_index.equals(x.index)
    assert result.provenance['fx_coordinate']=='arithmetic_forward_payoff_per_opening_reference_notional'


def test_residual_fx_omission_is_measurable_and_qis_risk_keeps_full_covariance():
    """An omitted residual/FX block changes active risk even when F and R are unchanged."""
    result=augment();h=fixture()[-1]
    c=result.residual_fx_covariance.to_numpy()
    np.testing.assert_allclose(result.covariance-result.without_residual_fx,c@h.to_numpy().T+h.to_numpy()@c.T)
    w=pd.Series([.1,.7,.1,.1],index=h.index);benchmark=pd.Series(.25,index=h.index)
    risk=result.to_risk_model()
    got=risk.compute_tre_at_date(benchmark,w,result.cutoff)
    d=(w-benchmark).to_numpy()
    expected=np.sqrt(sum(d[i]*result.covariance.iloc[i,j]*d[j] for i in range(4) for j in range(4)))
    assert got==pytest.approx(expected)
    assert abs(got/result.to_risk_model(include_residual_fx=False).compute_tre_at_date(benchmark,w,result.cutoff)-1)>.1
    assert risk.factor_covar is None and risk.residual_vars is None
    assert risk.factor_return_types['FX::BRL'] is qis.ReturnTypes.DIFFERENCE
    pd.testing.assert_frame_equal(risk.covar[result.cutoff],result.covariance)


def test_preserve_rejects_incompatible_cross_blocks_unless_scaling_is_explicit():
    """Smaller selected residual variance cannot silently retain impossible historical correlations."""
    parts=fixture(.01)
    with pytest.raises(ValueError,match='Incompatible preserved joint covariance'):
        augment(parts)
    spec=EquityFxRiskSpec('ME','preserve_local',window=80,min_observations=30,cross_covariance_policy='shrink_to_psd')
    result=augment(parts,spec=spec)
    assert 0<result.diagnostics['cross_covariance_scale']<1
    pd.testing.assert_frame_equal(result.residual_covariance,parts[0].residual_covariance)
    np.testing.assert_allclose(result.local_covariance,parts[0].asset_covariance,atol=1e-14)
    assert np.linalg.eigvalsh(result.joint_covariance).min()>-1e-12
    factor_block=result.joint_covariance.loc['factor','factor']
    pd.testing.assert_frame_equal(factor_block,parts[0].factor_covariance)


def test_joint_reestimate_records_local_block_change_and_matches_return_covariance():
    """Joint estimation changes selected marginals explicitly; it does not claim preservation."""
    parts=fixture(.2)
    result=augment(parts,policy='joint_reestimate')
    expected=np.cov(parts[2].to_numpy()+parts[3].to_numpy()@parts[4].to_numpy().T,rowvar=False)*12
    np.testing.assert_allclose(result.covariance,expected,rtol=1e-12,atol=1e-14)
    assert result.diagnostics['local_block_max_abs_change']>1e-3
    assert result.diagnostics['residual_block_max_abs_change']>1e-3
    assert result.provenance['local_residual_policy']=='empirical'


def test_future_returns_do_not_change_a_historical_snapshot():
    """Large post-cutoff data are never included in fitting moments."""
    parts=list(fixture());before=augment(parts)
    future=pd.date_range(parts[0].cutoff+pd.offsets.MonthEnd(),periods=4,freq='ME')
    for i in (1,2,3):
        parts[i]=pd.concat([parts[i],pd.DataFrame(100.,index=future,columns=parts[i].columns)],sort=False)
    after=augment(parts)
    pd.testing.assert_frame_equal(before.covariance,after.covariance)
    assert before.provenance==after.provenance


def test_currency_and_hedge_metadata_are_asof_and_reference_is_zero():
    """Currency spells and additional hedges use information known for the next interval."""
    dates=pd.to_datetime(['2020-01-15','2020-02-10','2021-01-01'])
    ccys=pd.DataFrame({'A':['EUR','BRL','EUR'],'B':['USD','USD','EUR']},index=dates)
    hedges=pd.DataFrame({'A':[0.,.25,1.],'B':[0.,0.,0.]},index=dates)
    h=build_currency_exposure(ccys,hedges,['EUR','BRL'],reference_ccy='USD',cutoff=pd.Timestamp('2020-02-29'))
    np.testing.assert_array_equal(h,[[0.,.75],[0.,0.]])
    with pytest.raises(ValueError,match='unavailable'):
        build_currency_exposure(ccys,hedges,['EUR','BRL'],reference_ccy='USD',cutoff=pd.Timestamp('2019-12-31'))
    with pytest.raises(ValueError,match='Missing FX'):
        build_currency_exposure(ccys,hedges,['EUR'],reference_ccy='USD',cutoff=pd.Timestamp('2020-02-29'))


@pytest.mark.parametrize('option',['metadata_available_date','data_available_date'])
def test_unavailable_inputs_fail(option):
    """Declared availability after the decision date must fail rather than being backdated."""
    with pytest.raises(ValueError,match='available at cutoff'):
        augment(**{option:pd.Timestamp('2030-01-01')})


def test_native_coordinates_cannot_be_relabelled():
    """A point-difference style factor must remain DIFFERENCE in the local covariance block."""
    with pytest.raises(ValueError,match='coordinates disagree'):
        augment(factor_return_types={'Market':qis.ReturnTypes.LOG,'Style':qis.ReturnTypes.LOG})


@pytest.mark.parametrize('change',['order','grid','infinite','frequency','missing','metadata'])
def test_invalid_or_insufficient_histories_fail(change):
    """No silent relabelling, date intersection, fill or pairwise-covariance fallback is allowed."""
    parts=list(fixture());spec=None
    if change=='order': parts[1]=parts[1].iloc[:,::-1]
    elif change=='grid': parts[3]=parts[3].iloc[1:]
    elif change=='infinite': parts[2].iloc[5,0]=np.inf
    elif change=='frequency': spec=EquityFxRiskSpec('QE','joint_reestimate',window=80,min_observations=30)
    elif change=='missing': parts[3].iloc[:60]=np.nan
    else: parts[4]=parts[4].iloc[::-1]
    with pytest.raises(ValueError):
        augment(parts,policy='joint_reestimate',**({} if spec is None else {'spec':spec}))


def test_one_common_missing_mask_is_used_for_every_moment():
    """Missing a single FX or equity observation removes that row from every block."""
    parts=list(fixture());parts[3].iloc[4,0]=np.nan;parts[2].iloc[6,2]=np.nan
    result=augment(parts,policy='joint_reestimate')
    keep=np.ones(80,dtype=bool);keep[[4,6]]=False
    values=parts[2].to_numpy()[keep]+parts[3].to_numpy()[keep]@parts[4].to_numpy().T
    np.testing.assert_allclose(result.covariance,np.cov(values,rowvar=False)*12,rtol=1e-12,atol=1e-14)
    assert result.diagnostics['complete_rows']==78
    assert result.diagnostics['dropped_dates']==parts[1].index[[4,6]].tolist()


def test_result_owns_all_public_data():
    """Caller edits cannot change stored covariance, provenance, masks or loadings."""
    result=augment();original=result.covariance
    for name in ('covariance','joint_covariance','raw_joint_covariance','residual_covariance','loadings'):
        copied = getattr(result,name)
        copied.iloc[:]=0.
    result.diagnostics['cross_covariance_scale']=0.
    result.provenance['local_model']['source']='changed'
    pd.testing.assert_frame_equal(result.covariance,original)
    assert result.diagnostics['cross_covariance_scale']==1.
    assert result.provenance['local_model']['source']!='changed'


def test_ewma_matches_an_independent_recursion_on_completed_linear_returns():
    """Annualisation and demean timing agree with a separately evaluated covariance recursion."""
    parts=fixture();spec=EquityFxRiskSpec('ME','joint_reestimate',window=80,min_observations=30,
                                       estimator='ewma',span=20,mean_adjustment='none')
    got=augment(parts,spec=spec)
    values=parts[2].to_numpy()+parts[3].to_numpy()@parts[4].to_numpy().T
    expected=np.zeros((4,4));decay=1-2/21
    for row in values: expected=decay*expected+(1-decay)*np.outer(row,row)
    np.testing.assert_allclose(got.covariance,expected*12,rtol=1e-12,atol=1e-14)
    centered=augment(parts,spec=replace(spec,mean_adjustment='ewma'))
    assert not centered.covariance.equals(got.covariance)


@pytest.mark.parametrize('kwargs',[
    {'covariance_policy':'unknown'}, {'window':1}, {'min_observations':90},
    {'estimator':'ewma'}, {'cross_covariance_policy':'silent_clip'},
    {'covariance_policy':'joint_reestimate','cross_covariance_policy':'shrink_to_psd'},
])
def test_invalid_specifications_fail(kwargs):
    """Every unimplemented or contradictory estimation policy fails at configuration time."""
    options={'frequency':'ME','covariance_policy':'preserve_local',**kwargs}
    with pytest.raises(ValueError): EquityFxRiskSpec(**options)

def test_zero_residual_support_cannot_hide_a_nonzero_fx_cross_block():
    """Scaling is not permission to invent variance in a zero-risk residual direction."""
    spec=EquityFxRiskSpec('ME','preserve_local',window=80,min_observations=30,cross_covariance_policy='shrink_to_psd')
    with pytest.raises(ValueError,match='outside the preserved marginal support'):
        augment(fixture(0.),spec=spec)


def test_mixed_local_cadences_are_rejected_before_augmentation():
    """A multi-cadence fitted block must not masquerade as a monthly equity fit."""
    parts=list(fixture())
    parts[0]._fields['provenance']['asset_observation_dates']={'ME':parts[0].cutoff,'QE':parts[0].cutoff}
    with pytest.raises(ValueError,match='responses must use the augmentation cadence'):
        augment(parts)


def test_observation_cannot_postdate_declared_data_availability():
    """An available snapshot cannot include a later training observation."""
    with pytest.raises(ValueError,match='observations exceed declared availability'):
        augment(data_available_date=fixture()[0].cutoff-pd.Timedelta(days=1))
