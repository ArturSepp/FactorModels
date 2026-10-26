"""Optional composed example: completed FM model -> OP solve -> existing QIS risk.

Install OP >=9.0.1 alongside FM >=0.3.0 for this example. Neither package
imports the other in its library. All data and forecasts here are synthetic.
"""
import numpy as np
import pandas as pd

from reference_workflows import run_local_test as fit_example


def run_local_test(case: int = 1) -> dict:
    """Inspect a supplied-input optimum and its QIS tracking error without refitting."""
    if case != 1:
        raise ValueError("Only the synthetic composed case 1 is available")
    from optimalportfolios.optimization.opportunity import OpportunityModel, solve_opportunity
    from optimalportfolios.optimization.portfolio_result import PortfolioOptimisationResult

    snapshot = fit_example()["snapshot"]
    assets = snapshot.asset_covariance.index
    alpha = pd.Series([.02, .01, .015, -.005], index=assets)
    benchmark = pd.Series(1. / len(assets), index=assets)
    supplied = OpportunityModel(covariance=snapshot.asset_covariance, alpha=alpha,
                                date=snapshot.cutoff)
    solution = solve_opportunity(supplied, benchmark, total_te=.04, solver="CLARABEL")
    assert solution.accepted and solution.weights is not None, solution.status
    np.testing.assert_allclose(solution.weights.sum(), 1., atol=1e-7)
    assert solution.weights.min() >= -1e-7 and solution.max_constraint_violation <= 1e-7
    risk = snapshot.to_risk_model()
    tracking_error = risk.compute_tre_at_date(benchmark, solution.weights, snapshot.cutoff)
    assert 0 < tracking_error <= .04 + 1e-7
    result = PortfolioOptimisationResult(weights=solution.weights, benchmark_weights=benchmark,
                                         risk_model=risk, risk_model_date=snapshot.cutoff)
    print({"status": solution.status, "annual_tracking_error": tracking_error,
           "objective": solution.objective_value})
    return {"snapshot": snapshot, "solution": solution, "portfolio": result,
            "risk_model": risk, "tracking_error": tracking_error}


if __name__ == "__main__":
    run_local_test()
