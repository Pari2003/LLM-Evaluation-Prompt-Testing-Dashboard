"""
Statistical Significance Testing.

Implements Welch's t-test and Cohen's d effect size from scratch using
Python stdlib (no scipy required). This fills a gap in existing LLM
evaluation tools — Promptfoo and DeepEval show raw scores but don't
test whether differences between prompt variants are statistically
significant or just noise.

Why this matters: with 3-5 repetitions per prompt variant, the difference
between variant A (score 0.87) and variant B (score 0.85) could easily be
random variation. Without significance testing, you might deploy the
wrong prompt.

Usage:
    from src.evaluation.significance import welch_t_test, cohens_d

    result = welch_t_test([0.87, 0.85, 0.89], [0.82, 0.80, 0.84])
    # result = {"t_statistic": 2.45, "p_value": 0.04, "significant": True, ...}
"""

from __future__ import annotations

import math
import statistics


def welch_t_test(
    sample_a: list[float],
    sample_b: list[float],
    alpha: float = 0.05,
) -> dict[str, float | bool]:
    """Perform Welch's t-test (unequal variance t-test) between two samples.

    Welch's t-test is appropriate when:
    - Sample sizes may differ
    - Variances may differ
    - Both conditions hold in prompt evaluation (different prompts may
      have very different variance in their scores)

    Args:
        sample_a: Composite scores for variant A.
        sample_b: Composite scores for variant B.
        alpha: Significance level (default 0.05).

    Returns:
        Dict with t_statistic, degrees_of_freedom, p_value, and significant.
    """
    n_a = len(sample_a)
    n_b = len(sample_b)

    # Need at least 2 samples per group for variance
    if n_a < 2 or n_b < 2:
        return {
            "t_statistic": 0.0,
            "degrees_of_freedom": 0.0,
            "p_value": 1.0,
            "significant": False,
        }

    mean_a = statistics.mean(sample_a)
    mean_b = statistics.mean(sample_b)
    var_a = statistics.variance(sample_a)
    var_b = statistics.variance(sample_b)

    # Handle zero-variance edge case
    se_a = var_a / n_a
    se_b = var_b / n_b
    se_sum = se_a + se_b

    if se_sum == 0:
        # Both samples have zero variance — scores are identical within each group
        if abs(mean_a - mean_b) < 1e-10:
            return {
                "t_statistic": 0.0,
                "degrees_of_freedom": float(n_a + n_b - 2),
                "p_value": 1.0,
                "significant": False,
            }
        else:
            # Different means with zero variance = perfectly significant
            return {
                "t_statistic": float("inf"),
                "degrees_of_freedom": float(n_a + n_b - 2),
                "p_value": 0.0,
                "significant": True,
            }

    # Welch's t-statistic
    t_stat = (mean_a - mean_b) / math.sqrt(se_sum)

    # Welch-Satterthwaite degrees of freedom
    df_num = se_sum**2
    df_denom = (se_a**2) / (n_a - 1) + (se_b**2) / (n_b - 1)
    df = df_num / df_denom if df_denom > 0 else 1.0

    # Two-tailed p-value from t-distribution
    p_value = _t_distribution_p_value(abs(t_stat), df)

    return {
        "t_statistic": round(t_stat, 4),
        "degrees_of_freedom": round(df, 2),
        "p_value": round(p_value, 6),
        "significant": p_value < alpha,
    }


def cohens_d(sample_a: list[float], sample_b: list[float]) -> tuple[float, str]:
    """Compute Cohen's d effect size between two samples.

    Cohen's d tells you the *magnitude* of the difference:
    - |d| < 0.2: negligible (the difference is trivially small)
    - 0.2 ≤ |d| < 0.5: small
    - 0.5 ≤ |d| < 0.8: medium
    - |d| ≥ 0.8: large (strong practical significance)

    Args:
        sample_a: Scores for variant A.
        sample_b: Scores for variant B.

    Returns:
        Tuple of (d_value, magnitude_string).
    """
    n_a = len(sample_a)
    n_b = len(sample_b)

    if n_a < 2 or n_b < 2:
        return 0.0, "negligible"

    mean_a = statistics.mean(sample_a)
    mean_b = statistics.mean(sample_b)
    var_a = statistics.variance(sample_a)
    var_b = statistics.variance(sample_b)

    # Pooled standard deviation
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_sd = math.sqrt(pooled_var) if pooled_var > 0 else 1e-10

    d = (mean_a - mean_b) / pooled_sd

    # Classify magnitude
    abs_d = abs(d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    return round(d, 4), magnitude


# ─── Internal: t-distribution p-value computation ─────────────────────────
# Implemented from scratch using the regularized incomplete beta function.
# This avoids requiring scipy while maintaining numerical accuracy.


def _t_distribution_p_value(t_abs: float, df: float) -> float:
    """Compute two-tailed p-value for |t| with df degrees of freedom.

    Uses the relationship between the t-distribution CDF and the
    regularized incomplete beta function:
        p = I(df / (df + t²), df/2, 1/2)

    Args:
        t_abs: Absolute value of the t-statistic.
        df: Degrees of freedom.

    Returns:
        Two-tailed p-value in [0, 1].
    """
    if df <= 0:
        return 1.0
    if t_abs == 0:
        return 1.0

    x = df / (df + t_abs * t_abs)
    p = _regularized_incomplete_beta(x, df / 2.0, 0.5)
    return max(0.0, min(1.0, p))


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Compute the regularized incomplete beta function I_x(a, b).

    Uses the continued fraction expansion (Lentz's method) which converges
    quickly for typical t-test parameters.

    Args:
        x: Upper limit of integration (0 ≤ x ≤ 1).
        a: First shape parameter (> 0).
        b: Second shape parameter (> 0).

    Returns:
        I_x(a, b) in [0, 1].
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use symmetry relation for better convergence
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _regularized_incomplete_beta(1.0 - x, b, a)

    # Log of the front factor: x^a * (1-x)^b / (a * B(a,b))
    log_front = a * math.log(x) + b * math.log(1.0 - x) - math.log(a) - _log_beta(a, b)

    # Continued fraction (Lentz's method)
    cf = _continued_fraction_beta(x, a, b)
    result = math.exp(log_front) * cf
    return max(0.0, min(1.0, result))


def _log_beta(a: float, b: float) -> float:
    """Log of the beta function: log(B(a, b)) = log(Γ(a)) + log(Γ(b)) - log(Γ(a+b))."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _continued_fraction_beta(
    x: float, a: float, b: float, max_iter: int = 200, tol: float = 1e-10
) -> float:
    """Evaluate the continued fraction for the incomplete beta function.

    Uses the modified Lentz's algorithm for numerical stability.
    """
    tiny = 1e-30
    f = tiny
    c = f
    d = 0.0

    for m in range(max_iter):
        if m == 0:
            numerator = 1.0
        else:
            k = m
            if k % 2 == 0:
                # Even terms
                j = k // 2
                numerator = (j * (b - j) * x) / ((a + 2 * j - 1) * (a + 2 * j))
            else:
                # Odd terms
                j = (k - 1) // 2
                numerator = -((a + j) * (a + b + j) * x) / ((a + 2 * j) * (a + 2 * j + 1))

        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d

        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny

        delta = c * d
        f *= delta

        if abs(delta - 1.0) < tol:
            break

    return f
