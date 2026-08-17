#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/*
 * Residual-based cyclic coordinate descent for
 *
 *   0.5 / n * ||y - X beta||^2 + lambda * sum_j rho(|beta_j|),
 *
 * where rho(t) = b*t + 0.5*sum_k w_k*relu(t-knot_k)^2 on
 * the canonical bounded domain 0 <= t <= knot[K-1]. X is supplied in
 * column-major feature layout: X[j*n + i] = X_row_major[i, j].
 */

static double segment_rho(double t, double h, double d, double c) {
    return c + d * t + 0.5 * h * t * t;
}

static double candidate_value(
    double t,
    double a,
    double u,
    double lambda,
    double h,
    double d,
    double c
) {
    const double delta = t - u;
    return 0.5 * a * delta * delta + lambda * segment_rho(t, h, d, c);
}

static double coordinate_minimizer(
    double a,
    double z,
    double lambda,
    const double *knots,
    const double *seg_h,
    const double *seg_d,
    const double *seg_c,
    int knot_count,
    double zero_tol
) {
    if (!(a > 1e-15) || !isfinite(a) || !isfinite(z)) {
        return 0.0;
    }
    const double u = fabs(z);
    double best_t = 0.0;
    double best_value = 0.5 * a * u * u;

    for (int segment = 0; segment < knot_count; ++segment) {
        const double lo = segment == 0 ? 0.0 : knots[segment - 1];
        const double hi = knots[segment];
        const double h = seg_h[segment];
        const double d = seg_d[segment];
        const double c = seg_c[segment];

        const double endpoint_lo = candidate_value(lo, a, u, lambda, h, d, c);
        if (isfinite(endpoint_lo) && endpoint_lo < best_value) {
            best_value = endpoint_lo;
            best_t = lo;
        }
        const double endpoint_hi = candidate_value(hi, a, u, lambda, h, d, c);
        if (isfinite(endpoint_hi) && endpoint_hi < best_value) {
            best_value = endpoint_hi;
            best_t = hi;
        }

        const double denominator = a + lambda * h;
        if (denominator > 1e-15 && isfinite(denominator)) {
            const double stationary = (a * u - lambda * d) / denominator;
            if (stationary >= lo && stationary <= hi && isfinite(stationary)) {
                const double value = candidate_value(
                    stationary, a, u, lambda, h, d, c
                );
                if (isfinite(value) && value < best_value) {
                    best_value = value;
                    best_t = stationary;
                }
            }
        }
    }

    if (best_t <= zero_tol) {
        return 0.0;
    }
    return signbit(z) ? -best_t : best_t;
}

static int build_segments(
    const double *knots,
    const double *weights,
    double b,
    int knot_count,
    double *seg_h,
    double *seg_d,
    double *seg_c
) {
    if (knot_count < 2 || fabs(knots[0]) > 1e-12) {
        return -1;
    }
    double sum_w = 0.0;
    double sum_wk = 0.0;
    double sum_wk2 = 0.0;
    for (int segment = 0; segment < knot_count; ++segment) {
        if (segment > 0) {
            const int active = segment - 1;
            const double weight = weights[active];
            const double knot = knots[active];
            sum_w += weight;
            sum_wk += weight * knot;
            sum_wk2 += weight * knot * knot;
        }
        if (!isfinite(knots[segment]) ||
            (segment > 0 && !(knots[segment] > knots[segment - 1]))) {
            return -2;
        }
        seg_h[segment] = sum_w;
        seg_d[segment] = b - sum_wk;
        seg_c[segment] = 0.5 * sum_wk2;
    }
    return 0;
}

int canonical_cd_solve(
    const double *x_columns,
    const double *y,
    int n,
    int p,
    double lambda,
    const double *knots,
    double b,
    const double *weights,
    int knot_count,
    const double *beta_init,
    int max_sweeps,
    double tolerance,
    double zero_tol,
    double *beta_out,
    int *sweeps_out,
    double *last_max_update_out,
    double *fixed_point_max_out
) {
    if (x_columns == NULL || y == NULL || knots == NULL || weights == NULL ||
        beta_init == NULL || beta_out == NULL || sweeps_out == NULL ||
        last_max_update_out == NULL || fixed_point_max_out == NULL ||
        n <= 0 || p <= 0 || knot_count < 2 || max_sweeps <= 0 ||
        !(lambda >= 0.0) || !(tolerance > 0.0) || !(zero_tol >= 0.0)) {
        return -10;
    }

    double *residual = (double *)malloc((size_t)n * sizeof(double));
    double *diagonal = (double *)malloc((size_t)p * sizeof(double));
    double *seg_h = (double *)malloc((size_t)knot_count * sizeof(double));
    double *seg_d = (double *)malloc((size_t)knot_count * sizeof(double));
    double *seg_c = (double *)malloc((size_t)knot_count * sizeof(double));
    if (residual == NULL || diagonal == NULL || seg_h == NULL ||
        seg_d == NULL || seg_c == NULL) {
        free(residual);
        free(diagonal);
        free(seg_h);
        free(seg_d);
        free(seg_c);
        return -11;
    }
    const int segment_status = build_segments(
        knots, weights, b, knot_count, seg_h, seg_d, seg_c
    );
    if (segment_status != 0) {
        free(residual);
        free(diagonal);
        free(seg_h);
        free(seg_d);
        free(seg_c);
        return -20 + segment_status;
    }

    const double bound = knots[knot_count - 1];
    memcpy(residual, y, (size_t)n * sizeof(double));
    for (int feature = 0; feature < p; ++feature) {
        const double *column = x_columns + (size_t)feature * (size_t)n;
        double norm2 = 0.0;
        for (int row = 0; row < n; ++row) {
            norm2 += column[row] * column[row];
        }
        diagonal[feature] = norm2 / (double)n;
        double value = beta_init[feature];
        if (!isfinite(value)) {
            free(residual);
            free(diagonal);
            free(seg_h);
            free(seg_d);
            free(seg_c);
            return -30;
        }
        value = fmax(-bound, fmin(bound, value));
        if (fabs(value) <= zero_tol || diagonal[feature] <= 1e-15) {
            value = 0.0;
        }
        beta_out[feature] = value;
        if (value != 0.0) {
            for (int row = 0; row < n; ++row) {
                residual[row] -= column[row] * value;
            }
        }
    }

    int converged = 0;
    int sweeps = 0;
    double last_max_update = INFINITY;
    for (sweeps = 1; sweeps <= max_sweeps; ++sweeps) {
        double max_update = 0.0;
        double max_abs_beta = 0.0;
        for (int feature = 0; feature < p; ++feature) {
            const double a = diagonal[feature];
            const double old_value = beta_out[feature];
            double new_value = 0.0;
            if (a > 1e-15) {
                const double *column = x_columns + (size_t)feature * (size_t)n;
                double dot = 0.0;
                for (int row = 0; row < n; ++row) {
                    dot += column[row] * residual[row];
                }
                const double z = old_value + dot / ((double)n * a);
                new_value = coordinate_minimizer(
                    a, z, lambda, knots, seg_h, seg_d, seg_c,
                    knot_count, zero_tol
                );
                const double update = new_value - old_value;
                if (update != 0.0) {
                    for (int row = 0; row < n; ++row) {
                        residual[row] -= column[row] * update;
                    }
                }
                if (fabs(update) > max_update) {
                    max_update = fabs(update);
                }
            } else if (old_value != 0.0) {
                max_update = fmax(max_update, fabs(old_value));
            }
            beta_out[feature] = new_value;
            max_abs_beta = fmax(max_abs_beta, fabs(new_value));
        }
        last_max_update = max_update;
        if (max_update <= tolerance * fmax(1.0, max_abs_beta)) {
            converged = 1;
            break;
        }
    }
    if (!converged) {
        sweeps = max_sweeps;
    }

    double fixed_point_max = 0.0;
    for (int feature = 0; feature < p; ++feature) {
        const double a = diagonal[feature];
        const double old_value = beta_out[feature];
        double coordinate_solution = 0.0;
        if (a > 1e-15) {
            const double *column = x_columns + (size_t)feature * (size_t)n;
            double dot = 0.0;
            for (int row = 0; row < n; ++row) {
                dot += column[row] * residual[row];
            }
            const double z = old_value + dot / ((double)n * a);
            coordinate_solution = coordinate_minimizer(
                a, z, lambda, knots, seg_h, seg_d, seg_c,
                knot_count, zero_tol
            );
        }
        fixed_point_max = fmax(
            fixed_point_max, fabs(coordinate_solution - old_value)
        );
    }

    *sweeps_out = sweeps;
    *last_max_update_out = last_max_update;
    *fixed_point_max_out = fixed_point_max;
    free(residual);
    free(diagonal);
    free(seg_h);
    free(seg_d);
    free(seg_c);
    return converged ? 0 : 1;
}
