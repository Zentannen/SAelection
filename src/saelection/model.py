"""Bayesian spatial model of VD-level vote shares, fitted with variational inference (NumPyro).

For VD i and category k (DA, ANC, ACTIONSA, MK, EFF, OTHER; ABSTAIN is the reference):

    z_ik   ~ Normal(eta_ik, sqrt(v_ik))         z = empirical alr of the VD result, v = its sampling variance
    eta_ik = A_k + delta_{m(i),k}               provincial level + municipal swing
           + b1_k * X1_ik + b2_k * X2_ik        past performance (previous LGE, previous national/provincial)
           + phi_ik                             VD residual with spatial covariance
    delta_mk ~ Normal(0, sigma_delta_k)
           + sum_j C1_kj X1_ij + C2_kj X2_ij    substitution: past strength of ideologically
                                                similar parties j (pairs in config/ideology.json)
           + psi_ik                             within-VD residual, correlated across parties:
    psi_i.   ~ MVN(0, diag(tau) Omega diag(tau)),  Omega ~ LKJ(2)   (negative = substitution)
    g_i      common VD turnout shock (added to all six alrs), also Leroux CAR
    phi_.k, g ~ Leroux CAR on the touching-VD graph:
               precision Q_k = [ (1 - lambda_k) I + lambda_k (D - W) ] / sigma_k^2

W is the contiguity matrix (VDs whose polygons touch), D its row sums. lambda_k in [0, 1] measures
how much the residuals of touching VDs co-vary (0 = independent VDs, -> 1 = intrinsic CAR). With
spatial[k] = False the term is dropped (lambda_k = 0). Parties are conditionally independent, so
per-party spatial on/off comparisons are exact.

Inference: SVI with a mean-field Normal guide (AutoNormal). log|Q| is computed exactly from the
eigenvalues of the graph Laplacian.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
from numpyro.infer import SVI, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoNormal
from scipy.sparse import coo_matrix

from .features import K, Design, alr, alr_var
from .data import PARTIES, ROOT


def similarity() -> np.ndarray:
    """Ideological similarity matrix (K, K) from config/ideology.json (zero diagonal)."""
    import json
    cfg = json.loads((ROOT / "config" / "ideology.json").read_text())
    idx = {p: i for i, p in enumerate(PARTIES)}
    S = np.zeros((K, K), np.float32)
    for a, b, w in cfg["pairs"]:
        S[idx[a], idx[b]] = S[idx[b], idx[a]] = w
    return S


# ----------------------------------------------------------------------------- graph helpers
def laplacian(edges: np.ndarray, n: int) -> np.ndarray:
    a = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).toarray()
    a = a + a.T
    return np.diag(a.sum(1)) - a


_EIG: dict = {}


def graph_eigen(edges: np.ndarray, n: int):
    """Eigen-decomposition of the touching-VD graph Laplacian (cached)."""
    key = (n, len(edges), int(edges.sum()))
    if key not in _EIG:
        mu, U = np.linalg.eigh(laplacian(edges, n))
        _EIG[key] = (np.clip(mu, 0.0, None), U)
    return _EIG[key]


# ----------------------------------------------------------------------------- model
def model(X1, X2, muni, n_muni, edges, mu, spatial, S, z=None, v=None, obs_mask=None, priors=None):
    n = X1.shape[0]
    p = priors or {}
    S = jnp.asarray(S)

    A = numpyro.sample("A", p.get("A", dist.Normal(0.0, 3.0).expand([K]).to_event(1)))
    b1 = numpyro.sample("b1", p.get("b1", dist.Normal(0.5, 1.0).expand([K]).to_event(1)))
    b2 = numpyro.sample("b2", p.get("b2", dist.Normal(0.5, 1.0).expand([K]).to_event(1)))
    sig_d = numpyro.sample("sig_d", p.get("sig_d", dist.HalfNormal(0.5).expand([K]).to_event(1)))
    # spatial fields: one per party + one common VD turnout factor (added to every party's alr)
    J = K + 1
    sigma = numpyro.sample("sigma", p.get("sigma", dist.HalfNormal(1.0).expand([J]).to_event(1)))
    lam = numpyro.sample("lam", p.get("lam", dist.Beta(1.0, 1.0).expand([J]).to_event(1)))
    lam = jnp.where(spatial, lam, 0.0)

    # substitution: past strength of ideologically similar parties (mask S, zero diagonal)
    C1 = numpyro.sample("C1", p.get("C1", dist.Normal(0.0, 1.0).expand([K, K]).to_event(2))) * 0.5 * S
    C2 = numpyro.sample("C2", p.get("C2", dist.Normal(0.0, 1.0).expand([K, K]).to_event(2))) * 0.5 * S
    # correlated within-VD party residuals (LKJ): negative correlations = substitution
    tau = numpyro.sample("tau", p.get("tau", dist.HalfNormal(0.5).expand([K]).to_event(1)))
    L = numpyro.sample("L_omega", p.get("L_omega", dist.LKJCholesky(K, 2.0)))
    psi = numpyro.sample("psi", dist.MultivariateNormal(jnp.zeros(K), scale_tril=tau[:, None] * L)
                         .expand([n]).to_event(1))

    d_raw = numpyro.sample("d_raw", dist.Normal(0.0, 1.0).expand([n_muni, K]).to_event(2))
    delta = d_raw * sig_d

    # Leroux CAR prior on phi (n, K+1), written as an explicit log density
    phi = numpyro.sample("phi", dist.ImproperUniform(dist.constraints.real_vector, (), (n, J)))
    diff2 = jnp.sum((phi[edges[:, 0]] - phi[edges[:, 1]]) ** 2, axis=0)          # (J,)
    quad = ((1.0 - lam) * jnp.sum(phi ** 2, axis=0) + lam * diff2) / sigma ** 2
    logdet = jnp.sum(jnp.log((1.0 - lam)[None, :] + lam[None, :] * mu[:, None]), axis=0) \
        - 2.0 * n * jnp.log(sigma)
    numpyro.factor("car", jnp.sum(0.5 * logdet - 0.5 * quad))

    # a field whose spatial term is switched off is removed entirely (its VD-level variation is
    # then carried by psi, which keeps the cross-party correlation)
    phi_on = phi * jnp.where(spatial, 1.0, 0.0)
    eta = A + delta[muni] + b1 * X1 + b2 * X2 + X1 @ C1.T + X2 @ C2.T + phi_on[:, :K] + phi_on[:, K:] + psi
    if z is not None:
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("z", dist.Normal(eta, jnp.sqrt(v)), obs=z)


def _init(site=None, n=None):
    if site is None:
        return partial(_init, n=n)
    if site["name"] == "phi":
        return jnp.zeros((n, K + 1), jnp.float32)
    if site["name"] == "psi":
        return jnp.zeros((n, K), jnp.float32)
    return init_to_median(site)


def fit(d: Design, spatial=None, obs_vd_mask=None, priors=None, seed=0, z=None, v=None,
        party_mask=None, num_draws=600, vi_steps=20000, vi_lr=0.02, verbose=False, method=None):
    """Fit to the design's target (or explicit z/v). Returns numpy posterior draws.

    method="vi" (default; env SAELECTION_METHOD) : SVI, mean-field Normal guide.
    method="nuts"                                : HMC/NUTS (exact but slow: ~35k latent dims;
                                                   use on Kaggle / a bigger machine)."""
    method = method or os.environ.get("SAELECTION_METHOD", "vi")
    # spatial: (K+1,) booleans = parties ..., common turnout factor
    spatial = np.ones(K + 1, bool) if spatial is None else np.asarray(spatial, bool)
    if len(spatial) == K:
        spatial = np.append(spatial, True)
    if z is None:
        z, v = alr(d.Y), alr_var(d.Y)
    obs = np.ones((d.n, K), bool)
    if obs_vd_mask is not None:
        obs &= obs_vd_mask[:, None]
    if party_mask is None:
        party_mask = d.contested if d.contested is not None else np.ones(K, bool)
    obs &= party_mask[None, :]
    obs &= np.isfinite(z)
    z = np.where(np.isfinite(z), z, 0.0)
    mu, _ = graph_eigen(d.edges, d.n)

    kw = dict(X1=jnp.asarray(d.X1, jnp.float32), X2=jnp.asarray(d.X2, jnp.float32),
              muni=jnp.asarray(d.muni), n_muni=len(d.muni_codes), edges=jnp.asarray(d.edges),
              mu=jnp.asarray(mu, jnp.float32), spatial=jnp.asarray(spatial), S=jnp.asarray(similarity()),
              z=jnp.asarray(z, jnp.float32), v=jnp.asarray(v, jnp.float32), obs_mask=jnp.asarray(obs),
              priors=priors)
    if method == "nuts":
        from numpyro.infer import MCMC, NUTS
        nchains = int(os.environ.get("SAELECTION_CHAINS", 4))
        mcmc = MCMC(NUTS(model, init_strategy=_init(n=d.n), target_accept_prob=0.85, max_tree_depth=8),
                    num_warmup=int(os.environ.get("SAELECTION_WARMUP", 500)), num_samples=max(1, num_draws // nchains),
                    num_chains=nchains, chain_method="parallel", progress_bar=verbose)
        mcmc.run(jax.random.PRNGKey(seed), **kw)
        s = {k: np.asarray(x) for k, x in mcmc.get_samples().items()}
        s["lam"] = np.where(spatial, s["lam"], 0.0)
        s["_elbo_final"] = float("nan")
        s["_diverging"] = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
        s["_spatial"] = spatial
        return s
    guide = AutoNormal(model, init_loc_fn=_init(n=d.n), init_scale=0.02)
    opt = optax.chain(optax.clip_by_global_norm(100.0),
                      optax.adam(optax.cosine_decay_schedule(vi_lr, vi_steps, alpha=0.02)))
    svi = SVI(model, guide, opt, Trace_ELBO(num_particles=1))
    res = svi.run(jax.random.PRNGKey(seed), vi_steps, progress_bar=verbose, **kw)
    draws = guide.sample_posterior(jax.random.PRNGKey(seed + 1), res.params, sample_shape=(num_draws,))
    s = {k: np.asarray(x) for k, x in draws.items() if k != "z"}
    s["lam"] = np.where(spatial, s["lam"], 0.0)
    loss = np.asarray(res.losses)
    s["_elbo_trace"] = -loss[:: max(1, len(loss) // 200)]
    s["_elbo_final"] = float(-loss[-500:].mean())
    s["_spatial"] = spatial
    return s


# ----------------------------------------------------------------------------- posterior helpers
def fixed_part(s, d: Design):
    """Everything in eta except the VD residuals phi + psi: (S, n, K)."""
    Sm = 0.5 * similarity()
    delta = s["d_raw"] * s["sig_d"][:, None, :]
    cross = (np.einsum("nj,skj->snk", d.X1, s["C1"] * Sm) + np.einsum("nj,skj->snk", d.X2, s["C2"] * Sm))
    return (s["A"][:, None, :] + delta[:, d.muni, :] + s["b1"][:, None, :] * d.X1
            + s["b2"][:, None, :] * d.X2 + cross)


def eta_from_samples(s, d: Design):
    ph = s["phi"] * np.asarray(s["_spatial"], float)
    return fixed_part(s, d) + ph[..., :K] + ph[..., K:] + s["psi"]


def omega(s):
    """Posterior draws of the within-VD cross-party residual correlation matrix."""
    L = s["L_omega"]
    return np.einsum("sij,skj->sik", L, L)


def pointwise_lpd(s, d: Design):
    """log p(z_ik | posterior), (n, K), integrating over the posterior draws."""
    z, v = alr(d.Y), alr_var(d.Y)
    eta = eta_from_samples(s, d)
    lp = -0.5 * (np.log(2 * np.pi * v)[None] + (z[None] - eta) ** 2 / v[None])
    return np.logaddexp.reduce(lp, axis=0) - np.log(lp.shape[0])


def sample_car(sigma, lam, edges, n, rng):
    """Draws from the Leroux CAR prior (exact, via the Laplacian eigenbasis).

    sigma, lam: (S, K) hyper-parameter draws. Returns (S, n, K)."""
    mu, U = graph_eigen(edges, n)
    S, Kk = sigma.shape
    out = np.empty((S, n, Kk), np.float32)
    for k in range(Kk):
        sd = sigma[:, k, None] / np.sqrt((1 - lam[:, k, None]) + lam[:, k, None] * mu[None, :])  # (S, n)
        out[:, :, k] = (rng.standard_normal((S, n)) * sd) @ U.T
    return out


def substitution_corr(Om: np.ndarray, tau=None) -> np.ndarray:
    """Partial correlations of the within-VD party residuals: the correlation between parties k
    and j holding the other parties (and so the shared turnout shock) fixed.
    Negative entries = substitution (one gains where the other loses)."""
    P = np.linalg.inv(Om)
    d = np.sqrt(np.diag(P))
    pc = -P / np.outer(d, d)
    np.fill_diagonal(pc, 1.0)
    return pc


# ----------------------------------------------------------------------------- exact CAR conditionals
def car_precision(edges: np.ndarray, n: int, sigma: float, lam: float) -> np.ndarray:
    return ((1.0 - lam) * np.eye(n) + lam * laplacian(edges, n)) / sigma ** 2


class CarConditional:
    """Exact Gaussian conditional of a Leroux CAR field on the nodes H given the nodes O:
        phi_H | phi_O ~ N( -Q_HH^{-1} Q_HO phi_O ,  Q_HH^{-1} ).
    Used for VDs with no data (held-out / not yet counted) instead of the mean-field VI
    approximation, which understates their uncertainty."""

    def __init__(self, edges, n, sigma, lam, H):
        H = np.asarray(H, bool)
        Q = car_precision(edges, n, sigma, lam)
        self.H, self.O = H, ~H
        Qhh = Q[np.ix_(H, H)]
        self.Lc = np.linalg.cholesky(Qhh)
        self.B = -np.linalg.solve(Qhh, Q[np.ix_(H, self.O)]) if self.O.any() else None
        Linv = np.linalg.solve(self.Lc, np.eye(H.sum()))
        self.var = np.sum(Linv ** 2, axis=0)            # diag(Q_HH^{-1})

    def mean(self, phi_O):                               # phi_O: (S, |O|) -> (S, |H|)
        if self.B is None:
            return np.zeros((phi_O.shape[0], self.H.sum()))
        return phi_O @ self.B.T

    def sample(self, phi_O, rng):
        eps = rng.standard_normal((phi_O.shape[0], self.H.sum()))
        # x = L^{-T} eps has covariance (L L^T)^{-1} = Q_HH^{-1}
        from scipy.linalg import solve_triangular
        noise = solve_triangular(self.Lc, eps.T, lower=True, trans="T").T
        return self.mean(phi_O) + noise


def heldout_lpd_exact(s, d: Design, held: np.ndarray):
    """Held-out log predictive density per party, (|H|, K), using the exact CAR conditional of
    each spatial field given the fitted fields at the observed VDs, and the exact marginal of
    the iid cross-party residual psi."""
    z, v = alr(d.Y), alr_var(d.Y)
    J = s["sigma"].shape[1]
    cm, cv = [], []
    on = np.asarray(s["_spatial"], bool)
    for j in range(J):
        if not on[j]:
            cm.append(np.zeros((s["phi"].shape[0], int(held.sum()))))
            cv.append(np.zeros(int(held.sum())))
            continue
        cc = CarConditional(d.edges, d.n, float(s["sigma"][:, j].mean()), float(s["lam"][:, j].mean()), held)
        cm.append(cc.mean(s["phi"][:, ~held, j]))
        cv.append(cc.var)
    fixed = fixed_part(s, d)[:, held, :]                                 # (S, H, K)
    mean = fixed + np.stack(cm[:K], -1) + cm[K][..., None]
    var = (v[held] + np.stack(cv[:K], -1) + cv[K][:, None] + (s["tau"].mean(0) ** 2)[None])[None]
    lp = -0.5 * (np.log(2 * np.pi * var) + (z[held][None] - mean) ** 2 / var)
    return np.logaddexp.reduce(lp, axis=0) - np.log(lp.shape[0])
