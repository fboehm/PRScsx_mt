#!/usr/bin/env python

"""
Markov Chain Monte Carlo (MCMC) sampler for multi-trait, cross-population polygenic prediction
with continuous shrinkage (CS) priors — PRS-CSx-MT.

Extends PRS-CSx by adding trait-specific shrinkage modulators psi_trait on top of the shared
local shrinkage psi, with optional sample overlap correction between traits.

When n_trait=1, this reduces exactly to the original PRS-CSx MCMC sampler.

"""


import numpy as np
from scipy import linalg
from scipy.stats import invwishart
from numpy import random
import gigrnd


def _safe_inv(R, jitter=1e-8):
    """Invert a (small) symmetric PD matrix, adding jitter if needed."""
    D = R.shape[0]
    try:
        return linalg.inv(R, check_finite=False)
    except Exception:
        return linalg.inv(R + jitter*np.eye(D), check_finite=False)


def _sample_corr(scatter, dof, nu0=None, psi0=None):
    """
    Parameter-expanded inverse-Wishart draw of a correlation matrix.

    Samples a covariance W ~ Inv-Wishart(nu0 + dof, psi0 + scatter), then rescales
    to a correlation matrix (Talhouk, Doucet & Murphy 2012). `scatter` is the
    D x D sum of outer products of the (psi-scaled) standardized effects.
    """
    D = scatter.shape[0]
    if nu0 is None:
        nu0 = D + 1.0
    if psi0 is None:
        psi0 = np.eye(D)
    post_df = nu0 + dof
    post_scale = psi0 + scatter
    post_scale = 0.5*(post_scale + post_scale.T)           # symmetrize
    if D == 1:
        return np.ones((1, 1))
    W = np.atleast_2d(invwishart.rvs(df=post_df, scale=post_scale))
    d = np.sqrt(np.clip(np.diag(W), 1e-12, None))
    R = W/np.outer(d, d)
    R = 0.5*(R + R.T)
    np.fill_diagonal(R, 1.0)
    return R


def mcmc(a, b, phi, snp_dict, beta_mrg, frq_dict, idx_dict, n, ld_blk, blk_size,
         n_iter, n_burnin, thin, pop, chrom, out_dir, out_name, meta, write_pst, seed,
         n_trait=1, rho_e=None, lambda_psi=1.0, cross_trait='mult'):
    """
    Multi-trait MCMC sampler.

    Parameters
    ----------
    a, b : float
        Hyperparameters for the continuous shrinkage prior.
    phi : float or None
        Global shrinkage parameter. If None, estimated from data.
    snp_dict : dict
        Unified SNP information dictionary.
    beta_mrg : dict
        Marginal betas. Keyed by (pp, tt) for multi-trait or pp for single-trait.
    frq_dict : dict
        Allele frequencies. Keyed by (pp, tt) for multi-trait or pp for single-trait.
    idx_dict : dict
        SNP index mapping. Keyed by pp (shared across traits).
    n : dict or list
        GWAS sample sizes. Keyed by (pp, tt) for multi-trait or indexed by pp.
    ld_blk : dict
        LD blocks. Keyed by pp.
    blk_size : dict
        Block sizes. Keyed by pp.
    n_iter, n_burnin, thin : int
        MCMC iteration parameters.
    pop : list
        Population labels.
    chrom : int
        Chromosome number.
    out_dir, out_name : str
        Output directory and file name prefix.
    meta : str
        'TRUE' or 'FALSE' for meta-analysis.
    write_pst : str
        'TRUE' or 'FALSE' for writing posterior samples.
    seed : int or None
        Random seed.
    n_trait : int
        Number of traits (default 1 for backward compatibility).
    rho_e : dict or None
        Per-population error correlation matrices (n_trait x n_trait).
        Keyed by pp. Only used when n_trait > 1. None means no overlap.
    lambda_psi : float
        Hyperparameter for trait-specific shrinkage modulators (default 1.0).
        Only used by cross_trait='mult'.
    cross_trait : str
        Cross-trait sharing scheme for multi-trait mode:
          'mult' : multiplicative scalar modulators psi * psi_trait (original;
                   sign-agnostic). Default.
          'mvcs' : multivariate continuous-shrinkage. Per-ancestry T x T signed
                   correlation matrix R_pop learned in-sampler; effects across
                   traits share the global-local scale psi_j jointly through R_pop.
                   Cross-ancestry borrowing remains via shared psi_j (as PRS-CSx).
          'kron' : separable Kronecker prior R = R_trait (X) R_anc. Signed
                   correlation across BOTH traits and ancestries; both factors
                   learned in-sampler. Ragged ancestry SNP sets handled by
                   presence-pattern caching + data augmentation.
        Note (mvcs/kron): sigma^2 is updated with the per-trait marginal prior;
        the cross-condition correlation enters the beta and R updates exactly.
    """
    print('... MCMC ...')
    if n_trait > 1:
        print('... multi-trait mode: %d traits ...' % n_trait)

    # seed
    if seed != None:
        random.seed(seed)

    # derived stats
    n_pst = int((n_iter-n_burnin)/thin)
    n_pop = len(pop)
    p_tot = len(snp_dict['SNP'])

    # Per (pp, tt) derived quantities
    p = {}
    het = {}
    for pp in range(n_pop):
        if n_trait == 1:
            p[pp] = len(beta_mrg[pp])
            het[pp] = np.sqrt(2.0*frq_dict[pp]*(1.0-frq_dict[pp]))
        else:
            for tt in range(n_trait):
                p[(pp, tt)] = len(beta_mrg[(pp, tt)])
                het[(pp, tt)] = np.sqrt(2.0*frq_dict[(pp, tt)]*(1.0-frq_dict[(pp, tt)]))

    n_blk = {}
    blk_diag_idx = {}
    for pp in range(n_pop):
        n_blk[pp] = len(ld_blk[pp])
        blk_diag_idx[pp] = [np.arange(blk_size[pp][kk]) for kk in range(n_blk[pp])]

    # n_grp: count of populations each SNP appears in
    n_grp = np.zeros(p_tot)
    for jj in range(p_tot):
        for pp in range(n_pop):
            if jj in idx_dict[pp]:
                n_grp[jj] += 1

    # Get sample size helper
    def get_n(pp, tt=0):
        if n_trait == 1:
            return n[pp] if isinstance(n, (list, dict)) else n
        else:
            return n[(pp, tt)]

    def get_p(pp, tt=0):
        if n_trait == 1:
            return p[pp]
        else:
            return p[(pp, tt)]

    def get_het(pp, tt=0):
        if n_trait == 1:
            return het[pp]
        else:
            return het[(pp, tt)]

    def get_beta_mrg(pp, tt=0):
        if n_trait == 1:
            return beta_mrg[pp]
        else:
            return beta_mrg[(pp, tt)]

    # initialization
    beta = {}
    sigma = {}
    if n_trait == 1:
        for pp in range(n_pop):
            beta[pp] = np.zeros((p[pp], 1))
            sigma[pp] = 1.0
    else:
        for pp in range(n_pop):
            for tt in range(n_trait):
                beta[(pp, tt)] = np.zeros((p[(pp, tt)], 1))
                sigma[(pp, tt)] = 1.0

    psi = np.ones((p_tot, 1))

    # Trait-specific modulators (only when n_trait > 1 and cross_trait='mult')
    if n_trait > 1 and cross_trait == 'mult':
        psi_trait = np.ones((p_tot, n_trait))

    # Correlation matrices for the multivariate-CS / Kronecker priors
    if n_trait > 1 and cross_trait == 'mvcs':
        R_cross = {pp: np.eye(n_trait) for pp in range(n_pop)}        # per-ancestry T x T
        Rinv_cross = {pp: np.eye(n_trait) for pp in range(n_pop)}
        R_cross_est = {pp: np.zeros((n_trait, n_trait)) for pp in range(n_pop)}
    if n_trait > 1 and cross_trait == 'kron':
        R_trait = np.eye(n_trait)            # T x T cross-trait correlation
        R_anc = np.eye(n_pop)                # K x K cross-ancestry correlation
        R_trait_est = np.zeros((n_trait, n_trait))
        R_anc_est = np.zeros((n_pop, n_pop))
        # global standardized-effect buffer alpha_glob[g, pop, trait]; NaN where absent
        alpha_glob = np.full((p_tot, n_pop, n_trait), np.nan)
        # ancestry-presence pattern for each global SNP (which pops contain it)
        pop_sets = [set(int(g) for g in idx_dict[pp]) for pp in range(n_pop)]
        g2l = [{int(g): i for i, g in enumerate(idx_dict[pp])} for pp in range(n_pop)]
        present_pops = [tuple(pp for pp in range(n_pop) if g in pop_sets[pp])
                        for g in range(p_tot)]
        patterns = sorted(set(present_pops))
        # global indices grouped by pattern; and, per pop, local indices grouped by pattern
        pat_gidx = {P: np.array([g for g in range(p_tot) if present_pops[g] == P], dtype=int)
                    for P in patterns}
        pat_lidx = {(P, pp): np.array([g2l[pp][int(g)] for g in pat_gidx[P]], dtype=int)
                    for P in patterns for pp in P}

    if phi == None:
        phi = 1.0; phi_updt = True
    else:
        phi_updt = False

    # Posterior sample storage for meta-analysis
    if (write_pst == 'TRUE') and (meta == 'TRUE'):
        beta_pst = {}
        if n_trait == 1:
            for pp in range(n_pop):
                beta_pst[pp] = np.zeros((p[pp], n_pst))
        else:
            for pp in range(n_pop):
                for tt in range(n_trait):
                    beta_pst[(pp, tt)] = np.zeros((p[(pp, tt)], n_pst))

    # space allocation for posterior estimates
    beta_est = {}
    beta_sq_est = {}
    sigma_est = {}
    if n_trait == 1:
        for pp in range(n_pop):
            beta_est[pp] = np.zeros((p[pp], 1))
            beta_sq_est[pp] = np.zeros((p[pp], 1))
            sigma_est[pp] = 0.0
    else:
        for pp in range(n_pop):
            for tt in range(n_trait):
                beta_est[(pp, tt)] = np.zeros((p[(pp, tt)], 1))
                beta_sq_est[(pp, tt)] = np.zeros((p[(pp, tt)], 1))
                sigma_est[(pp, tt)] = 0.0

    psi_est = np.zeros((p_tot, 1))
    phi_est = 0.0
    if n_trait > 1 and cross_trait == 'mult':
        psi_trait_est = np.zeros((p_tot, n_trait))

    # MCMC
    qq = 0
    for itr in range(1, n_iter+1):
        if itr % 100 == 0:
            print('--- iter-' + str(itr) + ' ---')

        if n_trait == 1:
            # === SINGLE-TRAIT MODE: exact PRS-CSx logic ===
            for pp in range(n_pop):
                mm = 0; quad = 0.0
                psi_pp = psi[idx_dict[pp]]
                for kk in range(n_blk[pp]):
                    if blk_size[pp][kk] == 0:
                        continue
                    else:
                        idx_blk = range(mm, mm+blk_size[pp][kk])
                        didx = blk_diag_idx[pp][kk]
                        dinvt = ld_blk[pp][kk].copy()
                        dinvt[didx, didx] += 1.0/psi_pp[idx_blk].ravel()
                        dinvt_chol = linalg.cholesky(dinvt, overwrite_a=True, check_finite=False)
                        beta_tmp = linalg.solve_triangular(dinvt_chol, beta_mrg[pp][idx_blk], trans='T', check_finite=False) \
                                   + np.sqrt(sigma[pp]/n[pp])*random.randn(blk_size[pp][kk], 1)
                        beta[pp][idx_blk] = linalg.solve_triangular(dinvt_chol, beta_tmp, trans='N', check_finite=False)
                        Cbeta = dinvt_chol @ beta[pp][idx_blk]
                        quad += np.dot(Cbeta.ravel(), Cbeta.ravel())
                        mm += blk_size[pp][kk]

                err = max(n[pp]/2.0*(1.0-2.0*sum(beta[pp]*beta_mrg[pp])+quad), n[pp]/2.0*sum(beta[pp]**2/psi_pp))
                sigma[pp] = 1.0/random.gamma((n[pp]+p[pp])/2.0, 1.0/err)

            # delta sampling
            delta = random.gamma(a+b, 1.0/(psi+phi))

            # psi (shared) sampling
            xx = np.zeros((p_tot, 1))
            for pp in range(n_pop):
                xx[idx_dict[pp]] += n[pp]*beta[pp]**2/sigma[pp]

            for jj in range(p_tot):
                while True:
                    try:
                        psi[jj] = gigrnd.gigrnd(a-0.5*n_grp[jj], 2.0*delta[jj,0], xx[jj,0])
                    except:
                        continue
                    else:
                        break
            psi[psi>1] = 1.0

            # phi (global) sampling
            if phi_updt == True:
                w = random.gamma(1.0, 1.0/(phi+1.0))
                phi = random.gamma(p_tot*b+0.5, 1.0/(delta.sum()+w))

            # posterior accumulation
            if (itr > n_burnin) and (itr % thin == 0):
                for pp in range(n_pop):
                    beta_est[pp] = beta_est[pp] + beta[pp]/n_pst
                    beta_sq_est[pp] = beta_sq_est[pp] + beta[pp]**2/n_pst
                    sigma_est[pp] = sigma_est[pp] + sigma[pp]/n_pst

                psi_est = psi_est + psi/n_pst
                phi_est = phi_est + phi/n_pst

                if (write_pst == 'TRUE') and (meta == 'TRUE'):
                    for pp in range(n_pop):
                        beta_pst[pp][:,[qq]] = beta[pp]
                    qq += 1

        else:
            # ===== MULTI-TRAIT MODE =====
            if cross_trait == 'mult':
                # Step 1: Beta sampling — loop over (pp, tt)
                for pp in range(n_pop):
                    for tt in range(n_trait):
                        mm = 0; quad = 0.0
                        psi_pp = psi[idx_dict[pp]]
                        psi_t = psi_trait[idx_dict[pp], tt:tt+1]
                        EFF_PSI_FLOOR = 1e-20
                        eff_psi_raw = psi_pp * psi_t
                        n_clipped = int(np.sum(eff_psi_raw < EFF_PSI_FLOOR))
                        if n_clipped > 0:
                            print('WARNING: %d SNPs hit eff_psi floor (%.1e) at iter %d, pop %s, trait %d'
                                  % (n_clipped, EFF_PSI_FLOOR, itr, pop[pp], tt))
                        eff_psi = np.maximum(eff_psi_raw, EFF_PSI_FLOOR)

                        n_kt = get_n(pp, tt)
                        p_kt = get_p(pp, tt)
                        sigma_kt = sigma[(pp, tt)]

                        # Compute adjusted marginal betas for sample overlap
                        beta_mrg_adj = get_beta_mrg(pp, tt).copy()
                        if rho_e is not None and tt > 0:
                            for tt_prev in range(tt):
                                rho_et = rho_e[pp][tt, tt_prev]
                                if rho_et != 0.0:
                                    beta_mrg_adj -= rho_et * beta[(pp, tt_prev)]

                        # Compute effective sigma for overlap correction
                        sigma_eff = sigma_kt
                        if rho_e is not None and tt > 0:
                            rho_sq_sum = 0.0
                            for tt_prev in range(tt):
                                rho_sq_sum += rho_e[pp][tt, tt_prev]**2
                            sigma_eff = sigma_kt * max(1.0 - rho_sq_sum, 0.01)

                        for kk in range(n_blk[pp]):
                            if blk_size[pp][kk] == 0:
                                continue
                            else:
                                idx_blk = range(mm, mm+blk_size[pp][kk])
                                didx = blk_diag_idx[pp][kk]
                                dinvt = ld_blk[pp][kk].copy()
                                dinvt[didx, didx] += 1.0/eff_psi[idx_blk].ravel()
                                dinvt_chol = linalg.cholesky(dinvt, overwrite_a=True, check_finite=False)
                                beta_tmp = linalg.solve_triangular(dinvt_chol, beta_mrg_adj[idx_blk], trans='T', check_finite=False) \
                                           + np.sqrt(sigma_eff/n_kt)*random.randn(blk_size[pp][kk], 1)
                                beta[(pp, tt)][idx_blk] = linalg.solve_triangular(dinvt_chol, beta_tmp, trans='N', check_finite=False)
                                Cbeta = dinvt_chol @ beta[(pp, tt)][idx_blk]
                                quad += np.dot(Cbeta.ravel(), Cbeta.ravel())
                                mm += blk_size[pp][kk]

                        # Step 2: Sigma sampling per (pp, tt)
                        err = max(n_kt/2.0*(1.0-2.0*sum(beta[(pp, tt)]*get_beta_mrg(pp, tt))+quad),
                                  n_kt/2.0*sum(beta[(pp, tt)]**2/eff_psi))
                        sigma[(pp, tt)] = 1.0/random.gamma((n_kt+p_kt)/2.0, 1.0/err)

                # Step 3: Delta sampling
                delta = random.gamma(a+b, 1.0/(psi+phi))

                # Step 4: Psi (shared) sampling — accumulate across ALL pops AND traits
                xx = np.zeros((p_tot, 1))
                for pp in range(n_pop):
                    for tt in range(n_trait):
                        n_kt = get_n(pp, tt)
                        psi_t_inv = 1.0 / psi_trait[idx_dict[pp], tt:tt+1]
                        xx[idx_dict[pp]] += n_kt * beta[(pp, tt)]**2 / sigma[(pp, tt)] * psi_t_inv

                for jj in range(p_tot):
                    while True:
                        try:
                            psi[jj] = gigrnd.gigrnd(a-0.5*n_grp[jj]*n_trait, 2.0*delta[jj,0], xx[jj,0])
                        except:
                            continue
                        else:
                            break
                psi[psi>1] = 1.0

                # Step 5: Psi_trait sampling — per trait, accumulate across pops
                for tt in range(n_trait):
                    xx_trait = np.zeros((p_tot, 1))
                    for pp in range(n_pop):
                        n_kt = get_n(pp, tt)
                        psi_inv = 1.0 / psi[idx_dict[pp]]
                        xx_trait[idx_dict[pp]] += n_kt * beta[(pp, tt)]**2 / sigma[(pp, tt)] * psi_inv

                    for jj in range(p_tot):
                        while True:
                            try:
                                psi_trait[jj, tt] = gigrnd.gigrnd(-0.5*n_grp[jj], 2.0*lambda_psi, xx_trait[jj,0])
                            except:
                                continue
                            else:
                                break
                psi_trait[psi_trait>1] = 1.0

                # Step 6: Phi (global) sampling
                if phi_updt == True:
                    w = random.gamma(1.0, 1.0/(phi+1.0))
                    phi = random.gamma(p_tot*b+0.5, 1.0/(delta.sum()+w))

            elif cross_trait == 'mvcs':
                # ---- multivariate-CS: per-ancestry T x T signed correlation R_pop ----
                # standardized-effect scale s_{pp,tt} = sqrt(sigma/n)
                s = {(pp, tt): np.sqrt(sigma[(pp, tt)]/get_n(pp, tt))
                     for pp in range(n_pop) for tt in range(n_trait)}

                # (a) update R_pop via parameter-expanded inverse-Wishart
                for pp in range(n_pop):
                    psi_pp = psi[idx_dict[pp]]                                 # (p_pp,1)
                    A = np.hstack([beta[(pp, tt)]/s[(pp, tt)] for tt in range(n_trait)])  # (p_pp,T)
                    scat = (A/psi_pp).T @ A                                    # T x T scatter
                    R_cross[pp] = _sample_corr(scat, dof=get_p(pp), nu0=n_trait+1.0)
                    Rinv_cross[pp] = _safe_inv(R_cross[pp])

                # (b) trait-conditional Gibbs for beta, then per-trait sigma
                for pp in range(n_pop):
                    Rinv = Rinv_cross[pp]
                    psi_pp = psi[idx_dict[pp]]
                    for tt in range(n_trait):
                        n_kt = get_n(pp, tt); p_kt = get_p(pp, tt); sigma_kt = sigma[(pp, tt)]
                        s_t = s[(pp, tt)]
                        diag_add = Rinv[tt, tt]/psi_pp                          # (p_pp,1) precision addend
                        cross = np.zeros_like(psi_pp)
                        for uu in range(n_trait):
                            if uu == tt:
                                continue
                            cross += Rinv[tt, uu]*(beta[(pp, uu)]/s[(pp, uu)])
                        # signed conditional-mean contribution (canonical form)
                        beta_mrg_adj = get_beta_mrg(pp, tt).copy() - (s_t/psi_pp)*cross
                        sigma_eff = sigma_kt
                        if rho_e is not None and tt > 0:
                            for tt_prev in range(tt):
                                rho_et = rho_e[pp][tt, tt_prev]
                                if rho_et != 0.0:
                                    beta_mrg_adj -= rho_et*beta[(pp, tt_prev)]
                            rho_sq_sum = sum(rho_e[pp][tt, tp]**2 for tp in range(tt))
                            sigma_eff = sigma_kt*max(1.0 - rho_sq_sum, 0.01)
                        mm = 0; quad = 0.0
                        for kk in range(n_blk[pp]):
                            if blk_size[pp][kk] == 0:
                                continue
                            idx_blk = range(mm, mm+blk_size[pp][kk])
                            didx = blk_diag_idx[pp][kk]
                            dinvt = ld_blk[pp][kk].copy()
                            dinvt[didx, didx] += diag_add[idx_blk].ravel()
                            dinvt_chol = linalg.cholesky(dinvt, overwrite_a=True, check_finite=False)
                            beta_tmp = linalg.solve_triangular(dinvt_chol, beta_mrg_adj[idx_blk], trans='T', check_finite=False) \
                                       + np.sqrt(sigma_eff/n_kt)*random.randn(blk_size[pp][kk], 1)
                            beta[(pp, tt)][idx_blk] = linalg.solve_triangular(dinvt_chol, beta_tmp, trans='N', check_finite=False)
                            Cbeta = dinvt_chol @ beta[(pp, tt)][idx_blk]
                            quad += np.dot(Cbeta.ravel(), Cbeta.ravel())
                            mm += blk_size[pp][kk]
                        # sigma: per-trait marginal prior (see docstring note)
                        err = max(n_kt/2.0*(1.0-2.0*sum(beta[(pp, tt)]*get_beta_mrg(pp, tt))+quad),
                                  n_kt/2.0*sum(beta[(pp, tt)]**2/psi_pp))
                        sigma[(pp, tt)] = 1.0/random.gamma((n_kt+p_kt)/2.0, 1.0/err)

                # (c) delta, shared psi (GIG with q_j = alpha^T Rinv alpha), phi
                delta = random.gamma(a+b, 1.0/(psi+phi))
                xx = np.zeros((p_tot, 1))
                for pp in range(n_pop):
                    Rinv = Rinv_cross[pp]
                    A = np.hstack([beta[(pp, tt)]/np.sqrt(sigma[(pp, tt)]/get_n(pp, tt))
                                   for tt in range(n_trait)])
                    q = np.einsum('it,tu,iu->i', A, Rinv, A)
                    xx[idx_dict[pp]] += q.reshape(-1, 1)
                for jj in range(p_tot):
                    while True:
                        try:
                            psi[jj] = gigrnd.gigrnd(a-0.5*n_grp[jj]*n_trait, 2.0*delta[jj,0], xx[jj,0])
                        except:
                            continue
                        else:
                            break
                psi[psi>1] = 1.0
                if phi_updt == True:
                    w = random.gamma(1.0, 1.0/(phi+1.0))
                    phi = random.gamma(p_tot*b+0.5, 1.0/(delta.sum()+w))

            elif cross_trait == 'kron':
                # ---- separable Kronecker prior R = R_trait (x) R_anc ----
                s = {(pp, tt): np.sqrt(sigma[(pp, tt)]/get_n(pp, tt))
                     for pp in range(n_pop) for tt in range(n_trait)}
                Rt_inv = _safe_inv(R_trait)                                    # T x T
                Ra_sub_inv = {P: _safe_inv(R_anc[np.ix_(P, P)]) for P in patterns}

                # refresh global standardized-effect buffer from current beta
                alpha_glob[:] = np.nan
                for pp in range(n_pop):
                    for tt in range(n_trait):
                        alpha_glob[idx_dict[pp], pp, tt] = (beta[(pp, tt)]/s[(pp, tt)]).ravel()

                # (b) condition-conditional Gibbs for beta over (pp, tt)
                for pp in range(n_pop):
                    for tt in range(n_trait):
                        n_kt = get_n(pp, tt); p_kt = get_p(pp, tt); sigma_kt = sigma[(pp, tt)]
                        s_t = s[(pp, tt)]
                        psi_pp = psi[idx_dict[pp]]
                        diag_add = np.zeros((p_kt, 1))
                        mean_lin = np.zeros((p_kt, 1))                          # signed conditional mean
                        for P in patterns:
                            if pp not in P:
                                continue
                            lidx = pat_lidx[(P, pp)]
                            if lidx.size == 0:
                                continue
                            gidx = pat_gidx[P]
                            Pl = list(P)
                            kl = Pl.index(pp)
                            rai = Ra_sub_inv[P]                                 # |P| x |P|
                            psi_g = psi[gidx, 0]                                # (m,)
                            diag_add[lidx, 0] = Rt_inv[tt, tt]*rai[kl, kl]/psi_g
                            Asub = alpha_glob[np.ix_(gidx, Pl, list(range(n_trait)))]   # (m,|P|,T)
                            inner = np.einsum('mkt,t->mk', Asub, Rt_inv[tt])     # (m,|P|)
                            full = inner @ rai[kl, :]                           # (m,)
                            self_term = rai[kl, kl]*Rt_inv[tt, tt]*alpha_glob[gidx, pp, tt]
                            cross = full - self_term
                            mean_lin[lidx, 0] = -(s_t/psi_g)*cross
                        beta_mrg_adj = get_beta_mrg(pp, tt).copy() + mean_lin
                        sigma_eff = sigma_kt
                        if rho_e is not None and tt > 0:
                            for tt_prev in range(tt):
                                rho_et = rho_e[pp][tt, tt_prev]
                                if rho_et != 0.0:
                                    beta_mrg_adj -= rho_et*beta[(pp, tt_prev)]
                            rho_sq_sum = sum(rho_e[pp][tt, tp]**2 for tp in range(tt))
                            sigma_eff = sigma_kt*max(1.0 - rho_sq_sum, 0.01)
                        mm = 0; quad = 0.0
                        for kk in range(n_blk[pp]):
                            if blk_size[pp][kk] == 0:
                                continue
                            idx_blk = range(mm, mm+blk_size[pp][kk])
                            didx = blk_diag_idx[pp][kk]
                            dinvt = ld_blk[pp][kk].copy()
                            dinvt[didx, didx] += diag_add[idx_blk].ravel()
                            dinvt_chol = linalg.cholesky(dinvt, overwrite_a=True, check_finite=False)
                            beta_tmp = linalg.solve_triangular(dinvt_chol, beta_mrg_adj[idx_blk], trans='T', check_finite=False) \
                                       + np.sqrt(sigma_eff/n_kt)*random.randn(blk_size[pp][kk], 1)
                            beta[(pp, tt)][idx_blk] = linalg.solve_triangular(dinvt_chol, beta_tmp, trans='N', check_finite=False)
                            Cbeta = dinvt_chol @ beta[(pp, tt)][idx_blk]
                            quad += np.dot(Cbeta.ravel(), Cbeta.ravel())
                            mm += blk_size[pp][kk]
                        err = max(n_kt/2.0*(1.0-2.0*sum(beta[(pp, tt)]*get_beta_mrg(pp, tt))+quad),
                                  n_kt/2.0*sum(beta[(pp, tt)]**2/psi_pp))
                        sigma[(pp, tt)] = 1.0/random.gamma((n_kt+p_kt)/2.0, 1.0/err)
                        # refresh buffer for this (pp,tt) with updated beta & sigma
                        s[(pp, tt)] = np.sqrt(sigma[(pp, tt)]/n_kt)
                        alpha_glob[idx_dict[pp], pp, tt] = (beta[(pp, tt)]/s[(pp, tt)]).ravel()

                # (c) delta, shared psi (GIG with present-submatrix Kron quadratic), phi
                delta = random.gamma(a+b, 1.0/(psi+phi))
                xx = np.zeros((p_tot, 1))
                for P in patterns:
                    gidx = pat_gidx[P]
                    if gidx.size == 0:
                        continue
                    Pl = list(P)
                    rai = Ra_sub_inv[P]
                    Asub = alpha_glob[np.ix_(gidx, Pl, list(range(n_trait)))]
                    q = np.einsum('mkt,tu,kc,mcu->m', Asub, Rt_inv, rai, Asub)
                    xx[gidx, 0] += q
                for jj in range(p_tot):
                    while True:
                        try:
                            psi[jj] = gigrnd.gigrnd(a-0.5*n_grp[jj]*n_trait, 2.0*delta[jj,0], xx[jj,0])
                        except:
                            continue
                        else:
                            break
                psi[psi>1] = 1.0
                if phi_updt == True:
                    w = random.gamma(1.0, 1.0/(phi+1.0))
                    phi = random.gamma(p_tot*b+0.5, 1.0/(delta.sum()+w))

                # (d) update Kronecker factors via PX inverse-Wishart, with data
                #     augmentation for SNPs absent in some ancestries
                Afull = np.where(np.isnan(alpha_glob), 0.0, alpha_glob)
                for P in patterns:
                    Pc = [pp for pp in range(n_pop) if pp not in P]
                    if len(Pc) == 0:
                        continue
                    gidx = pat_gidx[P]
                    Pl = list(P)
                    rai = Ra_sub_inv[P]
                    Apres = alpha_glob[np.ix_(gidx, Pl, list(range(n_trait)))]  # (m,|P|,T)
                    Bmat = R_anc[np.ix_(Pc, Pl)] @ rai                          # (|Pc|,|P|)
                    mean_rows = np.einsum('ck,mkt->mct', Bmat, Apres)           # (m,|Pc|,T)
                    schur = R_anc[np.ix_(Pc, Pc)] - R_anc[np.ix_(Pc, Pl)] @ rai @ R_anc[np.ix_(Pl, Pc)]
                    schur = 0.5*(schur + schur.T)
                    Lrow = linalg.cholesky(schur + 1e-10*np.eye(len(Pc)), lower=True, check_finite=False)
                    Lcol = linalg.cholesky(R_trait + 1e-10*np.eye(n_trait), lower=True, check_finite=False)
                    psi_g = psi[gidx, 0]
                    Z = random.randn(gidx.size, len(Pc), n_trait)
                    draw = np.einsum('cd,mde,fe->mcf', Lrow, Z, Lcol)           # MN(0, schur, R_trait)
                    draw = mean_rows + np.sqrt(psi_g)[:, None, None]*draw
                    Afull[np.ix_(gidx, Pc, list(range(n_trait)))] = draw
                Afull_w = Afull/psi.reshape(-1, 1, 1)
                S_anc = np.einsum('gkt,tu,gmu->km', Afull, Rt_inv, Afull_w)
                R_anc = _sample_corr(S_anc, dof=p_tot*n_trait, nu0=n_pop+1.0)
                Ra_inv_full = _safe_inv(R_anc)
                S_tr = np.einsum('gkt,km,gmu->tu', Afull, Ra_inv_full, Afull_w)
                R_trait = _sample_corr(S_tr, dof=p_tot*n_pop, nu0=n_trait+1.0)

            # Posterior accumulation (shared across modes)
            if (itr > n_burnin) and (itr % thin == 0):
                for pp in range(n_pop):
                    for tt in range(n_trait):
                        beta_est[(pp, tt)] = beta_est[(pp, tt)] + beta[(pp, tt)]/n_pst
                        beta_sq_est[(pp, tt)] = beta_sq_est[(pp, tt)] + beta[(pp, tt)]**2/n_pst
                        sigma_est[(pp, tt)] = sigma_est[(pp, tt)] + sigma[(pp, tt)]/n_pst

                psi_est = psi_est + psi/n_pst
                phi_est = phi_est + phi/n_pst
                if cross_trait == 'mult':
                    psi_trait_est = psi_trait_est + psi_trait/n_pst
                elif cross_trait == 'mvcs':
                    for pp in range(n_pop):
                        R_cross_est[pp] = R_cross_est[pp] + R_cross[pp]/n_pst
                elif cross_trait == 'kron':
                    R_trait_est = R_trait_est + R_trait/n_pst
                    R_anc_est = R_anc_est + R_anc/n_pst

                if (write_pst == 'TRUE') and (meta == 'TRUE'):
                    for pp in range(n_pop):
                        for tt in range(n_trait):
                            beta_pst[(pp, tt)][:,[qq]] = beta[(pp, tt)]
                    qq += 1

    # convert standardized beta to per-allele beta
    if n_trait == 1:
        for pp in range(n_pop):
            beta_est[pp] /= het[pp]
            beta_sq_est[pp] /= het[pp]**2

        if (write_pst == 'TRUE') and (meta == 'TRUE'):
            for pp in range(n_pop):
                beta_pst[pp] /= het[pp]
    else:
        for pp in range(n_pop):
            for tt in range(n_trait):
                beta_est[(pp, tt)] /= het[(pp, tt)]
                beta_sq_est[(pp, tt)] /= het[(pp, tt)]**2

        if (write_pst == 'TRUE') and (meta == 'TRUE'):
            for pp in range(n_pop):
                for tt in range(n_trait):
                    beta_pst[(pp, tt)] /= het[(pp, tt)]

    # meta-analysis
    if meta == 'TRUE':
        if n_trait == 1:
            vv = np.zeros((p_tot, 1))
            zz = np.zeros((p_tot, 1))
            for pp in range(n_pop):
                vv[idx_dict[pp]] += 1.0/(beta_sq_est[pp]-beta_est[pp]**2)
                zz[idx_dict[pp]] += 1.0/(beta_sq_est[pp]-beta_est[pp]**2)*beta_est[pp]
            mu = zz/vv

            if write_pst == 'TRUE':
                vv = np.zeros((p_tot, 1))
                zz = np.zeros((p_tot, n_pst))
                for pp in range(n_pop):
                    vv[idx_dict[pp]] += 1.0/(beta_sq_est[pp]-beta_est[pp]**2)
                    zz[idx_dict[pp],:] += 1.0/(beta_sq_est[pp]-beta_est[pp]**2)*beta_pst[pp]
                mu_pst = zz/vv
        else:
            mu = {}
            mu_pst = {}
            for tt in range(n_trait):
                vv = np.zeros((p_tot, 1))
                zz = np.zeros((p_tot, 1))
                for pp in range(n_pop):
                    vv[idx_dict[pp]] += 1.0/(beta_sq_est[(pp, tt)]-beta_est[(pp, tt)]**2)
                    zz[idx_dict[pp]] += 1.0/(beta_sq_est[(pp, tt)]-beta_est[(pp, tt)]**2)*beta_est[(pp, tt)]
                mu[tt] = zz/vv

                if write_pst == 'TRUE':
                    vv = np.zeros((p_tot, 1))
                    zz = np.zeros((p_tot, n_pst))
                    for pp in range(n_pop):
                        vv[idx_dict[pp]] += 1.0/(beta_sq_est[(pp, tt)]-beta_est[(pp, tt)]**2)
                        zz[idx_dict[pp],:] += 1.0/(beta_sq_est[(pp, tt)]-beta_est[(pp, tt)]**2)*beta_pst[(pp, tt)]
                    mu_pst[tt] = zz/vv

    # write posterior effect sizes
    if n_trait == 1:
        for pp in range(n_pop):
            if phi_updt == True:
                eff_file = out_dir + '/' + '%s_%s_pst_eff_a%d_b%.1f_phiauto_chr%d.txt' % (out_name, pop[pp], a, b, chrom)
            else:
                eff_file = out_dir + '/' + '%s_%s_pst_eff_a%d_b%.1f_phi%1.0e_chr%d.txt' % (out_name, pop[pp], a, b, phi, chrom)

            snp_pp = [snp_dict['SNP'][ii] for ii in idx_dict[pp]]
            bp_pp = [snp_dict['BP'][ii] for ii in idx_dict[pp]]
            a1_pp = [snp_dict['A1'][ii] for ii in idx_dict[pp]]
            a2_pp = [snp_dict['A2'][ii] for ii in idx_dict[pp]]

            with open(eff_file, 'w') as ff:
                for snp, bp, a1, a2, beta_val in zip(snp_pp, bp_pp, a1_pp, a2_pp, beta_est[pp]):
                    ff.write('%d\t%s\t%d\t%s\t%s\t%.6e\n' % (chrom, snp, bp, a1, a2, beta_val.item()))
    else:
        for pp in range(n_pop):
            for tt in range(n_trait):
                if phi_updt == True:
                    eff_file = out_dir + '/' + '%s_%s_trait%d_pst_eff_a%d_b%.1f_phiauto_chr%d.txt' % (out_name, pop[pp], tt, a, b, chrom)
                else:
                    eff_file = out_dir + '/' + '%s_%s_trait%d_pst_eff_a%d_b%.1f_phi%1.0e_chr%d.txt' % (out_name, pop[pp], tt, a, b, phi, chrom)

                snp_pp = [snp_dict['SNP'][ii] for ii in idx_dict[pp]]
                bp_pp = [snp_dict['BP'][ii] for ii in idx_dict[pp]]
                a1_pp = [snp_dict['A1'][ii] for ii in idx_dict[pp]]
                a2_pp = [snp_dict['A2'][ii] for ii in idx_dict[pp]]

                with open(eff_file, 'w') as ff:
                    for snp, bp, a1, a2, beta_val in zip(snp_pp, bp_pp, a1_pp, a2_pp, beta_est[(pp, tt)]):
                        ff.write('%d\t%s\t%d\t%s\t%s\t%.6e\n' % (chrom, snp, bp, a1, a2, beta_val.item()))

    if meta == 'TRUE':
        if n_trait == 1:
            if phi_updt == True:
                eff_file = out_dir + '/' + '%s_META_pst_eff_a%d_b%.1f_phiauto_chr%d.txt' % (out_name, a, b, chrom)
            else:
                eff_file = out_dir + '/' + '%s_META_pst_eff_a%d_b%.1f_phi%1.0e_chr%d.txt' % (out_name, a, b, phi, chrom)

            with open(eff_file, 'w') as ff:
                if write_pst == 'TRUE':
                    for snp, bp, a1, a2, beta_val in zip(snp_dict['SNP'], snp_dict['BP'], snp_dict['A1'], snp_dict['A2'], mu_pst):
                        ff.write(('%d\t%s\t%d\t%s\t%s' + '\t%.6e'*n_pst + '\n') % (chrom, snp, bp, a1, a2, *beta_val))
                else:
                    for snp, bp, a1, a2, beta_val in zip(snp_dict['SNP'], snp_dict['BP'], snp_dict['A1'], snp_dict['A2'], mu):
                        ff.write('%d\t%s\t%d\t%s\t%s\t%.6e\n' % (chrom, snp, bp, a1, a2, beta_val.item()))
        else:
            for tt in range(n_trait):
                if phi_updt == True:
                    eff_file = out_dir + '/' + '%s_META_trait%d_pst_eff_a%d_b%.1f_phiauto_chr%d.txt' % (out_name, tt, a, b, chrom)
                else:
                    eff_file = out_dir + '/' + '%s_META_trait%d_pst_eff_a%d_b%.1f_phi%1.0e_chr%d.txt' % (out_name, tt, a, b, phi, chrom)

                with open(eff_file, 'w') as ff:
                    if write_pst == 'TRUE':
                        for snp, bp, a1, a2, beta_val in zip(snp_dict['SNP'], snp_dict['BP'], snp_dict['A1'], snp_dict['A2'], mu_pst[tt]):
                            ff.write(('%d\t%s\t%d\t%s\t%s' + '\t%.6e'*n_pst + '\n') % (chrom, snp, bp, a1, a2, *beta_val))
                    else:
                        for snp, bp, a1, a2, beta_val in zip(snp_dict['SNP'], snp_dict['BP'], snp_dict['A1'], snp_dict['A2'], mu[tt]):
                            ff.write('%d\t%s\t%d\t%s\t%s\t%.6e\n' % (chrom, snp, bp, a1, a2, beta_val.item()))

    # print estimated phi
    if phi_updt == True:
        print('... Estimated global shrinkage parameter: %1.2e ...' % np.float64(phi_est).item())

    # write learned cross-condition correlation matrices
    if n_trait > 1 and cross_trait == 'mvcs':
        corr_file = out_dir + '/' + '%s_corr_mvcs_chr%d.txt' % (out_name, chrom)
        with open(corr_file, 'w') as ff:
            for pp in range(n_pop):
                ff.write('# posterior-mean cross-trait correlation, population %s\n' % pop[pp])
                for row in R_cross_est[pp]:
                    ff.write('\t'.join('%.4f' % v for v in row) + '\n')
        print('... Posterior-mean cross-trait correlation (pop %s):\n%s' % (pop[0], np.array2string(R_cross_est[0], precision=3)))
    if n_trait > 1 and cross_trait == 'kron':
        corr_file = out_dir + '/' + '%s_corr_kron_chr%d.txt' % (out_name, chrom)
        with open(corr_file, 'w') as ff:
            ff.write('# posterior-mean cross-trait correlation R_trait\n')
            for row in R_trait_est:
                ff.write('\t'.join('%.4f' % v for v in row) + '\n')
            ff.write('# posterior-mean cross-ancestry correlation R_anc (pop order: %s)\n' % ','.join(pop))
            for row in R_anc_est:
                ff.write('\t'.join('%.4f' % v for v in row) + '\n')
        print('... Posterior-mean R_trait:\n%s' % np.array2string(R_trait_est, precision=3))
        print('... Posterior-mean R_anc:\n%s' % np.array2string(R_anc_est, precision=3))

    print('... Done ...')
