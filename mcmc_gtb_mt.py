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
from numpy import random
import gigrnd


def mcmc(a, b, phi, snp_dict, beta_mrg, frq_dict, idx_dict, n, ld_blk, blk_size,
         n_iter, n_burnin, thin, pop, chrom, out_dir, out_name, meta, write_pst, seed,
         n_trait=1, rho_e=None, lambda_psi=1.0):
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

    # Trait-specific modulators (only when n_trait > 1)
    if n_trait > 1:
        psi_trait = np.ones((p_tot, n_trait))

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
    if n_trait > 1:
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
            # === MULTI-TRAIT MODE ===

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
                        # Adjust for correlated sampling noise from overlapping samples
                        # Subtract contribution of previously sampled traits
                        for tt_prev in range(tt):
                            rho_et = rho_e[pp][tt, tt_prev]
                            if rho_et != 0.0:
                                # Need to map beta from trait tt_prev into the LD block structure
                                beta_mrg_adj -= rho_et * beta[(pp, tt_prev)]

                    # Compute effective sigma for overlap correction
                    sigma_eff = sigma_kt
                    if rho_e is not None and tt > 0:
                        # Reduce sigma by (1 - sum of rho_e^2 with previous traits)
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

            # Step 3: Delta sampling (unchanged)
            delta = random.gamma(a+b, 1.0/(psi+phi))

            # Step 4: Psi (shared) sampling — accumulate across ALL pops AND traits
            xx = np.zeros((p_tot, 1))
            for pp in range(n_pop):
                for tt in range(n_trait):
                    n_kt = get_n(pp, tt)
                    # Divide out the trait-specific modulator
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

            # Step 5: Psi_trait (NEW) sampling — per trait, accumulate across pops
            for tt in range(n_trait):
                xx_trait = np.zeros((p_tot, 1))
                for pp in range(n_pop):
                    n_kt = get_n(pp, tt)
                    # Divide out the shared shrinkage
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

            # Step 6: Phi (global) sampling (unchanged)
            if phi_updt == True:
                w = random.gamma(1.0, 1.0/(phi+1.0))
                phi = random.gamma(p_tot*b+0.5, 1.0/(delta.sum()+w))

            # Posterior accumulation
            if (itr > n_burnin) and (itr % thin == 0):
                for pp in range(n_pop):
                    for tt in range(n_trait):
                        beta_est[(pp, tt)] = beta_est[(pp, tt)] + beta[(pp, tt)]/n_pst
                        beta_sq_est[(pp, tt)] = beta_sq_est[(pp, tt)] + beta[(pp, tt)]**2/n_pst
                        sigma_est[(pp, tt)] = sigma_est[(pp, tt)] + sigma[(pp, tt)]/n_pst

                psi_est = psi_est + psi/n_pst
                phi_est = phi_est + phi/n_pst
                psi_trait_est = psi_trait_est + psi_trait/n_pst

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

    print('... Done ...')
