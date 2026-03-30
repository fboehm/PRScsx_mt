#!/usr/bin/env python3
"""
simulate_indiv.py
-----------------
Individual-level simulation for the PRS-CSx-MT comparison study.

Unlike simulate_mt.py (which computes marginal summary statistics
analytically), this module:
  1. Simulates individual-level standardised genotype matrices from
     block-diagonal LD structure.
  2. Simulates multi-trait phenotypes from those genotypes.
  3. Runs univariate OLS GWAS per SNP — explicitly mimicking a GWAS
     restricted to HM3 SNPs (the SNP universe is the simulated HM3-like
     set from the start, so the HM3 filter is trivially satisfied).

Output files are format-identical to simulate_mt._write_sim_files(), so
the downstream PRSCSx / PRSCSx-MT pipeline (parse_genet_mt, mcmc_gtb_mt,
evaluate.py, aggregate.py) requires no changes.

Key difference for parse_genet_mt
----------------------------------
simulate_mt writes:  BETA = beta_mrg[i],  SE = 1/sqrt(n)  (constant)
This module writes:  BETA = bhat_ols[i],  SE = se_ols[i]  (per-SNP)

parse_genet_mt converts via  beta_std = BETA / SE / sqrt(n),
which equals the OLS t-statistic / sqrt(n) — the standardised marginal
effect expected by the PRSCSx model.  Both conventions converge to the
same value in large samples; the individual-level route also captures
finite-sample noise and MAF-dependent SE variation.
"""

import os
import numpy as np
from scipy import linalg
import h5py

from simulate_mt import _make_block_diagonal_ld


def simulate_indiv_gwas(
    n_snp=500, n_causal=50, n_pop=2, n_trait=2, n_gwas=None,
    block_size=50, ld_decay=0.5,
    rg=0.5, rho_pop=0.8, pi_causal=None,
    h2=None, rho_pheno=0.0, n_overlap=None,
    pop=None, out_dir=None, chrom=1, seed=42,
):
    """
    Simulate individual-level genotypes, multi-trait phenotypes, and
    univariate OLS GWAS summary statistics.

    Parameters
    ----------
    n_gwas : list of list [n_pop][n_trait]
        Number of *individuals* simulated for each (population, trait) pair.
        Independent cohorts are drawn for each combination (no overlap).
    (All other parameters are identical to simulate_mt.simulate_mt.
     rho_pheno and n_overlap are accepted but currently ignored.)

    Returns
    -------
    sim_data : dict
        Same structure as simulate_mt.simulate_mt(), with the addition of
        key 'gwas_se' holding per-SNP OLS standard errors.
    """
    rng = np.random.RandomState(seed)

    if n_gwas is None:
        n_gwas = [[50000] * n_trait for _ in range(n_pop)]
    if h2 is None:
        h2 = [0.5] * n_trait
    if pi_causal is None:
        pi_causal = n_causal / n_snp

    # ------------------------------------------------------------------
    # 1.  LD structure
    # ------------------------------------------------------------------
    R, ld_blocks, blk_sizes = _make_block_diagonal_ld(n_snp, block_size, ld_decay)

    # ------------------------------------------------------------------
    # 2.  True causal effects  (identical logic to simulate_mt)
    # ------------------------------------------------------------------
    causal_idx = rng.choice(n_snp, size=n_causal, replace=False)
    causal_idx.sort()

    if isinstance(rho_pop, (list, np.ndarray)):
        rho_pop_per_trait = list(rho_pop)
    else:
        rho_pop_per_trait = [rho_pop] * n_trait

    dim = n_pop * n_trait
    Sigma_kron = np.eye(dim)
    for pp1 in range(n_pop):
        for tt1 in range(n_trait):
            i1 = pp1 * n_trait + tt1
            for pp2 in range(n_pop):
                for tt2 in range(n_trait):
                    i2 = pp2 * n_trait + tt2
                    if i1 == i2:
                        continue
                    if pp1 == pp2:
                        Sigma_kron[i1, i2] = rg
                    elif tt1 == tt2:
                        Sigma_kron[i1, i2] = rho_pop_per_trait[tt1]
                    else:
                        Sigma_kron[i1, i2] = rg * np.sqrt(
                            rho_pop_per_trait[tt1] * rho_pop_per_trait[tt2]
                        )

    min_eig = np.min(np.linalg.eigvalsh(Sigma_kron))
    if min_eig < 1e-6:
        Sigma_kron += (1e-6 - min_eig) * np.eye(dim)
    Sigma_kron_chol = linalg.cholesky(Sigma_kron, lower=True)

    beta_causal = rng.randn(n_causal, dim) @ Sigma_kron_chol.T
    frq = rng.uniform(0.1, 0.9, n_snp)

    beta_true = {}
    for pp in range(n_pop):
        for tt in range(n_trait):
            b = np.zeros(n_snp)
            b[causal_idx] = beta_causal[:, pp * n_trait + tt]
            var_g = b @ R @ b
            if var_g > 0:
                b *= np.sqrt(h2[tt] / var_g)
            beta_true[(pp, tt)] = b

    # ------------------------------------------------------------------
    # 3.  Population labels
    # ------------------------------------------------------------------
    valid_pops = ['EUR', 'EAS', 'AFR', 'AMR', 'SAS']
    pop_labels = list(pop) if pop is not None else valid_pops[:n_pop]

    # ------------------------------------------------------------------
    # 4.  Per (population, trait): simulate individuals → OLS GWAS
    # ------------------------------------------------------------------
    beta_mrg  = {}
    gwas_se   = {}
    n_gwas_dict = {}

    for pp in range(n_pop):
        for tt in range(n_trait):
            n_indiv = n_gwas[pp][tt]
            n_gwas_dict[(pp, tt)] = n_indiv

            # 4a. Individual standardised genotypes (one block at a time)
            #
            # For each LD block with matrix R_b (bsize × bsize):
            #   X_b = Z @ chol(R_b)^T,   Z ~ N(0, I_{n_indiv × bsize})
            # This gives Corr(X_b[:,i], X_b[:,j]) ≈ R_b[i,j].
            X_cols = []
            for blk, bsize in zip(ld_blocks, blk_sizes):
                L = linalg.cholesky(blk + 1e-6 * np.eye(bsize), lower=True)
                X_cols.append(rng.randn(n_indiv, bsize) @ L.T)
            X = np.hstack(X_cols)    # (n_indiv, n_snp)

            # 4b. Phenotype
            #
            # y = X @ beta_true + noise,  Var(y) standardised to 1.
            y_g = X @ beta_true[(pp, tt)]
            var_g_sample = np.var(y_g)
            sigma_e = (
                np.sqrt(var_g_sample * (1.0 - h2[tt]) / h2[tt])
                if var_g_sample > 0 else 1.0
            )
            y = y_g + rng.randn(n_indiv) * sigma_e
            y_sd = np.std(y)
            if y_sd > 0:
                y = (y - np.mean(y)) / y_sd

            # 4c. Univariate OLS GWAS (vectorised over all n_snp SNPs)
            #
            # beta_hat_j = x_j'y / x_j'x_j
            # SE_j       = sqrt( RSS_j / ((n-2) * x_j'x_j) )
            # where RSS_j = y'y - beta_hat_j^2 * x_j'x_j
            xTx  = np.einsum('ij,ij->j', X, X)     # (n_snp,)
            xTy  = X.T @ y                           # (n_snp,)
            bhat = xTy / xTx                         # (n_snp,)

            RSS  = np.maximum(float(y @ y) - bhat**2 * xTx, 0.0)
            se   = np.sqrt(RSS / ((n_indiv - 2) * xTx))   # (n_snp,)

            beta_mrg[(pp, tt)] = bhat.reshape(-1, 1)
            gwas_se [(pp, tt)] = se

    # ------------------------------------------------------------------
    # 5.  SNP metadata
    # ------------------------------------------------------------------
    snp_info = {
        'SNP': ['rs%d' % (i + 1) for i in range(n_snp)],
        'A1':  ['A'] * n_snp,
        'A2':  ['G'] * n_snp,
        'BP':  list(range(1000, 1000 + n_snp * 1000, 1000)),
        'FRQ': frq,
        'CHR': [chrom] * n_snp,
    }

    sim_data = {
        'beta_true':  beta_true,
        'beta_mrg':   beta_mrg,
        'gwas_se':    gwas_se,
        'R':          R,
        'ld_blocks':  ld_blocks,
        'blk_sizes':  blk_sizes,
        'snp_info':   snp_info,
        'n_gwas':     n_gwas_dict,
        'pop':        pop_labels,
        'n_pop':      n_pop,
        'n_trait':    n_trait,
        'n_snp':      n_snp,
        'causal_idx': causal_idx,
        'frq':        frq,
    }

    if out_dir is not None:
        _write_indiv_gwas_files(sim_data, out_dir, chrom)

    return sim_data


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _write_indiv_gwas_files(sim_data, out_dir, chrom):
    """
    Write simulation output in PRSCSx-compatible format.

    Identical to simulate_mt._write_sim_files() except that:
      BETA = per-SNP OLS estimate   (not beta_mrg scaled by 1/sqrt(n))
      SE   = per-SNP OLS SE         (not a constant 1/sqrt(n))
    """
    os.makedirs(out_dir, exist_ok=True)

    n_pop    = sim_data['n_pop']
    n_trait  = sim_data['n_trait']
    n_snp    = sim_data['n_snp']
    pop      = sim_data['pop']
    snp_info = sim_data['snp_info']
    frq      = sim_data['frq']
    pop_upper = [p.upper() for p in pop]

    # ---- .bim file -------------------------------------------------------
    with open(os.path.join(out_dir, 'sim_data.bim'), 'w') as fh:
        for i in range(n_snp):
            fh.write('%d\t%s\t0\t%d\t%s\t%s\n' % (
                chrom, snp_info['SNP'][i], snp_info['BP'][i],
                snp_info['A1'][i], snp_info['A2'][i],
            ))

    # ---- snpinfo_mult_1kg_hm3 --------------------------------------------
    all_ref_pops = ['AFR', 'AMR', 'EAS', 'EUR', 'SAS']
    with open(os.path.join(out_dir, 'snpinfo_mult_1kg_hm3'), 'w') as fh:
        header = (['CHR', 'SNP', 'BP', 'A1', 'A2']
                  + ['FRQ_' + p for p in all_ref_pops]
                  + ['FLP_' + p for p in all_ref_pops])
        fh.write('\t'.join(header) + '\n')
        for i in range(n_snp):
            fh.write('%d\t%s\t%d\t%s\t%s' % (
                chrom, snp_info['SNP'][i], snp_info['BP'][i],
                snp_info['A1'][i], snp_info['A2'][i],
            ))
            for rp in all_ref_pops:
                fh.write('\t%.4f' % frq[i] if rp in pop_upper else '\t0.0000')
            for rp in all_ref_pops:
                fh.write('\t1' if rp in pop_upper else '\t0')
            fh.write('\n')

    # ---- summary statistics (one file per population × trait) ------------
    for pp in range(n_pop):
        for tt in range(n_trait):
            sst_path = os.path.join(out_dir, 'sst_%s_trait%d.txt' % (pop[pp], tt))
            bhat = sim_data['beta_mrg'][(pp, tt)]   # (n_snp, 1)
            se   = sim_data['gwas_se'][(pp, tt)]    # (n_snp,)
            with open(sst_path, 'w') as fh:
                fh.write('SNP\tA1\tA2\tBETA\tSE\n')
                for i in range(n_snp):
                    fh.write('%s\t%s\t%s\t%.6e\t%.6e\n' % (
                        snp_info['SNP'][i],
                        snp_info['A1'][i],
                        snp_info['A2'][i],
                        float(bhat[i, 0]),
                        float(se[i]),
                    ))

    # ---- LD block HDF5 reference files -----------------------------------
    for pp in range(n_pop):
        ld_dir = os.path.join(out_dir, 'ldblk_1kg_%s' % pop[pp].lower())
        os.makedirs(ld_dir, exist_ok=True)
        with h5py.File(os.path.join(ld_dir, 'ldblk_1kg_chr%d.hdf5' % chrom), 'w') as hf:
            snp_idx = 0
            for blk_num, (blk, bsize) in enumerate(
                zip(sim_data['ld_blocks'], sim_data['blk_sizes'])
            ):
                grp = hf.create_group('blk_%d' % (blk_num + 1))
                grp.create_dataset('ldblk', data=blk)
                grp.create_dataset(
                    'snplist',
                    data=[s.encode('UTF-8') for s in snp_info['SNP'][snp_idx:snp_idx + bsize]],
                )
                snp_idx += bsize

    # ---- true effect sizes (for evaluation) ------------------------------
    for pp in range(n_pop):
        for tt in range(n_trait):
            true_path = os.path.join(
                out_dir, 'true_effects_%s_trait%d.txt' % (pop[pp], tt)
            )
            beta_true = sim_data['beta_true'][(pp, tt)]
            with open(true_path, 'w') as fh:
                fh.write('SNP\tBETA_TRUE\n')
                for i in range(n_snp):
                    fh.write('%s\t%.6e\n' % (snp_info['SNP'][i], beta_true[i]))

    print('... simulation files written to %s ...' % out_dir)
