#!/usr/bin/env python

"""
Simulation framework for PRS-CSx-MT validation.

Generates synthetic multi-trait, multi-ancestry GWAS summary statistics with:
- Correlated causal effects via Kronecker structure Sigma_pop (x) Sigma_trait
- Block-diagonal LD with exponential decay
- Correlated sampling noise for sample overlap
- Output in PRS-CSx file format (sumstats, .bim, LD HDF5)

"""

import os
import numpy as np
from scipy import linalg
import h5py


def _make_ld_block(block_size, decay=0.5):
    """Create a block LD matrix with exponential decay."""
    R = np.zeros((block_size, block_size))
    for i in range(block_size):
        for j in range(block_size):
            R[i, j] = decay ** abs(i - j)
    return R


def _make_block_diagonal_ld(n_snp, block_size, decay=0.5):
    """Create a block-diagonal LD matrix."""
    n_blocks = int(np.ceil(n_snp / block_size))
    blocks = []
    sizes = []
    for b in range(n_blocks):
        bs = min(block_size, n_snp - b * block_size)
        blocks.append(_make_ld_block(bs, decay))
        sizes.append(bs)
    # Full LD matrix (for simulation)
    R = linalg.block_diag(*blocks)
    return R, blocks, sizes


def simulate_mt(n_snp=500, n_causal=50, n_pop=2, n_trait=2, n_gwas=None,
                block_size=50, ld_decay=0.5,
                rg=0.5, rho_pop=0.8, pi_causal=None,
                h2=None, rho_pheno=0.0, n_overlap=None,
                pop=None, out_dir=None, chrom=1, seed=42):
    """
    Generate synthetic multi-trait, multi-ancestry GWAS summary statistics.

    Parameters
    ----------
    n_snp : int
        Total number of SNPs.
    n_causal : int
        Number of causal SNPs.
    n_pop : int
        Number of populations.
    n_trait : int
        Number of traits.
    n_gwas : list of list, or None
        GWAS sample sizes, shape [n_pop][n_trait]. If None, defaults to
        [[50000]*n_trait]*n_pop.
    block_size : int
        LD block size for exponential-decay blocks.
    ld_decay : float
        LD decay rate within blocks.
    rg : float
        Genetic correlation between traits (for causal effects).
    rho_pop : float or list
        Cross-population correlation for causal effects. If a list, specifies
        per-trait cross-population correlations [rho_pop_t0, rho_pop_t1, ...].
    pi_causal : float or None
        Proportion of causal SNPs. If None, computed as n_causal/n_snp.
    h2 : list or None
        Heritability per trait. If None, defaults to [0.5]*n_trait.
    rho_pheno : float
        Phenotypic correlation (for simulating overlapping samples).
    n_overlap : list or None
        Number of overlapping samples per population. If None, no overlap.
    pop : list or None
        Population labels. Must be from {AFR, AMR, EAS, EUR, SAS} to match
        reference panel format. If None, defaults to ['EUR','EAS',...].
    out_dir : str or None
        If specified, write output files.
    chrom : int
        Chromosome label for output files.
    seed : int
        Random seed.

    Returns
    -------
    sim_data : dict
        Dictionary containing all simulation data:
        - 'beta_true': true causal effects, dict keyed by (pp, tt)
        - 'beta_mrg': marginal GWAS betas (standardized), dict keyed by (pp, tt)
        - 'R': LD matrix (shared across populations for simplicity)
        - 'ld_blocks': list of LD block matrices
        - 'blk_sizes': list of block sizes
        - 'snp_info': dict with SNP, A1, A2, BP, FRQ info
        - 'n_gwas': sample sizes dict keyed by (pp, tt)
    """
    rng = np.random.RandomState(seed)

    if n_gwas is None:
        n_gwas = [[50000]*n_trait for _ in range(n_pop)]
    if h2 is None:
        h2 = [0.5]*n_trait
    if pi_causal is None:
        pi_causal = n_causal / n_snp

    # Generate LD structure
    R, ld_blocks, blk_sizes = _make_block_diagonal_ld(n_snp, block_size, ld_decay)
    R_chol = linalg.cholesky(R + 1e-6*np.eye(n_snp), lower=True)

    # Select causal SNPs
    causal_idx = rng.choice(n_snp, size=n_causal, replace=False)
    causal_idx.sort()
    is_causal = np.zeros(n_snp, dtype=bool)
    is_causal[causal_idx] = True

    # Generate correlated causal effects
    # Support per-trait cross-population correlations
    if isinstance(rho_pop, (list, np.ndarray)):
        rho_pop_per_trait = rho_pop
    else:
        rho_pop_per_trait = [rho_pop] * n_trait

    # Build covariance matrix directly for flexibility
    # Ordering: [pop0_t0, pop0_t1, ..., pop1_t0, pop1_t1, ...]
    dim = n_pop * n_trait
    Sigma_kron = np.eye(dim)

    for pp1 in range(n_pop):
        for tt1 in range(n_trait):
            idx1 = pp1 * n_trait + tt1
            for pp2 in range(n_pop):
                for tt2 in range(n_trait):
                    idx2 = pp2 * n_trait + tt2
                    if idx1 == idx2:
                        continue  # diagonal is 1

                    if pp1 == pp2:
                        # Same population, different traits: use rg
                        Sigma_kron[idx1, idx2] = rg
                    elif tt1 == tt2:
                        # Same trait, different populations: use trait-specific rho_pop
                        Sigma_kron[idx1, idx2] = rho_pop_per_trait[tt1]
                    else:
                        # Different population AND different trait
                        # Use geometric mean of the two trait-specific rho_pop values times rg
                        rho_cross = rg * np.sqrt(rho_pop_per_trait[tt1] * rho_pop_per_trait[tt2])
                        Sigma_kron[idx1, idx2] = rho_cross

    # Ensure positive definiteness
    min_eig = np.min(np.linalg.eigvalsh(Sigma_kron))
    if min_eig < 1e-6:
        Sigma_kron += (1e-6 - min_eig) * np.eye(dim)

    Sigma_kron_chol = linalg.cholesky(Sigma_kron, lower=True)

    # Generate causal effects
    # Each causal SNP gets a (n_pop * n_trait) vector of effects
    beta_causal_raw = rng.randn(n_causal, n_pop * n_trait)
    beta_causal = beta_causal_raw @ Sigma_kron_chol.T

    # Scale effects by heritability
    beta_true = {}
    for pp in range(n_pop):
        for tt in range(n_trait):
            idx = pp * n_trait + tt
            b = np.zeros(n_snp)
            b[causal_idx] = beta_causal[:, idx]
            # Scale so that Var(X @ beta) ≈ h2[tt]
            # For standardized genotypes with LD: Var = beta' R beta
            var_g = b @ R @ b
            if var_g > 0:
                b *= np.sqrt(h2[tt] / var_g)
            beta_true[(pp, tt)] = b

    # Generate allele frequencies
    frq = rng.uniform(0.1, 0.9, n_snp)

    # Generate marginal GWAS summary statistics
    # beta_hat = R @ beta_true + noise
    # noise ~ N(0, (1/n) * R) for standardized effects
    beta_mrg = {}
    n_gwas_dict = {}
    for pp in range(n_pop):
        # Generate noise, potentially correlated across traits if there's overlap
        noise = {}
        for tt in range(n_trait):
            n_kt = n_gwas[pp][tt]
            n_gwas_dict[(pp, tt)] = n_kt
            # Standard GWAS noise: (1/sqrt(n)) * R^{1/2} @ z
            noise[tt] = (1.0 / np.sqrt(n_kt)) * R_chol @ rng.randn(n_snp)

        # Add correlated noise for overlapping samples
        if n_overlap is not None and rho_pheno != 0.0 and n_trait > 1:
            n_ov = n_overlap[pp] if isinstance(n_overlap, list) else n_overlap
            for tt1 in range(n_trait):
                for tt2 in range(tt1+1, n_trait):
                    n1 = n_gwas[pp][tt1]
                    n2 = n_gwas[pp][tt2]
                    rho_noise = rho_pheno * n_ov / np.sqrt(n1 * n2)
                    # Introduce correlation via shared noise component
                    shared = rng.randn(n_snp)
                    noise[tt1] += rho_noise * (1.0/np.sqrt(n1)) * R_chol @ shared
                    noise[tt2] += rho_noise * (1.0/np.sqrt(n2)) * R_chol @ shared

        for tt in range(n_trait):
            n_kt = n_gwas[pp][tt]
            # Marginal association: beta_mrg = R @ beta_true + noise
            beta_hat = R @ beta_true[(pp, tt)] + noise[tt]
            # Standardize: beta_std = beta_hat * sqrt(n) / sqrt(n) = beta_hat (already standardized)
            beta_mrg[(pp, tt)] = beta_hat.reshape(-1, 1)

    # Generate SNP info
    valid_pops = ['EUR', 'EAS', 'AFR', 'AMR', 'SAS']
    if pop is not None:
        pop_labels = pop
    else:
        pop_labels = valid_pops[:n_pop]
    snp_names = ['rs%d' % (i+1) for i in range(n_snp)]
    a1_alleles = ['A'] * n_snp
    a2_alleles = ['G'] * n_snp
    bp_positions = list(range(1000, 1000 + n_snp * 1000, 1000))

    snp_info = {
        'SNP': snp_names,
        'A1': a1_alleles,
        'A2': a2_alleles,
        'BP': bp_positions,
        'FRQ': frq,
        'CHR': [chrom] * n_snp,
    }

    sim_data = {
        'beta_true': beta_true,
        'beta_mrg': beta_mrg,
        'R': R,
        'ld_blocks': ld_blocks,
        'blk_sizes': blk_sizes,
        'snp_info': snp_info,
        'n_gwas': n_gwas_dict,
        'pop': pop_labels,
        'n_pop': n_pop,
        'n_trait': n_trait,
        'n_snp': n_snp,
        'causal_idx': causal_idx,
        'frq': frq,
    }

    # Write output files if directory specified
    if out_dir is not None:
        _write_sim_files(sim_data, out_dir, chrom)

    return sim_data


def _write_sim_files(sim_data, out_dir, chrom):
    """Write simulation data to PRS-CSx compatible file formats."""
    os.makedirs(out_dir, exist_ok=True)

    n_pop = sim_data['n_pop']
    n_trait = sim_data['n_trait']
    n_snp = sim_data['n_snp']
    pop = sim_data['pop']
    snp_info = sim_data['snp_info']
    frq = sim_data['frq']

    # Write .bim file
    bim_file = os.path.join(out_dir, 'sim_data.bim')
    with open(bim_file, 'w') as ff:
        for i in range(n_snp):
            ff.write('%d\t%s\t0\t%d\t%s\t%s\n' % (
                chrom, snp_info['SNP'][i], snp_info['BP'][i],
                snp_info['A1'][i], snp_info['A2'][i]))

    # Write reference panel SNP info (mimicking snpinfo_mult_1kg_hm3 format)
    all_ref_pops = ['AFR', 'AMR', 'EAS', 'EUR', 'SAS']
    ref_file = os.path.join(out_dir, 'snpinfo_mult_1kg_hm3')
    with open(ref_file, 'w') as ff:
        header_cols = ['CHR', 'SNP', 'BP', 'A1', 'A2']
        header_cols += ['FRQ_' + p for p in all_ref_pops]
        header_cols += ['FLP_' + p for p in all_ref_pops]
        ff.write('\t'.join(header_cols) + '\n')
        for i in range(n_snp):
            ff.write('%d\t%s\t%d\t%s\t%s' % (
                chrom, snp_info['SNP'][i], snp_info['BP'][i],
                snp_info['A1'][i], snp_info['A2'][i]))
            # Write frequency for each reference population
            # Use actual freq for simulated populations, 0 for others
            for rp in all_ref_pops:
                if rp.upper() in [p.upper() for p in pop]:
                    ff.write('\t%.4f' % frq[i])
                else:
                    ff.write('\t0.0000')
            # Write flip info (1 = no flip needed)
            for rp in all_ref_pops:
                if rp.upper() in [p.upper() for p in pop]:
                    ff.write('\t1')
                else:
                    ff.write('\t0')
            ff.write('\n')

    # Write summary statistics files
    for pp in range(n_pop):
        for tt in range(n_trait):
            sst_file = os.path.join(out_dir, 'sst_%s_trait%d.txt' % (pop[pp], tt))
            beta_mrg = sim_data['beta_mrg'][(pp, tt)]
            n_kt = sim_data['n_gwas'][(pp, tt)]
            with open(sst_file, 'w') as ff:
                ff.write('SNP\tA1\tA2\tBETA\tSE\n')
                for i in range(n_snp):
                    # Convert standardized beta back to raw beta for file format
                    # beta_std = beta / se / sqrt(n), so beta = beta_std * se * sqrt(n)
                    # For simulation, we set se = 1/sqrt(n), so beta_raw = beta_std
                    se = 1.0 / np.sqrt(n_kt)
                    beta_raw = beta_mrg[i, 0] * se * np.sqrt(n_kt)
                    ff.write('%s\t%s\t%s\t%.6e\t%.6e\n' % (
                        snp_info['SNP'][i], snp_info['A1'][i], snp_info['A2'][i],
                        beta_raw, se))

    # Write LD block HDF5 files
    for pp in range(n_pop):
        pop_lower = pop[pp].lower()
        ld_dir = os.path.join(out_dir, 'ldblk_1kg_%s' % pop_lower)
        os.makedirs(ld_dir, exist_ok=True)

        ld_file = os.path.join(ld_dir, 'ldblk_1kg_chr%d.hdf5' % chrom)
        ld_blocks = sim_data['ld_blocks']
        blk_sizes = sim_data['blk_sizes']

        with h5py.File(ld_file, 'w') as hf:
            snp_idx = 0
            for blk_num, (block, bsize) in enumerate(zip(ld_blocks, blk_sizes)):
                grp = hf.create_group('blk_%d' % (blk_num + 1))
                grp.create_dataset('ldblk', data=block)
                snp_list = snp_info['SNP'][snp_idx:snp_idx+bsize]
                grp.create_dataset('snplist', data=[s.encode('UTF-8') for s in snp_list])
                snp_idx += bsize

    # Write true effect sizes for evaluation
    for pp in range(n_pop):
        for tt in range(n_trait):
            true_file = os.path.join(out_dir, 'true_effects_%s_trait%d.txt' % (pop[pp], tt))
            beta_true = sim_data['beta_true'][(pp, tt)]
            with open(true_file, 'w') as ff:
                ff.write('SNP\tBETA_TRUE\n')
                for i in range(n_snp):
                    ff.write('%s\t%.6e\n' % (snp_info['SNP'][i], beta_true[i]))

    print('... simulation files written to %s ...' % out_dir)


def evaluate_prediction(beta_est, beta_true, R=None):
    """
    Evaluate estimated vs true effects.

    Parameters
    ----------
    beta_est : numpy array
        Estimated effect sizes (n_snp,) or (n_snp, 1).
    beta_true : numpy array
        True effect sizes (n_snp,).
    R : numpy array or None
        LD matrix. If provided, computes genetic prediction R^2.

    Returns
    -------
    metrics : dict
        Dictionary with evaluation metrics:
        - 'corr': Pearson correlation between estimated and true effects
        - 'mse': Mean squared error
        - 'r2_effects': R^2 of effect sizes
        - 'r2_genetic': R^2 of genetic values (if R provided)
    """
    beta_est = np.asarray(beta_est).flatten()
    beta_true = np.asarray(beta_true).flatten()

    # Pearson correlation
    if np.std(beta_est) > 0 and np.std(beta_true) > 0:
        corr = np.corrcoef(beta_est, beta_true)[0, 1]
    else:
        corr = 0.0

    # MSE
    mse = np.mean((beta_est - beta_true)**2)

    # R^2 of effect sizes
    ss_res = np.sum((beta_est - beta_true)**2)
    ss_tot = np.sum((beta_true - np.mean(beta_true))**2)
    r2_effects = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    metrics = {
        'corr': corr,
        'mse': mse,
        'r2_effects': r2_effects,
    }

    # R^2 of genetic values (using LD matrix)
    if R is not None:
        gv_est = R @ beta_est
        gv_true = R @ beta_true
        if np.std(gv_est) > 0 and np.std(gv_true) > 0:
            corr_gv = np.corrcoef(gv_est, gv_true)[0, 1]
            metrics['r2_genetic'] = corr_gv**2
        else:
            metrics['r2_genetic'] = 0.0

    return metrics


def run_simulation_study(scenarios=None, seed=42):
    """
    Run a set of simulation scenarios and report results.

    Parameters
    ----------
    scenarios : list of dict, or None
        Each dict contains parameters for simulate_mt(). If None, runs defaults.
    seed : int
        Base random seed.

    Returns
    -------
    results : list of dict
        Results for each scenario.
    """
    if scenarios is None:
        scenarios = [
            {'name': 'T=1 baseline', 'n_trait': 1, 'rg': 0.0},
            {'name': 'T=2 independent', 'n_trait': 2, 'rg': 0.0},
            {'name': 'T=2 correlated rg=0.5', 'n_trait': 2, 'rg': 0.5},
            {'name': 'T=2 correlated rg=0.8', 'n_trait': 2, 'rg': 0.8},
            {'name': 'T=2 overlap', 'n_trait': 2, 'rg': 0.5,
             'rho_pheno': 0.3, 'n_overlap': [20000, 20000]},
        ]

    results = []
    for ii, scenario in enumerate(scenarios):
        print('\n=== Scenario %d: %s ===' % (ii+1, scenario.get('name', 'unnamed')))
        name = scenario.pop('name', 'unnamed')

        sim = simulate_mt(seed=seed+ii, **scenario)

        # Evaluate true marginal associations as a baseline
        for pp in range(sim['n_pop']):
            for tt in range(sim['n_trait']):
                beta_mrg_flat = sim['beta_mrg'][(pp, tt)].flatten()
                beta_true_flat = sim['beta_true'][(pp, tt)]
                m = evaluate_prediction(beta_mrg_flat, beta_true_flat, R=sim['R'])
                print('  Pop %d, Trait %d: corr=%.4f, mse=%.6f, r2_eff=%.4f' %
                      (pp, tt, m['corr'], m['mse'], m['r2_effects']))
                if 'r2_genetic' in m:
                    print('    r2_genetic=%.4f' % m['r2_genetic'])

        results.append({'name': name, 'scenario': scenario, 'sim_data': sim})
        scenario['name'] = name  # restore

    return results


if __name__ == '__main__':
    print('Running PRS-CSx-MT simulation study...\n')
    run_simulation_study()
