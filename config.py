'''This module holds classes for the logging and model configuration
parameters that are set manually in here. There are also the filtering
functions used to preprocess the data

Learned negative binomial dispersion parameters
-----------------------------------------------
a0
	median: 3.021173158076349e-05
	mean: 3.039336514482573e-05
	25th percentile: 2.8907661553542307e-05
	75th percentile: 3.1848563862236224e-05
	acceptance rate: 0.3636
a1
	median: 0.03610445832385458
	mean: 0.036163868481381596
	25th percentile: 0.034369620675005035
	75th percentile: 0.0378392670993046
	acceptance rate: 0.5324

'''
from names import STRNAMES
import logging
import numpy as np
import pylab as pl
import sys
import math
import pandas as pd

# Parameters for a0 and a1
NEGBIN_A0 = 3.039336514482573e-05
NEGBIN_A1 = 0.036163868481381596
LEARNED_LOGNORMAL_SCALE = 0.3411239239789811

# File locations
GRAPH_NAME = 'graph'
MCMC_FILENAME = 'mcmc.pkl'
SUBJSET_FILENAME = 'subjset.pkl'
VALIDATION_SUBJSET_FILENAME = 'validate_subjset.pkl'
SYNDATA_FILENAME = 'syndata.pkl'
GRAPH_FILENAME = 'graph.pkl'
HDF5_FILENAME = 'traces.hdf5'
TRACER_FILENAME = 'tracer.pkl'
PARAMS_FILENAME = 'params.pkl'
FPARAMS_FILENAME = 'filtering_params.pkl'
SYNPARAMS_FILENAME = 'synthetic_params.pkl'
MLCRR_RESULTS_FILENAME = 'mlcrr_results.pkl'

PHYLOGENETIC_TREE_FILENAME = 'raw_data/phylogenetic_tree_branch_len_preserved.nhx'

def calculate_reads_a0a1(desired_percent_variation):
    '''
    At the full noise level, in terms of % noise of the signal at:
        10000 reads, the signal is ~10% noise
        100 reads, the signal is ~20% noise
        10 reads, the signal is ~30% noise

    When we scale the a0 and a1 terms, we are assuming that you want to
    scale the high abundance bacteria for that signal and we scale the
    a0 parameter such that they stay relative to each other.

    If == -1 we set to the full noise
    '''
    if desired_percent_variation == -1:
        desired_percent_variation = 0.05
    p = desired_percent_variation / 0.05
    return NEGBIN_A0*p, NEGBIN_A1*p

def isModelConfig(x):
    '''Checks if the input array is a model config object

    Parameters
    ----------
    x : any
        Instance we are checking

    Returns
    -------
    bool
        True if `x` is a a model config object
    '''
    return x is not None and issubclass(x.__class__, _BaseModelConfig)


class _BaseModelConfig(pl.Saveable):

    def __str__(self):
        s = '{}'.format(self.__class__.__name__)
        for k,v in vars(self).items():
            s += '\n\t{}: {}'.format(k,v)
        return s

    def suffix(self):
        raise NotImplementedError('Need to implement')


class ModelConfigICML(_BaseModelConfig):
    '''Configuration parameters for the model


    System initialization
    ---------------------
    SEED : int
        The random seed - set for all different modules (through pylab)
    DATA_FILENAME : str
        Location of the real data
    BURNIN : int
        Number of initial iterations to throw away
    N_SAMPLES : int
        Total number of iterations to perform
    CHECKPOINT : int
        This is the number of iterations that are saved to RAM until it is written
        to disk
    INTERMEDIATE_STEP : float, None
        This is the time step to do the intermediate points
        If there are no intermediate points added then we add no time points
    ADD_MIN_REL_ABUDANCE : bool
        If this is True, it will add the minimum relative abundance to all the data
    PROCESS_VARIANCE_TYPE : str
        Type of process variance to learn
        Options
            'homoscedastic'
                Not Implemented
            'heteroscedastic-global'
                Learn a heteroscadastic process variance (scales with the abundance)
                with global parameters (same v1 and v2 for every ASV)
            'heteroscedastic-per-asv'
                Learn v1 and v2 for each ASV separately
    DATA_DTYPE : str
        Type of data we are going to regress on
        Options:
            'abs': absolute abundance (qPCR*relative_abundance) data
            'rel': relative abundance data
            'raw': raw count data
    DIAGNOSTIC_VARIABLES : list(str)
        These are the names of the variables that you want to trace that are not
        necessarily variables we are learning. These are more for monitoring the
        inference
    QPCR_NORMALIZATION_MAX_VALUE : int, None
        Max value to set the qpcr value to. Rescale everything so that it is proportional
        to each other. If None there are no rescalings
    C_M : numeric
        This is the level of reintroduction of microbes each day

    Which parameters to learn
    -------------------------
    If the following parameters are true, then we add them to the inference order.
    These should all be `bool`s except for `INFERENCE_ORDER` which should be a list
    of `str`s.

    LEARN_BETA : bool
        Growth, self-interactions, interactions
    LEARN_CONCENTRATION : bool
        Concentration parameter of the clustering of the interactions
    LEARN_CLUSTER_ASSIGNMENTS : bool
        Cluster assignments to cluster the interactions
    LEARN_INDICATORS : bool
        Clustered interaction indicators of the interactions
    LEARN_INDICATOR_PROBABILITY : bool
        Probability of a positive interaction indicator
    LEARN_PRIOR_VAR_GROWTH : bool
        Prior variance of the growth
    LEARN_PRIOR_VAR_SELF_INTERACTIONS : bool
        Prior variance of the self-interactions
    LEARN_PRIOR_VAR_INTERACTIONS : bool
        Prior variance of the clustered interactions
    LEARN_PROCESS_VAR : bool
        Process variance parameters
    LEARN_FILTERING : bool
        Learn the auxiliary and the latent trajectory
    LEARN_PERT_VALUE : bool
        Magnitudes of the perturbation effects
    LEARN_PERT_INDICATOR : bool
        Clustered indicators of the perturbation effects
    LEARN_PERT_INDICATOR_PROBABILITY : bool
        Probability of a cluster ebing affected by the perturbation
    INFERENCE_ORDER : list
        This is the global order to learn the paramters. If one of the above parameters are
        False, then their respective name is removed from the inference order

    Initialization parameters
    -------------------------
    These are the arguments to send to the initialization. These should all be dictionaries
    which maps a string (argument) to its value to be passed into the `initialize` function.
    Last parameter is the initialization order, which is the order that we call the
    `initialize` function. These functions are called whether if the variables are being
    learned or not
    '''
    def __init__(self, output_basepath, data_path, data_seed, init_seed, a0, a1,
        n_samples, burnin, pcc, clustering_on):
        '''Initialize
        '''
        self.OUTPUT_BASEPATH = output_basepath
        self.DATA_PATH = data_path
        self.DATA_SEED = data_seed
        self.INIT_SEED = init_seed
        self.VALIDATION_SEED = 11195573

        self.DATA_FILENAME = 'pickles/real_subjectset.pkl'
        self.BURNIN = burnin
        self.N_SAMPLES = n_samples
        self.CHECKPOINT = 100
        self.ADD_MIN_REL_ABUNDANCE = False
        self.PROCESS_VARIANCE_TYPE = 'multiplicative-global'
        self.DATA_DTYPE = 'abs'
        self.DIAGNOSTIC_VARIABLES = ['n_clusters']
        self.DELETE_FIRST_TIMEPOINT = False

        self.GROWTH_TRUNCATION_SETTINGS = 'positive' #'in-vivo'
        self.SELF_INTERACTIONS_TRUNCATION_SETTINGS = 'positive'

        self.QPCR_NORMALIZATION_MAX_VALUE = 100
        self.C_M = 1e5

        # This is whether to use the log-scale dynamics or not
        self.DATA_LOGSCALE = True
        self.PERTURBATIONS_ADDITIVE = False

        self.MP_FILTERING = 'full'
        self.MP_INDICATORS = None
        self.MP_CLUSTERING = 'full-8'
        self.MP_ZERO_INFLATION = None
        self.RELATIVE_LOG_MARGINAL_INDICATORS = True
        self.RELATIVE_LOG_MARGINAL_PERT_INDICATORS = True
        self.RELATIVE_LOG_MARGINAL_CLUSTERING = False
        self.PERCENT_CHANGE_CLUSTERING = pcc

        self.NEGBIN_A0 = a0
        self.NEGBIN_A1 = a1
        self.CLUSTERING_ON = clustering_on
        self.N_QPCR_BUCKETS = 3
        
        self.LEARN = {
            STRNAMES.REGRESSCOEFF: True,
            STRNAMES.PROCESSVAR: True,
            STRNAMES.PRIOR_VAR_GROWTH: False,
            STRNAMES.PRIOR_VAR_SELF_INTERACTIONS: False,
            STRNAMES.PRIOR_VAR_INTERACTIONS: True,
            STRNAMES.PRIOR_VAR_PERT: True,
            STRNAMES.PRIOR_MEAN_GROWTH: True,
            STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS: True,
            STRNAMES.PRIOR_MEAN_INTERACTIONS: True,
            STRNAMES.PRIOR_MEAN_PERT: True,
            STRNAMES.FILTERING: True,
            STRNAMES.ZERO_INFLATION: False,
            STRNAMES.CLUSTERING: True, #clustering_on,
            STRNAMES.CONCENTRATION: True, #clustering_on,
            STRNAMES.CLUSTER_INTERACTION_INDICATOR: True,
            STRNAMES.INDICATOR_PROB: True,
            STRNAMES.PERT_INDICATOR: True,
            STRNAMES.PERT_INDICATOR_PROB: True,
            STRNAMES.QPCR_SCALES: False,
            STRNAMES.QPCR_DOFS: False,
            STRNAMES.QPCR_VARIANCES: False}

        self.INFERENCE_ORDER = [
            STRNAMES.CLUSTER_INTERACTION_INDICATOR,
            STRNAMES.INDICATOR_PROB,
            STRNAMES.PERT_INDICATOR,
            STRNAMES.PERT_INDICATOR_PROB,
            STRNAMES.REGRESSCOEFF,
            STRNAMES.PRIOR_MEAN_INTERACTIONS,
            STRNAMES.PRIOR_MEAN_PERT,
            STRNAMES.PRIOR_MEAN_GROWTH,
            STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS,
            STRNAMES.PRIOR_VAR_GROWTH,
            STRNAMES.PRIOR_VAR_SELF_INTERACTIONS,
            STRNAMES.PRIOR_VAR_INTERACTIONS,
            STRNAMES.PRIOR_VAR_PERT,
            STRNAMES.PROCESSVAR,
            STRNAMES.ZERO_INFLATION,
            STRNAMES.QPCR_SCALES,
            STRNAMES.QPCR_DOFS,
            STRNAMES.QPCR_VARIANCES,
            STRNAMES.FILTERING,
            STRNAMES.CLUSTERING,
            STRNAMES.CONCENTRATION]

        self.INITIALIZATION_KWARGS = {
            STRNAMES.QPCR_VARIANCES: {
                'value_option': 'empirical'},
            STRNAMES.QPCR_SCALES: {
                'value_option': 'prior-mean',
                'scale_option': 'empirical',
                'dof_option': 'diffuse',
                'proposal_option': 'auto',
                'target_acceptance_rate': 'optimal',
                'end_tune': 'half-burnin',
                'tune': 50,
                'delay':0},
            STRNAMES.QPCR_DOFS: {
                'value_option': 'diffuse',
                'low_option': 'valid',
                'high_option': 'med',
                'proposal_option': 'auto',
                'target_acceptance_rate': 'optimal',
                'end_tune': 'half-burnin',
                'tune': 50,
                'delay': 0},
            STRNAMES.PERT_VALUE: {
                'value_option': 'prior-mean',
                'delay':0},
            STRNAMES.PERT_INDICATOR_PROB: {
                'value_option': 'prior-mean',
                'hyperparam_option': 'weak-agnostic',
                'delay':0},
            STRNAMES.PERT_INDICATOR: {
                'value_option': 'all-off',
                'delay':0},
            STRNAMES.PRIOR_VAR_PERT: {
                'value_option': 'prior-mean',
                'scale_option': 'diffuse',
                'dof_option': 'diffuse',
                'delay': 0},
            STRNAMES.PRIOR_MEAN_PERT: {
                'value_option': 'prior-mean',
                'mean_option': 'zero',
                'var_option': 'diffuse',
                'delay':0},
            STRNAMES.PRIOR_VAR_GROWTH: {
                'value_option': 'prior-mean',
                'scale_option': 'inflated-median',
                'dof_option': 'diffuse',
                'proposal_option': 'tight',
                'target_acceptance_rate': 'optimal',
                'end_tune': 'half-burnin',
                'tune': 50,
                'delay':0},
            STRNAMES.PRIOR_MEAN_GROWTH: {
                'value_option': 'prior-mean',
                'mean_option': 'manual',
                'var_option': 'diffuse-linear-regression',
                'proposal_option': 'auto',
                'target_acceptance_rate': 0.44,
                'tune': 50,
                'end_tune': 'half-burnin',
                'truncation_settings': self.GROWTH_TRUNCATION_SETTINGS,
                'delay':0, 'mean': 1},
            STRNAMES.GROWTH_VALUE: {
                'value_option': 'linear-regression', #'prior-mean',
                'truncation_settings': self.GROWTH_TRUNCATION_SETTINGS,
                'delay': 0},
            STRNAMES.PRIOR_VAR_SELF_INTERACTIONS: {
                'value_option': 'prior-mean',
                'scale_option': 'inflated-median',
                'dof_option': 'diffuse',
                'proposal_option': 'tight',
                'target_acceptance_rate': 'optimal',
                'end_tune': 'half-burnin',
                'tune': 50,
                'delay':0},
            STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS: {
                'value_option': 'prior-mean',
                'mean_option': 'median-linear-regression',
                'var_option': 'diffuse-linear-regression',
                'proposal_option': 'auto',
                'target_acceptance_rate': 0.44,
                'tune': 50,
                'end_tune': 'half-burnin',
                'truncation_settings': self.SELF_INTERACTIONS_TRUNCATION_SETTINGS,
                'delay':0},
            STRNAMES.SELF_INTERACTION_VALUE: {
                'value_option': 'linear-regression',
                'truncation_settings': self.SELF_INTERACTIONS_TRUNCATION_SETTINGS,
                'delay': 0},
            STRNAMES.PRIOR_VAR_INTERACTIONS: {
                'value_option': 'auto',
                'dof_option': 'diffuse',
                'scale_option': 'same-as-aii',
                'mean_scaling_factor': 1,
                'delay': 0},
            STRNAMES.PRIOR_MEAN_INTERACTIONS: {
                'value_option': 'prior-mean',
                'mean_option': 'zero',
                'var_option': 'same-as-aii',
                'delay':0},
            STRNAMES.CLUSTER_INTERACTION_VALUE: {
                'value_option': 'all-off',
                'delay': 0},
            STRNAMES.CLUSTER_INTERACTION_INDICATOR: {
                'delay':0,
                'run_every_n_iterations': 1},
            STRNAMES.INDICATOR_PROB: {
                'value_option': 'auto',
                'hyperparam_option': 'weak-agnostic',
                'delay': 0},
            STRNAMES.FILTERING: {
                'x_value_option':  'loess',
                # 'q_value_option': 'coupling', #'loess',
                # 'hyperparam_option': 'manual',
                'tune': (int(self.BURNIN/2), 50),
                'a0': self.NEGBIN_A0,
                'a1': self.NEGBIN_A1,
                'v1': 1e-4,
                'v2': 1e-4,
                'proposal_init_scale':.001,
                'intermediate_interpolation': 'linear-interpolation',
                'intermediate_step': None, #('step', (1, None)), 
                'essential_timepoints': 'union',
                'delay': 1,
                'window': 6,
                'plot_initial': False,
                'target_acceptance_rate': 0.44},
            STRNAMES.ZERO_INFLATION: {
                'value_option': 'manual',
                'delay': 0},
            STRNAMES.CONCENTRATION: {
                'value_option': 'prior-mean',
                'hyperparam_option': 'diffuse',
                'delay': 0, 'n_iter': 20},
            STRNAMES.CLUSTERING: {
                'value_option': 'spearman',
                'delay': 2,
                'n_clusters': 10,
                'percent_mix': self.PERCENT_CHANGE_CLUSTERING,
                'run_every_n_iterations': 4},
            STRNAMES.REGRESSCOEFF: {
                'update_jointly_pert_inter': True,
                'update_jointly_growth_si': False,
                'tune': 50,
                'end_tune': 'half-burnin'},
            STRNAMES.PROCESSVAR: {
                # 'v1': 0.2**2,
                # 'v2': 1,
                # 'q_option': 'previous-t'}, #'previous-t'},
                'dof_option': 'diffuse', # 'half', 
                'scale_option': 'med',
                'value_option': 'prior-mean',
                'delay': 0}
        }

        self.INITIALIZATION_ORDER = [
            STRNAMES.FILTERING,
            STRNAMES.ZERO_INFLATION,
            STRNAMES.CONCENTRATION,
            STRNAMES.CLUSTERING,
            STRNAMES.PROCESSVAR,
            STRNAMES.PRIOR_MEAN_GROWTH,
            STRNAMES.PRIOR_VAR_GROWTH,
            STRNAMES.GROWTH_VALUE,
            STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS,
            STRNAMES.PRIOR_VAR_SELF_INTERACTIONS,
            STRNAMES.SELF_INTERACTION_VALUE,
            STRNAMES.PRIOR_MEAN_INTERACTIONS,
            STRNAMES.PRIOR_VAR_INTERACTIONS,
            STRNAMES.CLUSTER_INTERACTION_VALUE,
            STRNAMES.CLUSTER_INTERACTION_INDICATOR,
            STRNAMES.INDICATOR_PROB,
            STRNAMES.PRIOR_MEAN_PERT,
            STRNAMES.PRIOR_VAR_PERT,
            STRNAMES.PERT_INDICATOR,
			STRNAMES.PERT_VALUE,
            STRNAMES.PERT_INDICATOR_PROB,
            STRNAMES.REGRESSCOEFF,
            STRNAMES.QPCR_SCALES,
            STRNAMES.QPCR_DOFS,
            STRNAMES.QPCR_VARIANCES]

    def suffix(self):
        '''Create a suffix with the parameters
        '''
        perts = 'addit' if self.PERTURBATIONS_ADDITIVE else 'mult'
        
        s = '_ds{}_is{}_b{}_ns{}_co{}_perts{}'.format(
            self.DATA_SEED, self.INIT_SEED, self.BURNIN, self.N_SAMPLES,
            self.CLUSTERING_ON, perts)
        return s


class ModelConfigReal(_BaseModelConfig):
    '''Configuration parameters for the model


    System initialization
    ---------------------
    SEED : int
        The random seed - set for all different modules (through pylab)
    DATA_FILENAME : str
        Location of the real data
    BURNIN : int
        Number of initial iterations to throw away
    N_SAMPLES : int
        Total number of iterations to perform
    CHECKPOINT : int
        This is the number of iterations that are saved to RAM until it is written
        to disk
    INTERMEDIATE_STEP : float, None
        This is the time step to do the intermediate points
        If there are no intermediate points added then we add no time points
    UPDATE_DYNAMICS_JOINTLY : bool
        If True, we sample the growth, self_interactions, and interactions
        jointly from a multivariate normal distribution. If False, we sample
        the separately (growth from a positively truncated normal, self-interactions
        from a negatively truncated normal, and interactions from a multivariate normal)
    ADD_MIN_REL_ABUDANCE : bool
        If this is True, it will add the minimum relative abundance to all the data
    PROCESS_VARIANCE_TYPE : str
        Type of process variance to learn
        Options
            'homoscedastic'
                Not Implemented
            'heteroscedastic-global'
                Learn a heteroscadastic process variance (scales with the abundance)
                with global parameters (same v1 and v2 for every ASV)
            'heteroscedastic-per-asv'
                Learn v1 and v2 for each ASV separately
    DATA_DTYPE : str
        Type of data we are going to regress on
        Options:
            'abs': absolute abundance (qPCR*relative_abundance) data
            'rel': relative abundance data
            'raw': raw count data
    DIAGNOSTIC_VARIABLES : list(str)
        These are the names of the variables that you want to trace that are not
        necessarily variables we are learning. These are more for monitoring the
        inference
    QPCR_NORMALIZATION_MAX_VALUE : int, None
        Max value to set the qpcr value to. Rescale everything so that it is proportional
        to each other. If None there are no rescalings
    C_M : numeric
        This is the level of reintroduction of microbes each day

    Which parameters to learn
    -------------------------
    If the following parameters are true, then we add them to the inference order.
    These should all be `bool`s except for `INFERENCE_ORDER` which should be a list
    of `str`s.

    LEARN_BETA : bool
        Growth, self-interactions, interactions
    LEARN_CONCENTRATION : bool
        Concentration parameter of the clustering of the interactions
    LEARN_CLUSTER_ASSIGNMENTS : bool
        Cluster assignments to cluster the interactions
    LEARN_INDICATORS : bool
        Clustered interaction indicators of the interactions
    LEARN_INDICATOR_PROBABILITY : bool
        Probability of a positive interaction indicator
    LEARN_PRIOR_VAR_GROWTH : bool
        Prior variance of the growth
    LEARN_PRIOR_VAR_SELF_INTERACTIONS : bool
        Prior variance of the self-interactions
    LEARN_PRIOR_VAR_INTERACTIONS : bool
        Prior variance of the clustered interactions
    LEARN_PROCESS_VAR : bool
        Process variance parameters
    LEARN_FILTERING : bool
        Learn the auxiliary and the latent trajectory
    LEARN_PERT_VALUE : bool
        Magnitudes of the perturbation effects
    LEARN_PERT_INDICATOR : bool
        Clustered indicators of the perturbation effects
    LEARN_PERT_INDICATOR_PROBABILITY : bool
        Probability of a cluster ebing affected by the perturbation
    INFERENCE_ORDER : list
        This is the global order to learn the paramters. If one of the above parameters are
        False, then their respective name is removed from the inference order

    Initialization parameters
    -------------------------
    These are the arguments to send to the initialization. These should all be dictionaries
    which maps a string (argument) to its value to be passed into the `initialize` function.
    Last parameter is the initialization order, which is the order that we call the
    `initialize` function. These functions are called whether if the variables are being
    learned or not
    '''
    def __init__(self, output_basepath, data_seed, init_seed, burnin, n_samples, pcc,
        leave_out, max_n_asvs, cross_validate, use_bsub):
        '''Customization parameters

        Parameters
        ----------
        output_basepath : str
            This is the basepath to save the output
        data_seed, init_seed : int
            This is the seed to initialize the data and the model, respectively.
        burnin, n_samples : int
            These are the number of iterations to initially throw away and the
            total number of samples for the MCMC chain, respectively
        pcc : float
            [0,1]. (Percent Change Clustering). What proportion of the ASVs to
            change clusters during each iteration.
        leave_out : list(str), str, None
            These are the subject name/s to leave out during inference and to use to test
            the predictive accuracy. If `None` then we leave non of them out.
        '''

        self.OUTPUT_BASEPATH = output_basepath
        self.DATA_SEED = data_seed
        self.INIT_SEED = init_seed
        self.VALIDATION_SEED = 11195573

        self.CROSS_VALIDATE = cross_validate
        self.USE_BSUB = use_bsub

        self.N_CPUS = 12
        self.N_GBS = 10000

        self.DATA_FILENAME = 'pickles/real_subjectset.pkl'
        self.BURNIN = burnin
        self.N_SAMPLES = n_samples
        self.CHECKPOINT = 100
        self.ADD_MIN_REL_ABUNDANCE = False
        self.PROCESS_VARIANCE_TYPE = 'multiplicative-global'
        self.DATA_DTYPE = 'abs'
        self.DIAGNOSTIC_VARIABLES = ['n_clusters']
        self.DELETE_FIRST_TIMEPOINT = True

        self.QPCR_NORMALIZATION_MAX_VALUE = 100
        self.C_M = 1e5
        self.LEAVE_OUT = leave_out
        self.MAX_N_ASVS = max_n_asvs
        self.DATA_LOGSCALE = True
        self.PERTURBATIONS_ADDITIVE = False

        self.GROWTH_TRUNCATION_SETTINGS = 'positive'
        self.SELF_INTERACTIONS_TRUNCATION_SETTINGS = 'positive'

        self.MP_FILTERING = 'full'
        self.MP_INDICATORS = None
        self.MP_CLUSTERING = None #'full-8'
        self.MP_ZERO_INFLATION = None
        self.RELATIVE_LOG_MARGINAL_INDICATORS = True
        self.RELATIVE_LOG_MARGINAL_PERT_INDICATORS = True
        self.RELATIVE_LOG_MARGINAL_CLUSTERING = False
        self.PERCENT_CHANGE_CLUSTERING = pcc

        self.NEGBIN_A0 = NEGBIN_A0
        self.NEGBIN_A1 = NEGBIN_A1
        self.QPCR_NOISE_SCALE = LEARNED_LOGNORMAL_SCALE
        self.N_QPCR_BUCKETS = 3

        self.LEARN = {
            STRNAMES.REGRESSCOEFF: True,
            STRNAMES.PROCESSVAR: True,
            STRNAMES.PRIOR_VAR_GROWTH: False,
            STRNAMES.PRIOR_VAR_SELF_INTERACTIONS: False,
            STRNAMES.PRIOR_VAR_INTERACTIONS: True,
            STRNAMES.PRIOR_VAR_PERT: True,
            STRNAMES.PRIOR_MEAN_GROWTH: True,
            STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS: True,
            STRNAMES.PRIOR_MEAN_INTERACTIONS: True,
            STRNAMES.PRIOR_MEAN_PERT: True,
            STRNAMES.FILTERING: True,
            STRNAMES.ZERO_INFLATION: False,
            STRNAMES.CLUSTERING: True,
            STRNAMES.CONCENTRATION: True, 
            STRNAMES.CLUSTER_INTERACTION_INDICATOR: True,
            STRNAMES.INDICATOR_PROB: True,
            STRNAMES.PERT_INDICATOR: True,
            STRNAMES.PERT_INDICATOR_PROB: True,
            STRNAMES.QPCR_SCALES: False,
            STRNAMES.QPCR_DOFS: False,
            STRNAMES.QPCR_VARIANCES: False}

        self.INFERENCE_ORDER = [
            STRNAMES.CLUSTER_INTERACTION_INDICATOR,
            STRNAMES.INDICATOR_PROB,
            STRNAMES.PERT_INDICATOR,
            STRNAMES.PERT_INDICATOR_PROB,
            STRNAMES.REGRESSCOEFF,
            STRNAMES.PRIOR_MEAN_INTERACTIONS,
            STRNAMES.PRIOR_MEAN_PERT,
            STRNAMES.PRIOR_MEAN_GROWTH,
            STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS,
            STRNAMES.PRIOR_VAR_GROWTH,
            STRNAMES.PRIOR_VAR_SELF_INTERACTIONS,
            STRNAMES.PRIOR_VAR_INTERACTIONS,
            STRNAMES.PRIOR_VAR_PERT,
            STRNAMES.PROCESSVAR,
            STRNAMES.ZERO_INFLATION,
            STRNAMES.QPCR_SCALES,
            STRNAMES.QPCR_DOFS,
            STRNAMES.QPCR_VARIANCES,
            STRNAMES.FILTERING,
            STRNAMES.CLUSTERING,
            STRNAMES.CONCENTRATION]

        self.INITIALIZATION_KWARGS = {
            STRNAMES.QPCR_VARIANCES: {
                'value_option': 'empirical'},
            STRNAMES.QPCR_SCALES: {
                'value_option': 'prior-mean',
                'scale_option': 'empirical',
                'dof_option': 'diffuse',
                'proposal_option': 'auto',
                'target_acceptance_rate': 'optimal',
                'end_tune': 'half-burnin',
                'tune': 50,
                'delay':0},
            STRNAMES.QPCR_DOFS: {
                'value_option': 'diffuse',
                'low_option': 'valid',
                'high_option': 'med',
                'proposal_option': 'auto',
                'target_acceptance_rate': 'optimal',
                'end_tune': 'half-burnin',
                'tune': 50,
                'delay': 0},
            STRNAMES.PERT_VALUE: {
                'value_option': 'prior-mean',
                'delay':0},
            STRNAMES.PERT_INDICATOR_PROB: {
                'value_option': 'prior-mean',
                'hyperparam_option': 'strong-sparse',
                'N': 25,
                'delay':0},
            STRNAMES.PERT_INDICATOR: {
                'value_option': 'all-off',
                'delay':0},
            STRNAMES.PRIOR_VAR_PERT: {
                'value_option': 'prior-mean',
                'scale_option': 'diffuse',
                'dof_option': 'diffuse',
                'delay': 0},
            STRNAMES.PRIOR_MEAN_PERT: {
                'value_option': 'prior-mean',
                'mean_option': 'zero',
                'var_option': 'diffuse',
                'delay':0},
            STRNAMES.PRIOR_VAR_GROWTH: {
                'value_option': 'prior-mean',
                'scale_option': 'inflated-median',
                'dof_option': 'diffuse',
                'proposal_option': 'tight',
                'target_acceptance_rate': 'optimal',
                'end_tune': 'half-burnin',
                'tune': 50,
                'delay':0},
            STRNAMES.PRIOR_MEAN_GROWTH: {
                'value_option': 'prior-mean',
                'mean_option': 'manual',
                'var_option': 'diffuse-linear-regression',
                'proposal_option': 'auto',
                'target_acceptance_rate': 0.44,
                'tune': 50,
                'end_tune': 'half-burnin',
                'truncation_settings': self.GROWTH_TRUNCATION_SETTINGS,
                'delay':0, 'mean': 1},
            STRNAMES.GROWTH_VALUE: {
                'value_option': 'linear-regression', #'prior-mean',
                'truncation_settings': self.GROWTH_TRUNCATION_SETTINGS,
                'delay': 0},
            STRNAMES.PRIOR_VAR_SELF_INTERACTIONS: {
                'value_option': 'prior-mean',
                'scale_option': 'inflated-median',
                'dof_option': 'diffuse',
                'proposal_option': 'tight',
                'target_acceptance_rate': 'optimal',
                'end_tune': 'half-burnin',
                'tune': 50,
                'delay':0},
            STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS: {
                'value_option': 'prior-mean',
                'mean_option': 'median-linear-regression',
                'var_option': 'diffuse-linear-regression',
                'proposal_option': 'auto',
                'target_acceptance_rate': 0.44,
                'tune': 50,
                'end_tune': 'half-burnin',
                'truncation_settings': self.SELF_INTERACTIONS_TRUNCATION_SETTINGS,
                'delay':0},
            STRNAMES.SELF_INTERACTION_VALUE: {
                'value_option': 'linear-regression',
                'truncation_settings': self.SELF_INTERACTIONS_TRUNCATION_SETTINGS,
                'delay': 0},
            STRNAMES.PRIOR_VAR_INTERACTIONS: {
                'value_option': 'auto',
                'dof_option': 'diffuse',
                'scale_option': 'same-as-aii',
                'mean_scaling_factor': 1,
                'delay': 0},
            STRNAMES.PRIOR_MEAN_INTERACTIONS: {
                'value_option': 'prior-mean',
                'mean_option': 'zero',
                'var_option': 'same-as-aii',
                'delay':0},
            STRNAMES.CLUSTER_INTERACTION_VALUE: {
                'value_option': 'all-off',
                'delay': 0},
            STRNAMES.CLUSTER_INTERACTION_INDICATOR: {
                'delay':0,
                'run_every_n_iterations': 1},
            STRNAMES.INDICATOR_PROB: {
                'value_option': 'auto',
                'hyperparam_option': 'strong-sparse',
                'N': 25,
                'delay': 0},
            STRNAMES.FILTERING: {
                'x_value_option':  'loess',
                # 'q_value_option': 'coupling', #'loess',
                # 'hyperparam_option': 'manual',
                'tune': (int(self.BURNIN/2), 50),
                'a0': self.NEGBIN_A0,
                'a1': self.NEGBIN_A1,
                'v1': 1e-4,
                'v2': 1e-4,
                'proposal_init_scale':.001,
                'intermediate_interpolation': 'linear-interpolation',
                'intermediate_step': None, #('step', (1, None)), 
                'essential_timepoints': 'union',
                'delay': 1,
                'window': 6,
                'plot_initial': False,
                'target_acceptance_rate': 0.44},
            STRNAMES.ZERO_INFLATION: {
                'value_option': 'manual',
                'delay': 0},
            STRNAMES.CONCENTRATION: {
                'value_option': 'prior-mean',
                'hyperparam_option': 'diffuse',
                'delay': 0, 'n_iter': 20},
            STRNAMES.CLUSTERING: {
                'value_option': 'spearman', #'fixed-topology',
                'delay': 2,
                'n_clusters': 30,
                'value': 'output_real/pylab24/real_runs/strong_priors/healthy0_5_0.0001_rel_2_5/ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl',
                'percent_mix': self.PERCENT_CHANGE_CLUSTERING,
                'run_every_n_iterations': 4},
            STRNAMES.REGRESSCOEFF: {
                'update_jointly_pert_inter': True,
                'update_jointly_growth_si': False,
                'tune': 50,
                'end_tune': 'half-burnin'},
            STRNAMES.PROCESSVAR: {
                # 'v1': 0.2**2,
                # 'v2': 1,
                # 'q_option': 'previous-t'}, #'previous-t'},
                'dof_option': 'diffuse', # 'half', 
                'scale_option': 'med',
                'value_option': 'prior-mean',
                'delay': 0}
        }

        self.INITIALIZATION_ORDER = [
            STRNAMES.FILTERING,
            STRNAMES.ZERO_INFLATION,
            STRNAMES.CONCENTRATION,
            STRNAMES.CLUSTERING,
            STRNAMES.PROCESSVAR,
            STRNAMES.PRIOR_MEAN_GROWTH,
            STRNAMES.PRIOR_VAR_GROWTH,
            STRNAMES.GROWTH_VALUE,
            STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS,
            STRNAMES.PRIOR_VAR_SELF_INTERACTIONS,
            STRNAMES.SELF_INTERACTION_VALUE,
            STRNAMES.PRIOR_MEAN_INTERACTIONS,
            STRNAMES.PRIOR_VAR_INTERACTIONS,
            STRNAMES.CLUSTER_INTERACTION_VALUE,
            STRNAMES.CLUSTER_INTERACTION_INDICATOR,
            STRNAMES.INDICATOR_PROB,
            STRNAMES.PRIOR_MEAN_PERT,
            STRNAMES.PRIOR_VAR_PERT,
            STRNAMES.PERT_INDICATOR,
			STRNAMES.PERT_VALUE,
            STRNAMES.PERT_INDICATOR_PROB,
            STRNAMES.REGRESSCOEFF,
            STRNAMES.QPCR_SCALES,
            STRNAMES.QPCR_DOFS,
            STRNAMES.QPCR_VARIANCES]

    def suffix(self):
        '''Create a suffix with the parameters
        '''
        perts = 'addit' if self.PERTURBATIONS_ADDITIVE else 'mult'
        if self.LEAVE_OUT is not None:
            try:
                lo = str(tuple(list(self.LEAVE_OUT))).replace(',','_').replace(
                    '(','').replace(')','').replace(' ','')
            except:
                lo = str(self.LEAVE_OUT)
        else:
            lo = None
        s = 'ds{}_is{}_b{}_ns{}_lo{}_mo{}_log{}_perts{}'.format(
            self.DATA_SEED, self.INIT_SEED, self.BURNIN, self.N_SAMPLES, lo,
            self.MAX_N_ASVS, self.DATA_LOGSCALE, perts)
        return s

    def cv_suffix(self):
        '''Create a master suffix with the parameters
        '''
        perts = 'addit' if self.PERTURBATIONS_ADDITIVE else 'mult'
        s = 'ds{}_is{}_b{}_ns{}_mo{}_log{}_perts{}'.format(
            self.DATA_SEED, self.INIT_SEED, self.BURNIN, self.N_SAMPLES,
            self.MAX_N_ASVS, self.DATA_LOGSCALE, perts)
        return s

    def cv_single_suffix(self):
        '''Create a suffix for a single cv round
        '''
        return 'leave_out{}'.format(self.LEAVE_OUT)


class ModelConfigMDSINE1(_BaseModelConfig):
    '''Configuration parameters for the model


    System initialization
    ---------------------
    SEED : int
        The random seed - set for all different modules (through pylab)
    DATA_FILENAME : str
        Location of the real data
    BURNIN : int
        Number of initial iterations to throw away
    N_SAMPLES : int
        Total number of iterations to perform
    CHECKPOINT : int
        This is the number of iterations that are saved to RAM until it is written
        to disk
    INTERMEDIATE_STEP : float, None
        This is the time step to do the intermediate points
        If there are no intermediate points added then we add no time points
    UPDATE_DYNAMICS_JOINTLY : bool
        If True, we sample the growth, self_interactions, and interactions
        jointly from a multivariate normal distribution. If False, we sample
        the separately (growth from a positively truncated normal, self-interactions
        from a negatively truncated normal, and interactions from a multivariate normal)
    ADD_MIN_REL_ABUDANCE : bool
        If this is True, it will add the minimum relative abundance to all the data
    PROCESS_VARIANCE_TYPE : str
        Type of process variance to learn
        Options
            'homoscedastic'
                Not Implemented
            'heteroscedastic-global'
                Learn a heteroscadastic process variance (scales with the abundance)
                with global parameters (same v1 and v2 for every ASV)
            'heteroscedastic-per-asv'
                Learn v1 and v2 for each ASV separately
    DATA_DTYPE : str
        Type of data we are going to regress on
        Options:
            'abs': absolute abundance (qPCR*relative_abundance) data
            'rel': relative abundance data
            'raw': raw count data
    DIAGNOSTIC_VARIABLES : list(str)
        These are the names of the variables that you want to trace that are not
        necessarily variables we are learning. These are more for monitoring the
        inference
    QPCR_NORMALIZATION_MAX_VALUE : int, None
        Max value to set the qpcr value to. Rescale everything so that it is proportional
        to each other. If None there are no rescalings
    C_M : numeric
        This is the level of reintroduction of microbes each day

    Which parameters to learn
    -------------------------
    If the following parameters are true, then we add them to the inference order.
    These should all be `bool`s except for `INFERENCE_ORDER` which should be a list
    of `str`s.

    LEARN_BETA : bool
        Growth, self-interactions, interactions
    LEARN_CONCENTRATION : bool
        Concentration parameter of the clustering of the interactions
    LEARN_CLUSTER_ASSIGNMENTS : bool
        Cluster assignments to cluster the interactions
    LEARN_INDICATORS : bool
        Clustered interaction indicators of the interactions
    LEARN_INDICATOR_PROBABILITY : bool
        Probability of a positive interaction indicator
    LEARN_PRIOR_VAR_GROWTH : bool
        Prior variance of the growth
    LEARN_PRIOR_VAR_SELF_INTERACTIONS : bool
        Prior variance of the self-interactions
    LEARN_PRIOR_VAR_INTERACTIONS : bool
        Prior variance of the clustered interactions
    LEARN_PROCESS_VAR : bool
        Process variance parameters
    LEARN_FILTERING : bool
        Learn the auxiliary and the latent trajectory
    LEARN_PERT_VALUE : bool
        Magnitudes of the perturbation effects
    LEARN_PERT_INDICATOR : bool
        Clustered indicators of the perturbation effects
    LEARN_PERT_INDICATOR_PROBABILITY : bool
        Probability of a cluster ebing affected by the perturbation
    INFERENCE_ORDER : list
        This is the global order to learn the paramters. If one of the above parameters are
        False, then their respective name is removed from the inference order

    Initialization parameters
    -------------------------
    These are the arguments to send to the initialization. These should all be dictionaries
    which maps a string (argument) to its value to be passed into the `initialize` function.
    Last parameter is the initialization order, which is the order that we call the
    `initialize` function. These functions are called whether if the variables are being
    learned or not
    '''
    def __init__(self, output_basepath, data_seed, init_seed, burnin, n_samples, pcc,
        leave_out, max_n_asvs, cross_validate, use_bsub):
        '''Customization parameters

        Parameters
        ----------
        output_basepath : str
            This is the basepath to save the output
        data_seed, init_seed : int
            This is the seed to initialize the data and the model, respectively.
        burnin, n_samples : int
            These are the number of iterations to initially throw away and the
            total number of samples for the MCMC chain, respectively
        pcc : float
            [0,1]. (Percent Change Clustering). What proportion of the ASVs to
            change clusters during each iteration.
        leave_out : list(str), str, None
            These are the subject name/s to leave out during inference and to use to test
            the predictive accuracy. If `None` then we leave non of them out.
        '''

        self.OUTPUT_BASEPATH = output_basepath
        self.DATA_SEED = data_seed
        self.INIT_SEED = init_seed
        self.VALIDATION_SEED = 11195573

        self.CROSS_VALIDATE = cross_validate
        self.USE_BSUB = use_bsub

        self.N_CPUS = 12
        self.N_GBS = 10000

        self.DATA_FILENAME = 'pickles/subjset_cdiff.pkl'
        self.BURNIN = burnin
        self.N_SAMPLES = n_samples
        self.CHECKPOINT = 100
        self.ADD_MIN_REL_ABUNDANCE = False
        self.PROCESS_VARIANCE_TYPE = 'multiplicative-global'
        self.DATA_DTYPE = 'abs'
        self.DIAGNOSTIC_VARIABLES = ['n_clusters']
        self.DELETE_FIRST_TIMEPOINT = False
        self.DELETE_SHORT_DATAS = False

        self.QPCR_NORMALIZATION_MAX_VALUE = 100
        self.C_M = 1e5
        self.LEAVE_OUT = leave_out
        self.MAX_N_ASVS = max_n_asvs
        self.DATA_LOGSCALE = True
        self.PERTURBATIONS_ADDITIVE = False

        self.GROWTH_TRUNCATION_SETTINGS = 'positive'
        self.SELF_INTERACTIONS_TRUNCATION_SETTINGS = 'positive'

        self.MP_FILTERING = 'debug'
        self.MP_INDICATORS = None
        self.MP_CLUSTERING = None #'full-8'
        self.MP_ZERO_INFLATION = None
        self.RELATIVE_LOG_MARGINAL_INDICATORS = True
        self.RELATIVE_LOG_MARGINAL_PERT_INDICATORS = True
        self.RELATIVE_LOG_MARGINAL_CLUSTERING = False
        self.PERCENT_CHANGE_CLUSTERING = pcc

        self.NEGBIN_A0 = NEGBIN_A0
        self.NEGBIN_A1 = NEGBIN_A1
        self.QPCR_NOISE_SCALE = LEARNED_LOGNORMAL_SCALE
        self.N_QPCR_BUCKETS = 3

        self.LEARN = {
            STRNAMES.REGRESSCOEFF: True,
            STRNAMES.PROCESSVAR: True,
            STRNAMES.PRIOR_VAR_GROWTH: False,
            STRNAMES.PRIOR_VAR_SELF_INTERACTIONS: False,
            STRNAMES.PRIOR_VAR_INTERACTIONS: True,
            STRNAMES.PRIOR_VAR_PERT: True,
            STRNAMES.PRIOR_MEAN_GROWTH: True,
            STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS: True,
            STRNAMES.PRIOR_MEAN_INTERACTIONS: True,
            STRNAMES.PRIOR_MEAN_PERT: True,
            STRNAMES.FILTERING: True,
            STRNAMES.ZERO_INFLATION: False,
            STRNAMES.CLUSTERING: False,
            STRNAMES.CONCENTRATION: False, 
            STRNAMES.CLUSTER_INTERACTION_INDICATOR: True,
            STRNAMES.INDICATOR_PROB: True,
            STRNAMES.PERT_INDICATOR: True,
            STRNAMES.PERT_INDICATOR_PROB: True,
            STRNAMES.QPCR_SCALES: False,
            STRNAMES.QPCR_DOFS: False,
            STRNAMES.QPCR_VARIANCES: False}

        self.INFERENCE_ORDER = [
            STRNAMES.CLUSTER_INTERACTION_INDICATOR,
            STRNAMES.INDICATOR_PROB,
            STRNAMES.PERT_INDICATOR,
            STRNAMES.PERT_INDICATOR_PROB,
            STRNAMES.REGRESSCOEFF,
            STRNAMES.PRIOR_MEAN_INTERACTIONS,
            STRNAMES.PRIOR_MEAN_PERT,
            STRNAMES.PRIOR_MEAN_GROWTH,
            STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS,
            STRNAMES.PRIOR_VAR_GROWTH,
            STRNAMES.PRIOR_VAR_SELF_INTERACTIONS,
            STRNAMES.PRIOR_VAR_INTERACTIONS,
            STRNAMES.PRIOR_VAR_PERT,
            STRNAMES.PROCESSVAR,
            STRNAMES.ZERO_INFLATION,
            STRNAMES.QPCR_SCALES,
            STRNAMES.QPCR_DOFS,
            STRNAMES.QPCR_VARIANCES,
            STRNAMES.FILTERING,
            STRNAMES.CLUSTERING,
            STRNAMES.CONCENTRATION]

        self.INITIALIZATION_KWARGS = {
            STRNAMES.QPCR_VARIANCES: {
                'value_option': 'empirical'},
            STRNAMES.QPCR_SCALES: {
                'value_option': 'prior-mean',
                'scale_option': 'empirical',
                'dof_option': 'diffuse',
                'proposal_option': 'auto',
                'target_acceptance_rate': 'optimal',
                'end_tune': 'half-burnin',
                'tune': 50,
                'delay':0},
            STRNAMES.QPCR_DOFS: {
                'value_option': 'diffuse',
                'low_option': 'valid',
                'high_option': 'med',
                'proposal_option': 'auto',
                'target_acceptance_rate': 'optimal',
                'end_tune': 'half-burnin',
                'tune': 50,
                'delay': 0},
            STRNAMES.PERT_VALUE: {
                'value_option': 'prior-mean',
                'delay':0},
            STRNAMES.PERT_INDICATOR_PROB: {
                'value_option': 'prior-mean',
                'hyperparam_option': 'strong-sparse',
                'delay':0},
            STRNAMES.PERT_INDICATOR: {
                'value_option': 'all-off',
                'delay':0},
            STRNAMES.PRIOR_VAR_PERT: {
                'value_option': 'prior-mean',
                'scale_option': 'diffuse',
                'dof_option': 'diffuse',
                'delay': 0},
            STRNAMES.PRIOR_MEAN_PERT: {
                'value_option': 'prior-mean',
                'mean_option': 'zero',
                'var_option': 'diffuse',
                'delay':0},
            STRNAMES.PRIOR_VAR_GROWTH: {
                'value_option': 'manual',
                'scale_option': 'inflated-median',
                'dof_option': 'diffuse',
                'proposal_option': 'tight',
                'target_acceptance_rate': 'optimal',
                'end_tune': 'half-burnin',
                'value': 1,
                'tune': 50,
                'delay':0},
            STRNAMES.PRIOR_MEAN_GROWTH: {
                'value_option': 'prior-mean',
                'mean_option': 'manual',
                'var_option': 'diffuse-linear-regression',
                'proposal_option': 'auto',
                'target_acceptance_rate': 0.44,
                'tune': 50,
                'end_tune': 'half-burnin',
                'truncation_settings': self.GROWTH_TRUNCATION_SETTINGS,
                'delay':0, 'mean': 1},
            STRNAMES.GROWTH_VALUE: {
                'value_option': 'linear-regression', #'prior-mean',
                'truncation_settings': self.GROWTH_TRUNCATION_SETTINGS,
                'delay': 0},
            STRNAMES.PRIOR_VAR_SELF_INTERACTIONS: {
                'value_option': 'prior-mean',
                'scale_option': 'inflated-median',
                'dof_option': 'diffuse',
                'proposal_option': 'tight',
                'target_acceptance_rate': 'optimal',
                'end_tune': 'half-burnin',
                'tune': 50,
                'delay':0},
            STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS: {
                'value_option': 'prior-mean',
                'mean_option': 'median-linear-regression',
                'var_option': 'diffuse-linear-regression',
                'proposal_option': 'auto',
                'target_acceptance_rate': 0.44,
                'tune': 50,
                'end_tune': 'half-burnin',
                'truncation_settings': self.SELF_INTERACTIONS_TRUNCATION_SETTINGS,
                'delay':0},
            STRNAMES.SELF_INTERACTION_VALUE: {
                'value_option': 'linear-regression',
                'truncation_settings': self.SELF_INTERACTIONS_TRUNCATION_SETTINGS,
                'delay': 0},
            STRNAMES.PRIOR_VAR_INTERACTIONS: {
                'value_option': 'auto',
                'dof_option': 'diffuse',
                'scale_option': 'same-as-aii',
                'mean_scaling_factor': 1,
                'delay': 0},
            STRNAMES.PRIOR_MEAN_INTERACTIONS: {
                'value_option': 'prior-mean',
                'mean_option': 'zero',
                'var_option': 'same-as-aii',
                'delay':0},
            STRNAMES.CLUSTER_INTERACTION_VALUE: {
                'value_option': 'all-off',
                'delay': 0},
            STRNAMES.CLUSTER_INTERACTION_INDICATOR: {
                'delay':0,
                'run_every_n_iterations': 1},
            STRNAMES.INDICATOR_PROB: {
                'value_option': 'auto',
                'hyperparam_option': 'strong-sparse',
                'delay': 0},
            STRNAMES.FILTERING: {
                'x_value_option':  'loess',
                # 'q_value_option': 'coupling', #'loess',
                # 'hyperparam_option': 'manual',
                'tune': (int(self.BURNIN/2), 50),
                'a0': self.NEGBIN_A0,
                'a1': self.NEGBIN_A1,
                'v1': 1e-4,
                'v2': 1e-4,
                'proposal_init_scale':.001,
                'intermediate_interpolation': 'linear-interpolation',
                'intermediate_step': None, #('step', (1, None)), 
                'essential_timepoints': None,
                'delay': 1,
                'window': 6,
                'calculate_qpcr_loglik': True,
                'plot_initial': False,
                'target_acceptance_rate': 0.44},
            STRNAMES.ZERO_INFLATION: {
                'value_option': 'manual',
                'delay': 0},
            STRNAMES.CONCENTRATION: {
                'value_option': 'prior-mean',
                'hyperparam_option': 'diffuse',
                'delay': 0, 'n_iter': 20},
            STRNAMES.CLUSTERING: {
                'value_option': 'no-clusters',
                'delay': 2,
                'n_clusters': 30,
                'percent_mix': self.PERCENT_CHANGE_CLUSTERING,
                'run_every_n_iterations': 4},
            STRNAMES.REGRESSCOEFF: {
                'update_jointly_pert_inter': True,
                'update_jointly_growth_si': False,
                'tune': 50,
                'end_tune': 'half-burnin'},
            STRNAMES.PROCESSVAR: {
                # 'v1': 0.2**2,
                # 'v2': 1,
                # 'q_option': 'previous-t'}, #'previous-t'},
                'dof_option': 'diffuse', # 'half', 
                'scale_option': 'med',
                'value_option': 'prior-mean',
                'delay': 0}
        }

        self.INITIALIZATION_ORDER = [
            STRNAMES.FILTERING,
            STRNAMES.ZERO_INFLATION,
            STRNAMES.CONCENTRATION,
            STRNAMES.CLUSTERING,
            STRNAMES.PROCESSVAR,
            STRNAMES.PRIOR_MEAN_GROWTH,
            STRNAMES.PRIOR_VAR_GROWTH,
            STRNAMES.GROWTH_VALUE,
            STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS,
            STRNAMES.PRIOR_VAR_SELF_INTERACTIONS,
            STRNAMES.SELF_INTERACTION_VALUE,
            STRNAMES.PRIOR_MEAN_INTERACTIONS,
            STRNAMES.PRIOR_VAR_INTERACTIONS,
            STRNAMES.CLUSTER_INTERACTION_VALUE,
            STRNAMES.CLUSTER_INTERACTION_INDICATOR,
            STRNAMES.INDICATOR_PROB,
            STRNAMES.PRIOR_MEAN_PERT,
            STRNAMES.PRIOR_VAR_PERT,
            STRNAMES.PERT_INDICATOR,
			STRNAMES.PERT_VALUE,
            STRNAMES.PERT_INDICATOR_PROB,
            STRNAMES.REGRESSCOEFF,
            STRNAMES.QPCR_SCALES,
            STRNAMES.QPCR_DOFS,
            STRNAMES.QPCR_VARIANCES]

    def suffix(self):
        '''Create a suffix with the parameters
        '''
        perts = 'addit' if self.PERTURBATIONS_ADDITIVE else 'mult'
        if self.LEAVE_OUT is not None:
            try:
                lo = str(tuple(list(self.LEAVE_OUT))).replace(',','_').replace(
                    '(','').replace(')','').replace(' ','')
            except:
                lo = str(self.LEAVE_OUT)
        else:
            lo = None
        s = 'ds{}_is{}_b{}_ns{}_lo{}_mo{}_log{}_perts{}'.format(
            self.DATA_SEED, self.INIT_SEED, self.BURNIN, self.N_SAMPLES, lo,
            self.MAX_N_ASVS, self.DATA_LOGSCALE, perts)
        return s

    def cv_suffix(self):
        '''Create a master suffix with the parameters
        '''
        perts = 'addit' if self.PERTURBATIONS_ADDITIVE else 'mult'
        s = 'ds{}_is{}_b{}_ns{}_mo{}_log{}_perts{}'.format(
            self.DATA_SEED, self.INIT_SEED, self.BURNIN, self.N_SAMPLES,
            self.MAX_N_ASVS, self.DATA_LOGSCALE, perts)
        return s

    def cv_single_suffix(self):
        '''Create a suffix for a single cv round
        '''
        return 'leave_out{}'.format(self.LEAVE_OUT)


class MLCRRConfig(_BaseModelConfig):
    '''This is the configuration file for the Maximum Likelihood Constrained Ridge
    Regression model defined in the original MDSINE [1] model. This defines the 
    parameters for the model as well as the cross validation configuration.

    System parameters
    -----------------
    DATE_FILENAME

    DATA_DTYPE

    QPCR_nORMALIZATION_MAX_VALUE

    GROWTH_TRUNCATION_SETTINGS

    SELF_INTERACTION_TRUNCATION_SETTINGS

    Cross validation parameters
    ---------------------------
    CV_MAP_MIN, CV_MAP_MAX, CV_MAP_N : int
        Defines the numerical region over which the algorithm will
        search for regularization parameters settings. The region is
        from 10^CV_MAP_MIN to 10^CV_MAP_MAX with CV_MAP_N number of 
        spaces. 
        Example:
            CV_MAP_MIN = -3
            CV_MAP_MAX = 2
            CV_MAP_N = 5
            [0.0010, 0.0178, 0.3162, 5.6234, 100.0000]
    CV_REPLICATES : int
        This is the number of different "shuffles" or replications for cross-fold validation.
        This can and will be parallelized if N_CPUS > 1.
    CV_LEAVE_K_OUT : int
        Leave K out for cross validation: defaults to 1.

    Parameters
    ----------
    output_basepath : str
        Basepath to save everything to
    data_seed : int
        Seed to initialize the data to
    init_seed : int
        Seed to initialize the model to
    n_cpus : int
        Number of available processors available
    '''
    def __init__(self, output_basepath, data_seed, init_seed, data_path, n_cpus=1):

        self.OUTPUT_BASEPATH = output_basepath
        self.DATA_PATH = data_path
        self.DATA_SEED = data_seed
        self.INIT_SEED = init_seed
        self.N_CPUS = n_cpus

        self.DATA_FILENAME = '../pickles/real_subjectset.pkl'
        self.DATA_DTYPE = 'abs'
        self.QPCR_NORMALIZATION_MAX_VALUE = 100

        self.GROWTH_TRUNCATION_SETTINGS = 'positive'
        self.SELF_INTERACTION_TRUNCATION_SETTINGS = 'positive'

        # Cross validation parameters
        self.CV_MAP_MIN = -3
        self.CV_MAP_MAX = 2
        self.CV_MAP_N = 15
        self.CV_REPLICATES = 15
        self.CV_LEAVE_K_OUT = 1

    def suffix(self):
        '''Create a suffix for saving the runs
        '''
        mapmin = str(self.CV_MAP_MIN).replace('-','neg')
        mapmax = str(self.CV_MAP_MAX).replace('-','neg')
        s = '_ds{}_is{}_mmin{}_mmax{}_mn{}_cvnr{}_cvlko{}'.format(
            self.DATA_SEED, self.INIT_SEED, 
            mapmin, mapmax, self.CV_MAP_N, self.CV_REPLICATES,
            self.CV_LEAVE_K_OUT)
        return s


class SimulationConfig(_BaseModelConfig):
    '''These are the paramters used to make a synthetic dataset

    System Parameters
    -----------------
    pv_value : float
        What to set the process variance as
    simulation_dt : dt
        The smaller step size we use for froward integration so the integration
        does not become unstable
    n_days : int
        Total number of days to run the simulation for
    times : str, int, float
        How to generate the times
        if str:
            'darpa-study-sampling'
                Denser in the beginning, and around the ends of perturbations
        if int/float
            This is the density to sample at (0.5 means sample every half of
            a day)
    n_replicates : int
        How many replicates of subjects to run the inference with
    init_low, init_high : float
        The low and high to initialize the data at using a uniform
        distribution
    max_abundance : float
        Max abundance
    n_asvs : int
        How many ASVs to simulate
    healthy_patients : bool
        Which consortia of mice to use as noise approximators
    process_variance_level : float
        What to set the process variance to
    measurement_noise_level : float
        What to set the measurement noise to

    '''
    def __init__(self, times, n_replicates, n_asvs, healthy,
        process_variance_level, measurement_noise_level):
        self.PV_VALUE = process_variance_level**2
        self.SIMULATION_DT = 0.001
        self.N_DAYS = 'from-data'
        self.TIMES = times
        self.N_REPLICATES = n_replicates
        self.INIT_LOW = 1e5 #5e6
        self.INIT_HIGH = 1e7 #5e7
        self.MAX_ABUNDANCE = 1e8
        self.N_ASVS = n_asvs
        self.HEALTHY_PATIENTS = healthy
        self.PROCESS_VARIANCE_LEVEL = process_variance_level
        self.MEASUREMENT_NOISE_LEVEL = measurement_noise_level

        self.PERTURBATIONS = True #(0.3, '2', [0.1, 0.4, 0.5], [0.5, 1, 2], 0.1)

        self.NEGBIN_A0, self.NEGBIN_A1 = calculate_reads_a0a1(measurement_noise_level)
        self.QPCR_NOISE_SCALE = measurement_noise_level

    def suffix(self):
        max_abund = self.MAX_ABUNDANCE
        if max_abund is not None:
            max_abund = '{:.2E}'.format(max_abund)

        if self.PERTURBATIONS is None:
            perts = None
        else:
            perts = True
        s = '_nr{}_no{}_nd{}_ms{}_pv{}_ma{}_np{}_nt{}'.format(
            self.N_REPLICATES,
            self.N_ASVS,
            self.N_DAYS,
            self.MEASUREMENT_NOISE_LEVEL,
            self.PROCESS_VARIANCE_LEVEL,
            max_abund, perts,
            self.TIMES)
        return s


class FilteringConfig(pl.Saveable):
    '''These are the parameters for Filtering

    Different types of filtering
    ----------------------------
    `at_least_counts`
        For each ASV in the subjectset `subjset`, delete all ASVs that
        do not have at least a minimum number of counts `min_counts`
        for less than `min_num_subjects` subjects.

        Parameters
        ----------
        colonization_time : numeric, None
            This is the day that you want to start taking the relative abundance.
            We only lok at the relative abundance after the colonization period.
            If this is `None` then it is set to 0.
        min_counts : numeric
            This is the minimum number of counts it needs to have
        min_num_subjects : int
            This is the minimum number of subjects that there must be a relative
            abundance

    `consistency`
        Filters the subjects by looking at the consistency of the counts.
        There must be at least `min_num_counts` for at least
        `min_num_consecutive` consecutive timepoints for at least
        `min_num_subjects` subjects for the ASV to be classified as valid.

        Parameters
        ----------
        min_num_consecutive: int
            This is the minimum number of consecutive timepoints that there
            must be at least `min_num_counts`
        min_num_counts : int
            This is the minimum number of counts that there must be at each
            consecutive timepoint
        min_num_subjects : int, None
            This is how many subjects this must be true for for the ASV to be
            valid. If it is None then it only requires one subject.

    Additional Parameters
    ---------------------
    healthy : bool
        If True, do regression on the healthy patients
    '''
    def __init__(self, healthy):
        self.COLONIZATION_TIME = 5
        self.THRESHOLD = 0.0001
        self.DTYPE = 'rel'
        self.MIN_NUM_SUBJECTS = 2 #'all'
        self.MIN_NUM_CONSECUTIVE = 5
        self.HEALTHY = healthy

    def __str__(self):
        return 'healthy{}_{}_{}_{}_{}_{}'.format(
            self.HEALTHY,
            self.COLONIZATION_TIME,
            self.THRESHOLD,
            self.DTYPE,
            self.MIN_NUM_SUBJECTS,
            self.MIN_NUM_CONSECUTIVE)

    def suffix(self):
        return str(self)


class LoggingConfig(pl.Saveable):
    '''These are the parameters for logging

    FORMAT : str
        This is the logging format for stdout
    LEVEL : logging constant, int
        This is the level to log at for stdout
    NUMPY_PRINTOPTIONS : dict
        These are the printing options for numpy.
    '''
    def __init__(self):
        self.FORMAT = '%(levelname)s:%(module)s.%(lineno)s: %(message)s'
        self.LEVEL = logging.INFO
        self.NUMPY_PRINTOPTIONS = {
            'threshold': sys.maxsize, 'linewidth': sys.maxsize}

        logging.basicConfig(format=self.FORMAT, level=self.LEVEL)
        np.set_printoptions(**self.NUMPY_PRINTOPTIONS)
        pd.set_option('display.max_columns', None)


class NegBinConfig(_BaseModelConfig):
    '''Configuration class for learning the negative binomial dispersion
    parameters. Note that these parameters are learned offline.

    Parameters
    ----------
    seed : int
        Seed to start the inderence
    burnin, n_samples : int
        How many iterations for burn-in and total samples, respectively.
    basepath : str
        This is the basepath to save the graph. A separate folder within
        `basepath` will be created for the specific graph.
    synth : bool
        If True, run with the synthetic data, where the parameters needed
        to learn are `SYNTHETIC_A0` AND `SYNTHETIC_A1`.
    '''

    def __init__(self, seed, burnin, n_samples, basepath, synth):
        if basepath[-1] != '/':
            basepath += '/'

        self.RAW_COUNTS_FILENAME = 'raw_data/replicate_data/counts.txt'
        self.MAIN_INFERENCE_SUBJSET_FILENAME = 'pickles/real_subjectset.pkl'
        self.SEED = seed
        self.OUTPUT_BASEPATH = basepath
        self.BURNIN = burnin
        self.N_SAMPLES = n_samples
        self.CKPT = 100
        self.DADA_SEUQUENCE_COL_NAME = ['sequences']
        self.DADA_ASVSET_COL_NAMES = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus']
        self.REPLICATE_DATA_COLS = [
            (10, ['M2-D10-1A','M2-D10-1B','M2-D10-2A','M2-D10-2B','M2-D10-3A','M2-D10-3B']),	
            (8, ['M2-D8-1A',	'M2-D8-1B',	'M2-D8-2A',	'M2-D8-2B',	'M2-D8-3A',	'M2-D8-3B']),	
            (9, ['M2-D9-1A',	'M2-D9-1B',	'M2-D9-2A',	'M2-D9-2B',	'M2-D9-3A',	'M2-D9-3B'])]
        self.MOUSE_ID = '2'

        self.MP_FILTERING = 'full'

        # Synthetic arguments
        self.SYNTHETIC = synth
        self.SYNTHETIC_A0 = 1e-5
        self.SYNTHETIC_A1 = 0.056
        self.SYNTHETIC_N_REPLICATES = 3
        self.SYNTHETIC_DAYS = [10,9,8]

        self.INFERENCE_ORDER = [
            STRNAMES.NEGBIN_A0,
            STRNAMES.NEGBIN_A1,
            STRNAMES.FILTERING]

        self.LEARN = {
            STRNAMES.NEGBIN_A0: True,
            STRNAMES.NEGBIN_A1: True,
            STRNAMES.FILTERING: False}

        self.INITIALIZATION_ORDER = [
            STRNAMES.FILTERING,
            STRNAMES.NEGBIN_A0,
            STRNAMES.NEGBIN_A1
        ]

        self.INITIALIZATION_KWARGS = {
            STRNAMES.NEGBIN_A0: {
                'value': 1e-4,
                'truncation_settings': (0, 1e5),
                'tune': 50,
                'end_tune': int(self.BURNIN/2),
                'target_acceptance_rate': 'optimal', 
                'proposal_option': 'auto',
                'delay': 0},
            STRNAMES.NEGBIN_A1: {
                'value': 0.5,
                'tune': 50,
                'truncation_settings': (0, 1e5),
                'end_tune': int(self.BURNIN/2),
                'target_acceptance_rate': 'optimal', 
                'proposal_option': 'auto',
                'delay': 0},
            STRNAMES.FILTERING: {
                'tune': 50,
                'end_tune': int(self.BURNIN/2),
                'target_acceptance_rate': 'optimal', 
                'qpcr_variance_inflation': 100,
                'delay': 100}}

    def suffix(self):
        if self.SYNTHETIC:
            s = 'syn_nreps{}_ndays{}_'.format(self.SYNTHETIC_N_REPLICATES, 
                len(self.SYNTHETIC_DAYS))
        else:
            s = 'real_'

        s += 'seed{}_nb{}_ns{}'.format(self.SEED, self.BURNIN, self.N_SAMPLES)
        return s
        