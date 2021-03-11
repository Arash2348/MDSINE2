"""
Contains logic for initializing the clustering of an instantiated MDSINE2 model, by first learning perturbations
on individual OTUs (no interactions), then grouping the OTUs by perturbation sign.
"""
import os
from collections import defaultdict
import numpy as np

from mdsine2 import initialize_graph, Study, MDSINE2ModelConfig, run_graph
from mdsine2.pylab import BaseMCMC
from mdsine2.names import STRNAMES

from mdsine2.logger import logger


def sign_str(x: float):
    if x < 0:
        return "-"
    if x > 0:
        return "+"
    else:
        return "0"


def initialize_mdsine_from_perturbations(
        mcmc: BaseMCMC,
        cfg: MDSINE2ModelConfig,
        study: Study,
        n_samples: int=1000,
        burnin: int=0,
        checkpoint: int=500):
    '''
    Trains the model by first learning perturbations (no interactions)
    and divides up the OTUs into clusters based on perturbation magnitudes/signs.
    :param mcmc: The BaseMCMC object
    '''

    # ========= Run a copy of the model with interactions disabled.
    cfg_proxy = cfg.copy()
    cfg_proxy.OUTPUT_BASEPATH = os.path.join(cfg.OUTPUT_BASEPATH, "cluster-init")
    cfg_proxy.N_SAMPLES = n_samples
    cfg_proxy.BURNIN = burnin
    cfg_proxy.CHECKPOINT = checkpoint

    cfg_proxy.LEARN = {
        STRNAMES.GROWTH_VALUE: True,
        STRNAMES.SELF_INTERACTION_VALUE: True,
        STRNAMES.PERT_VALUE: True,
        STRNAMES.CLUSTER_INTERACTION_VALUE: False,
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
        STRNAMES.CLUSTER_INTERACTION_INDICATOR_PROB: True,
        STRNAMES.PERT_INDICATOR: True,
        STRNAMES.PERT_INDICATOR_PROB: True,
        STRNAMES.QPCR_SCALES: False,
        STRNAMES.QPCR_DOFS: False,
        STRNAMES.QPCR_VARIANCES: False}

    cfg_proxy.INFERENCE_ORDER = [
        STRNAMES.CLUSTER_INTERACTION_INDICATOR,
        STRNAMES.CLUSTER_INTERACTION_INDICATOR_PROB,
        STRNAMES.PERT_INDICATOR,
        STRNAMES.PERT_INDICATOR_PROB,
        STRNAMES.GROWTH_VALUE,
        STRNAMES.SELF_INTERACTION_VALUE,
        STRNAMES.PERT_VALUE,
        STRNAMES.CLUSTER_INTERACTION_VALUE,
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

    # Disable clustering.
    cfg_proxy.INITIALIZATION_KWARGS[STRNAMES.CLUSTER_INTERACTION_VALUE]['value_option'] = 'all-off'
    cfg_proxy.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value_option'] = 'no-clusters'
    cfg_proxy.LEARN[STRNAMES.CLUSTERING] = False
    cfg_proxy.INITIALIZATION_KWARGS[STRNAMES.PERT_INDICATOR_PROB] = {
        'value_option': 'manual',
        'value': 1,
        'hyperparam_option': 'manual',
        'a': 10000,
        'b': 1,
        'delay': 0
    }
    cfg_proxy.LEARN[STRNAMES.PERT_INDICATOR_PROB] = False

    mcmc_proxy = initialize_graph(params=cfg_proxy, graph_name=study.name, subjset=study)
    run_graph(mcmc_proxy, crash_if_error=True)

    # ========= Divide up OTUs based on perturbation signs.
    result_clustering = defaultdict(list)

    proxy_clustering_obj = mcmc_proxy.graph[STRNAMES.CLUSTERING].clustering
    pert_means = np.vstack(
        [pert.get_trace_from_disk()[n_samples // 2:, :].mean(axis=0) for pert in mcmc_proxy.graph.perturbations]
    )
    taxa_with_pert_signs = [
        (
            next(iter(cluster.members)),
            "".join([
                sign_str(pert_means[pidx, cidx])
                for pidx in range(pert_means.shape[0])
            ])
        )
        for cidx, cluster in enumerate(proxy_clustering_obj)
    ]

    for taxa_idx, pert_sign in taxa_with_pert_signs:
        result_clustering[pert_sign].append(taxa_idx)

    # ========= Save the result into original chain.
    result_clustering_arr = [
        c for _, c in result_clustering.items()
    ]

    cids = mcmc.graph[STRNAMES.CLUSTERING].value.fromlistoflists(result_clustering_arr)

    # Perturbation initialize to mean value across constituents of cluster.
    for cid, cluster in zip(cids, result_clustering_arr):
        for pidx, (pert, pert_proxy) in enumerate(zip(mcmc.graph.perturbations, mcmc_proxy.graph.perturbations)):
            pert_value = np.mean([
                pert_means[
                    pidx,
                    proxy_clustering_obj.member_idx_to_cidx(taxa_idx)  # Taxa idx -> Cluster ID -> Cluster IDX
                ] for taxa_idx in cluster
            ])
            logger.info("Initializing cluster {} with perturbation ({})={}".format(
                [mcmc.graph.data.taxa[oid].name for oid in cluster],
                pert.name,
                pert_value
            ))
            pert.magnitude.value[cid] = pert_value
