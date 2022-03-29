import mdsine2 as md2
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mdsine2.logger import logger
from mdsine2.names import STRNAMES
from .base import CLIModule
import pickle


# obtains posteriors and graphics -- DONE
def plot_posteriors(mcmc_syn, study):
    # Plot the posterior
    clustering = mcmc_syn.graph[STRNAMES.CLUSTERING_OBJ]
    coclusters = md2.summary(clustering.coclusters)['mean']
    md2.visualization.render_cocluster_probabilities(coclusters, taxa=study.taxa)
    interactions = mcmc_syn.graph[STRNAMES.INTERACTIONS_OBJ]
    A = md2.summary(interactions, set_nan_to_0=True)['mean']
    md2.visualization.render_interaction_strength(A, log_scale=True, taxa=study.taxa,
                                                  center_colors=True)
    bf = md2.generate_interation_bayes_factors_posthoc(mcmc_syn)
    md2.visualization.render_bayes_factors(bf, taxa=study.taxa)

# def calculate_AUCROC(mcmc_syn):
#     # METRIC 1: AUCROC
#     # ------
#     # Get truth
#     truth = semi_syn.model.interactions
#     # Get predicted, set self-interactions as diagonal
#     pred = mcmc_syn.graph[STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk()
#     si = mcmc_syn.graph[STRNAMES.SELF_INTERACTION_VALUE].get_trace_from_disk()
#     for i in range(len(study.taxa)):
#         pred[:, i, i] = si[:, i]
#     roc_per_gibb = md2.metrics.rocauc_posterior_interactions(
#         pred=pred, truth=truth, signed=True, per_gibb=True)
#     print('Average ROCAUC:', np.mean(roc_per_gibb))
#
# def calculate_RMSEinter():
#     # METRIC 2: RMSE of interactions
#     # --------------------
#     arr = np.zeros(pred.shape[0])
#     for gibb in range(len(arr)):
#         arr[gibb] = md2.metrics.RMSE(truth, pred[gibb])
#     print('Average RMSE error of interactions:', np.mean(arr))

def calculate_RMSEgrowth(mcmc_syn, semi_syn, rename_study, basepath):
    # METRIC 3: RMSE of growth values
    # ---------------------
    truth = semi_syn.model.growth
    pred = mcmc_syn.graph[STRNAMES.GROWTH_VALUE].get_trace_from_disk()
    print(pred.shape)
    print(truth.shape)
    arr = np.zeros(pred.shape[0])
    for gibb in range(len(arr)):
        arr[gibb] = md2.metrics.RMSE(truth, pred[gibb])
    print("This is the array:", arr)
    print('Average RMSE error of growth values', np.mean(arr))
    sns.boxplot(data=arr, color="#f045a7")
    sns.swarmplot(data=arr, color=".25")

    plt_name = rename_study + "-metric-img"
    plt_path = basepath + "/" + rename_study +"/" + plt_name
    plt.savefig(plt_path)


class MetricsCLI(CLIModule):
    def __init__(self, subcommand="metrics"):
        super().__init__(
            subcommand=subcommand,
            docstring=__doc__
        )

    def create_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            '--input-mcmc', '-im', type=str, dest='input_mcmc',
            required=True,
            help='This is the mcmc file to be used in to calculate metrics'
        )

        parser.add_argument(
            '--input-semi-syn', '-is', type=str, dest='input_semi_syn',
            required=True,
            help='This is the semi_syn file to be used in to calculate metrics'
        )

        parser.add_argument(
            '--seed', '-s', type=int, dest='seed',
            required=True,
            help='This is the seed to be used with the metrics program'
        )

        parser.add_argument(
            '--rename-study', type=str, dest='rename_study',
            required=False, default=None,
            help='Specify the name of the study to set'
        )
        parser.add_argument(
            '--basepath', '--output-basepath', '-b', type=str, dest='basepath',
            required=True,
            help='This is parent directory of the folder in which the metrics are saved'
        )

    def main(self, args: argparse.Namespace):
        print("Started to run main within metrics.py")
        logger.info('Loading mcmc file {}'.format(args.input_mcmc))
        logger.info('Loading semi_syn file {}'.format(args.input_semi_syn))

        #1) Grab the respective mcmc_syn file and semi_syn file
        mcmc_syn = md2.BaseMCMC.load(args.input_mcmc)
        semi_syn = pickle.load(open(args.input_semi_syn, "rb"))

        #2) Use Metric Calcuate function for Graphs
        calculate_RMSEgrowth(mcmc_syn, semi_syn, args.rename_study, args.basepath)



        # mcmc_syn = md2.BaseMCMC.load('/PHShome/as1010/synthetic_new_arash/mcmc.pkl')
        # semi_syn = pickle.load(open("/PHShome/as1010/synthetic_new_arash/semi_syn.pkl", "rb"))
        # calculate_RMSEgrowth(mcmc_syn, semi_syn)


        # # 1) load dataset
        # logger.info('Loading dataset {}'.format(args.input))
        # mcmc = md2.BaseMCMC.load(args.input) #might be study.load instead of BaseMCMC
        # study1 = md2.Study.load(args.input)
        # if args.rename_study is not None:
        #     if args.rename_study.lower() != 'none':
        #         study1.name = args.rename_study #might just want to delete 153-155
        #
        # print("This is study name:" + study1.name)
        # md2.seed(args.seed)




#####-------Previously how it was done----------
# mcmc_syn = md2.BaseMCMC.load('/PHShome/as1010/synthetic_new_arash/mcmc.pkl')
# semi_syn = pickle.load(open("/PHShome/as1010/synthetic_new_arash/semi_syn.pkl", "rb" ))
# calculate_RMSEgrowth(mcmc_syn, semi_syn)

