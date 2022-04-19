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

def calculate_RMSEgrowth(mcmc_syn_l, mcmc_syn_m, mcmc_syn_h, semi_syn, rename_study, basepath):
    # METRIC 3: RMSE of growth values
    # ---------------------
    truth = semi_syn.model.growth
    pred_l = mcmc_syn_l.graph[STRNAMES.GROWTH_VALUE].get_trace_from_disk()
    print("Pred_l:", pred_l)
    pred_m = mcmc_syn_m.graph[STRNAMES.GROWTH_VALUE].get_trace_from_disk()
    print("Pred_m:", pred_m)
    pred_h = mcmc_syn_h.graph[STRNAMES.GROWTH_VALUE].get_trace_from_disk()
    print("Pred_h:", pred_h)


    pred_list =[pred_l, pred_m, pred_h]
    array_list = []
    count = 1

    for i in pred_list:
        print(i.shape)
        print(truth.shape)
        arr = np.zeros(i.shape[0])
        for gibb in range(len(arr)):
            arr[gibb] = md2.metrics.RMSE(truth, i[gibb])
        print("This is the RMSE array of " + str(count), arr)
        print('Average RMSE error of growth values of ' + str(count), np.mean(arr))
        count = count + 1
        array_list.append(arr)

    print("This is full array_list:", array_list)
    sns.boxplot(data=array_list, color="#f045a7")
    sns.swarmplot(data=array_list, color=".25")

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
            '--input-mcmc-low', '-iml', type=str, dest='input_mcmc_low',
            required=True,
            help='This is the mcmc low file to be used in to calculate metrics'
        )

        parser.add_argument(
            '--input-mcmc-med', '-imm', type=str, dest='input_mcmc_med',
            required=True,
            help='This is the mcmc medium file to be used in to calculate metrics'
        )

        parser.add_argument(
            '--input-mcmc-high', '-imh', type=str, dest='input_mcmc_high',
            required=True,
            help='This is the mcmc high file to be used in to calculate metrics'
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
        logger.info('Loading mcmc file {}'.format(args.input_mcmc_low))
        logger.info('Loading mcmc file {}'.format(args.input_mcmc_med))
        logger.info('Loading mcmc file {}'.format(args.input_mcmc_high))
        logger.info('Loading semi_syn file {}'.format(args.input_semi_syn))

        #1) Grab the respective mcmc_syn file and semi_syn file
        mcmc_syn_low = md2.BaseMCMC.load(args.input_mcmc_low)
        print("Finished low")
        mcmc_syn_med = md2.BaseMCMC.load(args.input_mcmc_med)
        print("Finished medium")
        mcmc_syn_high = md2.BaseMCMC.load(args.input_mcmc_high)
        print("Finished high")
        semi_syn = pickle.load(open(args.input_semi_syn, "rb"))

        #2) Use Metric Calculate function for Graphs
        calculate_RMSEgrowth(mcmc_syn_low, mcmc_syn_med, mcmc_syn_high, semi_syn, args.rename_study, args.basepath)

        print("Finished main of metrics.py and entire program")


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

