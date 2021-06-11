import argparse
import random


class Parameters(object):
    def __init__(self, args, dataset):
        # dataset and method ==============================================================
        self.args = args
        self.dataset_obj = dataset
        self.method = args.method.lower()
        self.path = args.path
        self.dataset = args.dataset
        self.result_path = args.res_path + args.dataset + '/' + args.method + '/'
        self.result_folder = args.res_folder
        self.include_networks = eval(args.include_networks)

        # algo-parameters =======================================================
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.batch_size_seq = args.batch_size_seq
        self.valid_batch_siz = args.valid_batch_siz
        self.lr = args.lr
        self.optimizer = args.optimizer
        self.loss = args.loss
        if self.method in ['bpr']:
            self.loss = 'bpr'
        self.initializer = args.initializer
        self.stddev = args.stddev
        self.max_item_seq_length = args.max_item_seq_length
        self.load_embedding_flag = dataset.load_embedding_flag  # indicates extra-information

        # hyper-parameters ======================================================
        self.num_factors = args.num_factors
        self.num_layers = args.num_layers
        self.num_negatives = args.num_negatives
        self.num_negatives_seq = args.num_negatives_seq
        self.reg_w = args.reg_w
        self.reg_b = args.reg_b
        self.reg_lambda = args.reg_lambda
        self.margin = args.margin
        self.keep_prob = args.keep_prob

        # gnn ==============================================================
        self.hid_units = eval(args.hid_units)
        self.n_heads = eval(args.n_heads)
        self.gnn_keep_prob = args.gnn_keep_prob
        self.net_keep_prob = args.net_keep_prob
        self.d_k = args.d_k

        # valid test ============================================================
        self.at_k = args.at_k
        self.num_thread = args.num_thread
        self.epoch_mod = args.epoch_mod

        # Dataset counts ==============================================================
        self.num_user = dataset.num_user
        self.num_list = dataset.num_list
        self.num_item = dataset.num_item
        self.num_train_instances = len(dataset.trainArrTriplets[0])
        self.num_valid_instances = len(dataset.validNegativesDict.keys())
        self.num_test_instances = len(dataset.testNegativesDict.keys())
        self.num_nodes = self.num_user + self.num_list + self.num_item

        # data-structures ======================================================
        self.user_lists_dct = dataset.user_lists_dct
        self.list_items_dct = dataset.list_items_dct
        self.list_items_dct_train = dataset.list_items_dct_train
        self.list_user_dct = dataset.list_user_dct
        self.list_user_vec = dataset.list_user_vec
        self.train_matrix = dataset.train_matrix
        self.testNegativesDict = dataset.testNegativesDict
        self.validNegativesDict = dataset.validNegativesDict

        self.trainArrTriplets = dataset.trainArrTriplets
        self.validArrDubles = dataset.validArrDubles
        self.testArrDubles = dataset.testArrDubles
        self.train_matrix_item_seq = dataset.train_matrix_item_seq

        self.train_matrix_item_seq_for_test = dataset.train_matrix_item_seq_for_test

        # adj ===========
        if self.method in ['tenet']:
            self.user_adj_mat = dataset.user_adj_mat
            self.list_adj_mat = dataset.list_adj_mat
            self.item_adj_mat = dataset.item_adj_mat

        # new ================
        self.warm_start_gnn = args.warm_start_gnn
        self.include_hgnn = args.include_hgnn
        self.include_hgnn = True if args.include_hgnn == 'True' else False

    def get_args_to_string(self):
        args_str = str(random.randint(1, 1000000))
        return args_str


def parse_args():
    # dataset and method
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('--method', nargs='?', default='tenet', help='gmf,transformer,gnn,tenet')
    parser.add_argument('--path', nargs='?', default='/home/vijai/tenet/data_tenet/required/zhihu_small/tenet/zhihu/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='goodreads', help='Choose a dataset.')
    parser.add_argument('--res_path', nargs='?', default='/home/vijai/tenet/result_tenet/',
                        help='result path for plots and best error values.')
    parser.add_argument('--res_folder', nargs='?', default='test',
                        help='specific folder corresponding to different runs on different parameters.')
    parser.add_argument('--include_networks', nargs='?', default="['gnn', 'seq']",
                        help='include given networks in the model.')

    # algo-parameters
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size.')
    parser.add_argument('--batch_size_seq', type=int, default=256, help='Seq batch size.')
    parser.add_argument('--valid_batch_siz', type=int, default=32, help='Valid batch size.')
    parser.add_argument('--lr', type=float, default=.001, help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='adam', help='adam')
    parser.add_argument('--loss', nargs='?', default='ce', help='ce')
    parser.add_argument('--initializer', nargs='?', default='xavier', help='xavier')
    parser.add_argument('--stddev', type=float, default=0.02, help='stddev for normal and [min,max] for uniform')
    parser.add_argument('--max_item_seq_length', type=int, default=200, help='number of rated items to keep.')
    parser.add_argument('--load_embedding_flag', type=int, default=0,
                        help='0-->donot load embedding, 1-->load embedding for entities.')
    parser.add_argument('--at_k', type=int, default=5, help='@k for recall, map and ndcg, etc.')
    parser.add_argument('--knn_k', type=int, default=50, help='@k for knn.')
    parser.add_argument('--cosine', nargs='?', default='False', help='knn_graph cosine or not.')
    parser.add_argument('--embed_type', nargs='?', default='node2vec', help='Choose a dataset.')

    # hyper-parameters
    parser.add_argument('--num_factors', type=int, default=80, help='Embedding size.')
    parser.add_argument('--num_negatives', type=int, default=1, help='Negative instances in sampling.')
    parser.add_argument('--num_negatives_seq', type=int, default=2,
                        help='Negative instances in sampling for seq (done in main itself).')
    parser.add_argument('--reg_w', type=float, default=0.0000, help="Regularization for weight vector.")
    parser.add_argument('--reg_b', type=float, default=0.000, help="Regularization for user and item bias embeddings.")
    parser.add_argument('--reg_lambda', type=float, default=0.000,
                        help="Regularization lambda for user and item embeddings.")
    parser.add_argument('--margin', type=float, default=2.0, help='margin value for TripletMarginLoss.')
    parser.add_argument('--keep_prob', type=float, default=0.5, help='droupout keep probability in layers.')

    # gnn
    parser.add_argument('--num_layers', type=int, default=2, help='Number of hidden layers.')
    parser.add_argument('--hid_units', nargs='?', default='[48,32]', help='hidden units of GAT')
    parser.add_argument('--gnn_keep_prob', type=float, default=1.0,
                        help='proj keep probability in projection weights layers for reviews.')
    parser.add_argument('--net_keep_prob', type=float, default=1.0,
                        help='proj keep probability in projection weights layers for reviews.')

    # multi-head
    parser.add_argument('--n_heads', nargs='?', default='[1]', help='number of heads of GAT')
    parser.add_argument('--d_k', type=int, default=64, help='Number of hidden layers.')

    # valid and test
    parser.add_argument('--dataset_avg_flag_zero', type=int, default=0,
                        help='Dataset item embed zero (or) avg. zero --> 1, else avg')
    parser.add_argument('--epoch_mod', type=int, default=15, help='epoch mod --> to display valid and test error.')
    parser.add_argument('--num_thread', type=int, default=16, help='number of threads.')
    parser.add_argument('--comment', nargs='?', default='comment',
                        help='comments about the current experimental iterations.')

    # new
    parser.add_argument('--store_embedding', nargs='?', default='False',
                        help='whether to store user-list-item embeddings for knn_graph.')
    parser.add_argument('--knn_graph', nargs='?', default='False', help='knn_graph for tenet.')
    parser.add_argument('--user_adj_weights', nargs='?', default='False',
                        help='whether to use adjacency matrix weights for gnn.')
    parser.add_argument('--self_loop', nargs='?', default='True',
                        help='whether to use adjacency matrix weights for gnn.')

    # new
    parser.add_argument('--warm_start_gnn', type=int, default=100,
                        help='warm_start done on gnn part to give better embeddings to seq part.')
    parser.add_argument('--include_hgnn', nargs='?', default='True',
                        help='whether to include hgnn in gnn part of the network.')

    return parser.parse_args()
