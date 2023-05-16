import numpy as np
import pandas as pd
import networkx as nx
import os
from tqdm import tqdm


def monologue_converter(df):
    nodes = df.loc[:, ['말하는 사람', '듣는 사람']]
    nodes.loc[ nodes['듣는 사람'] == '독백', '듣는 사람' ] = nodes.loc[ nodes['듣는 사람'] == '독백', '말하는 사람']
    return nodes


class TheGlory:
    def __init__(self, file_list, data_dir):
        self.file_list = file_list
        self.data_dir = data_dir
        self.df_list = None  # Initialize df_list as None
        self.s1_df = None  # Initialize season 1 dataframe as None
        self.s2_df = None  # Initialize season 2 dataframe as None
        self.integrated_df = None  # Initialize integrated_df as None
        self.graph = None  # Initialize graph as None
        # self.random_graph = None  # Initialize Random Graphs. This will be used with df_list
        # self.s1_random_graph = None  # Initialize Random Graphs for Season 1. This will be used with s1_df
        # self.s2_random_graph = None  # Initialize Random Graphs for Season 2. This will be used with s2_df
        # self.total_random_graph = None  # Initialize Random Graphs for entire season.
        #                                 # This will be used with integrgated_df

    def data_load(self):
        df_list = []  # for separate data
        tmp_df = pd.DataFrame()  # for total data
        for i in range(len(self.file_list)):
            df_list.append(pd.read_excel(os.path.join(self.data_dir, self.file_list[i])))
        self.df_list = df_list  # save df_list to an instance variable
        print('Each of {} Episode DataFrames are Ready... '.format(len(df_list)))

        for i in range(len(df_list)):
            df_list[i]['episode'] = i+1
            tmp_df = pd.concat([tmp_df, df_list[i]], axis=0)

        self.integrated_df = tmp_df
        self.s1_df = tmp_df.loc[tmp_df['episode'] <= 8, :]
        self.s2_df = tmp_df.loc[tmp_df['episode'] > 8, :]
        print('Integrated DF that contains total episodes is Ready with Shape: {}'.format(self.integrated_df.shape))
        print('Season1 DF is Ready with Shape: {}'.format(self.s1_df.shape))
        print('Season2 DF is Ready with Shape: {}'.format(self.s2_df.shape))

    def graph_converter(self, df, direction=True):
        if self.df_list is None:
            raise ValueError("Data has not been loaded yet. Call the data_load() method first")

        df_new = df.copy()
        df_new.loc[:, ['말하는 사람', '듣는 사람']] = monologue_converter(df_new)

        node_list = np.unique(df_new['말하는 사람'].values.tolist() + df_new['듣는 사람'].values.tolist())
        edge_list = df_new[['말하는 사람', '듣는 사람']].values.tolist()

        src_tar = np.array([ row for row in list(df_new[['말하는 사람', '듣는 사람']].groupby(['말하는 사람', '듣는 사람']).value_counts().index)]).tolist()
        freq_list = np.array([ row for row in list(df_new[['말하는 사람', '듣는 사람']].groupby(['말하는 사람', '듣는 사람']).value_counts().values) ]).tolist()
        norm_freq_list = [ i / sum(freq_list) for i in freq_list ]  # normalize

        edge_weight_array = np.hstack((src_tar, np.array(norm_freq_list).reshape(len(src_tar), -1))) # normalize

        if direction:
            """for in-degree and out-degree"""
            g = nx.DiGraph()
            g.add_nodes_from(node_list)
            g.add_edges_from(edge_list)
            g.add_weighted_edges_from(edge_weight_array)
        else:
            """for undirected graph"""
            g = nx.Graph()
            g.add_nodes_from(node_list)
            g.add_edges_from(edge_list)
            g.add_weighted_edges_from(edge_weight_array)

        for _, _, d in g.edges(data=True):
            d['weight'] = np.float64(d['weight'])

        return g

    def degree_centrality(self, direction=True, cum=False):
        if self.df_list is None:
            raise ValueError("Data has not been loaded yet. Call the data_load() method first")

        total_df = pd.DataFrame()
        in_df = pd.DataFrame()
        out_df = pd.DataFrame()

        cum_total_df = pd.DataFrame()
        cum_in_df = pd.DataFrame()
        cum_out_df = pd.DataFrame()
        i = 1

        if direction:
            for df in tqdm(self.df_list):
                graph = self.graph_converter(df, direction)
                in_df = pd.concat([in_df, pd.Series(nx.in_degree_centrality(graph), name='ep_{}'.format(i))], axis=1)
                out_df = pd.concat([out_df, pd.Series(nx.out_degree_centrality(graph), name='ep_{}'.format(i))], axis=1)
                i += 1
            if cum:
                for i in range(1, in_df.shape[-1]+1):
                    cum_in_df['ep_{}'.format(i)] = in_df.iloc[:, 0:i].fillna(0).apply(lambda x: sum(x), axis=1)
                    cum_out_df['ep_{}'.format(i)] = out_df.iloc[:, 0:i].fillna(0).apply(lambda x: sum(x), axis=1)
                return cum_in_df, cum_out_df
            else:
                return in_df, out_df

        else:
            for df in tqdm(self.df_list):
                graph = self.graph_converter(df, direction)
                total_df = pd.concat([total_df, pd.Series(nx.degree_centrality(graph), name='ep_{}'.format(i))], axis=1)
                i += 1

            if cum:
                for i in range(1, total_df.shape[-1]+1):
                    cum_total_df['ep_{}'.format(i)] = total_df.iloc[:, 0:i].fillna(0).apply(lambda x: sum(x), axis=1)
                return cum_total_df
            else:
                return total_df

    def betweenness_centrality(self, cum=False):
        if self.df_list is None:
            raise ValueError("Data has not been loaded yet. Call the data_load() method first")

        btw_df = pd.DataFrame()
        cum_btw_df = pd.DataFrame()
        i = 1

        for df in tqdm(self.df_list):
            graph = self.graph_converter(df)
            btw_df = pd.concat([btw_df, pd.Series(nx.betweenness_centrality(graph, weight='weight'), name='ep_{}'.format(i))], axis=1)
            i += 1

        if cum:
            for i in range(1, btw_df.shape[-1]+1):
                cum_btw_df['ep_{}'.format(i)] = btw_df.iloc[:, 0:i].fillna(0).apply(lambda x: sum(x), axis=1)
            return cum_btw_df
        else:
            return btw_df

    def closeness_centrality(self, direction=True, cum=False):
        if self.df_list is None:
            raise ValueError("Data has not been loaded yet. Call the data_load() method first")

        cc_df = pd.DataFrame()
        cum_cc_df = pd.DataFrame()
        i = 1
        if direction:
            for df in tqdm(self.df_list):
                graph = self.graph_converter(df, direction)
                cc_df = pd.concat([cc_df, pd.Series(nx.closeness_centrality(graph), name='ep_{}'.format(i))], axis=1)
                i += 1
            if cum:
                for i in range(1, cc_df.shape[-1]+1):
                    cum_cc_df['ep_{}'.format(i)] = cc_df.iloc[:, 0:i].fillna(0).apply(lambda x: sum(x), axis=1)
                return cum_cc_df
            else:
                return cc_df
        else:
            raise ValueError("Closeness centrality is not defined for undirected graphs.")

    def pagerank(self, direction=True, cum=False):
        if self.df_list is None:
            raise ValueError("Data has not been loaded yet. Call the data_load() method first")

        pr_df = pd.DataFrame()
        cum_pr_df = pd.DataFrame()
        i = 1

        if direction:
            for df in tqdm(self.df_list):
                graph = self.graph_converter(df, direction)
                pr_df = pd.concat([pr_df, pd.Series(nx.pagerank(graph), name='ep_{}'.format(i))], axis=1)
                i += 1
            if cum:
                for i in range(1, pr_df.shape[-1]+1):
                    cum_pr_df['ep_{}'.format(i)] = pr_df.iloc[:, 0:i].fillna(0).apply(lambda x: sum(x), axis=1)
                return cum_pr_df
            else:
                return pr_df
        else:
            raise ValueError("Page Rank is not defined for undirected graphs.")

    def subgraph_centrality(self, direction=False, cum=False):
        if self.df_list is None:
            raise ValueError("Data has not been loaded yet. Call the data_load() method first")

        sc_df = pd.DataFrame()
        i = 1

        if cum:
            if direction:
                raise ValueError("Subgraph Centrality is not defined for directed graphs.")
            else:
                tmp_df = self.integrated_df
                length = len(np.unique(tmp_df['episode']))
                for i in range(1, length+1):
                    graph = self.graph_converter(tmp_df.loc[ tmp_df['episode'] <= i+1, :], direction)
                    sc_df = pd.concat([sc_df, pd.Series(nx.subgraph_centrality(graph), name='ep_{}'.format(i))], axis=1)
                    i += 1
                return sc_df
        else:
            if direction:
                raise ValueError("Subgraph Centrality is not defined for directed graphs.")
            else:
                for df in tqdm(self.df_list):
                    graph = self.graph_converter(df, direction)
                    sc_df = pd.concat([sc_df, pd.Series(nx.subgraph_centrality(graph), name='ep_{}'.format(i))], axis=1)
                    i += 1

                return sc_df


def triadic_significance(df_list, num_rg=1000):

    df_list_len = len(df_list)
    total_df = pd.DataFrame()

    i = 1
    for df in tqdm(df_list):
        df_new = df.copy()
        df_new.loc[:, ['말하는 사람', '듣는 사람']] = monologue_converter(df_new)

        node_list = np.unique(df_new['말하는 사람'].values.tolist() + df_new['듣는 사람'].values.tolist())
        edge_list = df_new[['말하는 사람', '듣는 사람']].values.tolist()

        g = nx.DiGraph()
        g.add_nodes_from(node_list)
        g.add_edges_from(edge_list)

        print('{} has {} nodes, {} edges'.format(i, g.number_of_nodes(), g.number_of_edges()))


        # random graph 먼저 생성
        num_rand_graphs = num_rg
        rand_graphs = [nx.gnm_random_graph(len(g.nodes), len(g.edges), directed=True) for i in range(num_rand_graphs)]
        # 랜덤 그래프에서 3-subgraph 출현 빈도 계산
        rand_subgraph_freqs = []
        for rand_graph in rand_graphs:
            sg = nx.triadic_census(rand_graph)
            rand_subgraph_freqs.append(sg)

        print('# of Random Graph Created --> ', len(rand_graphs))
        print('-----' * 3)

        # find all 3-subgraphs
        subgraphs = nx.triadic_census(g)

        # subgraph significance 계산
        subgraph_significance = {}
        z_scores = []  # normalized z-score
        for subgraph in subgraphs:
            rand_subgraph_freq = sum([rand_subgraph_freqs[i][subgraph] for i in range(num_rand_graphs)])
            mean = rand_subgraph_freq / num_rand_graphs  # 해당 패턴이 튀어 나올 평균값 (1000개의 랜덤 그래프니 1000으로 나눔)
            std = ((num_rand_graphs - 1) / num_rand_graphs) * sum([(rand_subgraph_freqs[i][subgraph] - mean)**2 for i in range(num_rand_graphs)]) ** 0.5
            if std == 0:
                z_score = 0
            else:
                z_score = (subgraphs[subgraph] - mean) / std
            z_scores.append(z_score)
            # subgraph_significance[subgraph] = z_score

        # calculate network significance profile (SP)
        sp = []
        z_squared_sum = sum([z**2 for z in z_scores])
        for z in z_scores:
            sp.append(z / z_squared_sum ** 0.5)

        j = 0
        for subgraph in subgraphs.keys():
            subgraph_significance[subgraph] = sp[j]
            j += 1

        if df_list_len == 1:
            total_df = pd.concat([total_df, pd.Series(subgraph_significance, name='entire_season'.format(i))], axis=1)
        elif df_list_len == 2:
            total_df = pd.concat([total_df, pd.Series(subgraph_significance, name='season_{}'.format(i))], axis=1)
        else:
            total_df = pd.concat([total_df, pd.Series(subgraph_significance, name='ep_{}'.format(i))], axis=1)
        i += 1

    return total_df