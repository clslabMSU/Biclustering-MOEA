from main import get_synthetic_datasets
from BiclusterSet import BiclusterSet

import numpy as np
from pickle import dumps, loads
import sqlite3
from sys import stdout
from time import time


def enhance_from_query(query: str, metric: str, num_to_keep: int, minimize: bool = True, absolute_value: bool = False) -> None:

    datasets = get_synthetic_datasets("synthetic/generated/narrow_trend_preserving/") + \
               get_synthetic_datasets("synthetic/generated/square_trend_preserving/") + \
               get_synthetic_datasets()

    # prepare database
    conn = sqlite3.connect("db/Pyclustering-DB.sqlite")
    c = conn.cursor()
    c.execute(query)

    rows = c.fetchall()
    rows_so_far = 1
    for row in rows:

        stdout.write("\rProcessing row %d..." % rows_so_far); stdout.flush()
        rows_so_far += 1

        biclusters = loads(row[9])

        metric_values = np.empty(len(biclusters))
        for i in range(len(biclusters)):
            if len(biclusters[i].genes())*len(biclusters[i].samples()) != 0:
                metric_values[i] = getattr(biclusters[i], metric)()
            else:
                print("Warning: algorithm %s produced bicluster of invalid size (rows_so_far=%d)" % (row[1], rows_so_far))
                metric_values[i] = np.inf if minimize else -np.inf

        if absolute_value:
            metric_values = [abs(i) for i in metric_values]

        # if we are maximizing metric, instead minimize negative of metric
        if not minimize:
            metric_values = [-i for i in metric_values]

        sorted_indices = np.argsort(metric_values)
        good_indices = np.where(sorted_indices < num_to_keep)[0]
        to_keep = BiclusterSet([biclusters[i] for i in good_indices])

        ground_truth_bics = list(filter(lambda dataset: dataset.name == row[3], datasets))[0].known_bics

        c.execute("INSERT INTO enhancement_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            time(),                                                     # timestamp
            row[1],                                                     # algorithm
            row[2],                                                     # algorithm_params
            row[3],                                                     # dataset_name
            row[4],                                                     # dataset_md5
            "$".join([row[5].split("$")[i] for i in good_indices]),     # biclusters_text
            len(to_keep),                                               # num_biclusters
            ",".join([str(len(x.genes())) for x in to_keep]),           # genes_per_bic
            ",".join([str(len(x.samples())) for x in to_keep]),         # samples_per_bic
            memoryview(dumps(to_keep)),                                 # biclusters_pickle
            to_keep.symmetric_relevance(ground_truth_bics),             # symmetric_relevance
            to_keep.symmetric_recovery(ground_truth_bics),              # symmetric_recovery
            row[12],                                                    # tag
            to_keep.relevance(ground_truth_bics),                       # relevance
            to_keep.recovery(ground_truth_bics),                        # recovery
            row[0],                                                     # timestamp_old
            metric                                                      # metric
        ))

        conn.commit()

    conn.close()


if __name__ == "__main__":

    QUERY = """
        SELECT * --algorithm, dataset_name, AVG(recovery), AVG(relevance), AVG(symmetric_recovery), AVG(symmetric_relevance), COUNT(dataset_name), AVG(num_biclusters), SUM(relevance/num_biclusters)/COUNT(relevance)
        FROM log
        WHERE (tag = '2018.06.08 Runs for Paper (Other Algorithms, New datasets) v3'
            OR tag = '2018.06.07 Runs for Paper (Other Algorithms, UniBic datasets)')
            --OR tag = '2018.06.03 Runs for Paper (NSGA_II, all measures, new datasets)'
            --OR tag = '2018.05.29 Runs for Paper (NSGA_II, all measures)')
        
        UNION
        
        SELECT * --algorithm, dataset_name, AVG(recovery), AVG(relevance), AVG(symmetric_recovery), AVG(symmetric_relevance), COUNT(dataset_name), AVG(num_biclusters), SUM(relevance/num_biclusters)/COUNT(relevance)
        FROM log
        WHERE algorithm = 'EvoBexpa'
            AND tag = '2018.05.29 Runs for Paper (EvoBexpa)'
        
        UNION
        
        SELECT * --'NSGA-II (ASR)', dataset_name, AVG(recovery), AVG(relevance), AVG(symmetric_recovery), AVG(symmetric_relevance), COUNT(dataset_name), AVG(num_biclusters), SUM(relevance/num_biclusters)/COUNT(relevance)
        FROM log
        WHERE (tag = '2018.06.03 Runs for Paper (NSGA_II, all measures, new datasets)'
              OR tag = '2018.05.29 Runs for Paper (NSGA_II, all measures)')
              AND algorithm_params LIKE '%ASR%'
        
        UNION
        
        SELECT * --'NSGA-II (SCS)', dataset_name, AVG(recovery), AVG(relevance), AVG(symmetric_recovery), AVG(symmetric_relevance), COUNT(dataset_name), AVG(num_biclusters), SUM(relevance/num_biclusters)/COUNT(relevance)
        FROM log
        WHERE (tag = '2018.06.03 Runs for Paper (NSGA_II, all measures, new datasets)'
              OR tag = '2018.05.29 Runs for Paper (NSGA_II, all measures)')
              AND algorithm_params LIKE '%SCS%'
        
        UNION
        
        SELECT * --'NSGA-II (VEt)', dataset_name, AVG(recovery), AVG(relevance), AVG(symmetric_recovery), AVG(symmetric_relevance), COUNT(dataset_name), AVG(num_biclusters), SUM(relevance/num_biclusters)/COUNT(relevance)
        FROM log
        WHERE (tag = '2018.06.03 Runs for Paper (NSGA_II, all measures, new datasets)'
              OR tag = '2018.05.29 Runs for Paper (NSGA_II, all measures)')
              AND algorithm_params LIKE '%VEt%'
        
        UNION
        
        SELECT * --'NSGA-II (TPC)', dataset_name, AVG(recovery), AVG(relevance), AVG(symmetric_recovery), AVG(symmetric_relevance), COUNT(dataset_name), AVG(num_biclusters), SUM(relevance/num_biclusters)/COUNT(relevance)
        FROM log
        WHERE (tag = '2018.06.03 Runs for Paper (NSGA_II, all measures, new datasets)'
              OR tag = '2018.05.29 Runs for Paper (NSGA_II, all measures)')
              AND algorithm_params LIKE '%TPC%'
        
        ORDER BY dataset_name ASC;
    """

    #enhance_from_query(QUERY, metric="ASR", num_to_keep=5, minimize=False, absolute_value=True)
    #enhance_from_query(QUERY, metric="SCS", num_to_keep=5, minimize=True, absolute_value=False)
    #enhance_from_query(QUERY, metric="VEt", num_to_keep=5, minimize=True, absolute_value=False)
    enhance_from_query(QUERY, metric="TPC", num_to_keep=5, minimize=False, absolute_value=False)
