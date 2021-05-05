from pyspark.sql.types import *
from optimus import Optimus
from optimus.helpers.json import json_enconding
from optimus.helpers.functions import deep_sort
import unittest
from pyspark.ml.linalg import Vectors, VectorUDT, DenseVector
import numpy as np
nan = np.nan
from optimus.engines.spark.ml import distancecluster as dc
op = Optimus(master='local')
class Test_df_distance_cluster(unittest.TestCase):
	maxDiff = None
	def test_levenshtein_cluster(self):
		actual_df =dc.levenshtein_cluster(source_df,'STATE')
		expected_value ={'Estado de México': {'similar': {'Distrito Federal': 710, 'Estado de México': 290}, 'count': 2, 'sum': 1000}, 'Distrito Federal': {'similar': {'Estado de México': 290, 'Distrito Federal': 710}, 'count': 2, 'sum': 1000}}
		self.assertDictEqual(deep_sort(expected_value),  deep_sort(actual_df))