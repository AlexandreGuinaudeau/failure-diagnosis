import os
import pandas as pd

RESULTS_PATH = os.path.realpath(os.path.join(__file__, "..", "results.csv"))

results = pd.read_csv(RESULTS_PATH)
d = {"num_tries": 500, "kind": "NUM"}
for k, v in d.items():
    results = results[results[k] == v]

cols = [u'<0.500000', u'<0.800000', u'<0.900000', u'<0.950000', u'<0.990000',
        u'kind', u'length_numerical', u'mean', u'median', u'std', u'model']
print results[cols]
