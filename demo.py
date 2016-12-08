"""
!user/bin/env python
_*_ coding: utf-8 _*_

Created on: Dec 02, 2016

Author: Peng Hao
Email: haopengbuaa@gmail.com

"""

import time
import pandas as pd
from collections import OrderedDict
from model.tagcdcf import TagCDCF


def run():
    # set model candidates
    algorithms = OrderedDict()
    algorithms['tagcdcf'] = run_tagcdcf
    # algorithms['tagicofi'] = run_tagicofi

    # Call recommendation models to run
    results = pd.DataFrame()
    for name, fit_predict in algorithms.items():
        start = time.time()
        rmse = fit_predict()
        spent_time = time.time() - start
        results.ix[name, 'time'] = spent_time
        results.ix[name, 'rmse'] = rmse

    print results


def run_tagcdcf():
    model = TagCDCF(reg_cross_u=0.001, reg_cross_i=0.001, reg_lambda=0.01, num_factor=10)
    model.fit()
    predictions = model.predict()
    rmse = model.evaluate()
    return rmse


def run_tagicofi():
    pass



if __name__ == '__main__':
    run()