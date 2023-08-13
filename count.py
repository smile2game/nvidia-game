import numpy
import torch

scores = [6.980088393125058,8.881376537147695,9.679177833079946,9.43168953092868,11.906907622922633,6.671339603312171,4.292077170903501,8.98155253993086,6.943475707855,12.913600915065706,12.167156630736132,9.253256340256165,2.4644891898240133,8.987209678569172,6.82824124343222,10.300536832226758,9.176423526919635,6.684904844145753,12.077863340309037,12.214466759097535,6.908038746667941,5.801727288970543,6.967825143235681,7.952273481695379,11.42350797818043,7.536683588747917,7.457147634578917,14.958996146430033,9.220029977394095,7.055474570446699,6.690683295861521,6.411754989084152,4.81318006494811,7.030679337233605,6.902336572531505,10.12826081644159]
latencys = [243.2079315185547,240.9040927886963,241.28103256225586,241.42098426818848,242.05589294433594,243.40510368347168,242.35105514526367,243.54100227355957,242.5549030303955,240.85497856140137,241.76788330078125,243.30997467041016,241.14608764648438,240.56696891784668,242.32888221740723,242.16413497924805,241.2889003753662,240.5109405517578,241.8069839477539,240.52786827087402,240.37694931030273,240.95582962036133,242.8262233734131,243.47805976867676,241.47582054138184,240.678071975708,240.71097373962402,239.42184448242188,240.39912223815918,239.69507217407227,242.2471046447754,241.6250705718994,240.25988578796387,241.5449619293213,239.33696746826172]
scores = numpy.array(scores)
latencys = numpy.array(latencys)
count = 0
for i in scores:
    if i >= 8:
        count += 1

print("",len(scores))
print("分数超出6.5分的个数:",count)
print("\n分数平均值:",scores.mean())
print("\n时间平均值:",latencys.mean())