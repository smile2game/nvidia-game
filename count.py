import numpy
import torch

scores = [9.67924590379479,10.293355525504396,10.460323282881708,6.936035375531576,9.002036825635402,7.988979023924162,14.377797413083417,7.788097056677987,17.591474899226267,9.478830602783315,12.966213373245523,5.93560766350691,14.232793834265888,10.090601826350161,16.102030324559568,10.099590017796361,9.446736308954327,17.1146727862264,16.231284362560025,8.475846083354329,8.499630315374075,11.973949052349814,12.076400925277898,13.6114562333021,7.154256541452534,8.295452834289696,14.854668087103892,9.815353227875947,9.682608851043575,6.468848673981737,7.354803178226011,7.444521845244999,11.842055152526722,7.907992170560781,14.314386230460435]
latencys = [242.6149845123291,242.3110008239746,246.44207954406738,244.23503875732422,243.52002143859863,244.25697326660156,243.4561252593994,243.38698387145996,241.4848804473877,243.2270050048828,241.66488647460938,243.21603775024414,241.38498306274414,245.04399299621582,244.19593811035156,243.3760166168213,244.87805366516113,243.1640625,243.36481094360352,242.88105964660645,241.87302589416504,242.91300773620605,242.13385581970215,243.18504333496094,246.11592292785645,244.17805671691895,245.47290802001953,243.50595474243164,245.0540065765381,245.24307250976562,242.9959774017334,245.84197998046875,243.58081817626953,242.35796928405762,242.03085899353027]
scores = numpy.array(scores)
latencys = numpy.array(latencys)
count = 0
for i in scores:
    if i >= 8:
        count += 1

print(len(scores))
print("分数超出预期个数:",count)
print("\n分数平均值:",scores.mean())
print("\n时间平均值:",latencys.mean())