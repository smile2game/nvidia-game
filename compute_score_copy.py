import numpy as np
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
import cv2
import datetime
from canny2image_TRT_copy import hackathon
import os 


block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = InceptionV3([block_idx]).to("cuda")

def PD(base_img, new_img):#PD函数是用来计算图像之间的感知距离（Perceptual Distance）的。感知距离是通过计算两个图像在InceptionV3模型中的特征表示之间的欧氏距离来衡量的。
    inception_feature_ref, _ = fid_score.calculate_activation_statistics([base_img], model, batch_size = 1, device="cuda")
    inception_feature, _ = fid_score.calculate_activation_statistics([new_img], model, batch_size = 1, device="cuda")
    pd_value = np.linalg.norm(inception_feature - inception_feature_ref)
    pd_string = F"Perceptual distance to: {pd_value:.2f}"
    print(pd_string)
    return pd_value

scores = []
latencys = []
hk = hackathon()
hk.initialize()
time_list = []
score_list = []
for i in range(20):
    path = "/home/player/pictures_croped/bird_"+ str(i) + ".jpg"
    img = cv2.imread(path)
    start = datetime.datetime.now().timestamp()
    new_img = hk.process(img,
            "a bird", 
            "best quality, extremely detailed", 
            "longbody, lowres, bad anatomy, bad hands, missing fingers", 
            1,   #按理说应该暂停了，怎么继续往下读取字符串
            256, 
            20,
            False, 
            1, #
            9, #
            2946901, 
            0.0, 
            100, 
            200)
    end = datetime.datetime.now().timestamp()
    cost = (end-start)*1000
    print("time cost is: \n加上引擎开跑！！用时:", cost)
    time_list.append(cost)
    new_path = "./bird_"+ str(i) + ".jpg"
    cv2.imwrite(new_path, new_img[0])
    # generate the base_img by running the pytorch fp32 pipeline (origin code in canny2image_TRT.py)
    base_path = path
    score = PD(base_path, new_path)
    score_list.append(score)
    print("得到的分数是: ", score)
    
print("------------\n----------时间列表为：",time_list)
print("------------\n----------平均时间为：",np.mean(time_list))
print("------------\n----------分数列表为：",score_list)
