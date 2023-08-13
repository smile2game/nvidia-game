import numpy as np
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
import cv2
import datetime
from canny2image_TRT import hackathon

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
for i in range(20):
    path = "/home/player/pictures_croped/bird_"+ str(i) + ".jpg"
    img = cv2.imread(path)
    start = datetime.datetime.now().timestamp()
    new_img = hk.process(img,
            "a bird", 
            "best quality, extremely detailed", 
            "longbody, lowres, bad anatomy, bad hands, missing fingers",  #
            1, #num_samples生成样本数量
            256, #image_resolution：图像的分辨率。
            20, #ddim_steps：DDIM采样的步数。
            False,  #guess_mode：一个布尔值，表示是否使用猜测模式。
            1, #strength：控制信号的强度。
            9, #scale：无条件控制信号的比例。
            2946901, #seed：随机数种子。
            0.0,  #eta：DDIM采样的步长。
            100, #low_threshold：Canny边缘检测的低阈值。
            200) #high_threshold：Canny边缘检测的高阈值。
    end = datetime.datetime.now().timestamp()
    cost = (end-start)*1000
    print("time cost is: \n加上引擎开跑！！用时:", cost)
    latencys.append(cost)
    new_path = "./pics_bird/bird_"+ str(i) + ".jpg"
    cv2.imwrite(new_path, new_img[0])
    # generate the base_img by running the pytorch fp32 pipeline (origin code in canny2image_TRT.py)
    base_path = "./torch_out/bird_"+ str(i) + ".jpg"
    score = PD(base_path, new_path)
    scores.append(score)
    print("得到的分数是: ", score)
print("------------\n----------时间列表为：",latencys)
print("------------\n----------放在服务器上平均时间为：",np.mean(latencys)+90)
print("------------\n----------分数列表为：",scores)
print("------------\n----------放在服务器上分数平均分：",np.mean(scores)+2)
count = 0
for i in scores:
    if i >= 6.5:
        count += 1
print("超出8.5的个数为:",count)


file_path = "for_sleep.txt"
with open(file_path,"a") as file:
    file.write("--------------------------------")
    file.write("\n时间列表:\n")
    for item in latencys:
        file.write(str(item) + "  ")
    file.write("\n时间平均值：\n" + str(np.mean(latencys)+90))

    file.write("\n得分列表:\n")
    for item in scores:
        file.write(str(item) + "  ")
    file.write("\n得分平均值：\n" + str(np.mean(scores)+2))
  


