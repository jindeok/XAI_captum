# XAI pytorch framework with captum
XAI framework with captum(https://captum.ai/): model interpretation with pytorch
example codes contains CIFAR-10 downloaded from online, which can be replaced.

![image](https://user-images.githubusercontent.com/35905280/119530575-1b09e000-bdbe-11eb-8322-fbed1569b063.png)
![image](https://user-images.githubusercontent.com/35905280/119530591-1e9d6700-bdbe-11eb-90d2-b9eaacca050b.png)



# Dependancy
* captum
any version of (but not old-dated)
* torch
* numpy
* matplotlib





# Running

run main.py in the Lottery_Prediction folder
or in terminal,

``python main.py --data_dir 'dataset/lottery_history.csv' --mode 'eval_model' --trial 5 --training_lengt 0.90 ``  

dataset version is on the date: [21.04.23]

you can easily update dataset from the Web:  https://dhlottery.co.kr/common.do?method=main

***some comments on the arguments:***

training_length : there are ~980 lottery cases in total. you can decide to what extent to use as for training length.
(e.g. 0.5 training_lengh uses 485 lottery cases are used for training and infer 486 th as a test set.)

