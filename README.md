# XAI pytorch framework with captum
XAI framework with captum(https://captum.ai/): model interpretation with pytorch
example codes contains CIFAR-10 downloaded from online, which can be replaced.



# Dependancy
* captum ``conda install captum -c pytorch``
* torch
* numpy
* matplotlib



# Running

run main.py 
or in terminal,

``python main.py --XAI_method 'IntegratedGradient' ``  

XAI method can be replaced by other XAI methods listed as follows:
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel

which will produce explained heatmap and original image accordingly

![image](https://user-images.githubusercontent.com/35905280/119530575-1b09e000-bdbe-11eb-8322-fbed1569b063.png)
![image](https://user-images.githubusercontent.com/35905280/119530591-1e9d6700-bdbe-11eb-90d2-b9eaacca050b.png)

