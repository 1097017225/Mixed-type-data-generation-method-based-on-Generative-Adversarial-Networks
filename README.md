# medgan
## 1. 在论文‘3.1’小节中，"where the value of the i th dimension indicates the number of occurrences (i.e., counts) of the i -th variable in the patient record",ith的值代表的是变量X中第i个维度记录的病人的某一属性的值吗？

## 2.在‘3.2’小节中，G can be trained to maximize log(D(G(z)) instead of minimizing log(1 − D(G(z))，是为了加大训练的梯度吗，使开始的训练步幅加大？

## 3. 在‘3.4’小节中，"This problem is denoted as mode collapse, which arises most likely due to the GAN’s optimization strategy often solving the max-min problem instead of the min-max problem",max-min和min-max有什么区别

## 4. 在‘3.4’小节中， "This is especially likely when mode collapse occurs because the ˆ p k ’s for most dimensions of the fake samples become dichotomized (either 0 or 1),whereas the ˆ p k ’s of real samples generally take on a value between 0 and 1. Therefore, if G wants to fool D , it will have to generate more diverse examples within the minibatch Dec(G(z 1 ,z 2 ,...))",Binary variables变量不就是在0和1中取值吗，为什么是在0和1之间取值？
