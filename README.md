# MF-Pytorch
Matrix Factorization for Recommendation with Pytorch

## Algoritm

**Predicting** a rating:
$$ \hat r_{ui} = \mu + b_u + b_i + q_i^Tp_u$$


**Optimize** the model:
$$ \min \sum (r_{ui} - \mu - b_u - b_i - q_i^Tp_u) + \mathcal{L2}$$

## Usage

...


## Test

#### Coat

> py .\main.py --dataset=coat --num=10 --iter=40 --lr=1e-3 --l2=1e-5 --binary=True

test rmse: `0.392228`, random rmse: `0.563749`
![](pics/coat_visual.png)

#### Yahoo! R3

> py .\main.py --dataset=yahoo --num=50 --iter=30 --lr=1e-3 --l2=1e-4 --binary=True

test rmse: `0.388629`, random rmse: `0.575621`
![](pics/yahoo_visual.png)

> py .\main.py --dataset=yahoo --num=50 --iter=30 --lr=1e-3 --l2=1e-4 --binary=False
> 
test rmse: `1.141142`, random rmse: `2.166870`
![](pics/raw_yahoo_visual.png)
#### KuaiRand

> py .\main.py --dataset=kuairand --key=is_click --num=50 --iter=50 --lr=1e-3 --l2=1e-5 --binary=True

test rmse: `0.382446`, random rmse: `0.577986`
![](pics/kuairand_is_click_visual.png)

> py .\main.py --dataset=kuairand --key=long_view --num=50 --iter=50 --lr=1e-3 --l2=1e-5 --binary=True

test rmse: `0.311767`, random rmse: `0.576226`
![](pics/kuairand_long_view_visual.png)

> py .\main.py --dataset=kuairand --key=is_like --num=50 --iter=50 --lr=1e-3 --l2=1e-5 --binary=True

test rmse: `0.080412`, random rmse: `0.576377`
![](pics/kuairand_is_like_visual.png)
