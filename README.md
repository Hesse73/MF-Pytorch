# MF-Pytorch
Matrix Factorization for Recommendation with Pytorch

## Algoritm

**Predict** a rating:
$$\hat r_{ui} = \mu + b_u + b_i + q_i^Tp_u$$


**Optimize** the model:
$$\min \sum (r_{ui} - \mu - b_u - b_i - q_i^Tp_u) + \mathcal{L2}$$

## Usage

Replace dataset directories in `config.py`, then use command below in **Test**.

## Test

### Coat

**Binary Dataset**

(Exit at epoch 22 with `early stop`)
> py .\main.py --dataset=coat --num=10 --iter=40 --lr=1e-3 --l2=1e-5 --early_stop --binary

test rmse: `0.392228`, random rmse: `0.576668`
![](pics/coat_visual.png)

**Raw Dataset**

(Exit at epoch 69 with `early stop`)
> py .\main.py --dataset=coat --num=10 --iter=80 --lr=1e-3 --l2=1e-5 --early_stop

test rmse: `1.090201`, random rmse: `2.328456`
![](pics/raw_coat_visual.png)

### Yahoo! R3

**Binary Dataset**

(Exit at epoch 25 with `early stop`)
> py .\main.py --dataset=yahoo --num=50 --iter=30 --lr=1e-3 --l2=1e-4 --early_stop --binary

test rmse: `0.388629`, random rmse: `0.579042`
![](pics/yahoo_visual.png)

**Raw Dataset**

(Exit at epoch 24 with `early stop`)
> py .\main.py --dataset=yahoo --num=50 --iter=30 --lr=1e-3 --l2=1e-4 --early_stop

test rmse: `1.141142`, random rmse: `2.567613`
![](pics/raw_yahoo_visual.png)
### KuaiRand

**Binary Dataset**

(Exit at epoch 9 with `early stop`)
> py .\main.py --dataset=kuairand --key=is_click --num=50 --iter=50 --lr=1e-3 --l2=1e-5 --early_stop --binary

test rmse: `0.382446`, random rmse: `0.577986`
![](pics/kuairand_is_click_visual.png)


(Exit at epoch 11 with `early stop`)
> py .\main.py --dataset=kuairand --key=long_view --num=50 --iter=50 --lr=1e-3 --l2=1e-5 --early_stop --binary

test rmse: `0.311767`, random rmse: `0.576226`
![](pics/kuairand_long_view_visual.png)


(Exit at epoch 20 with `early stop`)
> py .\main.py --dataset=kuairand --key=is_like --num=50 --iter=50 --lr=1e-3 --l2=1e-5 --early_stop --binary

test rmse: `0.080412`, random rmse: `0.576281`
![](pics/kuairand_is_like_visual.png)

**Continuous Dataset**

(Exit at epoch 5 with `early stop`)
> py .\main.py --dataset=kuairand --num=50 --iter=50 --lr=1e-3 --l2=1e-5 --early_stop 

test rmse: `0.399541`, random rmse: `0.614998`
![](pics/raw_kuairand_visual.png)
