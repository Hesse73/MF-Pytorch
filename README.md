# MF-Pytorch
Matrix Factorization for Recommendation with Pytorch

## Algoritm

**Predicting** a rating:
$$ \hat r_{ui} = \mu + b_u + b_i + q_i^Tp_u$$


**Optimize** the model:
$$ \min \sum (r_{ui} - \mu - b_u - b_i - q_i^Tp_u) + \mathcal{L2}$$

## Usage

...