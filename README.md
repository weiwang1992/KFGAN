# Improving Knowledge-aware Recommendation with Multi-level Interactive Contrastive Learning
This is our Pytorch implementation for the paper:


## Introduction


## Requirement
The code has been tested running under Python 3.8.16. The required packages are as follows:
- torch == 2.0.1
- numpy == 1.24.4
- sklearn == 1.2.2

## Usage
The hyper-parameter search range and optimal settings have been clearly stated in the codes (see the parser function in src/main.py).
* Train and Test

```
python main.py 
```


## Dataset

We provide three processed datasets: Book-Crossing, MovieLens-1M, and Last.FM.

We follow the paper " [Ripplenet: Propagating user preferences on the knowledge
graph for recommender systems](https://github.com/hwwang55/RippleNet)" to process data.


|                       |               | Book-Crossing | MovieLens-1M | Last.FM |
| :-------------------: | :------------ | ----------:   | --------: | ---------: |
| User-Item Interaction | #Users        |      17,860   |    6,036  |      1,872 |
|                       | #Items        |      14,967   |    2,445  |      3,846 |
|                       | #Interactions |     139,746   |  753,772  |      42,346|
|    Knowledge Graph    | #Entities     |      77,903   |    182,011|      9,366 |
|                       | #Relations    |          25   |         12|         60 |
|                       | #Triplets     |   151,500     |  1,241,996|     15,518 |


## Citation

If you want to use our codes in your research, please cite:
```

