# Domain Generalizable Person Search Using Unreal Training Dataset (AAAI2024)

This is an official repository for Domain Generalizable Person Search Using Unreal Dataset (AAAI2024).
### Dataset

Download the JTA dataset required for training at [here](https://github.com/fabbrimatteo/JTA-Dataset). 

We selected specific images from this dataset for training. 

The list of these images, including their annotations, can be found in `pickles/jta.pickle`. 

We also utilize `pickles/brisque.pickle` for our proposed method.

### Train
```
sh train.sh
```

