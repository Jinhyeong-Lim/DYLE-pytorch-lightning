# DYLE_pytorch-lightning
Reproduction of source code for ACL 2022 paper DYLE using Pytorch-Lightning: Dynamic latent extraction for abstract long input summary


## Folder Structure
- dataloaders: the python scripts to convert original dataset to the uniform format.
- oracle: Scripts to generate extractive oracles
- utils: Various utility functions, such as cleaning and rouge
- Experiment.py: Main file for our model
- config.py: Set model configuration
- Modules: Contains implementation of our dynamic extraction module
- test.py: Run test set
- train.py: Train the model

## Training and Evaluation

### Download the Datasets and Models
- Download QMSum dataset from https://github.com/Yale-LILY/QMSum
- Using Pyrouge Metric from https://github.com/bheinzerling/pyrouge
## Training

```
  python main.py --gpus -1 --distributed_backend dp
```

## Citation
```bibtex
@inproceedings{mao2021dyle,
  title={DYLE: Dynamic Latent Extraction for Abstractive Long-Input Summarization},
  author={Mao, Ziming and Wu, Chen Henry and Ni, Ansong and Zhang, Yusen and Zhang, Rui and Yu, Tao and Deb, Budhaditya and Zhu, Chenguang and Awadallah, Ahmed H and Radev, Dragomir},
  booktitle={ACL 2022},
  year={2022}
}
``` 
