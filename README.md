# ICNet

Code for "ICNet: Incorporating Indicator Words and Contexts to Identify Functional Description Information", IJCNN 2019

### Dependency

```
numpy
sklearn
jieba
pytorch
```

### Preprocess

```bash
cd preprocess
python3 pre_classify_data.py
python3 build_stop_word_list.py
python3 segment.py
python3 indicator_tagging.py
```

### Architecture

#### ICNet multi-tasks model

<div align=center>
<img src="./images/ICNet-multi-task.png" width="500px" />
</div>

```bash

```

#### ICNet ensemble model

<div align=center>
<img src="./images/ICNet-ensemble.png" width="500px" />
</div>

### Citation

If you find this work is useful in your research, please consider citing:

```
@inproceedings{liu2019icnet,
  title={ICNet: Incorporating Indicator Words and Contexts to Identify Functional Description Information},
  author={Qu Liu and Zhenyu Zhang and Yanzeng Li and Tingwen Liu and Diying Li and Jinqiao Shi},
  booktitle={2019 International Joint Conference on Neural Networks, {IJCNN} 2019},
  year={2019},
}
```

