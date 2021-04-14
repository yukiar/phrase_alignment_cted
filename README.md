# Compositional Phrase Alignment and Beyond

by Yuki Arase (Osaka University)

This repository provides an implementation of the phrase alignment method based on the constrained tree edit distance proposed in the following paper.

Yuki Arase and Jun'ichi Tsujii. 2020. [Compositional Phrase Alignment and Beyond](https://www.aclweb.org/anthology/2020.emnlp-main.125/). in Proc. of the Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1611-1623.  

## Prerequisites
Please make sure to install all the dependent libraries in ``requirements.txt``

SPADE and ESPADA datasets are downloadable from LDC (Linguistic Data Consortium)
* [SPADE](https://catalog.ldc.upenn.edu/LDC2018T09)
* ESPADA (will be released from LDC in May 2021)

Please place these corpora (xml files) in the ```data``` directory (this repository provides just samples for debugging).

## How to 
### Model names
The following labels correspond to models described in the paper.

* ```BERTTrainer```: BERT model with simple fine-tuning
* ```BERT1F```: The proposed model (BERT+SimMatrix)
* ```BERT1E```: The proposed model using [CLS] instead of SimMatrix (BERT+[CLS])

### Trained models
Trained models are distributed at [Zenodo](http://doi.org/10.5281/zenodo.4686663). Note that ALIR and ALIP values of these models are slightly different from what was reported in the paper because the paper reports the average performance of 10 models initialized with random seeds. 

### Fine-tune the BERT model
Pleaes set hyper-parameters as you want.
```
python ./fine-tune_bert.py --out_dir ../model/ --model_type BERT1F --pooling mean --train_epoch 100 --early_stop 5 --margin 1.0 --lr 3e-05 --ft_bert
```
### Alignment with the constrained tree edit distance algorithm
```
python ./main.py --out_dir ../out/ --model_dir ../model/ --model_name BERT1F_TripletMarginLoss_margin-1.0_lr-3e-05_mean_100_ft-bert-base-uncased.pkl --pooling mean --null_thresh 0.8
```

If you want to output alignments, please flag ```--decode``` option.

### Alignment with naive thresholding
```
python ./baseline_wo_ted.py --out_dir ../out/ --model_dir ../model/ --model_name BERT1F_TripletMarginLoss_margin-1.0_lr-3e-05_mean_100_ft-bert-base-uncased.pkl --pooling mean --null_thresh 0.6
```

### Alignment with FastText
Download a FastText model you like, and specify the path to the model ```--model_dir```
```
python ./main.py --out_dir ../out/ --model_name FastText --model_dir ../fasttext/crawl-300d-2M-subword.bin --pooling mean --null_thresh 0.8
python ./baseline_wo_ted.py --out_dir ../out/ --model_name FastText --model_dir ../fasttext/crawl-300d-2M-subword.bin --pooling mean --null_thresh 0.75
```

## Citation
When you use our codes in your projects, please cite the following paper.

Yuki Arase and Jun'ichi Tsujii. 2020. Compositional Phrase Alignment and Beyond. in Proc. of the Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1611-1623.  
```
@inproceedings{arase-tsujii-2020-compositional,
    title = "Compositional Phrase Alignment and Beyond",
    author = "Arase, Yuki  and
      Tsujii, Jun{'}ichi",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.125",
    doi = "10.18653/v1/2020.emnlp-main.125",
    pages = "1611--1623"
}
```
