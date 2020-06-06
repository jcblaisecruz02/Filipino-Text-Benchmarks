# Filipino-Text-Benchmarks

*This repository is a work in progress!*

This consolidated repository contains data and models from our two papers: 
* Establishing Baselines for Text Classification in Low-Resource Languages [(Cruz & Cheng, 2020)](https://arxiv.org/abs/2005.02068)
* Evaluating Language Model Finetuning Techniques for Low-resource Languages [(Cruz & Cheng, 2019)](https://arxiv.org/abs/1907.00409)

# Requirements
* PyTorch v1
* Transformers
* Optuna (optional, for hyperparameter search)
* tqdm
* NVIDIA GPU (all experiments were done on Tesla P100 GPUs)

# Reproducing Results
To finetune the models to a text classification dataset, you may use the provided ```train.py``` script. Make sure to remove the ```--lowercase``` flag when using cased models. To fit larger batch sizes on smaller GPUs, set the ```--accumulation``` argument to use gradient accumulation. Please see the script for a full list of command line arguments.

Here's an example that finetunes a small cased ELECTRA model on the Hatespeech dataset.

```
python train.py \
    --pretrained jcblaise/electra-tagalog-small-cased-discriminator \
    --train_data hatespeech/train.csv \
    --valid_data hatespeech/valid.csv \
    --test_data hatespeech/test.csv \
    --checkpoint model.pt \
    --do_train \
    --do_eval \
    --data_pct 1.0 \
    --msl 128 \
    --batch_size 32 \
    --add_token [LINK] \
    --add_token [HASHTAG] \
    --add_token [MENTION] \
    --weight_decay 8e-7 \
    --learning_rate 9e-5 \
    --adam_epsilon 1e-6 \
    --use_scheduler \
    --warmup_pct 0.1 \
    --epochs 3 \
    --seed 8139
```

Running this setup should give you a validation accuracy of 0.7620 and a testing accuracy of 0.7500.

The ```--label_column``` argument specifies the names of the columns that are considered targets (set to "label" by default). It can also take a comma-separated list of label columns to perform multilabel classification. Here's an example that finetunes a small ELECTRA model on the Dengue dataset.

```
python train.py \
    --pretrained jcblaise/electra-tagalog-small-cased-discriminator \
    --train_data dengue/train.csv \
    --valid_data dengue/valid.csv \
    --test_data dengue/test.csv \
    --label_column absent,dengue,health,mosquito,sick \
    --checkpoint model.pt \
    --do_train \
    --do_eval \
    --data_pct 1.0 \
    --msl 128 \
    --batch_size 32 \
    --add_token [LINK] \
    --add_token [HASHTAG] \
    --add_token [MENTION] \
    --weight_decay 8e-7 \
    --learning_rate 9e-5 \
    --adam_epsilon 1e-6 \
    --use_scheduler \
    --warmup_pct 0.1 \
    --epochs 3 \
    --seed 8139
```

This setup should yield a validation accuracy of 0.8313 and a test accuracy of 0.8553.

# Hyperparameter Search
You can perform hyperparameter search via Optuna using the same script. Toggle the ```--optimize_hyperparameters``` argument to use hyperparameter search. Searching for random seed (```---optimize_seed```), learning rate (```---optimize_learning_rate```), and weight decay (```---optimize_weight_decay```) are available out of the box. You can also toggle ```---dont_save``` to forego checkpoint saving to save on time and operations during long runs.

Here's a sample search for random seed run for 100 trials using an uncased ELECTRA model (don't forget to toggle ```--lowercase``` when using uncased models!)

```
python train.py \
    --pretrained jcblaise/electra-tagalog-small-uncased-discriminator \
    --lowercase \
    --train_data hatespeech/train.csv \
    --valid_data hatespeech/valid.csv \
    --do_train \
    --dont_save \
    --data_pct 1.0 \
    --msl 128 \
    --batch_size 32 \
    --accumulation 1 \
    --add_token [LINK] \
    --add_token [HASHTAG] \
    --add_token [MENTION] \
    --weight_decay 8e-7 \
    --learning_rate 9e-5 \
    --adam_epsilon 1e-6 \
    --use_scheduler \
    --warmup_pct 0.1 \
    --epochs 3 \
    --seed 42 \
    --optimize_hyperparameters \
    --study_name seed_search \
    --opt_n_trials 100 \
    --optimize_seed \
    --opt_seed_lowerbound 1 \
    --opt_seed_upperbound 9999
```

# Datasets
* **WikiText-TL-39** [`download`](https://s3.us-east-2.amazonaws.com/blaisecruz.com/datasets/wikitext-tl-39/wikitext-tl-39.zip)\
*Large Scale Unlabeled Corpora in Filipino*\
Large scale, unlabeled text dataset with 39 Million tokens in the training set. Inspired by the original [WikiText Long Term Dependency dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) (Merity et al., 2016). TL means "Tagalog." Originally published in [Cruz & Cheng (2019)](https://arxiv.org/abs/1907.00409).

* **Hate Speech Dataset** [`download`](https://s3.us-east-2.amazonaws.com/blaisecruz.com/datasets/hatenonhate/hatespeech_raw.zip)\
*Text Classification Dataset in Filipino* \
Contains 10k tweets (training set) that are labeled as hate speech or non-hate speech. Released with 4,232 validation and 4,232 testing samples. Collected during the 2016 Philippine Presidential Elections and originally used in Cabasag et al. (2019).

* **Dengue Dataset** [`download`](https://s3.us-east-2.amazonaws.com/blaisecruz.com/datasets/dengue/dengue_raw.zip)\
*Low-Resource Multiclass Text Classification Dataset in Filipino*\
Benchmark dataset for low-resource multiclass classification, with 4,015 training, 500 testing, and 500 validation examples, each labeled as part of five classes. Each sample can be a part of multiple classes. Collected as tweets and originally used in Livelo & Cheng (2018).

# Pretrained ELECTRA Models
We release new ELECTRA models in small and base configurations, with both the discriminator and generators available. All the models follow the same setups and were trained with the same hyperparameters as English ELECTRA models. Our models are available on HuggingFace Transformers and can be used on both PyTorch and Tensorflow.

**Discriminator Models**

* ELECTRA Base Cased Discriminator - [`jcblaise/electra-tagalog-base-cased-discriminator`](https://huggingface.co/jcblaise/electra-tagalog-base-cased-discriminator) 
* ELECTRA Base Uncased Discriminator - [`jcblaise/electra-tagalog-base-uncased-discriminator`](https://huggingface.co/jcblaise/electra-tagalog-base-uncased-discriminator) 
* ELECTRA Small Cased Discriminator - [`jcblaise/electra-tagalog-small-cased-discriminator`](https://huggingface.co/jcblaise/electra-tagalog-small-cased-discriminator) 
* ELECTRA Small Uncased Discriminator - [`jcblaise/electra-tagalog-small-uncased-discriminator`](https://huggingface.co/jcblaise/electra-tagalog-small-uncased-discriminator)

**Generator Models**

* ELECTRA Base Cased Generator - [`jcblaise/electra-tagalog-base-cased-generator`](https://huggingface.co/jcblaise/electra-tagalog-base-cased-generator) 
* ELECTRA Base Uncased Generator - [`jcblaise/electra-tagalog-base-uncased-generator`](https://huggingface.co/jcblaise/electra-tagalog-base-uncased-generator) 
* ELECTRA Small Cased Generator - [`jcblaise/electra-tagalog-small-cased-generator`](https://huggingface.co/jcblaise/electra-tagalog-small-cased-generator)
* ELECTRA Small Uncased Generator - [`jcblaise/electra-tagalog-small-uncased-generator`](https://huggingface.co/jcblaise/electra-tagalog-small-uncased-generator)

The models can be loaded using the code below:

```Python
from transformers import TFAutoModel, AutoModel, AutoTokenizer

# TensorFlow
model = TFAutoModel.from_pretrained('jcblaise/electra-tagalog-small-cased-generator', from_pt=True)
tokenizer = AutoTokenizer.from_pretrained('jcblaise/electra-tagalog-small-cased-generator', do_lower_case=False)

# PyTorch
model = AutoModel.from_pretrained('jcblaise/electra-tagalog-small-cased-generator')
tokenizer = AutoTokenizer.from_pretrained('jcblaise/electra-tagalog-small-cased-generator', do_lower_case=False)
```

# Pretrained BERT Models
We release four Tagalog BERT Base models and one Tagalog DistilBERT Base model. All the models use the same configurations as the original English BERT models. Our models are available on HuggingFace Transformers and can be used on both PyTorch and Tensorflow.

* BERT Base Cased - [`jcblaise/bert-tagalog-base-cased`](https://huggingface.co/jcblaise/bert-tagalog-base-cased) 
* BERT Base Uncased - [`jcblaise/bert-tagalog-base-uncased`](https://huggingface.co/jcblaise/bert-tagalog-base-uncased) 
* BERT Base Cased WWM - [`jcblaise/bert-tagalog-base-cased-WWM`](https://huggingface.co/jcblaise/bert-tagalog-base-cased-WWM) 
* BERT Base Uncased WWM - [`jcblaise/bert-tagalog-base-uncased-WWM`](https://huggingface.co/jcblaise/bert-tagalog-base-uncased-WWM) 
* DistilBERT Base Cased - [`jcblaise/distilbert-tagalog-base-cased`](https://huggingface.co/jcblaise/distilbert-tagalog-base-cased) 

The models can be loaded using the code below:

```Python
from transformers import TFAutoModel, AutoModel, AutoTokenizer

# TensorFlow
model = TFAutoModel.from_pretrained('jcblaise/bert-tagalog-base-cased', from_pt=True)
tokenizer = AutoTokenizer.from_pretrained('jcblaise/bert-tagalog-base-cased', do_lower_case=False)

# PyTorch
model = AutoModel.from_pretrained('jcblaise/bert-tagalog-base-cased')
tokenizer = AutoTokenizer.from_pretrained('jcblaise/bert-tagalog-base-cased', do_lower_case=False)
```

# Other Pretrained Models
* **ULMFiT-Tagalog** [`download`](https://s3.us-east-2.amazonaws.com/blaisecruz.com/ulmfit-tagalog/models/pretrained-wikitext-tl-39.zip)\
Tagalog pretrained AWD-LSTM compatible with the FastAI library. Originally published in [Cruz & Cheng (2019)](https://arxiv.org/abs/1907.00409).

# Citations
If you found our work useful, please make sure to cite!

```
@article{cruz2020establishing,
  title={Establishing Baselines for Text Classification in Low-Resource Languages},
  author={Cruz, Jan Christian Blaise and Cheng, Charibeth},
  journal={arXiv preprint arXiv:2005.02068},
  year={2020}
}
```

```
@article{cruz2019evaluating,
  title={Evaluating Language Model Finetuning Techniques for Low-resource Languages},
  author={Cruz, Jan Christian Blaise and Cheng, Charibeth},
  journal={arXiv preprint arXiv:1907.00409},
  year={2019}
}
```
