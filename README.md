# Filipino-Text-Benchmarks
Code and supplementary material for the paper "Establishing Baselines for Text Classification in Low-Resource Languages"

# Datasets
* **WikiText-TL-39** [`download`](https://storage.googleapis.com/blaisecruz/datasets/wikitext-tl-39/wikitext-tl-39.zip)\
*Large Scale Unlabeled Corpora in Filipino*\
Large scale, unlabeled text dataset with 39 Million tokens in the training set. Inspired by the original [WikiText Long Term Dependency dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) (Merity et al., 2016). TL means "Tagalog." Originally published in [Cruz & Cheng (2019)](https://arxiv.org/abs/1907.00409).

* **Hate Speech Dataset** [`download`](https://storage.googleapis.com/blaisecruz/datasets/hatenonhate/hatespeech_raw.zip)\
*Text Classification Dataset in Filipino* \
Contains 10k tweets (training set) that are labeled as hate speech or non-hate speech. Released with 4,232 validation and 4,232 testing samples. Collected during the 2016 Philippine Presidential Elections and originally used in Cabasag et al. (2019).

* **Dengue Dataset** [`download`](https://storage.googleapis.com/blaisecruz/datasets/dengue/dengue_raw.zip)\
*Low-Resource Multiclass Text Classification Dataset in Filipino*\
Benchmark dataset for low-resource multiclass classification, with 4,015 training, 500 testing, and 500 validation examples, each labeled as part of five classes. Each sample can be a part of multiple classes. Collected as tweets and originally used in Livelo & Cheng (2018).

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
tokenizer = AutoTokenizer.from_pretrained('jcblaise/bert-tagalog-base-cased')

# PyTorch
model = AutoModel.from_pretrained('jcblaise/bert-tagalog-base-cased')
tokenizer = AutoTokenizer.from_pretrained('jcblaise/bert-tagalog-base-cased')
```

# Other Pretrained Models
* **ULMFiT-Tagalog** [`download`](https://storage.googleapis.com/blaisecruz/ulmfit-tagalog/models/pretrained-wikitext-tl-39.zip)\
Tagalog pretrained AWD-LSTM compatible with the FastAI library. Originally published in [Cruz & Cheng (2019)](https://arxiv.org/abs/1907.00409).

# Citations
If you found our work useful, please make sure to cite!

```
@article{cruz2020establishing,
  title={Establishing Baselines for Text Classification in Low-Resource Languages},
  author={Cruz, Jan Christian Blaise and Cheng, Charibeth},
  journal={arXiv preprint arXiv:},
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
