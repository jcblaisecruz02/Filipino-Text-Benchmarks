# Filipino-Text-Benchmarks
Code and supplementary material for the paper "Establishing Baselines for Text Classification in Low-Resource Languages"

# Datasets
Hate Speech Dataset [[link]](https://storage.googleapis.com/blaisecruz/datasets/hatenonhate/hatespeech_raw.zip)

Dengue Dataset [[Link]](https://storage.googleapis.com/blaisecruz/datasets/dengue/dengue_raw.zip)

# Pretrained Models
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
