# label-embedding-initialisation

This project proposes a label embedding initialisation approach to improve multi-label classification (and especially for automated medical coding).

<p align="center">
    <img src="https://github.com/anonymised-account/label-embedding-medical-coding/blob/master/label-embedding-init-figure.PNG" width="400" title="Label Embedding Initialisation for Deep-Learning-Based Multi-Label Classification">
</p>

Key part of the implementation of label embedding initiailisation adapted from CAML ([models.py in CAML](https://github.com/jamesmullenbach/caml-mimic/blob/master/learn/models.py)).

```
# based on https://github.com/jamesmullenbach/caml-mimic/blob/master/learn/models.py
def _code_emb_init(self, code_emb, code_list):
        # code_list is a list of code having the same order as in multi-hot representation (sorted by frequency from high to low)
        code_embs = Word2Vec.load(code_emb)
        # bound for random variables for Xavier initialisation.
        bound = np.sqrt(6.0) / np.sqrt(self.num_labels + code_embs.vector_size)  
        weights = np.zeros(self.classifier.weight.size())
        n_exist, n_inexist = 0, 0
        for i in range(self.num_labels):
            code = code_list[i]
            if code in code_embs.wv.vocab:
                n_exist = n_exist + 1
                vec = code_embs.wv[code]
                #normalise to unit length
                weights[i] = vec / float(np.linalg.norm(vec) + 1e-6) 
            else:
                n_inexist = n_inexist + 1
                #using the original xavier uniform initialisation.
                weights[i] = np.random.uniform(-bound, bound, code_embs.vector_size);                 
        print("code exists embedding:", n_exist, " ;code not exist embedding:", n_inexist)
        
        # initialise label embedding for the weights in the final linear layer
        self.classifier.weight.data = torch.Tensor(weights).clone()
        print("final layer: code embedding initialised")
```

Label embedding initiailisation for BERT (adapting the [models.py in SimpleTransformers](https://github.com/ThilinaRajapakse/simpletransformers/blob/master/simpletransformers/custom_models/models.py))

```
# adapting https://github.com/ThilinaRajapakse/simpletransformers/blob/master/simpletransformers/custom_models/models.py
def _code_emb_init(self, config, code_emb, code_list):
        # code_list is a list of code sorted by frequency from high to low
        code_embs = Word2Vec.load(code_emb)
        std = config.initializer_range
        weights = np.zeros(self.classifier.weight.size())
        n_exist, n_inexist = 0, 0
        for i in range(self.num_labels):
            code = code_list[i]
            if code in code_embs.wv.vocab:
                n_exist = n_exist + 1
                vec = code_embs.wv[code]
                weights[i] = stats.zscore(weights[i])*std #standardise to the same as the originial initilisation in https://huggingface.co/transformers/_modules/transformers/modeling_bert.html
            else:
                n_inexist = n_inexist + 1
                weights[i] = np.random.normal(0, std, code_embs.vector_size);
        print("code exists embedding:", n_exist, " ;code not exist embedding:", n_inexist)
        self.classifier.weight.data = torch.Tensor(weights).clone()
        print("final layer: code embedding initialized")
```

# Requirements
* Python 3.6.*
* PyTorch 0.3.0 with [CAML](https://github.com/jamesmullenbach/caml-mimic) for CNN,BiGRU,CNN+att models for CNN,BiGRU,CNN+att models
* PyTorch 1.0.0+ for BERT models
* [Huggingface Transformers](https://github.com/huggingface/transformers) for BERT training and BioBERT model conversion to PyTorch
* [SimpleTransformers](https://github.com/ThilinaRajapakse/simpletransformers) 0.20.2 for Multi-Label Classfication with BERT models
* [gensim](https://radimrehurek.com/gensim/) for pre-training label embeddings with the word2vec algorithm
* [BioBERT](https://github.com/dmis-lab/biobert) for pre-trained BioBERT models.

# Dataset and preprocessing
We use [the MIMIC-III dataset](https://mimic.physionet.org/) with the preprocessing steps from [CAML](https://github.com/jamesmullenbach/caml-mimic).

# Using BioBERT
See answer from https://github.com/huggingface/transformers/issues/457#issuecomment-518403170.

# Acknowledgement
* MIMIC-III dataset is from https://mimic.physionet.org/ after request and training.
* Thanks for the kind answers from [SimpleTransformers](https://github.com/ThilinaRajapakse/simpletransformers).
