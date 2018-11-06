# WIP
Implementation of the *Transformer* model in the paper:

> Ashish Vaswani, et al. ["Attention is all you need."](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) NIPS 2017.

![Transformer model](https://lilianweng.github.io/lil-log/assets/images/transformer.png)

Check my blog post on attention and transformer:
* [Attention? Attention!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)

Implementations that helped me:
* https://github.com/Kyubyong/transformer/
* https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
* http://nlp.seas.harvard.edu/2018/04/01/attention.html


A couple of tricking points in the implementation.

* How to construct the mask correctly?
* How to correctly shift decoder input (as training input) and decoder target (as ground truth in the loss function)?
* How to make the prediction in an autoregressive way?
* Keeping the embedding of `<pad>` as a constant zero vector is sorta important.

    