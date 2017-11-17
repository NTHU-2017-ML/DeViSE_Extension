# MileStone2 - DeViSE on CIFAR10

## Core visual model
1. Environment Setting:
    - PyTorch: We choose **[PyTorch]** (see this link for installation instructions) as our deep learning framework.
    
2. Dataset: [CIFAR10]
    - [PyTorch provided CIFAR10]
    
    We use normalized version of CIFAR10 provided by PyTorch, which is very easy to download.
    
3. CNN model: ResNet-18
    - Github: [3rd party version of ResNet-18 on CIFAR10]
    
4. How to use it?
    - `git clone` this project, and `cd` into `devise` folder.
    - Now first run `devise_pretrain_main.py` to pretrain the core visual model. It will save the best version checkpoint.
    - After having a satisfying pretrained model saved, run `devise_finetune_main.py` to finetune on it. And that's all!

## Word2Vec Pre-trained model
1. Environment Setting:
    - Gensim: We use **gensim** to load the pretrain vector.
    ```
    pip install gensim
    ```

2. Dataset(English): [Word2Vec pretrained vector]
    - Github: [fastText Pre-trained]
    - Resource: [Pre-train vector dataset]

    We use the model pretrained by Facebook Research, **fastText**.
    It embeds words to vectors with dimension of 300.

3. How to use it?
    - Load the pretrained model through **gensim**.
    ```
    import gensim
    model = gensim.models.KeyedVectors.load_word2vec_format("wiki.en.vec", binary=False)
    ```
    - Get the embedding vectors of CIFAR10's class labels.
    ```
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for i in range(len(classes)):
        output[i] = model.wv[classes[i]]
    ```
    - Write the embedding vectors of those classes to a pickle file.
     ```
    import pickle

    output_file = open('output.pkl', 'wb')
    pickle.dump(output, output_file, protocol=2)
    output_file.close()
    ```
    - The saved pickle could be used as a lookup table for the *core visual model*.


[Word2Vec pretrained vector]: https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec
[fastText Pre-trained]: https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
[PyTorch]:http://pytorch.org/
[CIFAR10]:https://www.cs.toronto.edu/~kriz/cifar.html
[PyTorch provided CIFAR10]:http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#loading-and-normalizing-cifar10
[3rd party version of ResNet-18 on CIFAR10]:https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
[Pre-train vector dataset]: https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
