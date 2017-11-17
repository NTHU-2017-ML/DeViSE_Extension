# MileStone2 - DeViSE on CIFAR10

## Core visual model

## Word2Vec Pre-trained model
1. Environment Setting:
    - Gensim: We use **gensim** to load the pretrain vector.
    ```
    pip install gensim
    ```

2. Data Set: [Word2Vec pretrained vector]
    - github: [fastText Pre-trained]

    We use the model pretrained by Facebook Research which is called **fastText**.
    It embeds words to vectors with dimension of 300.

3. How to use it?
    - Load the pretrained model through **gensim**.
    ```
    import gensim
    model = gensim.models.KeyedVectors.load_word2vec_format("wiki.en.vec", binary=False)
    ```
    - Get the embedding vectors of CIFAR10's class labels 
    ```
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for i in range(len(classes)):
        output[i] = model.wv[classes[i]]
    ```
    - Write the embedding vectors of those classes to a pickle file
     ```
    import pickle

    output_file = open('output.pkl', 'wb')
    pickle.dump(output, output_file, protocol=2)
    output_file.close()
    ```
    - The saved pickle could be used as a lookup table for the *core visual model*


[Word2Vec pretrained vector]: https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec
[fastText Pre-trained]: https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
