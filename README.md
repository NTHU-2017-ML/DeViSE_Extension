<a name="MileStone2"/>
# MileStone2
[MileStone2] (#MileStone2)
    [Image Pre-train Mdel] (##Image Pre-train Mdel)
    [Word2Vec Pre-train vector] (##Word2Vec Pre-train vector)
    
<a name="Image Pre-train Mdel"/>
## Image Pre-train Mdel

<a name="Word2Vec Pre-train vector"/>
## Word2Vec Pre-train vector
1. Environment Setting:
    - Gensim: We use the gensim to load the pretrain vector.
    ```
    pip install gensim
    ```

2. Data Set: [Word2Vec pretrain vector]
    - github: [fastText Pre-train]

    We use the facebook's research that is called fastText.
    It provide us 300-D word's embedding space.

3. How to use?
    - Load the model by gensim.
    ```
    import gensim
    model = gensim.models.KeyedVectors.load_word2vec_format("wiki.en.vec", binary=False)
    ```
    - Use the Image dataset's label to get the target embedding space.
    ```
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for i in range(len(classes)):
        output[i] = model.wv[classes[i]]
    ```
    - Write the target embedding space to picke file
     ```
    import pickle

    output_file = open('output.pkl', 'wb')
    pickle.dump(output, output_file, protocol=2)
    output_file.close()
    ```
    - Target embedding space could provide DeViSE model lookup table


[Word2Vec pretrain vector]: https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec
[fastText Pre-train]: https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
