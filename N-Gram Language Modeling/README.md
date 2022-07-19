## Assignment 2 README





To run the code : 

```python3 main.py --model <model> --smoothing <value> --set <file>```



This command will run the specified model with the smoothing value and compute the perplexity on the specified dataset.



the ``model`` parameter can be specified with:

- unigram

- bigram

- trigram

- interpolate



the ```smoothing``` parameter can be any number greater than zero and specified the value for $\alpha$



the ```set``` parameter can be specified with the names below to compute the perplexity of the given dataset

- train

- dev

- test

  






