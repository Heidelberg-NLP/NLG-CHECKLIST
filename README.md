# A Testsuite for AMR Evaluation Metrics

This testsuite aims at giving developers the opportunity to compare evaluation metrics to both human judgement and some commonly used
NLG or STS metrics, regarding specific linguistic phenomena. 

Its main objectives are:

- To help users develop new and improve old metrics by providing them with helpful information such as:
    - Statistics about the annotated scores for each phenomenon
    - Human Judgement and scores for a selection of NLG and STS metrics for individual test cases
    - Average Scores for each phenomenon
    - Correlation matrizes including correlation between all metrics and human judgement for each phenomenon and overall
- To make contributing to the testsuite easy and simple

## Installation of Requirements

Note that there are two requirement files in this repository, `requirements.txt` and `requirements_add.txt`.
If you only wish to run the testsuite or add a new metric, you just need to install `requirements.txt` by typing.

```
pip install -r requirements.txt
```

You also have to install the [`amr-utils`](https://github.com/ablodge/amr-utils) package.

If you wish to add new test cases that have not yet been evaluated by the metrics used in the testsuite, you will have to install requirements_add.txt with the following command:

```
pip install -r requirements_add.txt
```

Furthermore, you will need to clone the [`MFScore`](https://github.com/Heidelberg-NLP/MFscore) repository and download [`Meteor Version 1.5`](https://www.cs.cmu.edu/~alavie/METEOR/). Please follow the instructions and install their requirements as well. Both repositories have to be located in the parent directory of the testsuite.

Furthermore, you will need to download the 300 dimensional GloVe word embeddings in order to be able to test with S2match and MF score. They have to be located in a directory `vectors` in the parent directory as well.

## Running the Testsuite

If you just want to run the testsuite with the avaliable metrics, you can simply go into the directory `scripts` and execute the testsuite in the terminal by running the command:

```
python3 test_suite.py -m
```

You will get a short overview of the testsuite and the overall results on the console. The results of the testsuite for each phenomenon will be written into text files in the directory ```Results``` or, if you choose to output HTML files, ```Results_HTML```. You can do so by simply adding the flag ```-html``` at the end of your command.

In order to give you an impression of the testsuite's output, the directories ```Results``` and ```Results_HTML``` already contain the testsuite's results with the metrics listed below.

Note, that you can decide which metrics you want to include in your run of the testsuite by commenting the ones you don't want to consider with a '#'.


If you want to test your own metric, you will first need to compute the scores for each test case. You can find the data you need in `data/content_test_cases.json`. If you like, you can use get_your_scores.py as a template to implement your metric. It will output a JSON file with the test cases' IDs as keys and a dictionary containing the metric's score as value. You can even test multiple metrics at once by adding multiple scores to the score dictionary for each test case. Similar to the ```metrics.txt``` file, you can choose which of these metrics you want to test by writing them into the ```my_metrics.txt``` file.

Then run the following command in the `scripts` directory:

```
python3 test_suite.py path_to_your_JSON
```

## Data Sets

The data present in the testsuite was sampled from the following data sets:

- STS (Semantic Textual Similarity) Data Sets (2012-2017)
- SICK (Sentences Involving Compositional Knowldedge) Data Set

## Metrics

The follwing metrics are employed in the testsuite are the following:

- BERT Score
- BLEU 
- chrF++
- Meteor
- MF Score
- MF Score (&beta;=0)
- MF Score (&beta;=&infin;)
- S-BERT (roberta-large)
- S-BERT (roberta-base)
- S-BERT (bert-large)
- S-BERT (distilbert-base)
- S2match
- Smatch
