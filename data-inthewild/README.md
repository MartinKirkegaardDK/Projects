# Data in the Wild project
This repo contains the files necessary to perform our scraping, preprocessing and producing the figures from our analysis. The entire folder structure is shown below.

The ``data`` directory contains a folder ``raw`` with the raw scraped data, a folder ``images`` with the scraped images, a folder ``interim`` with data used for processing together with unmerged partly processed data and a folder ``processed`` with our final processed data. Most of the scraped data is in ``json`` format, with top-level keys corresponding to the URLs of the scraped recipes. The images in their own folder have corresponding file-names.

The ``notebooks`` directory contains jupyter notebooks with the code used for processing and visualizing the data and training machine learning models. Some pull functions from the ``.py`` files in the ``src`` directory. Within the ``notebooks`` directory, the notebooks are categorized into ``scraping`` (self-explanatory), ``preprocessing`` (for the code used to create the final processed ``data/processed/data.json`` file), ``analysis_and_visualization`` (for describing and visualizing this final data set) and ``machine_learning`` (for the code used to train the models for predicting cuisines from ingredients and images, respectively). 

The machine learning models themselves, both for predicting cuisines and for tagging the ingredient data, are stored in the ``ml_models`` directory, if it would be too slow to retrain them every time the notebooks are run.

The ``visualization`` contains the visualizations used in our report.

## How to run

Install the requirements by running

```
pip install -r requirements.txt
```

Then all the notebooks should be able to run, but only the ones in ``analysis_and_visualization`` should be necessary to produce the statistics and visualizations used in the report.

## Folder structure
```
├───data
│   ├───images
│   ├───interim
│   │   ├───annotation
│   │   │   ├───annotations
│   │   │   └───spacy
│   │   ├───geographical_tags
│   │   └───scraping
│   ├───processed
│   └───raw
├───ml_models
│   ├───annotation
│   │   └───models
│   │       ├───model-best
│   │       │   ├───ner
│   │       │   ├───tok2vec
│   │       │   └───vocab
│   │       └───model-last
│   │           ├───ner
│   │           ├───tok2vec
│   │           └───vocab
│   └───images
├───notebooks
│   ├───analysis_and_visualization
│   ├───machine_learning
│   ├───preprocessing
│   │   ├───annotation
│   │   └───geographical_tags
│   └───scraping
├───src
└───visualizations
    ├───geographical_tags
    ├───images
    └───ingredients
```