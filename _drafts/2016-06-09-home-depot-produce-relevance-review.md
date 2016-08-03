---
layout: post
title: "Home Depot Product Search"
author: denxx
modified:
categories: blog
excerpt: "We often associate Home Depot with shelves of tools and appliances, some of which we didn't even know existed. However, there is a data science side of Home Depot, which was recently showcased on Kaggle. This post will review the Home Depot Product Relevance Challenge and describe some of the approaches used by the participants."
tags: ["ecommerce", "relevance", "kaggle"]
image:
  feature:
icon: "homedepot.png"
date: 2016-08-01
---

The Home Depot Product Search Relevance Kaggle competition offered participants a chance to build a model, which predicts relevance of products that are returned in a response to the user query. This is a typical e-commerce search scenario, where the products returned in response to user's query are structured objects, represented by a collection of properties (e.g. brand, size, type, etc.), which brings certain specifics compared to classical document search. A common approach to e-commerce search is to use an out-of-the-box search engine (e.g. [Apache Solr](http://lucene.apache.org/solr/){:target="_blank"} or [Elastic Search](https://www.elastic.co/){:target="_blank"}) to retrieve a set of candidate products using keywords mentioned in different properties, and then apply a post-processing machine learning model to rerank the products. The goal of the Home Depot competition was exactly to build such a post-processing model to predict relevance of products to given queries.

## Data

For the challenge [Home Depot provided a dataset](https://www.kaggle.com/c/home-depot-product-search-relevance/data){:target="_blank"}, that contains pairs of user search queries and products, that might be relevant to these queries. Each such pair was examined by at least three human raters, who assigned integer scores from 1 (not relevant) to 3 (highly relevant). The scores from different raters were averaged. Organizers made available [instructions](https://www.kaggle.com/c/home-depot-product-search-relevance/download/relevance_instructions.docx){:target="_blank"} for human raters who judged the relevance of products to the queries.

While the user queries are simply strings, that contain one or more terms, products are more complex. Each product has a product id, title (e.g. "Delta Vero 1-Handle Shower Only Faucet Trim Kit in Chrome (Valve Not Included)"), a more verbose description (e.g. "Update your bathroom with the Delta Vero Single-Handle Shower Faucet Trim Kit in Chrome ...") and one or more attributes, which could be brand name, weight, color, etc. While some attributes have a clear name (e.g. "MFG Brand Name"), others are simply bullet points, that you might expect to see under the product description on a website (e.g. "Bullet01" -> "Versatile connector for various 90° connections and home repair projects").


## Evaluation

The goal of the challenge is to predict the relevance score for (product, search query) pairs from the test set. The main quality metric for the submissions was root mean squared error (RMSE): $$\textrm{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (rel_i - \hat{rel}_i)^2}$$, where $$rel_i$$ is the average human relevance score, and $$\hat{rel}_i$$ is the model prediction.


## Data Exploration

Before making any predictions it's helpful to [look on the data](https://en.wikipedia.org/wiki/Exploratory_data_analysis){:target="_blank"} to get some insights and ideas. Many participants posted their explaratory scripts and notebooks, e.g. [HomeDepot First Data Exploration](https://www.kaggle.com/briantc/home-depot-product-search-relevance/homedepot-first-dataexploreation-k){:target="_blank"}.

Let's look into the distribution of relevance labels across the training set:
{% highlight python %}
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

train_data = pd.read_csv("train.csv", encoding="ISO-8859-1")
sns.countplot(x="relevance", data=train_df, palette="Greens_d")
plt.show()
{% endhighlight %}

![Histogram of relevance scores]({{ site.url }}/images/home_depot/relevance_plot.png)

As we can see, most of the products in the dataset are relevant to the corresponding queries, and the average relevance score is 2.38 (median = 2.33).

Next, let's see what percentage of products occur in train, test or both datasets:
{% highlight python %}
# Using matplotlib_venn library: https://github.com/konstantint/matplotlib-venn
from matplotlib_venn import venn2
venn2([set(train_df["product_uid"]), set(test_df["product_uid"])],
	set_labels=('train', 'test'))
plt.show()
{% endhighlight %}

![Venn diagram of products in train and test sets]({{ site.url }}/images/home_depot/product_ids_venn.png)

And the same for queries:

{% highlight python %}
# Using matplotlib_venn library: https://github.com/konstantint/matplotlib-venn
from matplotlib_venn import venn2
venn2([set(train_df["search_term"]), set(test_df["search_term"])],
	set_labels=('train', 'test'))
plt.show()
{% endhighlight %}

![Venn diagram of queries in train and test sets]({{ site.url }}/images/home_depot/queries_venn.png)

The percentages of queries and products that appear in train only and in test only are quite important. It's natural to expect that predictions for new products and queries might be harder to make, than for queries and products which the model has seen during training. Therefore, a good training-validation splits (or cross-validation splits) would maintain the ratio of repeated and new queries and products. Otherwise, the parameters tuned on the validation set can be far from optimal, and we will get a lower leaderboard score.

{% highlight python %}
# Split products into buckets based on their product_uid
train_df["bucket"] = np.floor(train_df["product_uid"] / 1000)
sns.pointplot(x="bucket", y="relevance", data=train_df[["bucket", "relevance"]])
plt.show()
{% endhighlight %}

An interesting observation was made by some of the participants, who looked at the relevance scores for products with different product ids:

![Histogram of relevance scores for different product_uid]({{ site.url }}/images/home_depot/relevance_bucket_plot.png)

This unusual dependence between relevance and product ids was also exploited, e.g. by adding the id itself or a binary indicators as features.


## Pre-processing

A quick look at raw queries and product descriptions reveals several typical problems, that might affect the performance of a model:

* Quite often queries contain typos, which can cause problems, for example, for counting the number of query terms matched in the question. Example queries: "lawn sprkinler", "portable air condtioners", etc.
There are some typical typos, that occur again and again in the dataset, and one simple approach was to build a list of terms that frequently occur in queries, but doesn't occur in product descriptions and manually fix these names. It's also not hard at all to implement your own spelling corrector, e.g. [spelling corrector described by Peter Norvig](http://norvig.com/spell-correct.html){:target="_blank"}. Another approach was to [collect spelling corrections using Google](https://www.kaggle.com/steubk/home-depot-product-search-relevance/fixing-typos){:target="_blank"}.

* Products have many numeric characteristics with different units. Unfortunately, the ways these numbers and units are written in products and queries isn't standard. 
Most of the participants standartized these measures and units manually, for example:
{% highlight python %}
def fix_units(s, replacements = {"'|in|inches|inch": "in",
                    "''|ft|foot|feet": "ft",
                    "pound|pounds|lb|lbs": "lb",
                    "volt|volts|v": "v",
                    "watt|watts|w": "w",
                    "ounce|ounces|oz": "oz",
                    "gal|gallon": "gal",
                    "m|meter|meters": "m",
                    "cm|centimeter|centimeters": "cm",
                    "mm|milimeter|milimeters": "mm",
                    "yd|yard|yards": "yd",
                    }):

    # Add spaces after measures
    regexp_template = r"([/\.0-9]+)[-\s]*({0})([,\.\s]|$)"
    regexp_subst_template = "\g<1> {0} "

    s = re.sub(r"([^\s-])x([0-9]+)", "\g<1> x \g<2>", s).strip()

    # Standartize unit names
    for pattern, repl in replacements.iteritems():
        s = re.sub(regexp_template.format(pattern), regexp_subst_template.format(repl), s)

    s = re.sub(r"\s\s+", " ", s).strip()
    return s
{% endhighlight %}

* As we all know, words have different forms: singular and plural, verb forms and tenses, etc. All these variations can cause us troubles with word comparisons. A common strategy to deal with these situations is to [stem](https://en.wikipedia.org/wiki/Stemming){:target="_blank"} words, i.e. reduce word forms to their roots. There are multiple algorithms to do it, implemented in different languages. In Python one can use the [nltk](http://www.nltk.org/){:target="_blank"} library:
{% highlight python %}
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
text = [stemmer.stem(word.lower()) for word in word_tokenize(text)]
{% endhighlight %}


## Features

Now, as we pre-processed the data, we can switch to feature engineering. The task is to estimate the relevance of the product to the given query, or in other words if the given product is a good match for the query.

The first idea is to count how many terms of the query matches in the product description. One way to do this is to represent a product and query as bag-of-word vectors, where each dimension corresponds to a particular term in the vocabulary. Then we can compute [**cosine similarity**](https://en.wikipedia.org/wiki/Cosine_similarity){:target="_blank"} of these vectors. However, as you can imagine the word "the" is less interesting than the word "conditioner". A common way to estimate word importance is to compute its [inverse document frequency](http://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html){:target="_blank"}. IDF scores could be incorporated into product or query word vectors (see [**tf-idf**](https://en.wikipedia.org/wiki/Tf%E2%80%93idf){:target="_blank"}). Information retrieval research knows many other retrieval metrics besides tf-idf, for example: [BM-25](https://en.wikipedia.org/wiki/Okapi_BM25){:target="_blank"}, [divergence from randomness](https://en.wikipedia.org/wiki/Divergence-from-randomness_model){:target="_blank"}, [sequential dependency model](http://ciir.cs.umass.edu/pubfiles/ir-387.pdf){:target="_blank"}, etc. I'm not aware if any of the participants experimented with these models.

Unfortunately, these IR measures by default consider synonyms as different words and don't account for word relatedness. There are different strategies, that allow one to account for semantic similarity of words: [WordNet](https://wordnet.princeton.edu/){:target="_blank"} synonyms, hypernyms and hyponyms can be used to check if a word is related to another one, or we can use some kind of distributed word representation. **Distributed word representations** map terms to vectors in lower dimensional space (compared to the size of the vocabulary). In this lower dimensional space the distance between vectors, corresponding to related terms, is lower than for some unrelated terms. To build such representations one can use certain dimenstionality reduction techniques such as [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis){:target="_blank"}, or neural network based word embeddings, e.g. [Word2Vec](https://code.google.com/archive/p/word2vec/){:target="_blank"} or [GloVe](http://nlp.stanford.edu/projects/glove/){:target="_blank"}. The later already contain pre-trained word vectors, which you can simply download and start using. For Python enthusiast we can recommend to take a look at the [gensim library](https://radimrehurek.com/gensim/){:target="_blank"}, which has LSA, word2vec and other similar models implemented.

A particularly neat idea to compute similarity between longer pieces of text given word embeddings is so called [word movers distance](http://mkusner.github.io/publications/WMD.pdf){:target="_blank"}. The idea of this method is borrowed from the [earth movers distance](https://en.wikipedia.org/wiki/Earth_mover%27s_distance){:target="_blank"} and it estimates the total "effort" required to move words of one piece of text, e.g. query, to the other, e.g. product description. The ```wmdistance``` method in word2vec class from gensim implements this algorithm. Here is a short example:
{% highlight python %}
from gensim.models import Word2Vec
from nltk.corpus import stopwords

# Load word2vec model
model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
# Normalize the embedding vectors
model.init_sims(replace=True)

# Prepare sentences
stop_words = stopwords.words('english')
sentence1 = [w for w in "portable airconditioner".split() if w not in stop_words]
sentence2 = [w for w in "spt 8,000 btu portable air conditioner".split() if w not in stop_words]

print 'Word movers distance = %.3f' % model.wmdistance(sentence1, sentence2)
{% endhighlight %}

As we noted earlier, user queries contain a lot of typos. One idea was to fix the typos as the pre-processing step. Alternatively (or in addition), we can use fuzzy string matching. We can allow terms to match if their [edit distance](https://en.wikipedia.org/wiki/Edit_distance){:target="_blank"} is less than a certain thresholds, or we can count character n-gram instead of complete term matches. More specifically, instead of splitting query and product description into words, we can split into character n-grams (e.g. 3,4,5,6-grams). This bag of n-grams can be used similarly to bag of words, e.g. to compute cosine similarity or simply the number of matched n-grams.

Another useful idea is that not all parts of the product description and of the query are equally useful. It was noted, that matched types of products and brands are very useful for predicting the product relevance. A good strategy is to have different features for matching some specific attributes, e.g. product type, brand, size, color, etc. The model can then learn different weights for different types of matches. 
Unfortunately, these attributes are not always implicitly given. The product attributes file contains a mixture of various textual information. Therefore, some heuristic rules can be used to extract color, size (e.g. dictionary of colors, presence of some specific units), etc. The brand attribute was actually given:
{% highlight python %}
attributes_data = pd.read_csv(attributes_path, encoding="ISO-8859-1").dropna()
brand_data = attributes_data[attributes_data['name'] == 'MFG Brand Name'][["product_uid", "value"]].rename(
	columns={"value": "brand"})
{% endhighlight %}

Besides that one can imagine that matched adjective, such as "portable", might be less important than a matched noun, e.g. "conditioner". Therefore, we can use a tagger to annotate words in product descriptions and queries with their part-of-speech (POS) tags. Luckily, Nltk library, which we used before for stemming, has a POS tagger (here is the [list of part-of-speech tags](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html){:target="_blank"}):
{% highlight python %}
import nltk
text = nltk.word_tokenize("And now for something completely different")
nltk.pos_tag(text)
# Returns
# [('And', 'CC'), ('now', 'RB'), ('for', 'IN'), ('something', 'NN'), ('completely', 'RB'), ('different', 'JJ')]
{% endhighlight %}


Finally, let me describe some insight from the interview of the winners of this challenge. As we have seen in the exploratory analysis, many of the queries appear multiple times, so do products. So, for a random test instance there is a chance that we had some instances with the same query or product in the training data. We would like to use this somehow. The winning team clustered the products for each query and computed cluster centroids. Then, at test time they computed the similarity of the current product to the query centroid and used this as a feature. Please refer to the [winning team interview](http://blog.kaggle.com/2016/05/18/home-depot-product-search-relevance-winners-interview-1st-place-alex-andreas-nurlan/){:target="_blank"} for details.

Of course, this is just a small set of features that can be and were used to represent product-query pairs for relevance prediction. Many of the participants shared their detailed list of features on the [HomeDepot competition forum](https://www.kaggle.com/c/home-depot-product-search-relevance/forums/t/20427/congrats-to-the-winners/){:target="_blank"}.


## Algorithms


Most of the participants of the challenge posed this problem as a regression problem, i.e. predicting a numeric relevance score. Participants explored many different algorithms: from linear regression to [random forest](https://en.wikipedia.org/wiki/Random_forest){:target="_blank"} and [gradient boosted regression trees](https://en.wikipedia.org/wiki/Gradient_boosting){:target="_blank"}.
Top scoring teams didn't stop here, and as common in Kaggle competitions, built an [ensemble of different approaches](https://en.wikipedia.org/wiki/Ensemble_learning){:target="_blank"}. The winners of the competitions stacked three layers of models. On the lowest layer they build 300 different regression models using various combinations of features. On the next layer 63 models combined predictions of the lowest layer models, and finally these 63 models were combined into a single model using ridge regression. [The interview of the winning team on Kaggle blog](http://blog.kaggle.com/2016/05/18/home-depot-product-search-relevance-winners-interview-1st-place-alex-andreas-nurlan/){:target="_blank"} has pretty nice diagram of the overall approach. 

## Links

- [HomeDepot challenge 1st place team interview](http://blog.kaggle.com/2016/05/18/home-depot-product-search-relevance-winners-interview-1st-place-alex-andreas-nurlan/)
- [HomeDepot challenge 2nd place team interview](http://blog.kaggle.com/2016/06/15/home-depot-product-search-relevance-winners-interview-2nd-place-thomas-sean-qingchen-nima/)
- [HomeDepot challenge 3rd place team interview](http://blog.kaggle.com/2016/06/01/home-depot-product-search-relevance-winners-interview-3rd-place-team-turing-test-igor-kostia-chenglong/) and [Github code repository with documentation](https://github.com/ChenglongChen/Kaggle_HomeDepot)
- Our own humble attempt at the challenge can be found in [DataScienceBar github](https://github.com/datasciencebar/HomeDepotKaggle).
