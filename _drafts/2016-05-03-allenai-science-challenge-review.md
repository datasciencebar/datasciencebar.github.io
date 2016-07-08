---
layout: post
title: "The Allen AI Science Challenge"
author: denxx
modified:
categories: blog
excerpt: "In this post we will review the Allen AI Science Challenge, which finished a couple of months ago on Kaggle. Participants developed systems to automatically answer multiple-choice 8th grade science questions."
tags: ["question answering", "kaggle"]
image:
  feature:
icon: "allenai_science_challenge.png"
date: 2016-05-24
---

![Allen AI Science Challenge](/images/allenai_science_challenge_big.png "Allen AI Science Challenge")

Recently the term of "Artificial Intelligence" infiltrated the titles of many news stories. [Self-driving cars](https://en.wikipedia.org/wiki/Autonomous_car){:target="_blank"}, [computer programs playing Atari games](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf){:target="_blank"} better than humans can do, [AlphaGo](https://en.wikipedia.org/wiki/AlphaGo){:target="_blank"} beating the world champion in the game of Go and many other recent achievements make us believe that we are on the verge of [Technological singularity](https://en.wikipedia.org/wiki/Technological_singularity){:target="_blank"}. However, an experts in machine learning and artificial intelligence are slightly more skeptical, for example: ["Why AlphaGo is not AI"](http://spectrum.ieee.org/automaton/robotics/artificial-intelligence/why-alphago-is-not-ai){:target="_blank"}, ["AlphaGo is not the solution to AI"](http://hunch.net/?p=3692542){:target="_blank"} and ["What counts as artificially intelligent? AI and deep learning, explained"](http://www.theverge.com/2016/2/29/11133682/deep-learning-ai-explained-machine-learning){:target="_blank"}. [Deep learning](https://en.wikipedia.org/wiki/Deep_learning){:target="_blank"} made it possible to train complicated models for various perception tasks, such as speech and image recognition, but we are still to see a model that will show a deeper understanding of the world and that can reason beyond the explicitly present information. In particular, one of such tasks where machines still lag far behind is text understanding. Despite the tremendous success of [IBM Watson](http://www.ibm.com/smarterplanet/us/en/ibmwatson/what-is-watson.html){:target="_blank"} defeating best human competitors in the Jeopardy! TV show, the techniques used to generate the answers are rather shallow and are still based on the simple text matches[^1].

> Oren Etzioni, CEO of [Allen Institute of Artificial Intelligence (AI2)](http://allenai.org/){:target="_blank"} said “IBM has announced that Watson is ‘going to college’ and ‘diagnosing patients’. But before college and medical school — let’s make sure Watson can ace the 8th grade science test. We challenge Watson, and all other interested parties — take the Allen AI Science Challenge.”

In October 2015 AI2 invited[^2] the researchers around the world to participate in the [Allen AI Science Challenge](https://www.kaggle.com/c/the-allen-ai-science-challenge){:target="_blank"} hosted on [Kaggle](http://kaggle.com){:target="_blank"}.

## Data

The [dataset](https://www.kaggle.com/c/the-allen-ai-science-challenge/data){:target="_blank"} provided for this competition contains a collection of multiple choice questions from a typical US 8th grade science corriculum. Each question has four possible answers, only one of which is correct. The training data contained of 2,500 questions, validation set - 8,132 questions and test set - 21,298 questions. Unfortunately, the exact dataset was licensed for this challenge only and was removed from the Kaggle website. AI2 released[^3] a smaller datasets of similar nature on their website.

Here is an example science question:

> Which of the following tools is most useful for tightening a small mechanical fastener?<br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A.&nbsp; chisel  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;B.&nbsp; pliers  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;C.&nbsp; sander  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;D.&nbsp; saw

The strategy of memorizing answers to the questions from the training set wouldn't be very helpful for answering new unseen questions. Therefore, a question answering system has to involve an external data sources. The rules of the challenge allowed participants to use any publicly available dictionaries and text collections. As examples, organizers recommended the following list of [knowledge resources](http://aclweb.org/aclwiki/index.php?title=RTE_Knowledge_Resources#Publicly_available_Resources){:target="_blank"} and [textbooks](http://www.ck12.org/){:target="_blank"}. Most of the text corpora, such as Wikipedia, are usually quite big, and, therefore, one needs to preprocess and  [index](https://en.wikipedia.org/wiki/Search_engine_indexing){:target="_blank"} these text documents in order to use them subsequently in the question answering system. There is a large number of [open source search systems](https://en.wikipedia.org/wiki/List_of_search_engines#Open_source_search_engines){:target="_blank"} available, and [Apache Lucene](https://lucene.apache.org/){:target="_blank"} is one of the most popular.

## Task and Evaluation

The task was to return a single answer (A, B, C or D) for each of the test questions. The official [AI Challenge performance metric](https://www.kaggle.com/c/the-allen-ai-science-challenge/details/evaluation){:target="_blank"} was the accuracy $$acc=\frac{\#correct}{\#total}$$, i.e. the fraction of the correctly answered questions. Therefore, everybody could at least get a score around 0.25 by rangom guessing.

Oh, did I forget to mention money? 1st Place - **$50,000**, 2nd Place - **$20,000**, 3rd Place - **$10,000**.

## Results

The final states of the [public](https://www.kaggle.com/c/the-allen-ai-science-challenge/leaderboard/public){:target="_blank"} and [private](https://www.kaggle.com/c/the-allen-ai-science-challenge/leaderboard/private){:target="_blank"} leaderboards differ a little bit. As we can see, the winners were able to get the accuracy of 0.59308, which is not even *D* in [the US grading system](https://en.wikipedia.org/wiki/Academic_grading_in_the_United_States){:target="_blank"}.


#### Baseline
Organizers provided a baseline approach, which is based on using [Lucene](https://lucene.apache.org/){:target="_blank"} over a [Wikipedia index](https://github.com/lemire/IndexWikipedia){:target="_blank"} and achieved a public leaderboard score of 0.4325. The idea is to query the index with a concatenation of the question and each answer and compare the retrieval scores of the top matched document. There are multiple variations of the baseline, e.g. using different retrieval metrics, using top-N document scores, etc. One particularly useful idea is to index paragraphs instead of the full documents. Wikipedia documents are usually long, but in order to answer a particular question we would search for for a specific fact, that is most likely expressed within a single passage or even a sentence. Another direction for the improvement would be incorporating additional text collections, such as  [CK-12 textbooks](http://ck12.org){:target="_blank"} and other science books, flashcards, etc.

#### Approaches

Many insipirations for the participating models were drawn from the research paper ["Combining Retrieval, Statistics, and Inference to Answer Elementary Science Questions"](http://web.engr.illinois.edu/~khashab2/files/2015_aristo/2015_aristo_aaai-2016.pdf){:target="_blank"} by Peter Clark, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Oyvind Tafjord, and Peter Turney, which appeared at [AAAI, 2016](http://www.aaai.org/Conferences/AAAI/aaai16.php){:target="_blank"}. This paper focuses on 4th grade science exams as a target task, but can obviously skip a couple of years and be applied to 8th grade exams. The described system represents an ensemble of 3 different layers of knowledge representation: *text*, *corpus statistics* and *knowledge base*.

![Artitecture of Allen AI Aristo system](/images/allenai_aristo.png "Artitecture of Allen AI Aristo system")

- **Text**-based module is an implementation of [Information Retrieval (IR)](https://en.wikipedia.org/wiki/Information_retrieval){:target="_blank"} approach, which is based on the idea that the correct answer terms are likely to appear near the question terms. In the described implementation Lucene created an index of sentences from the text corpora ([CK12 textbooks](http://ck12.org){:target="_blank"} and crawled web corpus). Each answer is scored based on the top retrieved sentence that mention at least one non-stopword from question and answer. Lucene implements basic [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf){:target="_blank"} and [BM-25](https://en.wikipedia.org/wiki/Okapi_BM25){:target="_blank"} retrieval score models, which doesn't account for matched term proximity. There are more advanced models, such as [Positional Language Model](http://sifaka.cs.uiuc.edu/~ylv2/pub/sigir09-plm.pdf){:target="_blank"} or [Sequential Dependency Model](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.61.1097&rep=rep1&type=pdf){:target="_blank"}. Unfortunately, we aren't aware of any experiments with such models during the challenge.
- **Corpus statistics**-based module takes the idea of question and answer term coocurrences to the next level. Rather than scoring the answer by the top retrieved sentence, one can consider the statistics based on the whole corpus. [Pointwise Mutual Information (PMI)](http://en.wikipedia.org/wiki/Pointwise_mutual_information){:target="_blank"} is one way to measure associations. This module scores each answer using average PMI between question and answer unigrams, bigrams, trigrams and skip-bigrams. Another implemented approach, that falls into this category is based on cosine similarities between question and answer word embeddings. Many competitors used pre-trained word vectors, such as [word2vec](https://code.google.com/archive/p/word2vec/){:target="_blank"} or [Glove](http://nlp.stanford.edu/projects/glove/){:target="_blank"}. The paper describes a couple of ways question and answer can be represented (average pairwise cosine similarity of terms, or sum of term embeddings), and multiple similarity scores are combined together using a trained SVM model. This approach alone could achieve a score in a range of 0.3*, but it was useful in a combination with other techniques. Embeddings are usually built using the idea that similar words are used in a similar context. Another related approached, described on the [Kaggle forum](https://www.kaggle.com/c/the-allen-ai-science-challenge/forums/t/18983/0-39-quick-solution){:target="_blank"} used distances between question and answer terms in a corpus (Wikipedia) to score each candidate. The described heuristic led to a score of 0.39.
- **Knowledge base** module contains two approaches: probabilistic logic rules and integer linear programming. Rule-based approach works over a [knowledge base](https://en.wikipedia.org/wiki/Knowledge_base){:target="_blank"}, extracted using a hand-craften set of rules, that look for cause, purpose, requirement, condition and similar relations in science corpus. You can see an example of rule and extracted knowledge on page 3 of the paper. After such knowledge base is constructed, a [textual entailment algorithm](https://en.wikipedia.org/wiki/Textual_entailment){:target="_blank"} is used to score question-answer pair by attempting to derive them from this knowledge. The last approach of the paper operates over a set of extracted tables, each of which correspond to a single predicate (e.g. location). In this approach question-answering is treated as a global optimization over tables using integer linear programming techniques.

The paper describes ablation experiments, that study the importances of each of the components. The results suggest that PMI and IR-based components have the strongest effect on the overall system performance. However, there are certain types of questions, that are better handled by knowledge base approaches, and therefore the combination of all components significantly outperforms each of them individually.


#### Winners

The official summary of the Allen AI Science Challenge results are published by Allen AI in their ["Moving Beyond the Turing Test with the Allen AI Science Challenge"](http://arxiv.org/pdf/1604.04315v2.pdf){:target="_blank"} paper. There is no surprise that the winning approaches represent a big ensebles of different features and approaches. Most participants emphasized a big role of corpora, and all winning systems used a combination of Wikipedia, science textbooks and other online resources. Unfortunately, we still don't have a good tool for deep text-understanding and reasoning, as shallow IR-based techniques turned out to be among the most useful signals, and as pointed out by the winner, alone reach the score of 0.55. 

Organizers asked the winning teams to open source their approaches, which they did and we can dig deeper into their implementations:

1. Cardal: [https://github.com/Cardal/Kaggle_AllenAIscience](https://github.com/Cardal/Kaggle_AllenAIscience){:target="_blank"}
2. PoweredByTalkWalker: [https://github.com/bwilbertz/kaggle_allen_ai](https://github.com/bwilbertz/kaggle_allen_ai){:target="_blank"}
3. Alejandro Mosquera: [https://github.com/amsqr/Allen_AI_Kaggle](https://github.com/amsqr/Allen_AI_Kaggle){:target="_blank"}

## Conclusion

Allen AI Science Challenge demonstrated that there is still a long way to go before automatic question answering system will be able to reason about the world. Approaches, that try to work with structured representation of knowledge currently doesn't perform as good as relatively simple information retrieval techniques on average. In the [challenge overview paper](http://arxiv.org/pdf/1604.04315v2.pdf){:target="_blank"} Allen AI mentioned that they are going to launch a new, $1 million challenge, with a goal of moving from information retrieval into intelligent reasoning. Let's keep track of news from Allen AI2 by following [their Twitter!](http://twitter.com/allenai_org){:target="_blank"}.

---

[^1]: [http://ieeexplore.ieee.org/xpl/tocresult.jsp?isnumber=6177717](http://ieeexplore.ieee.org/xpl/tocresult.jsp?isnumber=6177717)
[^2]: [AI2 Competition Press Release](http://allenai.org/content/articles/AI2KaggleCompetitionPressRelease.pdf)
[^3]: [http://allenai.org/data.html](http://allenai.org/data.html)