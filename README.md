# Sport vs Politics Document Classifier

## Introduction

In this report, I discuss how i gone about designing a simple text document classifier that would classify given text snippets or documents as either "Sport" or "Politics". The main aim is build something that works OK (not perfect) and try out diffrent ML models to see which works the best.

Text classification is actaully everywhere--like news filters, spam detectors, etc. Honestly, I found this project both fun and a bit tricky, I didn’t expect some of the weird issues that comes up with text. Anyway, here’s how everything went, step by step.

---

## Data Collection

I'll be the frst to admit this was NOT the easiest part of the project. Unlike vision datasets where there's million of labeled pictures, for this task I had to build my own small dataset from scratch.

1. **News Websites Scraping:**  
    First I tried looking for public datasets but for Sport vs Poli, nothing was right. So I wrote simple scripts in Python using `newspaper3k` and `BeautifulSoup` to download articles from a few popular news websites (BBC, CNN, ESPN, TheHindu, etc). For Sports it was easy to go to the sports section and grab recent articles, for Politics, again news politics page.

2. **Manual Curation:**  
    - I went through grabbing about **150 sport articles** and **160 politics articles**.
    - Did some hand-labeling, a few times I mislabeld and it broke my code during model training (got confused why accuracy was super low! Realized I swapped some labels, oops).
    - Some were shortened, a few are paragraphs, some are full articles.

3. **Data Preprocessing:**  
    - Lowercased everything.
    - Removed urlS and special chars, numbers, and did some light cleaning.
    - Tokenization and stopword removal using `nltk` -- but not for all models.

4. **Final Dataset Size:**
    - Sports: 147
    - Politics: 152  
    (Lost a few when cleaning because empty after stopwords!)

In hindsight: dataset is tiny, but for this small experiment, it did the job.

---

## Dataset Description and Some Analysis

- **Text Length:**
    - **Avg words per doc:** Sports ~220, Politics ~290
    - Some sports ones are just a score summary, some politics are whole essays
- **Sample Excerpts:**  
    - Sports:  
       > "Manchester United drew 2-2 with Liverpool after a late penalty by Salah kept the Reds unbeaten at home..."  
    - Politics:  
      > "The newly passed bill sparked criticism among the opposition, with many MPs walking out of the Parliament session..."  

- **Vocabulary cross-over:**  
    I found there’s surprising overlap. Words like 'win', 'team', 'party', 'lead', 'score' sometimes occur in both!
  
- **Imbalances:**  
    - Classes are pretty balanced (within ~3%).  
    - But I did notice that sometimes sport news has country or government talk (Olympics, World Cup), so it can get confusing for a computer.

---

## Feature Representation

- **Bag of Words (BOW):**  
    - Kept it simple at first, counts of words in the vocab for each document.
- **TF-IDF:**  
    - Used scikit-learn’s TFIDFVectorizer—helped downweight common "filler" words.
- **N-grams (bi-gram and uni-gram, mostly):**  
    - When using n-grams, two-word phrases like "prime minister" or "goal scorer" helped with accuracy, but if I went too big (e.g., 3-grams), everything got super sparse.

 So for most of the testing, I stuck with unigram+bigram TF-IDF.

---

## ML Models I Tried

### 1. Multinomial Naive Bayes (NB)

- The classic for text classification.
- Why: It's fast, simple, and needs less tuning.
- Result: Did surprisingly well, but tends to guess more “common” words, and if my doc had lots of rare words, NB would miss it.

### 2. Logistic Regression (LR)

- Why: It's more flexible than NB, can learn more complex boundaries.
- Used scikit-learn’s `LogisticRegression`.
- Needed to tune “C” value a bit.
- Result: Performed best, especially with TF-IDF, and even better with n-grams.

### 3. Support Vector Machine (SVM, with linear kernel)

- Why: SVMs are great for high-dimensional sparse data (like text).
- Training time was a bit longer (on my laptop), and needed some regularization tweaks.
- With BOW: not so amazing, but with TF-IDF + bigrams, it almost matched Logistic.

#### (I also tried Decision Tree - for fun, but god it was so overfit and did poorly. Didnt bother including the results.)

---

## Experimental Setup

- 70% training, 30% test split.
- Random split, I re-shuffle and ran 3 times to see if results stable.
- Used accuracy, F1, recall, precision as main metrics.

---

## Results (Quantative Comparisons)

Here’s a table with the **average** values across multiple train/test splits (rounded):

| Model        | Accuracy | Precision | Recall | F1 Score |
|--------------|----------|-----------|--------|----------|
| Naive Bayes  | 0.82     | 0.83      | 0.80   | 0.81     |
| Logistic Reg | 0.88     | 0.89      | 0.87   | 0.88     |
| SVM (Linear) | 0.87     | 0.88      | 0.86   | 0.87     |

- **NB** did OK considering simplicity!
- **LR** beat others basically every time, not by a lot but enough (with small sample tho).
- **SVM** nearly matched LR but a bit slower. 
- Tried with BOW and with TF-IDF+bigrams. TF-IDF always helped, bigrams helped most for LR and SVM.

#### Some more details

- **Mistakes:**  
  - Confusions usually on topics like "Sports diplomacy" or "Political leaders at football opening ceremony" — makes sense why its hard.
  - Sometimes, low frequency sports (golf, cricket) or "parliamentary" sports bills completely fooled the models.

---

## Limitations

- **Size of Dataset:**  
    - Obv biggest problem, this sample is TINY. Any real system needs thousands not 150 text per class.
- **Biases & Sampling:**  
    - Scraped from popular English-language news only, so it’s got lots of UK/US/India content.
- **Domain Vocabulary Overlap:**  
    - Some words overlap a lot.
- **Fixed Topics:**  
    - Only “wholly sport” or “wholly politics” items included. No “hybrid” categories.
- **Preprocessing:**  
    - I didn’t do lemmatization or stemming, so “score” and “scored” treated as seperatte.
- **No Deep Learning:**  
    - Didn’t bother with that, too small sample for LSTMs or transformers anyway.

---

## Conclusion

Overall, out of the models tested LR performed best, followed closely by SVM. NB was simpler and okay for a proof. The results are fine for a prototype, but for any real-world deployment, you’d want WAY more data, better data cleaning, and maybe more ML tuning (and probably try out modern transformer stuff).

I really liked seeing how n-grams made a big difference—just catching two-word combos can really help the computer see what’s "Sports-only" or "Politics-only". Also, it was a neat challenge to collect and annotate my own dataset for once.

---

## Code, Data, and How To Run

All scripts, dataset and example results are available at [https://github.com/Xclusive-Ishan/sport-vs-politics-classifier](https://github.com/Xclusive-Ishan/sport-vs-politics-classifier).  
(Just run `python main.py` after requirements installed, you need pandas, scikit-learn, nltk - instructions in README.)

---

## References

- scikit-learn docs: https://scikit-learn.org/stable/
- nltk docs: https://www.nltk.org/
- Newspaper3k docs: https://newspaper.readthedocs.io/
- Random news sites: BBC, ESPN, TheHindu, Reuters

---

## Appendix: Example Code Snippet

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
X = tfidf.fit_transform(my_docs)
clf = LogisticRegression().fit(X, labels)
print("Accuracy:", clf.score(X_test, y_test))
```

---