# Compare sentiment analysis performance using GCP API and Transformers with lyrics from songs by Yoasobi
My nineth project on github. Comparing Japanese sentiment analysis result using 1. NLP API provided by Google Cloud Platform and 2. BERT model provided by Transformers, with song lyrics from famous Japanese group Yoasobi.

Feel free to provide comments, I just started learning Python last year and I am now concentrating on data anylysis, web presentation and deep learning application.

## Simple Introduction for Two Methods

### Google Cloud Platform NLP API
GCP provides NLP API in different categories, one I usually use is sentiment analysis, which can determine the feeling of a text in a form of score and magnitude.

### Transformers with BERT
BERT is a great NLP pretrained model provided by Google, and Transformers is also a great library for fast development on NLP task. This time I have used a pretrained model named daigo/bert-base-japanese-sentiment and a pretrained tokenizer named bert-base-japanese-whole-word-masking.

### Difference
Apart from the analysis result, the main difference between two methods are
1. **GCP NLP API is not free**, they provide free quota (around 5,000 sentences) per month but it should not be enough for real life use case. While **using public pretrained models and Transformers are totally free**, though the company Hugging Face who provides the library do have some plans for using like private models and datasets.
2. **Using NLP API by GCP rarely consume computing resources**, basically it just calling a API and get back the result. But **using pretrained models and Transformers do consume some computing resources**, I will also show the time consumed for using Transformers below, but certainly the resources needed for training a model or fine-tuning a model would be completely different from directly using the pretrained model to do analysis.

## Process

### Introduction
I am not going to write how to get the GCP NLP API and Transformers. For the GCP NLP API, you can just go to the GCP documentation and search for it, it should takes less than 10 minutes to create a service account and get the API json key. For Transformers, as I am using Tensorflow 2, I do not know how well the lastest version of Transformers support for Tensorflow 2, for the sake of compatibility, I just use a old version (transformers==2.10.0) for it.

For the analysis target, I am going to use few song lyrics from famous Japanese group Yoasobi. Lyrics are always a good and a difficult target for NLP analysis because
1. Lyrics may change with the trend
2. Lyrics may use some words that are unusual
3. Lyrics are usually emotional

And for the test, it divides in 3 parts
1. Part 1: Words, which can be treated as a sanity check
2. Part 2: Short sentence
3. Part 3: Complete sentence or a paragraph of the lyric

### Sample code for GCP NLP API

Import necessary library
```
import time
from google.cloud import language_v1
import os
```

Define the API, to avoid touching the API limit, I always add 0.1 second buffer
```
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "nlp.json"
client = language_v1.LanguageServiceClient()

def api_call(text):
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT, language='ja')

    response_sentiment = client.analyze_sentiment(request={'document': document, 'encoding_type': language_v1.EncodingType.UTF8})

    result = []
    result.append(text)
    result.append(response_sentiment.document_sentiment.score)
    
    time.sleep(0.1)
    return result
```

Here is the word list (Part 1)
```
word = ['優れる','嬉しい','有名','ナイス','夜景','冷静','大人','法律','慎重','日本','革命','チーター','泣かせ','崩れる','落とす','痛み','厳しい']
```

Run the test
```
print('テキスト　スコア')
for i in word:
    result = api_call(i)
    print(result)
```

### Sample code for Transformers

Import necessary library
```
from transformers import pipeline, AutoModelForSequenceClassification, BertJapaneseTokenizer
```

Build the pipeline using 2 pretrained models
```
model = AutoModelForSequenceClassification.from_pretrained('daigo/bert-base-japanese-sentiment') 
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
nlp = pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)
```

Here is the word list (Part 1)
```
word = ['優れる','嬉しい','有名','ナイス','夜景','冷静','大人','法律','慎重','日本','革命','チーター','泣かせ','崩れる','落とす','痛み','厳しい']
```

Run the test
```
print('テキスト　スコア')
for i in word:
    print(nlp(i))
```

## Result
