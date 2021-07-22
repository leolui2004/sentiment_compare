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

## Analysis Process

### Introduction
I am not going to write how to get the GCP NLP API and Transformers. For the GCP NLP API, you can just go to the GCP documentation and search for it, it should takes less than 10 minutes to create a service account and get the API json key. For Transformers, as I am using Tensorflow 2, I do not know how well the lastest version of Transformers support for Tensorflow 2, for the sake of compatibility, I just use a old version (transformers==2.10.0) for it.

For the analysis target, I am going to use few song lyrics from famous Japanese group Yoasobi. Lyrics are always a good and a difficult target for NLP analysis because
1. Lyrics may change with the trend
2. Lyrics may use some words that are unusual
3. Lyrics are usually emotional
4. Lyrics can have different meanings when read by different people

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

### Sample Lyrics

Sentence list for Part 2 and Part 3
```
lyrics_short = ['素敵な日になっていく', # もう少しだけ
               '喜びはめぐる',
               '慌ただしく過ぎる朝',
               '気持ちが沈んでいく朝',
               'もう少しだけ',
               '喜びが広がる',
               '小さな幸せを見つけられますように',
               '涙流すことすら無いまま', # たぶん
               '目を閉じたまま考えてた',
               '悪いのは誰だ 分かんないよ',
               '誰のせいでもない',
               '一人で迎えた朝',
               '仕方がないよきっと',
               '優しさの日々を辛い日々と感じてしまった',
               '少し冷えた朝だ']

lyrics_long = ['どこか虚しいような そんな気持ち つまらないな でもそれでいい そんなもんさ これでいい', # 群青
               '知らず知らず隠してた 本当の声を響かせてよ、ほら',
               '好きなものを好きだと言う 怖くて仕方ないけど 本当の自分 出会えた気がしたんだ',
               '思うようにいかない、今日も また慌ただしくもがいてる',
               '悔しい気持ちも ただ情けなくて 涙が出る',
               '踏み込むほど 苦しくなる 痛くもなる',
               '好きなことを続けること それは「楽しい」だけじゃない 本当にできる？ 不安になるけど',
               '何枚でも ほら何枚でも 自信がないから描いてきたんだよ',
               '周りを見たって 誰と比べたって 僕にしかできないことはなんだ',
               '大丈夫、行こう、あとは楽しむだけだ',
               'さよならだけだった', # 夜に駆ける
               '初めて会った日から 僕の心の全てを奪った',
               'どこか儚い空気を纏う君は 寂しい目をしてたんだ',
               '涙が零れそうでも ありきたりな喜びきっと二人なら見つけられる',
               '見惚れているかのような恋するような そんな顔が嫌いだ',
               '君の為に用意した言葉どれも届かない',
               '終わりにしたい だなんてさ 釣られて言葉にした時 君は初めて笑った',
               '騒がしい日々に笑えなくなっていた 僕の目に映る君は綺麗だ',
               '明けない夜に溢れた涙も 君の笑顔に溶けていく',
               '繋いだ手を離さないでよ 二人今、夜に駆け出していく',
               'ただその真っ黒な目から 涙溢れ落ちないように', # 怪物
               'この間違いだらけの世界の中 君には笑ってほしいから',
               'もう誰も傷付けない 強く強くなりたいんだよ 僕が僕でいられるように',
               'ありのまま生きることが正義か 騙し騙し生きるのは正義か 僕の在るべき姿とはなんだ 本当の僕は何者なんだ 教えてくれよ',
               '不器用だけれど いつまでも君とただ 笑っていたいから']
```

## Result

### Result image for the test
![image](https://github.com/leolui2004/sentiment_compare/blob/main/pic/yoasobi_1.png)

This is the result after running the test, but it is difficult to see the result so I will put the result in a table and add some visulization

### Part 1 Result (Word)
![image](https://github.com/leolui2004/sentiment_compare/blob/main/pic/yoasobi_11.png)

The result is surprising, with Transformers using those 2 pretrained models failed on some word analysis. Words like「痛み」(pain) which should be a negative word, 「大人」(adult), 「日本」(japan) which should be an neutral word, but it failed to score it appropriately.

While the GCP NLP API is also not that perfect (e.g. +0.4 for「崩れる」(collapse)), on average it did provide a good result on Part 1.

### Part 2 Result (Short Sentence)
![image](https://github.com/leolui2004/sentiment_compare/blob/main/pic/yoasobi_12.png)

The average absolute score difference is further larger when comes to short sentence. However this time both GCP and Transformers have some good and bad results respectively. Here I pick up a few for further explain.

素敵な日になっていく (It will be a wonderful day)
GCP correctly scored a +0.90, while Transformers scored a -0.66, which is completely wrong

気持ちが沈んでいく朝 (The morning when I feel depressed)
Maybe GCP failed to recognize the word 「気持ちが沈む」(feeling down), in this case Transformers did a well job

悪いのは誰だ 分かんないよ (I don't know who is bad)
This is interesting, both GCP and Transformers scored in totally opposite direction

優しさの日々を辛い日々と感じてしまった (I felt the days of kindness as painful days)
This one is also tricky, both positive word (「優しさ」(kindness)) and negative word (「辛い」(painful)) are included in the sentence, and GCP failed to recognize the sentence structure

### Part 3 Result (Long Sentence)
![image](https://github.com/leolui2004/sentiment_compare/blob/main/pic/yoasobi_13.png)

Part 3 is the most difficult part and even different people may have different interpretation. I am not going to explain one by one, but the trend is that GCP tends to be more conservative, with around half of the them with scores in between +0.2 to -0.2, while Transformers always score in high magnitude.

## Performance (Speed) Test
Finally I also did a speed test for Transformers, with repeating the analysis for 1,000 times, and it shows that it is as fast as using API, with just completed the task in less than 30 seconds.

![image](https://github.com/leolui2004/sentiment_compare/blob/main/pic/yoasobi_5.png)
