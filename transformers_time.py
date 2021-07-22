from transformers import pipeline, AutoModelForSequenceClassification, BertJapaneseTokenizer
import datetime
    
model = AutoModelForSequenceClassification.from_pretrained('daigo/bert-base-japanese-sentiment') 
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
nlp = pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)

lyrics_long = ['どこか虚しいような そんな気持ち つまらないな でもそれでいい そんなもんさ これでいい', # 群青
               '知らず知らず隠してた 本当の声を響かせてよ、ほら',
               '好きなものを好きだと言う 怖くて仕方ないけど 本当の自分 出会えた気がしたんだ',
               '思うようにいかない、今日も また慌ただしくもがいてる',
               '悔しい気持ちも ただ情けなくて 涙が出る',
               '踏み込むほど 苦しくなる 痛くもなる',
               '好きなことを続けること それは「楽しい」だけじゃない 本当にできる？ 不安になるけど',
               '何枚でも ほら何枚でも 自信がないから描いてきたんだよ',
               '周りを見たって 誰と比べたって 僕にしかできないことはなんだ',
               '大丈夫、行こう、あとは楽しむだけだ',]

print(datetime.datetime.now())
for i in range(100):
    for j in lyrics_long:
        nlp(j)
print(datetime.datetime.now())
