import time
from google.cloud import language_v1
import os

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

word = ['優れる',
       '嬉しい',
       '有名',
       'ナイス',
       '夜景',
       '冷静',
       '大人',
       '法律',
       '慎重',
       '日本',
       '革命',
       'チーター',
       '泣かせ',
       '崩れる',
       '落とす',
       '痛み',
       '厳しい']

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


print('テキスト　スコア')
for i in word:
    result = api_call(i)
    print(result)

print('テキスト　スコア')
for j in lyrics_short:
    result = api_call(j)
    print(result)

print('テキスト　スコア')
for k in lyrics_long:
    result = api_call(k)
    print(result)