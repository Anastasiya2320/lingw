#with open("C:/Users/ноут/Desktop/lingv/mymy1.txt", encoding="utf-8") as file:

with open("C:/Users/ноут/Desktop/lingv/mymy1.txt", "r", encoding="utf-8") as file:
    c = file.read()
    tokens = c.split(" ")
    print(tokens)



import re
with open("C:/Users/ноут/Desktop/lingv/mymy1.txt", "r", encoding="utf-8") as file:
    text = file.read()
    tokens = re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",text)
    print(tokens)



import nltk
nltk.download('punkt')
with open("C:/Users/ноут/Desktop/lingv/mymy1.txt", "r", encoding="utf-8") as file:
    a = file.read()
    tokens = nltk.word_tokenize(a)
    print(tokens)



file = open("C:/Users/ноут/Desktop/lingv/mymy1.txt", encoding="utf-8")
words = 0
for line in file:
    words += len(line.split())
print("Words:", words)



from collections import Counter
with open("C:/Users/ноут/Desktop/lingv/mymy1.txt", encoding="utf-8") as file:
    text = file.read()
words = text.lower().split()
words = [word.strip('.,!?()[]{}"\'') for word in words]
word_counter = Counter(words)
total_words = sum(word_counter.values())
print("Общее количество слов в тексте:", total_words)



with open("C:/Users/ноут/Desktop/lingv/mymy1.txt", encoding="utf-8") as file:
    text = file.read()
n = ''.join([char for char in text if char.isalpha() or char.isspace()])
print(n)



import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
with open("C:/Users/ноут/Desktop/lingv/mymy1.txt", encoding="utf-8") as file:
    text = file.read()
stop_words = set(stopwords.words('russian'))
word_tokens = word_tokenize(text)
filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
clean_text = ' '.join(filtered_text)
print(clean_text)



import nltk
from nltk.stem.snowball import RussianStemmer
nltk.download('punkt')
stemmer = RussianStemmer()
with open("C:/Users/ноут/Desktop/lingv/mymy1.txt", encoding="utf-8") as file:
    text = file.read()
words = nltk.word_tokenize(text)
stemmed_words = [stemmer.stem(word) for word in words]
stemmed_text = ' '.join(stemmed_words)
print(stemmed_text)



import pymorphy2
morph = pymorphy2.MorphAnalyzer()
with open("C:/Users/ноут/Desktop/lingv/mymy1.txt", encoding="utf-8") as file:
    text = file.read()
words = text.split()
lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
lemmatized_text = ' '.join(lemmatized_words)
print(lemmatized_text)



import string
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
with open("C:/Users/ноут/Desktop/lingv/mymy1.txt", encoding="utf-8") as file:
    text = file.read()
def preprocess_text(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    return words
processed_text = preprocess_text(text)
word_counts = Counter(processed_text)
def plot_word_frequency(word_freq, n):
    most_common_words = word_freq.most_common(n)
    words, counts = zip(*most_common_words)
    plt.figure(figsize=(12, 6))
    plt.bar(words, counts, color='skyblue')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(f'Top {n} Most Common Words')
    plt.xticks(rotation=45)
    plt.show()
plot_word_frequency(word_counts, 5)



import nltk
import matplotlib.pyplot as plt
with open("C:/Users/ноут/Desktop/lingv/mymy1.txt", encoding="utf-8") as file:
    text = file.read()
tokens = nltk.word_tokenize(text)
text = nltk.Text(tokens)
words_to_plot = ['Герасим', 'очень', 'барыни']
indexes = [index for index, word in enumerate(text) if word in words_to_plot]
for i, word in enumerate(words_to_plot):
    word_indexes = [idx for idx in indexes if text[idx] == word]
    plt.scatter(word_indexes, [i] * len(word_indexes), label=word)
plt.yticks(range(len(words_to_plot)), words_to_plot)
plt.xlabel('Распределение слов в тексте')
plt.title('Дисперсионный график')
plt.show()



import pymorphy2
morph = pymorphy2.MorphAnalyzer()
with open('C:/Users/ноут/Desktop/lingv/mymy1.txt', 'r') as file:
    text = file.read()
words = text.split()
for word in words:
    parsed_word = morph.parse(word)[0]  
    print(f'Слово: {word}, Лемма: {parsed_word.normal_form}, Часть речи: {parsed_word.tag.POS}, Падеж: {parsed_word.tag.case}')



import pymorphy2
import matplotlib.pyplot as plt
morph = pymorphy2.MorphAnalyzer()
with open("C:/Users/ноут/Desktop/lingv/mymy1.txt", encoding="utf-8") as file:
     text = file.read()
words = text.split()
pos_list = []
for word in words:
    parsed_word = morph.parse(word)[0]  
    pos = parsed_word.tag.POS
    if pos is not None:
        pos_list.append(pos)
pos_freq = {pos: pos_list.count(pos) for pos in set(pos_list)}
plt.figure(figsize=(10, 6))
plt.bar(pos_freq.keys(), pos_freq.values(), color='skyblue')
plt.xlabel('Часть речи')
plt.ylabel('Частотность')
plt.title('Распределение частотности частей речи в тексте')
plt.show()



import nltk
with open("C:/Users/ноут/Desktop/lingv/mymy1.txt", encoding="utf-8") as file:
    text = file.read()
tokens = nltk.word_tokenize(text)
text = nltk.Text(tokens)
word_to_check = 'он'
print('Слова с похожим контекстом на слово "{}":'.format(word_to_check))
text.similar(word_to_check)
print('\nОбщие контексты для слов "Герасим" и "Немой":')
concordance_list = text.concordance_list(word_to_check)
for entry in concordance_list:
    print(entry.line)
print('\nЧасто встречающиеся словосочетания (коллокации):')
text.collocations()



import pymorphy2
morph = pymorphy2.MorphAnalyzer()
lemma_cache = {}
def lemmatize_word(word):
    if word in lemma_cache:
        return lemma_cache[word]
    else:
        parsed_word = morph.parse(word)[0]
        lemma = parsed_word.normal_form
        lemma_cache[word] = lemma
        return lemma
with open('C:/Users/ноут/Desktop/lingv/mymy1.txt', 'r', encoding='utf-8') as file:
    text = file.read()
words = text.split()
for word in words:
    lemma = lemmatize_word(word)
    print(f'Слово: {word}, Лемма: {lemma}')



import spacy
nlp = spacy.load("en_core_web_sm")
with open("C:/Users/ноут/Desktop/lingv/mymy1.txt", encoding="utf-8") as file:
    text = file.read()
doc = nlp(text)
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)



import matplotlib.pyplot as plt
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
with open("C:/Users/ноут/Desktop/lingv/mymy1.txt", encoding="utf-8") as file:
    text = file.read()
tokens = word_tokenize(text.lower())
tokens = [word for word in tokens if word.isalpha()]
stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens if word not in stop_words and len(word) >= 3]
freq_dist = FreqDist(tokens)
plt.figure(figsize=(12, 6))
freq_dist.plot(20, cumulative=False)
plt.title('Top 20 частотных слов (длина >= 3)')
plt.xlabel('Слово')
plt.ylabel('Частота')
plt.show()     



import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import math
nltk.download('stopwords')
with open(r'C:/Users/ноут/Desktop/lingv/mymy1.txt', encoding='utf-8') as doc_1, open(r'C:/Users/ноут/Desktop/lingv/lokon.txt', encoding='utf-8') as doc_2:
    line1 = doc_1.read()
    line2 = doc_2.read()
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
tokens_1 = tokenizer.tokenize(line1.lower())
tokens_2 = tokenizer.tokenize(line2.lower())
text_with_no_stopwords_1 = [word for word in tokens_1 if word not in stop_words]
text_with_no_stopwords_2 = [word for word in tokens_2 if word not in stop_words]
bag_of_words_1 = set(text_with_no_stopwords_1)
bag_of_words_2 = set(text_with_no_stopwords_2)
dict_1 = dict.fromkeys(bag_of_words_1, 0)
dict_2 = dict.fromkeys(bag_of_words_2, 0)
for word in text_with_no_stopwords_1:
    dict_1[word] += 1
for word in text_with_no_stopwords_2:
    dict_2[word] += 1
def compute_term_frequency(word_dictionary, bag_of_words):
    term_frequency_dictionary = {}
    length_of_bag_of_words = len(bag_of_words)
    for word, count in word_dictionary.items():
        term_frequency_dictionary[word] = count / float(length_of_bag_of_words)
    return term_frequency_dictionary
def compute_inverse_document_frequency(full_doc_list):
    idf_dict = {}
    length_of_doc_list = len(full_doc_list)
    for doc in full_doc_list:
        for word, value in doc.items():
            if value > 0:
                idf_dict[word] = idf_dict.get(word, 0) + 1
    for word, value in idf_dict.items():
        idf_dict[word] = math.log(length_of_doc_list / float(value))
    return idf_dict
final_idf_dict = compute_inverse_document_frequency([dict_1, dict_2])
print("TF для документа 1:", compute_term_frequency(dict_1, text_with_no_stopwords_1))
print("TF для документа 2:", compute_term_frequency(dict_2, text_with_no_stopwords_2))
print("IDF для обоих документов:", final_idf_dict)
for word in bag_of_words_2:
    dict_2[word] = 1
print(dict_1, dict_2)
def compute_term_frequency(word_dictionary, bag_of_words):
    term_frequency_dictionary = {}
    length_of_bag_of_words = len(bag_of_words)
    for word, count in word_dictionary.items():
        term_frequency_dictionary[word] = count / float(length_of_bag_of_words)
    return term_frequency_dictionary
print(compute_term_frequency(dict_1, bag_of_words_1), (dict_2, bag_of_words_2))
import math
def compute_inverse_document_frequency(full_doc_list):
    idf_dict = {}
    length_of_doc_list = len(full_doc_list)
    idf_dict = dict.fromkeys(full_doc_list[0].keys(), 0)
    for word, value in idf_dict.items():
        idf_dict[word] = math.log(length_of_doc_list / (float(value)+1))
    return idf_dict
final_idf_dict = compute_inverse_document_frequency([dict_1])
print(final_idf_dict)



from pymystem3 import Mystem
mystem = Mystem()
with open("C:/Users/ноут/Desktop/lingv/mymy1.txt", encoding="utf-8") as file:
    text = file.read()
lemmas = mystem.lemmatize(text)
analysis = mystem.analyze(text)
print(lemmas)
print(analysis)



from pymystem3 import Mystem
import matplotlib.pyplot as plt
mystem = Mystem()
with open("C:/Users/ноут/Desktop/lingv/mymy1.txt", encoding="utf-8") as file:
    text = file.read()
lemmas = mystem.lemmatize(text)
target_lemma = "сердце"
lemma_counts = [lemmas[:i+1].count(target_lemma) for i in range(len(lemmas))]
plt.plot(range(len(lemmas)), lemma_counts)
plt.xlabel('Порядковый номер слова в тексте')
plt.ylabel('Частотность леммы "{}"'.format(target_lemma))
plt.title('Изменение частотности леммы "{}" в тексте'.format(target_lemma))
plt.show()



import pymorphy2
def process_text(text):
    morph = pymorphy2.MorphAnalyzer()
    words = text.split()
    processed_words = []
    for word in words:
        parsed_word = morph.parse(word)[0]
        if parsed_word.tag.POS in {'NOUN', 'ADJF'}:
            inflected_word = parsed_word.inflect({'ablt'})
        elif parsed_word.tag.POS == 'VERB':
            inflected_word = parsed_word.inflect({'past', 'femn'})
        else:
            inflected_word = None
        if inflected_word:
            processed_word = inflected_word.word
        else:
            processed_word = word 
        processed_words.append(processed_word)
    processed_text = ' '.join(processed_words)
    return processed_text
with open("C:/Users/ноут/Desktop/lingv/mymy1.txt", encoding="utf-8") as file:
    text = file.read()
processed_text = process_text(text)
print(processed_text)
modified_text = processed_text.replace("лекарство", "гроза")
print(modified_text)



import spacy
nlp = spacy.load("ru_core_news_sm")
def extract_named_entities(text):
    doc = nlp(text)
    named_entities = []
    for entity in doc.ents:
        named_entities.append((entity.text, entity.label_))
    return named_entities
with open("C:/Users/ноут/Desktop/lingv/mymy1.txt", encoding="utf-8") as file:
    text = file.read()
named_entities = extract_named_entities(text)
print(named_entities)



import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import math
nltk.download('stopwords')
with open(r'C:/Users/ноут/Desktop/lingv/mymy1.txt', encoding='utf-8') as doc_1, open(r'C:/Users/ноут/Desktop/lingv/lokon.txt', encoding='utf-8') as doc_2:
    line1 = doc_1.read()
    line2 = doc_2.read()
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
tokens_1 = tokenizer.tokenize(line1.lower())
tokens_2 = tokenizer.tokenize(line2.lower())
text_with_no_stopwords_1 = [word for word in tokens_1 if word not in stop_words]
text_with_no_stopwords_2 = [word for word in tokens_2 if word not in stop_words]
bag_of_words_1 = set(text_with_no_stopwords_1)
bag_of_words_2 = set(text_with_no_stopwords_2)
dict_1 = dict.fromkeys(bag_of_words_1, 0)
dict_2 = dict.fromkeys(bag_of_words_2, 0)
for word in text_with_no_stopwords_1:
    dict_1[word] += 1
for word in text_with_no_stopwords_2:
    dict_2[word] += 1
def compute_term_frequency(word_dictionary, bag_of_words):
    term_frequency_dictionary = {}
    length_of_bag_of_words = len(bag_of_words)
    for word, count in word_dictionary.items():
        term_frequency_dictionary[word] = count / float(length_of_bag_of_words)
    return term_frequency_dictionary
def compute_inverse_document_frequency(full_doc_list):
    idf_dict = {}
    length_of_doc_list = len(full_doc_list)
    for doc in full_doc_list:
        for word, value in doc.items():
            if value > 0:
                idf_dict[word] = idf_dict.get(word, 0) + 1
    for word, value in idf_dict.items():
        idf_dict[word] = math.log(length_of_doc_list / float(value))
    return idf_dict
final_idf_dict = compute_inverse_document_frequency([dict_1, dict_2])
print("TF для документа 1:", compute_term_frequency(dict_1, text_with_no_stopwords_1))
print("TF для документа 2:", compute_term_frequency(dict_2, text_with_no_stopwords_2))
print("IDF для обоих документов:", final_idf_dict)
for word in bag_of_words_2:
    dict_2[word] = 1
print(dict_1, dict_2)
def compute_term_frequency(word_dictionary, bag_of_words):
    term_frequency_dictionary = {}
    length_of_bag_of_words = len(bag_of_words)
    for word, count in word_dictionary.items():
        term_frequency_dictionary[word] = count / float(length_of_bag_of_words)
    return term_frequency_dictionary
print(compute_term_frequency(dict_1, bag_of_words_1), (dict_2, bag_of_words_2))
import math
def compute_inverse_document_frequency(full_doc_list):
    idf_dict = {}
    length_of_doc_list = len(full_doc_list)
    idf_dict = dict.fromkeys(full_doc_list[0].keys(), 0)
    for word, value in idf_dict.items():
        idf_dict[word] = math.log(length_of_doc_list / (float(value)+1))
    return idf_dict
final_idf_dict = compute_inverse_document_frequency([dict_1])
print(final_idf_dict)



import stanza
stanza.download(lang="ru", package=None, processors={"tokenize":""})
if __name__ == "__main__":
    with open("C:/Users/ноут/Desktop/lingv/mymy1.txt", encoding="utf-8") as file:
        text = file.read()
    ppln = stanza.Pipeline(
        lang='ru',
        processors='tokenize',
        tokenize_model_path=text,
        use_gpu=True
    )
    SOME_TEXT = "Your Russian text goes here"
    doc = ppln(SOME_TEXT)
    print(doc)