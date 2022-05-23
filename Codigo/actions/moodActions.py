from .personalInformationActions import UserInformation
from random import randint
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, FollowupAction, ConversationPaused
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModel, TFAutoModel, pipeline
import spacy
from spacy.lang.es.examples import sentences 
import spacy_spanish_lemmatizer
nlp_spacy = spacy.load("es_core_news_sm")
import re, nltk
from collections import OrderedDict
from operator import itemgetter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import ngrams
import pandas
nltk.set_proxy('http://proxy.example.com:3128', ('USERNAME', 'PASSWORD'))
nltk.download('popular', quiet=True) # for downloading packages
nltk.download('wordnet') # first-time use only
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import SnowballStemmer
from random import seed
from random import randint
from nltk.util import ngrams
import math
from collections import Counter
from nltk import cluster
import io
import random
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os

from nltk.tokenize import word_tokenize

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
warnings.filterwarnings('ignore')
from spacy.pipeline import EntityRuler
from nltk.stem import SnowballStemmer
spanish_stemmer = SnowballStemmer('spanish')
lemmatizer = WordNetLemmatizer() 
firstMood = ""
PREGUNTASELEGIDAS = []
QUESTIONS = [
    "¿Con qué frecuencia tiene poco interés o placer en realizar cosas?",
    "En las dos últimas semanas, ¿con qué frecuencia se ha sentido decaído/a, deprimido/a o sin esperanzas?",
    "¿Qué dificultad ha tenido para conciliar el sueño o, en caso opuesto, en levantarse de la cama?",
    "En las dos últimas semanas, ¿con qué frecuencia ha experimentado cansancio o falta de energía?",
    "¿Con qué frecuencia cree que ha sentido falta o exceso de apetito?",
    "En las dos últimas semanas, ¿con qué recurrencia se ha sentido mal consigo mismo/a, "
    "que es un fracaso o qué le ha fallado a sus seres queridos?",
    "¿Con cuánta dificultad se ha enfrentado para centrarse en actividades, como leer o ver la televisión?",
    "¿Con qué abundancia cree que se ha movido o hablado tan despacio/rápido que otras personas "
    "lo puedan haber notado?",
    "En las dos últimas semanas, ¿con qué frecuencia ha tenido pensamientos que impliquen autolesión o que "
    "impliquen que estaría mejor muerto/a?",
]
QUESTIONSRELACIONES = [
    "Cuentame de los problemas que llegaste a tener esta semana",
    "2Cuentame de los problemas que llegaste a tener esta semana",
    "3Cuentame de los problemas que llegaste a tener esta semana",
    "4Cuentame de los problemas que llegaste a tener esta semana",
]
QUESTIONSANSIEDAD = [
    "¿Qué síntomas físicos tienes?",
    "2¿Qué síntomas físicos tienes?",
    "3¿Qué síntomas físicos tienes?",
    "4¿Qué síntomas físicos tienes?",
]
QUESTIONSDEPRESION = [
    "Cuentame si tuviste pensamientos que impliquen autolesión",
    "2Cuentame si tuviste pensamientos que impliquen autolesión",
    "3Cuentame si tuviste pensamientos que impliquen autolesión",
    "4Cuentame si tuviste pensamientos que impliquen autolesión",
]

f1=open('/Users/daniela.beltran/Desktop/maestria/Tesis/Codigo/Corporas/ansiedad.txt','r',errors = 'ignore')
raw_ansiedad=f1.read()
raw_ansiedad = raw_ansiedad.lower()# converts to lowercase
raw_spacy_ansiedad = raw_ansiedad

f2=open('/Users/daniela.beltran/Desktop/maestria/Tesis/Codigo/Corporas/depresion.txt','r',errors = 'ignore')
raw_depresion=f2.read()
raw_depresion = raw_depresion.lower()# converts to lowercase
raw_spacy_depresion = raw_depresion

f3=open('/Users/daniela.beltran/Desktop/maestria/Tesis/Codigo/Corporas/relaciones.txt','r',errors = 'ignore')
raw_relaciones=f3.read()
raw_relaciones = raw_relaciones.lower()# converts to lowercase
raw_spacy_relaciones = raw_relaciones

f4=open('/Users/daniela.beltran/Desktop/maestria/Tesis/Codigo/Corporas/suicidio.txt','r',errors = 'ignore')
raw_suicidio=f4.read()
raw_suicidio = raw_suicidio.lower()# converts to lowercase
raw_spacy_suicidio = raw_suicidio

path = '/Users/daniela.beltran/Desktop/maestria/Tesis/Codigo/Corporas/'
list_book = {}
for subdir, dirs, files in os.walk(path):
  for file in files:
    if file != '.DS_Store':
        f = open(path + file, "r", encoding='latin-1')
        list_book[file] = f.read().lower()   
        #print('\t', list_book[file][:100])
        f.close()
vectorizer = TfidfVectorizer()
corpus_tfidf = vectorizer.fit_transform(list_book.values())
main_doc1 = nlp_spacy(raw_spacy_ansiedad)
main_doc2 = nlp_spacy(raw_spacy_depresion)
main_doc3 = nlp_spacy(raw_spacy_relaciones)
main_doc4 = nlp_spacy(raw_spacy_suicidio)
main_doc_no_stop_words1 = nlp_spacy(' '.join([str(t) for t in main_doc1 if not t.is_stop]))
main_doc_no_stop_words2 = nlp_spacy(' '.join([str(t) for t in main_doc2 if not t.is_stop]))
main_doc_no_stop_words3 = nlp_spacy(' '.join([str(t) for t in main_doc3 if not t.is_stop]))
main_doc_no_stop_words4 = nlp_spacy(' '.join([str(t) for t in main_doc4 if not t.is_stop]))

ansiedad_stop_tokens = str(main_doc_no_stop_words1)
text_file_ansiedad = open("/Users/daniela.beltran/Desktop/maestria/Tesis/Codigo/CorporasStop/ansiedad.txt", "w")
n = text_file_ansiedad.write(ansiedad_stop_tokens)
text_file_ansiedad.close()

depresion_stop_tokens = str(main_doc_no_stop_words2)
text_file_depresion = open("/Users/daniela.beltran/Desktop/maestria/Tesis/Codigo/CorporasStop/depresion.txt", "w")
n = text_file_depresion.write(depresion_stop_tokens)

relaciones_stop_tokens = str(main_doc_no_stop_words3)
text_file_relaciones = open("/Users/daniela.beltran/Desktop/maestria/Tesis/Codigo/CorporasStop/relaciones.txt", "w")
n = text_file_relaciones.write(relaciones_stop_tokens)
text_file_relaciones.close()

suicidio_stop_tokens = str(main_doc_no_stop_words4)
text_file_suicidio = open("/Users/daniela.beltran/Desktop/maestria/Tesis/Codigo/CorporasStop/suicidio.txt", "w")
n = text_file_suicidio.write(suicidio_stop_tokens)
text_file_suicidio.close()

path = '/Users/daniela.beltran/Desktop/maestria/Tesis/Codigo/CorporasStop/'
list_book_lower_stop = {}
for subdir, dirs, files in os.walk(path):
  for file in files:
    if file != '.DS_Store':
        #print("segundo archivo")
        f = open(path + file, "r", encoding='latin-1')
        list_book_lower_stop[file] = f.read().lower()   
        #print('\t', list_book_lower_stop[file][:100])
        f.close()
corpus_tfidf_lower_stop = vectorizer.fit_transform(list_book_lower_stop.values())


main_doc1_lemma=[]
for token in nlp_spacy(main_doc_no_stop_words1):
    main_doc1_lemma.append(token.lemma_)

main_doc2_lemma=[]
for token in nlp_spacy(main_doc_no_stop_words2):
    main_doc2_lemma.append(token.lemma_)

main_doc3_lemma=[]
for token in nlp_spacy(main_doc_no_stop_words3):
    main_doc3_lemma.append(token.lemma_)

main_doc4_lemma=[]
for token in nlp_spacy(main_doc_no_stop_words4):
    main_doc4_lemma.append(token.lemma_)


main_doc1_lemma=" ".join(main_doc1_lemma)
main_doc2_lemma=" ".join(main_doc2_lemma)
main_doc3_lemma=" ".join(main_doc3_lemma)
main_doc4_lemma=" ".join(main_doc4_lemma)


main_doc1_lemma=nlp_spacy(main_doc1_lemma)
main_doc2_lemma=nlp_spacy(main_doc2_lemma)
main_doc3_lemma=nlp_spacy(main_doc3_lemma)
main_doc4_lemma=nlp_spacy(main_doc4_lemma)


ansiedad_lemma=str(main_doc1_lemma)
depresion_lemma=str(main_doc2_lemma)
relaciones_lemma=str(main_doc3_lemma)
suicidio_lemma=str(main_doc4_lemma)


def sent_tokenizer(raw_input):
    new_input = raw_input.split("\n")
    return new_input


main_doc1_entity_final = sent_tokenizer(raw_spacy_ansiedad)
main_doc2_entity_final = sent_tokenizer(raw_spacy_depresion)
main_doc3_entity_final = sent_tokenizer(raw_spacy_relaciones)
main_doc4_entity_final = sent_tokenizer(raw_spacy_suicidio)



text_file_ansiedad = open("/Users/daniela.beltran/Desktop/maestria/Tesis/Codigo/CorporasStopLemas/ansiedad.txt", "w")
n = text_file_ansiedad.write(ansiedad_lemma)
text_file_ansiedad.close()
text_file_depresion = open("/Users/daniela.beltran/Desktop/maestria/Tesis/Codigo/CorporasStopLemas/depresion.txt", "w")
n = text_file_depresion.write(depresion_lemma)
text_file_depresion.close()
text_file_relaciones = open("/Users/daniela.beltran/Desktop/maestria/Tesis/Codigo/CorporasStopLemas/relaciones.txt", "w")
n = text_file_relaciones.write(relaciones_lemma)
text_file_relaciones.close()
text_file_suicidio = open("/Users/daniela.beltran/Desktop/maestria/Tesis/Codigo/CorporasStopLemas/suicidio.txt", "w")
n = text_file_suicidio.write(suicidio_lemma)
text_file_suicidio.close()

path = '/Users/daniela.beltran/Desktop/maestria/Tesis/Codigo/CorporasStopLemas/'
list_book_stop_lemas = {}
for subdir, dirs, files in os.walk(path):
  for file in files:
    #print("archivo 3")
    print(file)
    f = open(path + file, "r", encoding='latin-1')
    list_book_stop_lemas[file] = f.read().lower()   
    #print('\t', list_book_stop_lemas[file][:100])
    f.close()
    #print(list_book_stop_lemas)

corpus_tfidf_stop_lemas = vectorizer.fit_transform(list_book_stop_lemas.values())


mi_first = True;
PREGUNTASTWOMOODS = ["¿Puedes contarme un poco más?"]

HIQUESTIONS = ["¿Cómo puedo ayudarte?",
"¿Sobre qué te gustaría conversar?",
"¿En qué te gustaría enfocarte hoy?",
"¿Qué puedo hacer por ti?",
"¿En qué puedo apoyarte?",
"¿Cómo estás hoy?",
"¿Qué ha pasado?",
"¿Cómo te has sentido los últimos días?",]

QUESTIONSCONTINUE = ["Cuentame, ¿algo más te preocupa?",
"Cuentame un poco más de los problemas que llegaste a tener esta semana",
"¿Cómo te hace sentir este problema?",
"¿Cuál es el problema desde tu punto de vista?",]

myFirstMoods =[]
myFinalMood = ''
myTwoMoods = False
conversationWithUser =[]
mood = []
question_more = 0

# si no coincide 2 vez contactar, recuerda que tienes disponible la linea de emergencia
class ConversationUser:
    def __init__(self, conversation,tema):
        self.conversation = conversation
        self.tema = tema
    
    def getConversation():
        global conversationWithUser
        messageConversation = ""
        for c in conversationWithUser:
            messageConversation = messageConversation + c
        return messageConversation
    def getCategoTema():
        global myFinalMood
        global myConversation
        global mood
        
        return mood

myUser = UserInformation("","","","","")
myConversation = ConversationUser("","")

class ActionAskMood(Action):

    def name(self) -> Text:
        return "action_ask_mood"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        global myFirstMoods
        global myFinalMood
        global myTwoMoods
        global conversationWithUser
        global mood
        global myUser
        global myConversation
        global mi_first
        global question_more
        mi_first = True;
        myFirstMoods =[]
        myFinalMood = ''
        myTwoMoods = False
        conversationWithUser =[]
        mood = []
        question_more = 0
        myConversation = ConversationUser("","")


        dispatcher.utter_message(text=HIQUESTIONS[randint(0, len(HIQUESTIONS)-1)])

        return []

class ActionGetFirstMood(Action):

    def name(self) -> Text:
        return "action_get_first_mood"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        #dispatcher.utter_message(text="estoy en el action action_get_first_mood")
        global myFirstMoods
        global myTwoMoods
        global PREGUNTASELEGIDAS
        global conversationWithUser
        current_state = tracker.current_state()
        latest_message = current_state["latest_message"]["text"]
        conversationWithUser.append(latest_message)
        global mi_first
        global myFinalMood
        global mood
        if myTwoMoods:
            current_state = tracker.current_state()
            latest_message = current_state["latest_message"]["text"]
            myMoodIs = chooseMood(latest_message)
            if len(myMoodIs) > 1 :
                for x in myMoodIs:
                    for y in myFirstMoods:
                        if x == y:
                            myFinalMood = y
            else:
                for y in myFirstMoods:
                    if y == myMoodIs[0]:
                        print("estoy mirando mi mood")
                        myFinalMood = y
                        myConversation.tema = myFinalMood
            if myFinalMood == 'depresion':
                myFinalMood = 'depresion'
            if myFinalMood == '':
                myFinalMood = 'depresion'
            if myFinalMood == 'ansiedad':
                PREGUNTASELEGIDAS=QUESTIONSANSIEDAD
            if myFinalMood == 'depresion':
                PREGUNTASELEGIDAS=QUESTIONSDEPRESION
            if myFinalMood == 'relaciones':
                PREGUNTASELEGIDAS=QUESTIONSRELACIONES
            if myFinalMood == 'suicidio':
                dispatcher.utter_message(text="Por favor contacta la liena 123")
                return [FollowupAction("action_depression_end")]
            myConversation.tema = myFinalMood
            mood = arrayCategoSearch(myFinalMood)
        else:               
            if mi_first:
                #firstMood = simil_spacy(latest_message)
                myMoodIs = chooseMood(latest_message)
                if len(myMoodIs) > 1 :
                    myFirstMoods = myMoodIs
                    #dispatcher.utter_message(text="Lanzar una pregunta")
                    myTwoMoods = True
                    return [FollowupAction("action_ask_question_two_moods")]
                else: 
                #simil_jaccard(latest_message)
                    print(myMoodIs[0])
                    if myMoodIs[0] == 'ansiedad':
                        PREGUNTASELEGIDAS=QUESTIONSANSIEDAD
                    if myMoodIs[0] == 'depresion':
                        PREGUNTASELEGIDAS=QUESTIONSDEPRESION
                    if myMoodIs[0] == 'relaciones':
                        PREGUNTASELEGIDAS=QUESTIONSRELACIONES
                    if myMoodIs[0] == 'suicidio':
                        conversationWithUser.append("Por favor contacta la liena 123")
                        return [FollowupAction("action_depression_end")]
                        
                myConversation.tema = myFinalMood       
                mood = arrayCategoSearch(myFinalMood)
                mi_first = True
           
        return [FollowupAction("action_recursos")]

class ActionDepressionEnd(Action):

    def name(self) -> Text:
        return "action_depression_end"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Por favor contacta la linea 123")

        return [FollowupAction("action_more_conversation_more_recursos")]

class ActionAskQuestion(Action):

    def name(self) -> Text:
        return "action_ask_question_continue_conversation"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
       
        question_more = tracker.get_slot('question_id')
        if question_more == len(QUESTIONSCONTINUE):
            myConversation.conversation = conversationWithUser
            return [FollowupAction("action_end_conversation")]
        else: 
            dispatcher.utter_message(text=QUESTIONSCONTINUE[question_more])
            current_state = tracker.current_state()
            latest_message = current_state["latest_message"]["text"]
            conversationWithUser.append(latest_message)
            conversationWithUser.append(QUESTIONSCONTINUE[question_more])
        return [SlotSet('question_id', question_more + 1)]


class ActionAskQuestionTwoMoods(Action):

    def name(self) -> Text:
        return "action_ask_question_two_moods"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        global PREGUNTASTWOMOODS
        global myTwoMoods
        if myTwoMoods:
            current_question = tracker.get_slot('question_two_moods_id')
            dispatcher.utter_message(text= PREGUNTASTWOMOODS[current_question])
            conversationWithUser.append(PREGUNTASTWOMOODS[current_question])
        return []


class ActionMoodTwoMoods(Action):

    def name(self) -> Text:
        return "action_mood_two_moods"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        global myFirstMoods
        global myTwoMoods
        myFinalMood = ''
        global mood
        
        if myTwoMoods:
            current_state = tracker.current_state()
            latest_message = current_state["latest_message"]["text"]
            conversationWithUser.append(latest_message)
            myMoodIs = chooseMood(latest_message)
            if len(myMoodIs) > 1 :
                for x in myMoodIs:
                    for y in myFirstMoods:
                        if x == y:
                            print("en la seleccion final del mood")
                            print(y)
                            myFinalMood = y
        else:
            for y in myFirstMoods:
                if y == myMoodIs[0]:
                    myFinalMood = y
        if myFinalMood == '':
            myFinalMood = 'depresion'
        if myFinalMood == 'depresion':
            myFinalMood = 'depresion'
        if myFinalMood == 'ansiedad':
            PREGUNTASELEGIDAS=QUESTIONSANSIEDAD
        if myFinalMood == 'depresion':
            PREGUNTASELEGIDAS=QUESTIONSDEPRESION
        if myFinalMood == 'relaciones':
            PREGUNTASELEGIDAS=QUESTIONSRELACIONES
        if myFinalMood == 'suicidio':
            conversationWithUser.append("Por favor contacta la liena 123")
            return [FollowupAction("action_depression_end")]
        myConversation.tema = myFinalMood
        mood = arrayCategoSearch(myFinalMood)

        return [FollowupAction("action_ask_question_two_moods")]

#CATEGORIZADOR 1 - lower

def lowerSpacy(sentence):
    textLower = sentence.lower()
    search_doc = nlp_spacy(textLower)
    main_doc1 = nlp_spacy(raw_spacy_ansiedad)
    main_doc2 = nlp_spacy(raw_spacy_depresion)
    main_doc3 = nlp_spacy(raw_spacy_relaciones)
    main_doc4 = nlp_spacy(raw_spacy_suicidio)
    print("Similitud con Ansiedad:")
    print(search_doc.similarity(main_doc1))
    print("Similitud con Depresion:")
    print(search_doc.similarity(main_doc2))
    print("Similitud con Relaciones:")
    print(search_doc.similarity(main_doc3))
    print("Similitud con Suicidio:")
    print(search_doc.similarity(main_doc4))
    response = {'ansiedad': search_doc.similarity(main_doc1), 'depresion': search_doc.similarity(main_doc2), 'relaciones': search_doc.similarity(main_doc3), 'suicidio': search_doc.similarity(main_doc4)}
    return response

def lowerStopSpacy(sentence):
    test_raw = sentence.lower()
    search_doc = nlp_spacy(test_raw)
    search_doc_no_stop_words = nlp_spacy(' '.join([str(t) for t in search_doc if not t.is_stop]))
    print("Similitud con Ansiedad:")
    print(search_doc_no_stop_words.similarity(main_doc_no_stop_words1))
    print("Similitud con Depresion:")
    print(search_doc_no_stop_words.similarity(main_doc_no_stop_words2))
    print("Similitud con Relaciones:")
    print(search_doc_no_stop_words.similarity(main_doc_no_stop_words3))
    print("Similitud con Suicidio:")
    print(search_doc_no_stop_words.similarity(main_doc_no_stop_words4))
    response = {'ansiedad': search_doc_no_stop_words.similarity(main_doc_no_stop_words1), 'depresion': search_doc_no_stop_words.similarity(main_doc_no_stop_words2), 'relaciones': search_doc_no_stop_words.similarity(main_doc_no_stop_words3), 'suicidio': search_doc_no_stop_words.similarity(main_doc_no_stop_words4)}
    return response

def lowerStopLemaSpacy(sentence):
    test_raw = sentence.lower()
    search_doc = nlp_spacy(test_raw)
    search_doc_no_stop_words = nlp_spacy(' '.join([str(t) for t in search_doc if not t.is_stop]))
    var_test=[]
    for token in nlp_spacy(search_doc_no_stop_words):
        var_test.append(token.lemma_)
    var_str=" ".join(var_test)
    var_str=nlp_spacy(var_str)

    print("=================================")
    print("Similitud con Ansiedad:")
    print(var_str.similarity(main_doc1_lemma))
    print("Similitud con Depresion:")
    print(var_str.similarity(main_doc2_lemma))
    print("Similitud con Relaciones:")
    print(var_str.similarity(main_doc3_lemma))
    print("Similitud con Suicidio:")
    print(var_str.similarity(main_doc4_lemma))
    response = {'ansiedad': var_str.similarity(main_doc1_lemma), 'depresion': var_str.similarity(main_doc2_lemma), 'relaciones': var_str.similarity(main_doc3_lemma), 'suicidio': var_str.similarity(main_doc4_lemma)}
    return response


def categorizerLower(sentence):
    response = []
    print("*****CATEGORIZADOR 1*****")
    print("SPACY LOWER")
    spacyResponse = lowerSpacy(sentence)
    print("SPACY LOWER STOP")
    cosenoResponse = lowerStopLemaSpacy(sentence)
    print("SPACY LOWER STOP LEMA")
    jaccardResponse = lowerStopLemaSpacy(sentence)
    response.append(spacyResponse)
    response.append(cosenoResponse)
    response.append(jaccardResponse)
    return response

#CATEGORIZADOR 2 - lower and stopwords

def lowerCoseno(sentence):
    text_query = sentence.lower()
    query_tfidf = vectorizer.transform([text_query])
    query_tfidf2 = vectorizer.transform([list_book['ansiedad.txt']])
    cosine_response = cosine_similarity(query_tfidf, query_tfidf2)
    query_corporas_relacion = vectorizer.transform([list_book['relaciones.txt']])
    cosine_response_relaciones = cosine_similarity(query_tfidf, query_corporas_relacion)
    query_corporas_depresion = vectorizer.transform([list_book['depresion.txt']])
    cosine_response_depresion = cosine_similarity(query_tfidf, query_corporas_depresion)
    query_corporas_suicidio = vectorizer.transform([list_book['suicidio.txt']])
    cosine_response_suicidio = cosine_similarity(query_tfidf, query_corporas_suicidio)
    ansiedad = cosine_response[0,0]
    relaciones = cosine_response_relaciones[0,0]
    depresion = cosine_response_depresion[0,0]
    suicidio = cosine_response_suicidio[0,0]
    print("similitud de coseno:")
    print("ansiedad: ")
    print(ansiedad)
    print("depresion: ")
    print(depresion)
    print("relaciones: ")
    print(relaciones)
    print("suicidio: ")
    print(suicidio)
    response = {'ansiedad': ansiedad, 'depresion': depresion, 'relaciones': relaciones, 'suicidio': suicidio}
    return response



def lowerStopCoseno(sentence):
    search_doc = nlp_spacy(sentence)
    search_doc_no_stop_words = nlp_spacy(' '.join([str(t) for t in search_doc if not t.is_stop]))
    content_es_input = str(search_doc_no_stop_words)
    text_query = content_es_input
    query_tfidf = vectorizer.transform([text_query])
    
    query_tfidf2 = vectorizer.transform([list_book_lower_stop['ansiedad.txt']])
    cosine_response = cosine_similarity(query_tfidf, query_tfidf2)
    query_corporas_relacion = vectorizer.transform([list_book_lower_stop['relaciones.txt']])
    cosine_response_relaciones = cosine_similarity(query_tfidf, query_corporas_relacion)
    query_corporas_depresion = vectorizer.transform([list_book_lower_stop['depresion.txt']])
    cosine_response_depresion = cosine_similarity(query_tfidf, query_corporas_depresion)
    query_corporas_suicidio = vectorizer.transform([list_book_lower_stop['suicidio.txt']])
    cosine_response_suicidio = cosine_similarity(query_tfidf, query_corporas_suicidio)
    ansiedad = cosine_response[0,0]
    relaciones = cosine_response_relaciones[0,0]
    depresion = cosine_response_depresion[0,0]
    suicidio = cosine_response_suicidio[0,0]
    print("similitud de coseno:")
    print("ansiedad: ")
    print(ansiedad)
    print("depresion: ")
    print(depresion)
    print("relaciones: ")
    print(relaciones)
    print("suicidio: ")
    print(suicidio)

    response = {'ansiedad': ansiedad, 'depresion': depresion, 'relaciones': relaciones, 'suicidio': suicidio}
    return response

def loweStopLemaCoseno(sentence):
    test_raw = sentence.lower()
    search_doc = nlp_spacy(test_raw)
    search_doc_no_stop_words = nlp_spacy(' '.join([str(t) for t in search_doc if not t.is_stop]))
    var_test=[]
    for token in nlp_spacy(search_doc_no_stop_words):
        var_test.append(token.lemma_)
    var_str=" ".join(var_test)
    var_str=nlp_spacy(var_str)
    text_query = str(var_str)
    query_tfidf = vectorizer.transform([text_query])
    query_tfidf2 = vectorizer.transform([list_book_stop_lemas['ansiedad.txt']])
    cosine_response = cosine_similarity(query_tfidf, query_tfidf2)
    query_corporas_relacion = vectorizer.transform([list_book_stop_lemas['relaciones.txt']])
    cosine_response_relaciones = cosine_similarity(query_tfidf, query_corporas_relacion)
    query_corporas_depresion = vectorizer.transform([list_book_stop_lemas['depresion.txt']])
    cosine_response_depresion = cosine_similarity(query_tfidf, query_corporas_depresion)
    query_corporas_suicidio = vectorizer.transform([list_book_stop_lemas['suicidio.txt']])
    cosine_response_suicidio = cosine_similarity(query_tfidf, query_corporas_suicidio)
    ansiedad = cosine_response[0,0]
    relaciones = cosine_response_relaciones[0,0]
    depresion = cosine_response_depresion[0,0]
    suicidio = cosine_response_suicidio[0,0]


    print("similitud de coseno:")
    print("ansiedad: ")
    print(ansiedad)
    print("depresion: ")
    print(depresion)
    print("relaciones: ")
    print(relaciones)
    print("suicidio: ")
    print(suicidio)
    response = {'ansiedad': ansiedad, 'depresion': depresion, 'relaciones': relaciones, 'suicidio': suicidio}
    
    #max_key = max(response, key=response.get)
    #print(max_key)
    return response

def categorizerLowerStop(sentence):
    response = []
    print("*****CATEGORIZADOR 2*****")
    print("COSENO LOWER")
    spacyResponse = lowerCoseno(sentence)
    print("COSENO LOWER STOP")
    cosenoResponse = lowerStopCoseno(sentence)
    print("COSENO LOWER STOP LEMA")
    jaccardResponse = loweStopLemaCoseno(sentence)
    response.append(spacyResponse)
    response.append(cosenoResponse)
    response.append(jaccardResponse)
    return response

#CATEGORIZADOR 3 - lower, stopwords and lematizacion

def lowerJaccard(sentence):
    test_raw = sentence.lower()
    ansiedad_bigrams = list(ngrams(raw_spacy_ansiedad, 2))
    depresion_bigrams = list(ngrams(raw_spacy_depresion, 2))
    relaciones_bigrams = list(ngrams(raw_spacy_relaciones, 2))
    suicidio_bigrams = list(ngrams(raw_spacy_suicidio, 2))
    input_bigrams = list(ngrams(test_raw, 2))
    intersection_ansiedad = len(list(set(ansiedad_bigrams).intersection(set(input_bigrams))))
    union_ansiedad = (len(set(ansiedad_bigrams)) + len(set(input_bigrams))) - intersection_ansiedad
    simil_ansiedad= float(intersection_ansiedad) / union_ansiedad
    intersection_depresion = len(list(set(depresion_bigrams).intersection(set(input_bigrams))))
    union_depresion = (len(set(depresion_bigrams)) + len(set(input_bigrams))) - intersection_depresion
    simil_depresion= float(intersection_depresion) / union_depresion
    intersection_relaciones = len(list(set(relaciones_bigrams).intersection(set(input_bigrams))))
    union_relaciones = (len(set(relaciones_bigrams)) + len(set(input_bigrams))) - intersection_relaciones
    simil_relaciones= float(intersection_relaciones) / union_relaciones
    intersection_suicidio = len(list(set(suicidio_bigrams).intersection(set(input_bigrams))))
    union_suicidio = (len(set(suicidio_bigrams)) + len(set(input_bigrams))) - intersection_suicidio
    simil_suicidio= float(intersection_suicidio) / union_suicidio
    print("ansiedad: ")
    print(simil_ansiedad)
    print("depresion: ")
    print(simil_depresion)
    print("relaciones: ")
    print(simil_relaciones)
    print("suicidio: ")
    print(simil_suicidio)
    response = {'ansiedad': simil_ansiedad, 'depresion': simil_depresion, 'relaciones': simil_relaciones, 'suicidio': simil_suicidio}
    return response
    

def lowerStopJaccard(sentence):
    test_raw = sentence.lower()
    search_doc = nlp_spacy(sentence)
    search_doc_no_stop_words = nlp_spacy(' '.join([str(t) for t in search_doc if not t.is_stop]))
    content_es_input = str(search_doc_no_stop_words)
    ansiedad_bigrams = list(ngrams(ansiedad_stop_tokens, 1))
    depresion_bigrams = list(ngrams(depresion_stop_tokens, 1))
    relaciones_bigrams = list(ngrams(relaciones_stop_tokens, 1))
    suicidio_bigrams = list(ngrams(suicidio_stop_tokens, 1))
    input_bigrams = list(ngrams(content_es_input, 1))
    intersection_ansiedad = len(list(set(ansiedad_bigrams).intersection(set(input_bigrams))))
    union_ansiedad = (len(set(ansiedad_bigrams)) + len(set(input_bigrams))) - intersection_ansiedad
    simil_ansiedad= float(intersection_ansiedad) / union_ansiedad
    intersection_depresion = len(list(set(depresion_bigrams).intersection(set(input_bigrams))))
    union_depresion = (len(set(depresion_bigrams)) + len(set(input_bigrams))) - intersection_depresion
    simil_depresion= float(intersection_depresion) / union_depresion
    intersection_relaciones = len(list(set(relaciones_bigrams).intersection(set(input_bigrams))))
    union_relaciones = (len(set(relaciones_bigrams)) + len(set(input_bigrams))) - intersection_relaciones
    simil_relaciones= float(intersection_relaciones) / union_relaciones
    intersection_suicidio = len(list(set(suicidio_bigrams).intersection(set(input_bigrams))))
    union_suicidio = (len(set(suicidio_bigrams)) + len(set(input_bigrams))) - intersection_suicidio
    simil_suicidio= float(intersection_suicidio) / union_suicidio
    print("ansiedad: ")
    print(simil_ansiedad)
    print("depresion: ")
    print(simil_depresion)
    print("relaciones: ")
    print(simil_relaciones)
    print("suicidio: ")
    print(simil_suicidio)
    response = {'ansiedad': simil_ansiedad, 'depresion': simil_depresion, 'relaciones': simil_relaciones, 'suicidio': simil_suicidio}
    return response

def loweStopLemaJaccard(sentence):
    test_raw = sentence.lower()
    search_doc = nlp_spacy(test_raw)
    search_doc_no_stop_words = nlp_spacy(' '.join([str(t) for t in search_doc if not t.is_stop]))
    var_test=[]
    for token in nlp_spacy(search_doc_no_stop_words):
        var_test.append(token.lemma_)
    var_str=" ".join(var_test)
    var_str=nlp_spacy(var_str)
    ansiedad_bigrams = list(ngrams(ansiedad_lemma, 1))
    depresion_bigrams = list(ngrams(depresion_lemma, 1))
    relaciones_bigrams = list(ngrams(relaciones_lemma, 1))
    suicidio_bigrams = list(ngrams(suicidio_lemma, 1))

    input_bigrams = list(ngrams(str(var_str), 1))

    intersection_ansiedad = len(list(set(ansiedad_bigrams).intersection(set(input_bigrams))))
    union_ansiedad = (len(set(ansiedad_bigrams)) + len(set(input_bigrams))) - intersection_ansiedad
    simil_ansiedad= float(intersection_ansiedad) / union_ansiedad

    intersection_depresion = len(list(set(depresion_bigrams).intersection(set(input_bigrams))))
    union_depresion = (len(set(depresion_bigrams)) + len(set(input_bigrams))) - intersection_depresion
    simil_depresion= float(intersection_depresion) / union_depresion

    intersection_relaciones = len(list(set(relaciones_bigrams).intersection(set(input_bigrams))))
    union_relaciones = (len(set(relaciones_bigrams)) + len(set(input_bigrams))) - intersection_relaciones
    simil_relaciones= float(intersection_relaciones) / union_relaciones

    intersection_suicidio = len(list(set(suicidio_bigrams).intersection(set(input_bigrams))))
    union_suicidio = (len(set(suicidio_bigrams)) + len(set(input_bigrams))) - intersection_suicidio
    simil_suicidio= float(intersection_suicidio) / union_suicidio

    print("ansiedad: ")
    print(simil_ansiedad)
    print("depresion: ")
    print(simil_depresion)
    print("relaciones: ")
    print(simil_relaciones)
    print("suicidio: ")
    print(simil_suicidio)
    response = {'ansiedad': simil_ansiedad, 'depresion': simil_depresion, 'relaciones': simil_relaciones, 'suicidio': simil_suicidio}

    return response

def categorizerLowerStopLema(sentence):
    response = []
    print("*****CATEGORIZADOR 3*****")
    print("JACCARD LOWER")
    spacyResponse = lowerJaccard(sentence)
    print("JACARD LOWER STOP")
    cosenoResponse = lowerStopJaccard(sentence)
    print("JACCARD LOWER STOP LEMA")
    jaccardResponse = loweStopLemaJaccard(sentence)
    response.append(spacyResponse)
    response.append(cosenoResponse)
    response.append(jaccardResponse)
    return response
# categorizador 4
def categorizer4(sentence):
    print("CATEGORIZADOR 4")

    test_raw = sentence.lower()
    search_doc = nlp_spacy(test_raw)
    search_doc_no_stop_words = nlp_spacy(' '.join([str(t) for t in search_doc if not t.is_stop]))
    var_test=[]
    for token in nlp_spacy(search_doc_no_stop_words):
        var_test.append(token.lemma_)
    var_str=" ".join(var_test)
    var_str=nlp_spacy(var_str)
    

    ansiedad_total_ent = []
    depresion_total_ent = []
    relaciones_total_ent = []
    suicidio_total_ent = []

    for word in var_str:
        print(f'{word.text:{12}} {word.pos_:{10}}')
        ansiedad_total_ent.append(len([w for w in main_doc1_lemma if w.text==word.text and w.tag_== word.tag_]))  
        depresion_total_ent.append(len([w for w in main_doc2_lemma if w.text==word.text and w.tag_== word.tag_]))
        relaciones_total_ent.append(len([w for w in main_doc3_lemma if w.text==word.text and w.tag_== word.tag_]))
        suicidio_total_ent.append(len([w for w in main_doc4_lemma if w.text==word.text and w.tag_== word.tag_]))  

    print("ansiedad")
    print(ansiedad_total_ent)
    print(sum(ansiedad_total_ent))
    print("depresion")
    print(depresion_total_ent)
    print(sum(depresion_total_ent))
    print("relaciones")
    print(relaciones_total_ent)
    print(sum(relaciones_total_ent))
    print("suicidio")
    print(suicidio_total_ent)
    print(sum(suicidio_total_ent))
    response = {'ansiedad': sum(ansiedad_total_ent), 'depresion': sum(depresion_total_ent), 'relaciones': sum(relaciones_total_ent), 'suicidio': sum(suicidio_total_ent)}
    responseArray = []
    responseArray.append(response)
    return responseArray

def categorizer5(sentence):
    print("CTEGORIZADOR 5")
    ruler = nlp_spacy.add_pipe("entity_ruler")

    for c in main_doc1_entity_final:
        ruler.add_patterns([{"label": "ANSIE", "pattern": c}]) # Llamaremos CAR a los cargos reconocidos

    for d in main_doc2_entity_final:
        ruler.add_patterns([{"label": "DEPRE", "pattern": d}]) # Llamaremos CAR a los cargos reconocidos

    for e in main_doc3_entity_final:
        ruler.add_patterns([{"label": "RELAC", "pattern": e}]) # Llamaremos CAR a los cargos reconocidos

    for f in main_doc4_entity_final:
        ruler.add_patterns([{"label": "SUICI", "pattern": f}]) # Llamaremos CAR a los cargos reconocidos



    search_doc = nlp_spacy(sentence)

    ansiedad_total_ent= 0
    depresion_total_ent= 0
    relaciones_total_ent= 0
    suicidio_total_ent=0

    for ent in search_doc.ents:
        print('{:10} | {:50}'.format(ent.label_, ent.text))
        if ent.label_ == 'ANSIE':
            ansiedad_total_ent = ansiedad_total_ent + 1
        if ent.label_ == 'DEPRE':
            depresion_total_ent = depresion_total_ent + 1
        if ent.label_ == 'RELAC':
            relaciones_total_ent = relaciones_total_ent + 1
        if ent.label_ == 'SUICI':
            suicidio_total_ent = suicidio_total_ent + 1

        #ansiedad_total_ent.append(len([w for w in main_doc1.ents if w.text==ent.text and w.label_==ent.label_]))  
        #depresion_total_ent.append(len([w for w in main_doc2.ents if w.text==ent.text and w.label_==ent.label_]))
        #relaciones_total_ent.append(len([w for w in main_doc3.ents if w.text==ent.text and w.label_==ent.label_]))
        #suicidio_total_ent.append(len([w for w in main_doc4.ents if w.text==ent.text and w.label_==ent.label_]))  

    print("ansiedad")
    print(ansiedad_total_ent)
    print("depresion")
    print(depresion_total_ent)
    print("relaciones")
    print(relaciones_total_ent)
    print("suicidio")
    print(suicidio_total_ent)
    nlp_spacy.remove_pipe('entity_ruler')
    response = {'ansiedad': ansiedad_total_ent, 'depresion': depresion_total_ent, 'relaciones': relaciones_total_ent, 'suicidio': suicidio_total_ent}
    responseArray = []
    responseArray.append(response)
    return responseArray


## COMPARACION 1: POR CADA PRUEBA SACAR LA CATEGORIA QUE MÁS SE REPITE
def mostCommon(dictMostCommon):
    max_key = max(dictMostCommon, key=dictMostCommon.get)

    return max_key


def mostCommonCategorizer(arrayCategorizer):
    ansiedad = 0
    depresion = 0
    relaciones = 0
    suicidio = 0
    response = ''

    spacyMostCommon = mostCommon(arrayCategorizer[0])
    if spacyMostCommon == 'ansiedad':
        ansiedad = ansiedad + 1
    if spacyMostCommon == 'depresion':
        depresion = depresion + 1
    if spacyMostCommon == 'relaciones':
        relaciones = relaciones + 1
    if spacyMostCommon == 'suicidio':
        suicidio = suicidio + 1

    if len(arrayCategorizer) > 2 : 
        jaccardCommon = mostCommon(arrayCategorizer[1])
        if jaccardCommon == 'ansiedad':
            ansiedad = ansiedad + 1
        if jaccardCommon == 'depresion':
            depresion = depresion + 1
        if jaccardCommon == 'relaciones':
            relaciones = relaciones + 1
        if jaccardCommon == 'suicidio':
            suicidio = suicidio + 1

        if ansiedad >= depresion and ansiedad >= relaciones and ansiedad >= suicidio:
            response = 'ansiedad'
        if depresion >= ansiedad and depresion >= relaciones and depresion >= suicidio:
            response = 'depresion'
        if relaciones >= depresion and relaciones >= ansiedad and relaciones >= suicidio:
            response = 'relaciones'
        if suicidio >= depresion and suicidio >= relaciones and suicidio >= ansiedad:
            response = 'suicidio'
        cosenoCommon = mostCommon(arrayCategorizer[2])
        if cosenoCommon == 'ansiedad':
            ansiedad = ansiedad + 1
        if cosenoCommon == 'depresion':
            depresion = depresion + 1
        if cosenoCommon == 'relaciones':
            relaciones = relaciones + 1
        if cosenoCommon == 'suicidio':
            suicidio = suicidio + 1
    if ansiedad >= depresion and ansiedad >= relaciones and ansiedad >= suicidio:
        response = 'ansiedad'
    if depresion >= ansiedad and depresion >= relaciones and depresion >= suicidio:
        response = 'depresion'
    if relaciones >= depresion and relaciones >= ansiedad and relaciones >= suicidio:
        response = 'relaciones'
    if suicidio >= depresion and suicidio >= relaciones and suicidio >= ansiedad:
        response = 'suicidio'

    return response

def globalMostCommon(responseFirstCategorizer, responseSecondCategorizer, responseThirdCategorizer, responseFourthCategorizer, responseFifthCategorizer):
    ansiedad = 0
    depresion = 0
    relaciones = 0
    suicidio = 0
    response = ''
    myMood = {'depresion': depresion}
    myResponseMood = []
    mostCommonFirst = mostCommonCategorizer(responseFirstCategorizer)
    print("el mas comun en el categorizador 1 es")
    print(mostCommonFirst)
    if mostCommonFirst == 'ansiedad':
        ansiedad = ansiedad + 1
    if mostCommonFirst == 'depresion':
        depresion = depresion + 1
    if mostCommonFirst == 'relaciones':
        relaciones = relaciones + 1
    if mostCommonFirst == 'suicidio':
        suicidio = suicidio + 1
    mostCommonSecond = mostCommonCategorizer(responseSecondCategorizer)
    print("el mas comun en el categorizador 2 es")
    print(mostCommonSecond)
    if mostCommonSecond == 'ansiedad':
        ansiedad = ansiedad + 1
    if mostCommonSecond == 'depresion':
        depresion = depresion + 1
    if mostCommonSecond == 'relaciones':
        relaciones = relaciones + 1
    if mostCommonSecond == 'suicidio':
        suicidio = suicidio + 1
    mostCommonThird = mostCommonCategorizer(responseThirdCategorizer)
    print("el mas comun en el categorizador 3 es")
    print(mostCommonThird)
    if mostCommonThird == 'ansiedad':
        ansiedad = ansiedad + 1
    if mostCommonThird == 'depresion':
        depresion = depresion + 1
    if mostCommonThird == 'relaciones':
        relaciones = relaciones + 1
    if mostCommonThird == 'suicidio':
        suicidio = suicidio + 1
    mostCommonFourth = mostCommonCategorizer(responseFourthCategorizer)
    print("el mas comun en el categorizador 4 es")
    print(mostCommonFourth)
    if mostCommonFourth == 'ansiedad':
        print("en el 4 entre a ansiedad")
        ansiedad = ansiedad + 1
    if mostCommonFourth == 'depresion':
        print("en el 4 entre a depresion")
        depresion = depresion + 1
    if mostCommonFourth == 'relaciones':
        print("en el 4 entre a relaciones")
        relaciones = relaciones + 1
    if mostCommonFourth == 'suicidio':
        print("en el 4 entre a suicidio")
        suicidio = suicidio + 1
    mostCommonFifth = mostCommonCategorizer(responseFifthCategorizer)
    print("el mas comun en el categorizador 5 es")
    print(mostCommonFifth)
    if mostCommonFifth == 'ansiedad':
        ansiedad = ansiedad + 1
    if mostCommonFifth == 'depresion':
        depresion = depresion + 1
    if mostCommonFifth == 'relaciones':
        relaciones = relaciones + 1
    if mostCommonFifth == 'suicidio':
        suicidio = suicidio + 1

    if ansiedad >= depresion and ansiedad >= relaciones and ansiedad >= suicidio:
        response = 'ansiedad'
        myMood = {'ansiedad': ansiedad}
        myResponseMood.append(myMood)
    if depresion >= ansiedad and depresion >= relaciones and depresion >= suicidio:
        response = 'depresion'
        myMood = {'depresion': depresion}
        myResponseMood.append(myMood)
    if relaciones >= depresion and relaciones >= ansiedad and relaciones >= suicidio:
        response = 'relaciones'
        myMood = {'relaciones': relaciones}
        myResponseMood.append(myMood)
    if suicidio >= depresion and suicidio >= relaciones and suicidio >= ansiedad:
        response = 'suicidio'
        myMood = {'suicidio': suicidio}
        myResponseMood.append(myMood)
    print("mi resultado final de el mas comun en los 5 categorizadores es:")
    print(myResponseMood)
    return myResponseMood 

## COMPARACION 2: POR CADA FUNCION MIRAR SI LA DISSTANCIA ENTRE DOS CATEGORIAS ES MUY CORTA
                # SI ESO OCURRE EN TODOS LOS DE UN CATEGORIZADOR UNA NUEVA PREGUNTA
def distanceFunction(responseCategorizer, responseArrayMostCommon):
    ansiedadA = responseCategorizer['ansiedad'] 
    depresionA = responseCategorizer['depresion'] 
    relacionesA = responseCategorizer['relaciones'] 
    suicidioA = responseCategorizer['suicidio']
    maxDistance = 0.07
    response = []
    ansiedad = 0
    depresion = 0
    relaciones = 0
    suicidio = 0
    
    arrayDistancesMostCommon = OrderedDict()
    arrayDistancesMostCommon['ansiedad'] = ansiedadA
    arrayDistancesMostCommon['depresion'] = depresionA
    arrayDistancesMostCommon['relaciones'] = relacionesA
    arrayDistancesMostCommon['suicidio'] = suicidioA
    print(arrayDistancesMostCommon)
    
    print(arrayDistancesMostCommon)
    mySortedMostCommon = OrderedDict(sorted(arrayDistancesMostCommon.items(), key =lambda kv:(kv[1], kv[0])))
    print("revisando esto")
    valuesFinalMostCommon = list(mySortedMostCommon)
    valueskeysMostCommon = list(mySortedMostCommon.keys())
    valuesMostCommon = list(mySortedMostCommon.values())
    print(valuesFinalMostCommon[0])
    responseMostCommon = []
    responseMostCommon.append(valuesFinalMostCommon[len(valuesFinalMostCommon)-1])
    responseMostCommon.append(valuesFinalMostCommon[len(valuesFinalMostCommon)-2])
    mresponseValxMostCommon1 =  responseMostCommon[0]
    mresponseValxMostCommon2 = responseMostCommon[1]


    mresponseValxMostCommon2Key = valueskeysMostCommon[len(valueskeysMostCommon)-1]
    mresponseValxMostCommon1Key = valueskeysMostCommon[len(valueskeysMostCommon)-2]
    mresponseValxMostCommon2Value = valuesMostCommon[len(valuesMostCommon)-1]
    mresponseValxMostCommon1Velue = valuesMostCommon[len(valuesMostCommon)-2]


    print("estoy revisando la funcion de distancias")
    print(responseArrayMostCommon)
    array2MostCommon = False
    if abs(mresponseValxMostCommon2Value - mresponseValxMostCommon1Velue) <= maxDistance:
        print("quedo el mimsmo arreglo")
        array2MostCommon = True
    myResponse = []
    for responseMax in responseArrayMostCommon:
        
        mresponseValx =  list(responseMax.keys())
        for mresponseVal in mresponseValx:

            if mresponseVal == mresponseValxMostCommon1Key or  mresponseVal == mresponseValxMostCommon2Key:
                if array2MostCommon:
                    response = responseMostCommon
                else:

                    if mresponseVal == 'ansiedad':
                        if abs(ansiedad - depresion) <= maxDistance :
                            ansiedad = ansiedad + 1
                            depresion = depresion + 1
                        if abs(ansiedad - relaciones) <= maxDistance :
                            ansiedad = ansiedad + 1
                            relaciones = relaciones + 1
                        if abs(ansiedad - suicidio) <= maxDistance :
                            ansiedad = ansiedad + 1
                            suicidio = suicidio + 1
                    if mresponseVal == 'depresion':
                        if abs(ansiedad - depresion) <= maxDistance :
                            ansiedad = ansiedad + 1
                            depresion = depresion + 1
                        if abs(depresion - relaciones) <= maxDistance :
                            depresion = depresion + 1
                            relaciones = relaciones + 1
                        if abs(depresion - suicidio) <= maxDistance :
                            depresion = depresion + 1
                            suicidio = suicidio + 1
                    if mresponseVal == 'relaciones':
                        if abs(ansiedad - relaciones) <= maxDistance :
                            ansiedad = ansiedad + 1
                            relaciones = relaciones + 1
                        if abs(depresion - relaciones) <= maxDistance :
                            relaciones = relaciones + 1
                            depresion = depresion + 1
                        if abs(relaciones - suicidio) <= maxDistance :
                            relaciones = relaciones + 1
                            depresion = depresion + 1  
                    if mresponseVal == 'suicidio':
                        if abs(ansiedad - suicidio) <= maxDistance :
                            ansiedad = ansiedad + 1
                            suicidio = suicidio + 1
                        if abs(depresion - suicidio) <= maxDistance :
                            suicidio = suicidio + 1
                            depresion = depresion + 1
                        if abs(relaciones - suicidio) <= maxDistance :
                            suicidio = suicidio + 1
                            relaciones = relaciones + 1
            if (mresponseValxMostCommon1Key == 'ansiedad' or  mresponseValxMostCommon2Key == 'ansiedad') and mresponseVal == 'depresion':
                    if mresponseVal == 'ansiedad':
                        if abs(ansiedad - depresion) <= maxDistance :
                            ansiedad = ansiedad + 1
                            depresion = depresion + 1
                        if abs(ansiedad - relaciones) <= maxDistance :
                            ansiedad = ansiedad + 1
                            relaciones = relaciones + 1
                        if abs(ansiedad - suicidio) <= maxDistance :
                            ansiedad = ansiedad + 1
                            suicidio = suicidio + 1
                    if mresponseVal == 'depresion':
                        if abs(ansiedad - depresion) <= maxDistance :
                            ansiedad = ansiedad + 1
                            depresion = depresion + 1
                        if abs(depresion - relaciones) <= maxDistance :
                            depresion = depresion + 1
                            relaciones = relaciones + 1
                        if abs(depresion - suicidio) <= maxDistance :
                            depresion = depresion + 1
                            suicidio = suicidio + 1
                    if mresponseVal == 'relaciones':
                        if abs(ansiedad - relaciones) <= maxDistance :
                            ansiedad = ansiedad + 1
                            relaciones = relaciones + 1
                        if abs(depresion - relaciones) <= maxDistance :
                            relaciones = relaciones + 1
                            depresion = depresion + 1
                        if abs(relaciones - suicidio) <= maxDistance :
                            relaciones = relaciones + 1
                            depresion = depresion + 1  
                    if mresponseVal == 'suicidio':
                        if abs(ansiedad - suicidio) <= maxDistance :
                            ansiedad = ansiedad + 1
                            suicidio = suicidio + 1
                        if abs(depresion - suicidio) <= maxDistance :
                            suicidio = suicidio + 1
                            depresion = depresion + 1
                        if abs(relaciones - suicidio) <= maxDistance :
                            suicidio = suicidio + 1
                            relaciones = relaciones + 1

    print("viendo las distancias mi arreglo quedo")
    arrayDistances = OrderedDict()
    arrayDistances['ansiedad'] = ansiedad
    arrayDistances['depresion'] = depresion
    arrayDistances['relaciones'] = relaciones
    arrayDistances['suicidio'] = suicidio
    print(arrayDistances)
    
    print(arrayDistances)
    mySorted = OrderedDict(sorted(arrayDistances.items(), key =lambda kv:(kv[1], kv[0])))

    valuesFinal = list(mySorted)
    print(valuesFinal)
    response.append(valuesFinal[len(valuesFinal)-1])
    response.append(valuesFinal[len(valuesFinal)-2])
    return response
def chooseMood(sentence):
    global mood
    responseFirstCategorizer = categorizerLower(sentence)
    responseSecondCategorizer = categorizerLowerStop(sentence)
    responseThirdCategorizer = categorizerLowerStopLema(sentence)
    responseFourthCategorizer = categorizer4(sentence)
    responseFifthCategorizer = categorizer5(sentence)

    needMoreInfo = 0
    responseArray = []
    ansiedad = 0
    depresion = 0
    relaciones = 0
    suicidio = 0
    
    responseGlobalCommon = globalMostCommon(responseFirstCategorizer,responseSecondCategorizer,responseThirdCategorizer,responseFourthCategorizer,responseFifthCategorizer)
    responseGlobalCommonValue = list(responseGlobalCommon[0].values())
    if  len(responseGlobalCommon) == 1 and responseGlobalCommonValue[0] >= 3:
            myglobalCat =  list(responseGlobalCommon[0].keys())
            responseArray.append(myglobalCat[0])
            print("Si me decidi ")
            print(responseArray[0])
            mood = arrayCategoSearch(responseArray[0])
    else:
        distanceFirstCategorizerLower = distanceFunction(responseFirstCategorizer[0], responseGlobalCommon)
        distanceFirstCategorizerLowerStop = distanceFunction(responseFirstCategorizer[1], responseGlobalCommon)
        distanceFirstCategorizerLowerStopLema = distanceFunction(responseFirstCategorizer[2], responseGlobalCommon)
        
        if len(distanceFirstCategorizerLower) > 1:
            needMoreInfo = needMoreInfo + 1
            if distanceFirstCategorizerLower[0] == 'ansiedad' or distanceFirstCategorizerLower[1] == 'ansiedad' :
                 ansiedad = ansiedad + 1
            if distanceFirstCategorizerLower[0] == 'depresion' or distanceFirstCategorizerLower[1] == 'depresion':
                depresion = depresion + 1
            if distanceFirstCategorizerLower[0] == 'relaciones' or distanceFirstCategorizerLower[1] == 'relaciones':
                relaciones = relaciones + 1
            if distanceFirstCategorizerLower[0] == 'suicidio' or distanceFirstCategorizerLower[1] == 'suicidio':
                suicidio = suicidio + 1
        else:
            if distanceFirstCategorizerLower[0] == 'ansiedad':
                 ansiedad = ansiedad + 1
            if distanceFirstCategorizerLower[0] == 'depresion':
                depresion = depresion + 1
            if distanceFirstCategorizerLower[0] == 'relaciones':
                relaciones = relaciones + 1
            if distanceFirstCategorizerLower[0] == 'suicidio':
                suicidio = suicidio + 1

        if len(distanceFirstCategorizerLowerStop) > 1:
            needMoreInfo = needMoreInfo + 1
            if distanceFirstCategorizerLowerStop[0] == 'ansiedad' or distanceFirstCategorizerLowerStop[1] == 'ansiedad' :
                 ansiedad = ansiedad + 1
            if distanceFirstCategorizerLowerStop[0] == 'depresion' or distanceFirstCategorizerLowerStop[1] == 'depresion':
                depresion = depresion + 1
            if distanceFirstCategorizerLowerStop[0] == 'relaciones' or distanceFirstCategorizerLowerStop[1] == 'relaciones':
                relaciones = relaciones + 1
            if distanceFirstCategorizerLowerStop[0] == 'suicidio' or distanceFirstCategorizerLowerStop[1] == 'suicidio':
                suicidio = suicidio + 1
        else: 
            if distanceFirstCategorizerLowerStop[0] == 'ansiedad':
                 ansiedad = ansiedad + 1
            if distanceFirstCategorizerLowerStop[0] == 'depresion':
                depresion = depresion + 1
            if distanceFirstCategorizerLowerStop[0] == 'relaciones':
                relaciones = relaciones + 1
            if distanceFirstCategorizerLowerStop[0] == 'suicidio':
                suicidio = suicidio + 1

        if len(distanceFirstCategorizerLowerStopLema) > 1:
            needMoreInfo = needMoreInfo + 1
            if distanceFirstCategorizerLowerStopLema[0] == 'ansiedad' or distanceFirstCategorizerLowerStopLema[1] == 'ansiedad' :
                 ansiedad = ansiedad + 1
            if distanceFirstCategorizerLowerStopLema[0] == 'depresion' or distanceFirstCategorizerLowerStopLema[1] == 'depresion':
                depresion = depresion + 1
            if distanceFirstCategorizerLowerStopLema[0] == 'relaciones' or distanceFirstCategorizerLowerStopLema[1] == 'relaciones':
                relaciones = relaciones + 1
            if distanceFirstCategorizerLowerStopLema[0] == 'suicidio' or distanceFirstCategorizerLowerStopLema[1] == 'suicidio':
                suicidio = suicidio + 1
        else:
            if distanceFirstCategorizerLowerStopLema[0] == 'ansiedad':
                 ansiedad = ansiedad + 1
            if distanceFirstCategorizerLowerStopLema[0] == 'depresion':
                depresion = depresion + 1
            if distanceFirstCategorizerLowerStopLema[0] == 'relaciones':
                relaciones = relaciones + 1
            if distanceFirstCategorizerLowerStopLema[0] == 'suicidio':
                suicidio = suicidio + 1



        distanceSecondCategorizerLower =distanceFunction(responseSecondCategorizer[0],responseGlobalCommon)
        distanceSecondCategorizerLowerStop= distanceFunction(responseSecondCategorizer[1],responseGlobalCommon)
        distanceSecondCategorizerLowerStopLemas= distanceFunction(responseSecondCategorizer[2],responseGlobalCommon)
        if len(distanceSecondCategorizerLower) > 1:
            needMoreInfo = needMoreInfo + 1
            if distanceSecondCategorizerLower[0] == 'ansiedad' or distanceSecondCategorizerLower[1] == 'ansiedad' :
                 ansiedad = ansiedad + 1
            if distanceSecondCategorizerLower[0] == 'depresion' or distanceSecondCategorizerLower[1] == 'depresion':
                depresion = depresion + 1
            if distanceSecondCategorizerLower[0] == 'relaciones' or distanceSecondCategorizerLower[1] == 'relaciones':
                relaciones = relaciones + 1
            if distanceSecondCategorizerLower[0] == 'suicidio' or distanceSecondCategorizerLower[1] == 'suicidio':
                suicidio = suicidio + 1
        else:
            if distanceSecondCategorizerLower[0] == 'ansiedad':
                 ansiedad = ansiedad + 1
            if distanceSecondCategorizerLower[0] == 'depresion':
                depresion = depresion + 1
            if distanceSecondCategorizerLower[0] == 'relaciones':
                relaciones = relaciones + 1
            if distanceSecondCategorizerLower[0] == 'suicidio':
                suicidio = suicidio + 1


        if len(distanceSecondCategorizerLowerStop) > 1:
            needMoreInfo = needMoreInfo + 1
            if distanceSecondCategorizerLowerStop[0] == 'ansiedad' or distanceSecondCategorizerLowerStop[1] == 'ansiedad' :
                 ansiedad = ansiedad + 1
            if distanceSecondCategorizerLowerStop[0] == 'depresion' or distanceSecondCategorizerLowerStop[1] == 'depresion':
                depresion = depresion + 1
            if distanceSecondCategorizerLowerStop[0] == 'relaciones' or distanceSecondCategorizerLowerStop[1] == 'relaciones':
                relaciones = relaciones + 1
            if distanceSecondCategorizerLowerStop[0] == 'suicidio' or distanceSecondCategorizerLowerStop[1] == 'suicidio':
                suicidio = suicidio + 1
        else: 
            if distanceSecondCategorizerLowerStop[0] == 'ansiedad':
                 ansiedad = ansiedad + 1
            if distanceSecondCategorizerLowerStop[0] == 'depresion':
                depresion = depresion + 1
            if distanceSecondCategorizerLowerStop[0] == 'relaciones':
                relaciones = relaciones + 1
            if distanceSecondCategorizerLowerStop[0] == 'suicidio':
                suicidio = suicidio + 1

        if len(distanceSecondCategorizerLowerStopLemas) > 1:
            needMoreInfo = needMoreInfo + 1
            if distanceSecondCategorizerLowerStopLemas[0] == 'ansiedad' or distanceSecondCategorizerLowerStopLemas[1] == 'ansiedad' :
                 ansiedad = ansiedad + 1
            if distanceSecondCategorizerLowerStopLemas[0] == 'depresion' or distanceSecondCategorizerLowerStopLemas[1] == 'depresion':
                depresion = depresion + 1
            if distanceSecondCategorizerLowerStopLemas[0] == 'relaciones' or distanceSecondCategorizerLowerStopLemas[1] == 'relaciones':
                relaciones = relaciones + 1
            if distanceSecondCategorizerLowerStopLemas[0] == 'suicidio' or distanceSecondCategorizerLowerStopLemas[1] == 'suicidio':
                suicidio = suicidio + 1
        else: 
            if distanceSecondCategorizerLowerStopLemas[0] == 'ansiedad':
                 ansiedad = ansiedad + 1
            if distanceSecondCategorizerLowerStopLemas[0] == 'depresion':
                depresion = depresion + 1
            if distanceSecondCategorizerLowerStopLemas[0] == 'relaciones':
                relaciones = relaciones + 1
            if distanceSecondCategorizerLowerStopLemas[0] == 'suicidio':
                suicidio = suicidio + 1

        distanceThirdCategorizerLower = distanceFunction(responseThirdCategorizer[0],responseGlobalCommon)
        distanceThirdCategorizerLowerStop = distanceFunction(responseThirdCategorizer[1],responseGlobalCommon)
        distanceThirdCategorizerLowerStopLemas = distanceFunction(responseThirdCategorizer[1],responseGlobalCommon)
        if len(distanceThirdCategorizerLower) > 1:
            needMoreInfo = needMoreInfo + 1
            if distanceThirdCategorizerLower[0] == 'ansiedad' or distanceThirdCategorizerLower[1] == 'ansiedad' :
                 ansiedad = ansiedad + 1
            if distanceThirdCategorizerLower[0] == 'depresion' or distanceThirdCategorizerLower[1] == 'depresion':
                depresion = depresion + 1
            if distanceThirdCategorizerLower[0] == 'relaciones' or distanceThirdCategorizerLower[1] == 'relaciones':
                relaciones = relaciones + 1
            if distanceThirdCategorizerLower[0] == 'suicidio' or distanceThirdCategorizerLower[1] == 'suicidio':
                suicidio = suicidio + 1
        else:
            if distanceThirdCategorizerLower[0] == 'ansiedad':
                 ansiedad = ansiedad + 1
            if distanceThirdCategorizerLower[0] == 'depresion':
                depresion = depresion + 1
            if distanceThirdCategorizerLower[0] == 'relaciones':
                relaciones = relaciones + 1
            if distanceThirdCategorizerLower[0] == 'suicidio':
                suicidio = suicidio + 1

        if len(distanceThirdCategorizerLowerStop) > 1:
            needMoreInfo = needMoreInfo + 1
            if distanceThirdCategorizerLowerStop[0] == 'ansiedad' or distanceThirdCategorizerLowerStop[1] == 'ansiedad' :
                 ansiedad = ansiedad + 1
            if distanceThirdCategorizerLowerStop[0] == 'depresion' or distanceThirdCategorizerLowerStop[1] == 'depresion':
                depresion = depresion + 1
            if distanceThirdCategorizerLowerStop[0] == 'relaciones' or distanceThirdCategorizerLowerStop[1] == 'relaciones':
                relaciones = relaciones + 1
            if distanceThirdCategorizerLowerStop[0] == 'suicidio' or distanceThirdCategorizerLowerStop[1] == 'suicidio':
                suicidio = suicidio + 1
        else: 
            if distanceThirdCategorizerLowerStop[0] == 'ansiedad':
                 ansiedad = ansiedad + 1
            if distanceThirdCategorizerLowerStop[0] == 'depresion':
                depresion = depresion + 1
            if distanceThirdCategorizerLowerStop[0] == 'relaciones':
                relaciones = relaciones + 1
            if distanceThirdCategorizerLowerStop[0] == 'suicidio':
                suicidio = suicidio + 1

        if len(distanceThirdCategorizerLowerStopLemas) > 1:
            needMoreInfo = needMoreInfo + 1
            if distanceThirdCategorizerLowerStopLemas[0] == 'ansiedad' or distanceThirdCategorizerLowerStopLemas[1] == 'ansiedad' :
                 ansiedad = ansiedad + 1
            if distanceThirdCategorizerLowerStopLemas[0] == 'depresion' or distanceThirdCategorizerLowerStopLemas[1] == 'depresion':
                depresion = depresion + 1
            if distanceThirdCategorizerLowerStopLemas[0] == 'relaciones' or distanceThirdCategorizerLowerStopLemas[1] == 'relaciones':
                relaciones = relaciones + 1
            if distanceThirdCategorizerLowerStopLemas[0] == 'suicidio' or distanceThirdCategorizerLowerStopLemas[1] == 'suicidio':
                suicidio = suicidio + 1
        else:
            if distanceThirdCategorizerLowerStopLemas[0] == 'ansiedad':
                 ansiedad = ansiedad + 1
            if distanceThirdCategorizerLowerStopLemas[0] == 'depresion':
                depresion = depresion + 1
            if distanceThirdCategorizerLowerStopLemas[0] == 'relaciones':
                relaciones = relaciones + 1
            if distanceThirdCategorizerLowerStopLemas[0] == 'suicidio':
                suicidio = suicidio + 1

        distanceFourthCategorizerLower = distanceFunction(responseFourthCategorizer[0],responseGlobalCommon)
        if len(distanceFourthCategorizerLower) > 1:
            needMoreInfo = needMoreInfo + 1
            if distanceFourthCategorizerLower[0] == 'ansiedad' or distanceFourthCategorizerLower[1] == 'ansiedad' :
                 ansiedad = ansiedad + 1
            if distanceFourthCategorizerLower[0] == 'depresion' or distanceFourthCategorizerLower[1] == 'depresion':
                depresion = depresion + 1
            if distanceFourthCategorizerLower[0] == 'relaciones' or distanceFourthCategorizerLower[1] == 'relaciones':
                relaciones = relaciones + 1
            if distanceFourthCategorizerLower[0] == 'suicidio' or distanceFourthCategorizerLower[1] == 'suicidio':
                suicidio = suicidio + 1
        else: 
            if distanceFourthCategorizerLower[0] == 'ansiedad':
                 ansiedad = ansiedad + 1
            if distanceFourthCategorizerLower[0] == 'depresion':
                depresion = depresion + 1
            if distanceFourthCategorizerLower[0] == 'relaciones':
                relaciones = relaciones + 1
            if distanceFourthCategorizerLower[0] == 'suicidio':
                suicidio = suicidio + 1

        distanceFifthCategorizerLower = distanceFunction(responseFifthCategorizer[0],responseGlobalCommon)
        if len(distanceFifthCategorizerLower) > 1:
            needMoreInfo = needMoreInfo + 1
            if distanceFifthCategorizerLower[0] == 'ansiedad' or distanceFifthCategorizerLower[1] == 'ansiedad' :
                 ansiedad = ansiedad + 1
            if distanceFifthCategorizerLower[0] == 'depresion' or distanceFifthCategorizerLower[1] == 'depresion':
                depresion = depresion + 1
            if distanceFifthCategorizerLower[0] == 'relaciones' or distanceFifthCategorizerLower[1] == 'relaciones':
                relaciones = relaciones + 1
            if distanceFifthCategorizerLower[0] == 'suicidio' or distanceFifthCategorizerLower[1] == 'suicidio':
                suicidio = suicidio + 1
        else: 
            if distanceFifthCategorizerLower[0] == 'ansiedad':
                 ansiedad = ansiedad + 1
            if distanceFifthCategorizerLower[0] == 'depresion':
                depresion = depresion + 1
            if distanceFifthCategorizerLower[0] == 'relaciones':
                relaciones = relaciones + 1
            if distanceFifthCategorizerLower[0] == 'suicidio':
                suicidio = suicidio + 1
        
    
        arrayDistances = OrderedDict()
        arrayDistances['ansiedad'] = ansiedad
        arrayDistances['depresion'] = depresion
        arrayDistances['relaciones'] = relaciones
        arrayDistances['suicidio'] = suicidio
        if arrayDistances['ansiedad'] >  arrayDistances['depresion'] and  arrayDistances['ansiedad'] >  arrayDistances['relaciones'] and  arrayDistances['ansiedad'] >  arrayDistances['suicidio']:
            responseArray.append('ansiedad')
            mood = arrayCategoSearch('ansiedad')
        elif arrayDistances['depresion'] >  arrayDistances['ansiedad'] and  arrayDistances['depresion'] >  arrayDistances['relaciones'] and  arrayDistances['depresion'] >  arrayDistances['suicidio']:
            responseArray.append('depresion')
            mood = arrayCategoSearch('depresion')
        elif arrayDistances['relaciones'] >  arrayDistances['ansiedad'] and  arrayDistances['relaciones'] >  arrayDistances['depresion'] and  arrayDistances['relaciones'] >  arrayDistances['suicidio']:
            responseArray.append('relaciones')
            mood = arrayCategoSearch('relaciones')
        elif arrayDistances['suicidio'] >  arrayDistances['ansiedad'] and  arrayDistances['suicidio'] >  arrayDistances['depresion'] and  arrayDistances['suicidio'] >  arrayDistances['relaciones']:
            responseArray.append('suicidio')
            mood = arrayCategoSearch('suicidio')
        else:

            print(arrayDistances)
            mySorted = OrderedDict(sorted(arrayDistances.items(), key =lambda kv:(kv[1], kv[0])))

            valuesFinal = list(mySorted)
            print(valuesFinal)
                    
            responseArray.append(valuesFinal[2])
            responseArray.append(valuesFinal[3])
            print("no me decidi aun tengo que ver en estas dos ")
            print(responseArray[0])
            print(responseArray[1])
       
                
            
    return responseArray

def arrayCategoSearch(categoria):
    global myFinalMood
    global myConversation
    global mood
    print("en getCategoTema")
    print(myConversation.tema)
    print(myFinalMood)
    if categoria == 'depresion':
        mood.append("Tristeza")
        mood.append("Trastorno depresivo")
        mood.append("trastorno depresivo")
                
    if categoria == 'ansiedad':
        mood.append("Ansiedad")
        mood.append("Higiene del sueño")
        mood.append("Miedo")
    if categoria == 'relaciones':
        mood.append("Conflicto familiar")
        mood.append("Características familiares")
        mood.append("Crianza")
        mood.append("Relaciones de hermanos")
            
    return mood

