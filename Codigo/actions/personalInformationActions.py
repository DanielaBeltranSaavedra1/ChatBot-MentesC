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
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import ngrams
import pandas
nltk.download('popular', quiet=True) # for downloading packages
nltk.download('wordnet') # first-time use only
from nltk.stem import SnowballStemmer

USERPERSONALINFO = []

class UserInformation:
    def __init__(self, name, birthday, mail, phone, doc):
        self.name = name
        self.birthday = birthday
        self.mail = mail
        self. phone = phone
        self. doc = doc

    def getMyUser():
        global myUser
        return myUser

myUser = UserInformation("","","","","")
class ActionAskPersonalInformation(Action):
	
    def name(self) -> Text:
        return "action_ask_personal_information"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            global myUser
            global USERPERSONALINFO
            USERPERSONALINFO = []
            myUser = UserInformation("","","","","")
            dispatcher.utter_message(text="Hola! mi nombre es Mia, antes de comenzar por favor ingresa tus datos personales")
            dispatcher.utter_message(text="Ingresa tu nombre completo:")
            
            return []

class ActionAskPersonalInformation_fecha(Action):
    
    def name(self) -> Text:
        return "action_ask_personal_information_fecha"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            current_state = tracker.current_state()
            latest_message = current_state["latest_message"]["text"]
            USERPERSONALINFO.append(latest_message)
            dispatcher.utter_message(text="Ingresa tu fecha de nacimiento (Por favor ingresa la fecha con el formato DD/MM/AAAA):")
            
            return []

class ActionAskPersonalInformation_numero(Action):
    
    def name(self) -> Text:
        return "action_ask_personal_information_numero"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            current_state = tracker.current_state()
            latest_message = current_state["latest_message"]["text"]
            USERPERSONALINFO.append(latest_message)
            dispatcher.utter_message(text="Ingresa tu número de telefono (Por favor agrega el indicativo +57):")
            
            return []

class ActionAskPersonalInformation_mail(Action):
    
    def name(self) -> Text:
        return "action_ask_personal_information_mail"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            current_state = tracker.current_state()
            latest_message = current_state["latest_message"]["text"]
            USERPERSONALINFO.append(latest_message)
            dispatcher.utter_message(text="Ingresa tu correo electrónico:")
            
            return []

class ActionAskPersonalInformation_documento(Action):
    
    def name(self) -> Text:
        return "action_ask_personal_information_documento"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            current_state = tracker.current_state()
            latest_message = current_state["latest_message"]["text"]
            USERPERSONALINFO.append(latest_message)
            dispatcher.utter_message(text="Ingresa tu número de documento (Por favor agrega el tipo de documento CC/TI/CE/PASAPORTE):")
            
            return []


class ActionActiveChatbot(Action):

    def name(self) -> Text:
        return "action_active_chatbot"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        global myUser
        current_state = tracker.current_state()
        latest_message = current_state["latest_message"]["text"]
        USERPERSONALINFO.append(latest_message)
        if len(USERPERSONALINFO) == 5:
            myUser.name = USERPERSONALINFO[0]
            myUser.birthday = USERPERSONALINFO[1]
            myUser.phone = USERPERSONALINFO[2]
            myUser.mail = USERPERSONALINFO[3]
            myUser.doc = USERPERSONALINFO[4]

        return [FollowupAction("utter_ask_question_request")]

      
