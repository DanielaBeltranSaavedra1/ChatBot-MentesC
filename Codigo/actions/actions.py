from .personalInformationActions import UserInformation
from .moodActions import ConversationUser
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
import smtplib
import random
import numpy as np
import pandas as pd


class ActionRecursos(Action):

    def name(self) -> Text:
        return "action_recursos"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        tema = ConversationUser.getCategoTema()
        myUser = UserInformation.getMyUser()
        bithday = myUser.birthday.split('/')
        year = ''
        if len(bithday) > 1:
            year = bithday[2]
        nombre = myUser.name.split(" ")
        dispatcher.utter_message(text="Aquí hay algo para que puede ayudarte")
        dispatcher.utter_message(text=selectResource(tema, year))
        #dispatcher.utter_message(text=nombre[0] + " ¿Te gustaría que un profesional te contacte el siguiente día hábil para que brinde mayor atención?")

        return [FollowupAction("action_more_conversation_more_recursos")]
        
class ActionAskNeedMoreConversation(Action):

    def name(self) -> Text:
        return "action_more_conversation_more_recursos"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        myUser = UserInformation.getMyUser()
        nombre = myUser.name.split(" ")
        dispatcher.utter_message(text=nombre[0] + " ¿Te gustaría seguir hablando conmigo?, podría recomendarte algún otro recurso que puede ayudarte")

        return []

class ActionEndConversation(Action):

    def name(self) -> Text:
        return "action_end_conversation"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        myUser = UserInformation.getMyUser()
        nombre = myUser.name.split(" ")
        dispatcher.utter_message(text=nombre[0] + " ¿Te gustaría que un profesional te contacte el siguiente día hábil para que brinde mayor atención?")

        return []

class ActionEndConversationMoreAtention(Action):

    def name(self) -> Text:
        return "action_end_conversation_more_atention"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Perfecto, un profesional se pondrá en contacto contigo!!")
        myUser = UserInformation.getMyUser()
        nombre = myUser.name.split(" ")
        dispatcher.utter_message(text=nombre[0]+ "Recuerda que acá estoy disponible para volver a hablar contigo ")
        conversationUser = ConversationUser.getConversation()
        message = '<html><body><h1>Hola! este correo es automático</h1><h2>A continuación tienes los datos del usuario que tuvo una conversación con el chatbot de Mentes y requiere más atención</h2><p><b>Nombre:</b></p></body></html>'
        sendMail(myUser.name,myUser.mail,myUser.phone,myUser.birthday,myUser.doc,conversationUser,'Si','Conversación con usuario que necesita más atención')
        sendMailToUser(myUser.name,myUser.mail,"Solicitud de contacto con profesional del Portal Mentes Colectivas")
        return []

class ActionEndConversationNoMoreAtention(Action):

    def name(self) -> Text:
        return "action_end_conversation_no_more_atention"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Está bien. Recuerda que puedes hablar conmigo en cualquier momento!")
        myUser = UserInformation.getMyUser()
        conversationUser = ConversationUser.getConversation()
        message = '<html><body><h1>Hola! este correo es automático</h1><h2>A continuación tienes los datos del usuario que tuvo una conversación con el chatbot de Mentes y requiere más atención</h2><p><b>Nombre:</b></p></body></html>'
        sendMail(myUser.name,myUser.mail,myUser.phone,myUser.birthday,myUser.doc,conversationUser,'No','Conversación con usuario que no necesita más atención')
        return []
class ActionSendConversationDepression(Action):

    def name(self) -> Text:
        return "action_send_conversation_depression"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        myUser = UserInformation.getMyUser()
        conversationUser = ConversationUser.getConversation()
        message = '<html><body><h1>Hola! este correo es automático</h1><h2>A continuación tienes los datos del usuario que tuvo una conversación con el chatbot de Mentes el cual tuvo una clasificación de suicidio</h2><p><b>Nombre:</b></p></body></html>'
        sendMail(myUser.name,myUser.mail,myUser.phone,myUser.birthday,myUser.doc,conversationUser,'Si','Conversación con usuario que fue clasificado su respuesta con el tema de suicidio')
        return []


def sendMail(nombre,correo,telefono,fechaNacimiento,documento,conversacion,needAction, subject):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    me = "chatbotmentescolectivas2022@gmail.com"
    my_password = r"Javeriana2022"
    you = "chatbotmentescolectivas2022@gmail.com"
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = me
        msg['To'] = you
        html = '<!DOCTYPE html><html><body><h1>Hola! este correo es automático</h1><h2>A continuación tienes los datos del usuario que tuvo una conversación con el chatbot de Mentes</h2><table style="width:100%"><tr><th>Nombre</th><th>Correo</th><th>Telefono</th><th>Fecha de nacimiento</th><th>Documento de identidad</th><th>Conversación</th><th>Requiere más atención</th></tr><tr><td>{name}</td><td>{mail}</td><td>{phone}</td><td>{birthday}</td><td>{doc}</td><td>{conversation}</td><td>{atention}</td></tr></table></body></html>'.format(name= nombre, mail = correo , phone = telefono, birthday = fechaNacimiento, doc = documento, conversation = conversacion,atention = needAction)
        part2 = MIMEText(html, 'html')
        msg.attach(part2)
        s = smtplib.SMTP_SSL('smtp.gmail.com')
        s.login(me, my_password)
        s.sendmail(me, you, msg.as_string())
        s.quit()

        print("Correo enviado correctamente")
    except:
        print("Fallo al momento de enviar el correo")

def sendMailToUser(nombre,correo, subject):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    me = "chatbotmentescolectivas2022@gmail.com"
    my_password = r"Javeriana2022"
    you = correo
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = me
        msg['To'] = you
        html = '<!DOCTYPE html><html><body><h1>Hola {name}!</h1><h3>Este correo es para confirmarte que la solicitud para ser contactado por un profesional ya fué enviada, se contactarán contigo el siguiente día hábil!!. Sin embargo, recuerda que tienes las líneas de atención 123 si necesitas una atención inmediata</h3><h2>Saludos, Mia chabot del Portal Mentes Colectivas.</h2></body></html>'.format(name= nombre)
        part2 = MIMEText(html, 'html')
        msg.attach(part2)
        s = smtplib.SMTP_SSL('smtp.gmail.com')
        s.login(me, my_password)
        s.sendmail(me, you, msg.as_string())
        s.quit()

        print("Correo enviado correctamente")
    except:
        print("Fallo al momento de enviar el correo")

def selectResource(tema, year):
    import pandas as pd
    file_errors_location = '/Users/daniela.beltran/Desktop/maestria/Tesis/Codigo/Recursos/RecursosDoc.xlsx'
    resources = pd.read_excel(file_errors_location)
    toSelect = []
    age = getAge(year)
    print("el tema elegido al final fue")
    print(tema)
    for x in resources['SUBTEMA']:
        t = x.replace("; ", ";")
        y = t.split(';')
        
        if any(map(lambda each: each in y, tema)):
            
            toSelect.append(x)
    selectRows = resources[resources.SUBTEMA.isin(toSelect)]
    
    row = []
    finalRowsResources = selectRows
    if age <= 5: 
        row.append('0-5')
        finalRowsResources= selectRows[selectRows.a05 == 'Sí']
    if age >=6 and age <=11:
        row.append('6-11')
        finalRowsResources= selectRows[selectRows.a611 == 'Sí']
    if age >=12 and age <=18:
        row.append('12-18')
        finalRowsResources= selectRows[selectRows.a1218 == 'Sí']
    if age >=19 and age <=26:
        row.append('19-26')
        finalRowsResources= selectRows[selectRows.a1926 == 'Sí']
    if age >=27 and age <=59:
        row.append('27-59')
        finalRowsResources= selectRows[selectRows.a2759 == 'Sí']
    if age >=60:
        row.append('60')
        finalRowsResources= selectRows[selectRows.a60 == 'Sí']
    
    
    

    selectResources = selectRows['Enlace'].values.tolist()
    
    return selectResources[randint(0, len(selectResources)-1)]
def getAge(year):
    print(year)
    #userAge = 2022 - int(year)
    return 0





