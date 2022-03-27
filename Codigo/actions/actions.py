
from random import randint
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, FollowupAction, ConversationPaused
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModel, TFAutoModel, pipeline

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


class ActionAskQuestion(Action):

    def name(self) -> Text:
        return "action_ask_question"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        current_question = tracker.get_slot('question_id')
        if current_question == len(QUESTIONS):
            return [FollowupAction("action_end_conversation")]
        dispatcher.utter_message(text=f"**Pregunta {current_question + 1} de {len(QUESTIONS)}** \n --- \n" +
                                      QUESTIONS[current_question])
        return [SlotSet('question_id', current_question + 1)]

class ActionStartQuestions(Action):

    def name(self) -> Text:
        return "action_start_questions"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="Muy bien, ¡comencemos!")
        return [SlotSet('state', 'questions')]


class ActionEndConversation(Action):

    def name(self) -> Text:
        return "action_end_conversation"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        is_asking = tracker.get_slot('state') == 'questions'
        if is_asking:
            dispatcher.utter_message(text="Gracias por responder a mis preguntas 😁")
        else:
            dispatcher.utter_message(text="Está bien. ¡Espero hablar contigo en otro momento!")
           

        return [ConversationPaused()]

"""
class  example(Action):

    def name(self) -> Text:
        return "example"
  
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        nlp1 = pipeline( 'question-answering', model='mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es', 
            tokenizer=('mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es',{"use_fast": False}))
        return  [SlotSet(nlp1({'question': 'question','context': context_tristeza}))]"""





        
        