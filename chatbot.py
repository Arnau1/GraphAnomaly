import re
import numpy as np
import random

greeting = ["Hello! How can I assist you today?", "Hi there! What brings you here?",  "Welcome! What can I help you with?"]
farewell = ["Farewell! If you need anything else, feel free to reach out.", "Goodbye! Have a great day!", "Until next time! If you ever need assistance, I'll be here."]
help = ["What do you need assistance with?", "How can I help you?", "Explain me what do you need help with."]
photo = ["Okay, please send me a photo of what you've described.", "Alright, could you send me an evidence of what you've mentioned."]
unknown = ["I'm sorry, I didn't understand that. Could you please rephrase it differently?", "I'm not quite sure what you're asking. Could you clarify it?", "Sorry, I'm not sure what you mean. Could you try asking in a different way?"]

def chatbot(input_message):
    print('Patient: ', input_message)
    n = np.random.randint(0,3)
    output_message = 'Doctor: '

    if re.search('([Hh]ello|[Hh]i|[Hh]ey|[Gg]ood\s(?:morning|afternoon|evening)|[Gg]reetings)$', input_message):
        output_message += greeting[n]
        return output_message

    elif re.search('([Gg]oodbye(.*)|[Bb]ye(.*)|[Ss]ee you later(.*)|[Tt]ake care(.*)|[Ff]arewell(.*)|[Hh]ave a (?:good|great) day(.*)|[Ss]ee you (=:soon|later)(.*))$', input_message):
        output_message += farewell[n]
        return output_message
    
    elif re.search('(help|support|assistance|ask(?:s|ed|ing))', input_message):
        output_message += help[n]
        return output_message
    
    elif re.search("notic(?:e|ed|ing|es|e's)|fe(?:el|lt|eling|els)|experienc(?:e|es|ed|ing)|encounter(?:s|ed|ing)|observ(?:es|ed|ing)|worr(?:y|ies|ied|ying)|concern(?:ed|ing)", input_message):
        
        if re.search('pigmentations?|spots?|marks?|lesions?|skin', input_message):
            output_message += "Okay, please send me a photo of what you've mentioned. Visual information will help me assist you better."
            return output_message
        
        elif re.search('cough|sneeze|breathing|sound|cold|sore throat|throat|hoarse voice|voice|congestion|swallowing|difficulty(?:swallowing|breathing)', input_message):
            output_message += "Alright, I would need you to provide me an audio recording of your breathing to better assess what is happening."
            return output_message
        
        else:
            output_message += unknown[n]
            return output_message

    
    else:
        output_message += unknown[n]
        return output_message

# inputs = ["Hi",
#           "I need some help with something",
#           "Lately, I've been noticing some dark spots in my skin.",
#           "Also, I am concerned because I've been having difficulty breathing.",
#           "Thank you so much, bye!",
#           "I am hungry"]

# for input in inputs:
#   print(chatbot(input))