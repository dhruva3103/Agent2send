#!/usr/bin/env python3
import json
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# py parlai/chat_service/tasks/overworld_demo/run.py --debug --verbose
import os
import re
from os.path import exists

from parlai.core.worlds import World
from parlai.chat_service.services.messenger.worlds import OnboardWorld
from parlai.core.agents import create_agents_from_shared
import torch
import torch.nn as nn
from transformers import BertTokenizer
from .mental_bert import BertForSequenceClassificationImproved
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import pandas as pd

device = torch.device("cpu")
# tokenizer_bert = BertTokenizer.from_pretrained('./customworld/mental_bert_swmh_improved')
# model_bert = BertForSequenceClassificationImproved.from_pretrained('./customworld/mental_bert_swmh_improved')
# model_bert.to(device)


#### emotion analysis with distilbert model #############
tokenizer_bert = AutoTokenizer.from_pretrained("bdotloh/distilbert-base-uncased-empathetic-dialogues-context")

model_bert = AutoModelForSequenceClassification.from_pretrained(
    "bdotloh/distilbert-base-uncased-empathetic-dialogues-context")
model_bert.to(device)


# ---------- Chatbot demo ---------- #
class MessengerBotChatOnboardWorld(OnboardWorld):
    """
    Example messenger onboarding world for Chatbot Model.
    """

    @staticmethod
    def generate_world(opt, agents):
        return MessengerBotChatOnboardWorld(opt=opt, agent=agents[0])

    def parley(self):
        self.episodeDone = True


class MessengerBotChatTaskWorld(World):
    """
    Example one person world that talks to a provided agent (bot).
    """

    MAX_AGENTS = 1
    MODEL_KEYS = ['blender_90M', 'wizard_of_wikipedia']

    def __init__(self, opt, agent, bot):
        self.agent = agent
        self.episodeDone = False
        self.model = bot
        self.first_time = True

    @staticmethod
    def generate_world(opt, agents):
        if opt['models'] is None:
            raise RuntimeError("Model must be specified")
        return MessengerBotChatTaskWorld(
            opt,
            agents[0],
            create_agents_from_shared(
                [opt['shared_bot_params'][keys] for keys in MessengerBotChatTaskWorld.MODEL_KEYS]
            ),
        )

    @staticmethod
    def assign_roles(agents):
        agents[0].disp_id = 'ChatbotAgent'

    def inference(self, model, tokenizer, text):
        """
        Inference function for a pretrained model.
        """
        model.eval()
        tokens_tensor = tokenizer(
            text,
            max_length=model.config.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            predictions = model(**tokens_tensor).logits
            probs = nn.functional.softmax(predictions, dim=-1)
            predictions = torch.max(probs, dim=-1)
            prob, class_id = predictions.values.cpu().detach().item(), predictions.indices.cpu().detach().item()
            class_name = model.config.id2label[class_id]
            confidences = {model.config.id2label[i]: p for i, p in enumerate(probs.cpu().detach().numpy()[0])}

        return class_name, prob, confidences

    def parley(self):
        if self.first_time:
            self.agent.observe(
                {
                    'id': 'World',
                    'text': 'Welcome to the ParlAI Chatbot demo. '
                            'You are now paired with a bot - feel free to send a message.'
                            'Type [DONE] to finish the chat, or [RESET] to reset the dialogue history.',
                }
            )
            self.first_time = False
        a = self.agent.act()
        if a is not None:
            if '[DONE]' in a['text']:
                self.episodeDone = True
            elif '[RESET]' in a['text']:
                self.model.reset()
                self.agent.observe({"text": "[History Cleared]", "episode_done": False})
            else:
                class_name, prob, confidences = self.inference(model_bert, tokenizer_bert, a['text'])
                filename = f"confidences/thecookiedistilbert.csv"
                input_log_file = f"outputs/{a['payload']}.txt"

                content = (
                        f"--- Input Text: {a['text']}" + "\n\n"
                                                         f"Prediction: {class_name} | Confidence: {prob}"
                                                         "--- Confidence for each classes\n"
                                                         f"{list(confidences.items())}" + "\n"
                )
                print("content", content)
                self.record_confidences_score(confidences, filename)
                label, max_avg = self.get_largest_label(filename)
                self.input_logger(input_log_file, content)

                print(f"The Label with largest average is : {label}", flush=True)

                self.model[0].observe(a)
                response = self.model[0].act()
                response.force_set('text', response['text'])
                if label:
                    # print("label",label.split('.')[1].capitalize())
                    confidence_score = (
                            "Average:\n"
                            "Confidence for each classes\n"
                            f"{pd.DataFrame({'class': list(confidences.keys()), 'confidence': list(confidences.values())})}"
                            + "\n\n"
                              f"Prediction: {label} | Confidence: {confidences[label]}" + "\n"
                            # f"Call chat bot model for {label.split('.')[1].capitalize()}" + "\n\n"
                              f"Call chat bot model for {label.capitalize()}" + "\n\n"
                    )

                    self.writer(input_log_file, confidence_score)
                    # Choose the model based on the label
                    if label == "self.depression":
                        self.model[1].observe(a)
                        response = self.model[1].act()

                    model_bert_res = f"""
                    --- Input Text {a['text']}
                    --- Prediction: {class_name} | Confidence: {prob}
                    --- Confidence for each class
                    {pd.DataFrame({'class': list(confidences.keys()), 'confidence': list(confidences.values())})}
                    --- Maximum Average {max_avg}
                    """
                    response.force_set('text', response['text'] + model_bert_res)

                print("===response====")
                print(response)
                print("~~~~~~~~~~~")
                self.agent.observe(response)

                print("===MENTAL MODEL====", flush=True)
                print(f"--- Input Text {a['text']}", flush=True)
                print(f"--- Prediction: {class_name} | Confidence: {prob}", flush=True)
                print(f"--- Confidence for each class", flush=True)
                print(pd.DataFrame({"class": list(confidences.keys()), "confidence": list(confidences.values())}),
                      flush=True)
                return json.dumps(pd.DataFrame({"class": list(confidences.keys()), "confidence": list(confidences.values())}),
                    flush=True)

    def input_logger(self, filename, content):
        """
        Logger method for the input text which calles all the helper method to format the text
        and write to a file

        :params filename: The filename the content to be written in.
        :params content: The final formatted text to be written to the file.
        """
        current_input_number = self.get_last_input_number(filename) + 1
        ordinary_number = self.get_ordinary_number(current_input_number)
        input_str = self.format_input(current_input_number, ordinary_number, content)
        if current_input_number == 1:
            input_str = "User input recording\n\n" f"{input_str}" + "\n"
        self.writer(filename, input_str)

    def writer(self, filename, content):
        """
        Write the content to the filename

        :params filename: The filename the content to be written in.
        :params content: The final formatted text to be written to the file.
        """
        with open(filename, "a+") as f:
            f.writelines(content)

    def get_ordinary_number(self, input_number):
        """
        Returns ordinary number for the input integer number

        :params input_number: Integer input number
        :return: string which is ordinary number representation for the input_number
        """
        if input_number % 10 == 1 and input_number != 11:
            return f"{input_number}st"
        elif input_number % 10 == 2 and input_number != 12:
            return f"{input_number}nd"
        elif input_number % 10 == 3 and input_number != 13:
            return f"{input_number}rd"
        else:
            return f"{input_number}th"

    def format_input(self, input_number, ordinary_number, content):
        """
        Returns a formatted string

        :params input_number: Integer input number for the text
        :params ordinary_number: Ordinay number representation of the input number Ex. 1st, 2nd ..
        :params content: A formatted string which contains the user text and prediction condifences fore each classes

        :return: A formatted string Ex.
        Input 2:
        This is 2nd message

        --- Input Text: Hello

        Prediction: self.depression | Confidence: 0.5590712428092957--- Confidence for each classes
        [('self.Anxiety', 0.09287001), ('self.bipolar', 0.17255585), ('self.depression', 0.55907124), ('self.SuicideWatch', 0.14899628), ('self.offmychest', 0.026506605)]
        """
        new_line = "\n"
        return (
            f"Input {input_number}:{new_line}"
            f"This is {ordinary_number} message{new_line}{new_line}"
            f"{content}{new_line}"
        )

    def get_last_input_number(self, filename):
        """
        Get the last input number with regex search
        The regex returns [' 1', ' 2'] and we take the max from the list

        :params filename: the text filename which store input and confidence scores
        :returns last_inpu_number: The last input number in the text file
        """
        try:
            with open(filename, "r") as f:
                input_numbers = re.findall("Input\s\d+", f.read())
                input_numbers = [int(num.split(" ")[1]) for num in input_numbers]
                return max(input_numbers)
        except FileNotFoundError:
            return 0

    def get_largest_label(self, filename):
        """
        Take the latest 5 values from the csv file and calculate the label with the largest average value
        It will calculate the average when the input is multiple of 5 only

        :params filename: The csv filename
        :return tuple: (The label with the largest average value, and the max_average value)
        """
        print("filename", filename)
        label_with_highest_avg = None
        # if not exists(filename):
        #     return label_with_highest_avg, 0
        df = pd.read_csv(filename)
        print("df", df)
        # if len(df) < 36 or len(df) % 36 != 0:
        #     return label_with_highest_avg, 0
        latest_values = df.tail(5).to_dict("list")
        max_average = float("-inf")
        for key, value in latest_values.items():
            current_average = float(sum(value) / len(value))
            if current_average > max_average:
                max_average = current_average
                label_with_highest_avg = key
        return label_with_highest_avg, max_average

    def record_confidences_score(self, confidences, filename):
        """
        Add confidence score to a csv file and append as they occur

        :params confidences: The confidences score
        :params filename: The csv filename
        """

        df = pd.DataFrame([confidences])
        if os.path.exists(filename):
            df.to_csv(filename, mode="a+", index=False, header=False)
        else:
            df.to_csv(filename, index=False)

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        self.agent.shutdown()


# ---------- Overworld -------- #
class MessengerOverworld(World):
    """
    World to handle moving agents to their proper places.
    """

    def __init__(self, opt, agent):
        self.agent = agent
        self.opt = opt
        self.first_time = True
        self.episodeDone = False

    @staticmethod
    def generate_world(opt, agents):
        return MessengerOverworld(opt, agents[0])

    @staticmethod
    def assign_roles(agents):
        for a in agents:
            a.disp_id = 'Agent'

    def episode_done(self):
        return self.episodeDone

    def parley(self):
        if self.first_time:
            self.agent.observe(
                {
                    'id': 'Overworld',
                    'text': 'Welcome to the overworld for the ParlAI messenger '
                            'chatbot demo. Please type "begin" to start, or "exit" to exit',
                    'quick_replies': ['begin', 'exit'],
                }
            )
            self.first_time = False
        a = self.agent.act()
        if a is not None and a['text'].lower() == 'exit':
            self.episode_done = True
            return 'EXIT'
        if a is not None and a['text'].lower() == 'begin':
            self.episodeDone = True
            return 'default'
        elif a is not None:
            self.agent.observe(
                {
                    'id': 'Overworld',
                    'text': 'Invalid option. Please type "begin".',
                    'quick_replies': ['begin'],
                }
            )

