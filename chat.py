#!/usr/bin/env python
# -*- coding: utf8 -*-
import random
import json
import discord
from discord.ext import commands
import torch
from model import NeuralNet


from nltk_utils import bag_of_words, tokenize

token = 'Nzg4NzU0ODgxMjA3ODYxMjQ4.X9oHOg.d-0iZ9LjTnIx_MRX30XZkht0uFs'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

class Bot(commands.Bot):

    def __init__(self):
        super().__init__(command_prefix='8', pm_help=None, description="Türkçe Discord Bot")


    async def on_ready(self):
        print("Ben bir botum")
        print(self.user.name)
        print(self.user.id)


    async def on_message(self, message):
        if message.author.bot:
            return

        print(message.content)
   
    
        # sentence = "Merhaba"
        sentence = str(message.content)

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
                    await message.channel.send(f"{random.choice(intent['responses'])}")
        else:
            print(f"{bot_name}: Anlayamadim. Tekrar sorar misin?")
            await message.channel.send(f"{random.choice(intent['responses'])}")


     
bot_name = "RoboCan"
print("Haydi konuşalım! (çıkış yapmak için => 'quit')")
bot  = Bot()
bot.run(token)
