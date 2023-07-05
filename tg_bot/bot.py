import sys
sys.path.insert(0, "..")

from telebot.async_telebot import AsyncTeleBot
import os
from skimage.io import imread
from skimage.transform import resize
import torch
import matplotlib.pyplot as plt
from models.generator import Generator
import numpy as np


f = open('tokens.txt', 'r', encoding='UTF-8')
token = f.readline()
f.close()
bot = AsyncTeleBot(token)
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

model = Generator()
model.load_state_dict(torch.load('../weights/gen_50k_2', map_location=torch.device('cpu')))
model.eval()


@bot.message_handler(commands=["start", "!help"])
async def start(m):
    await bot.send_message(m.chat.id, "Нарисуй изображение, где на !белом! "
                                      "фоне будет нарисован черный контур "
                                      "какой-либо обуви(желательно изображение 256х256)."
                                      "Скинь его боту файлом и получи сгенерированную обувь.")


@bot.message_handler(content_types=["document"])
async def handle_document(m):
    file_info = await bot.get_file(m.document.file_id)
    downloaded_file = await bot.download_file(file_info.file_path)
    with open('pic.png', 'wb') as new_file:
        new_file.write(downloaded_file)
    image = imread('pic.png')
    os.remove('pic.png')
    image = np.array(image, np.float32) / 255
    image = resize(image, (256, 256), anti_aliasing=True)
    image = torch.from_numpy(image).permute(2, 0, 1)
    image = torch.reshape(model(image[None, :, :, :]), (3, 256, 256)).permute(1, 2, 0).detach().numpy()
    image = image * stats[1][0] + stats[0][0]
    plt.imsave('pic.png', image)
    await bot.send_photo(m.chat.id, open('pic.png', 'rb'))
    os.remove('pic.png')


@bot.message_handler(content_types=["photo"])
async def handle_photo(m):
    await bot.send_message(m.chat.id, "Это не файл...")


@bot.message_handler()
async def handle_message(m):
    await bot.send_message(m.chat.id, "Это даже не картинка...")
