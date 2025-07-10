# -*- coding: utf-8 -*-
"""
Created on 2025/07/10 15:16:59
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import os
from os import path


def get_index(n):
    string = str(n)
    if len(string) < 4:
        string = "0" * (4 - len(string)) + string

    index = ""
    for s in string:
        index += (s + "-")

    return index


new = """---
title: "{}"
excerpt: "None<br/><img src='/images/nikon/.jpg'>"
collection: photography
---

None"""


images_list = ["Xi'an-Bell-Tower",
               "Xi'an-TV-Tower",
               "Taiyuan-South-Railway-Station",
               "Qinling-Wildlife-Park",
               "Turdus-Mandarinus-in-Fengqing-Park",
               "Xiaohe-Ecological-Park",
               "Xi'an-Academy-Gate",
               "Duck-in-the-Lotus",
               "Xi'an-Saige-Parrot",
               "Taiyuan-Shouyi-Gate",
               "Xi'an-Xingqing-Palace",
               "Street-Photography-with-Bai-Xiang",
               "Xi'an-Daming-Palace",
               "Dreamland-Anime-Exhibition-Free-Travel",
               "Fengqing-Park",
               "Xidian-University-Drama-Troupe",
               "Baixiang-Simon",
               "Nipponia-Nippon-at-Qinling-Wildlife-Park",
               "Duck-Silhouettes-at-the-Xi'an-Expo-Park",
               "Monkey-Fight",
               "Taiyuan-Shanxi-Hotel",
               "Cyanopica-Cyanus",
               "Office-Building-on-Keji-Road",
               "Moon",
               "Whirlpool-of-Flower-Sea",
               "Red-Panda",
               "Pelecanus-Onocrotalus", ]

images_dict = {}
for image in images_list:
    images_dict[image] = None

for file in os.listdir("."):
    if file.endswith(".md"):
        name = file[20:-3]
        with open(file, "r", encoding="utf-8") as f:
            data = f.read()
        images_dict[name] = data
        os.remove(file)

for number, key in enumerate(images_list):

    value = images_dict[key]
    index = get_index(number)

    file_name = f"photography-{index}{key}.md"

    with open(file_name, "w", encoding="utf-8") as f:
        f.write(value if value is not None else new.format(key))


print(images_dict)


