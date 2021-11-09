# -*- coding: utf-8 -*-
# @Time    : 2021/11/9 3:13
# @Author  : LIU YI

#python 画柱状图折线图
#-*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
from matplotlib.font_manager import FontProperties
import pandas as pd

color_list = ["#FBF46D", "#2F86A6", "#34BE82", "#FF9292", "#90AACB"]


data = pd.read_csv("intro.csv")
data = data.multiply(100)
data.index = ["User0", "User1", "User2", "User3", "User4"]


data.plot(kind = "bar", fontsize = 15, color = color_list, width= 0.8 )
# plt.figure(figsize=(10,10))
# width = 0.3
# n = 2
# plt.ylabel("Model Performance (%)")
# plt.bar(color = color_list)
plt.grid(axis = "y", linestyle = "-.")
plt.ylabel("Model Performance (%)", fontsize = 15)
plt.legend(fontsize = 12)
plt.xticks(rotation = 0)
# plt.colorbar(color_list)

#
plt.savefig("fig1.png", dpi = 600, format = 'png')
plt.show()