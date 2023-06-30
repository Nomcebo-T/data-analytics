# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 18:01:57 2023

@author: Game
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Step 1: Read the contents of the "un_declaration_hr_text_data.txt" file.
with open(r"C:\Users\Game\OneDrive\data_analytics\Projects\Text mining\datasets\un_declaration_hr_text_data.txt") as file:
 text = file.read()
# Step 2: Preprocess the text by removing punctuation, converting to lowercase, and tokenizing into words.
tokens = word_tokenize(text.lower())
words = [word for word in tokens if word.isalpha()]

# Step 3: Remove stop words using a predefined list of stop words.
stop_words = set(stopwords.words("english"))
filtered_words = [word for word in words if word not in stop_words]

# Step 4: Count the frequency of each word in the text.
word_freq = FreqDist(filtered_words)

# Step 5: Generate a word cloud using the most frequent words.
wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)

# Step 6: Generate a bar plot of the top 25 frequent words.
top_words = word_freq.most_common(25)
labels, frequencies = zip(*top_words)

plt.figure(figsize=(10, 6))
plt.bar(labels, frequencies)
plt.xticks(rotation=90)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 25 Frequent Words")
plt.tight_layout()
plt.savefig(r"C:\Users\Game\OneDrive\data_analytics\Projects\Text mining\output\most_freq_terms.png")
plt.show()
