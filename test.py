import pandas as pd

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

df = pd.DataFrame(['This is really good',
				  'Every day is a new day',
				  ], columns=['text'])
print(df, end='\n\n')
MAX_NUM_WORDS = 3

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

tokenizer.fit_on_texts(df.text)
sequences = tokenizer.texts_to_sequences(df.text)
print(sequences, end='\n\n')

padded = pad_sequences(sequences, maxlen=3, padding='post')
print('After Padding: \n',padded, end='\n\n')

print(tokenizer.word_index)
print(len(tokenizer.word_index))

num_words = min(MAX_NUM_WORDS, len(tokenizer.word_index) + 1)
embedding_matrix = np.zeros((num_words, 10))
print(embedding_matrix.shape)
