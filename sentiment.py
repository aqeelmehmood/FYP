import numpy as np 
import pandas as pd  
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.metrics import roc_curve, auc
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix
from nltk.tokenize import RegexpTokenizer
import re
import string
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import warnings
warnings.filterwarnings("ignore")
# %matplotlib inline

#%%

data = pd.read_csv("Data1.csv", encoding = "ISO-8859-1", engine="python")


#%%
data

#%%
data.columns = ["label", "time", "date", "query", "username", "text"]


#%%
print('lenght of data is', len(data))
#%%
data.info()

#%%
data=data[['text','label']]


#%%
data['label'][data['label']==4]=1


#%%
data_pos = data[data['label'] == 1]
data_neg = data[data['label'] == 0]


#%%
data_pos = data_pos.iloc[:int(20000)]
data_neg = data_neg.iloc[:int(20000)]



#%%
data = pd.concat([data_pos, data_neg])


#%%
data['text']=data['text'].str.lower()


#%%
data['text'].tail()


#%%
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_list = stopwords.words('english')


#%%
from nltk.corpus import stopwords
", ".join(stopwords.words('english'))
#%%
STOPWORDS = set(stopwords.words('english'))
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

#%%
data['text'] = data['text'].apply(lambda text: cleaning_stopwords(text))

#%%
data['text'].head()

#%%
english_punctuations = string.punctuation
punctuations_list = english_punctuations

#%%
def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

#%%
data['text']= data['text'].apply(lambda x: cleaning_punctuations(x))

#%%
data['text'].tail()

#%%
def cleaning_URLs(data):
    return re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',data)
#%%
data['text'] = data['text'].apply(lambda x: cleaning_URLs(x))


#%%
data['text'].tail()


#%%
def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)

#%%
data['text'] = data['text'].apply(lambda x: cleaning_numbers(x))
data['text'].tail()



#%%
tokenizer = RegexpTokenizer(r'\w+')
data['text'] = data['text'].apply(tokenizer.tokenize)


#%%

data['text'].head()

#%%
st = nltk.PorterStemmer()
def stemming_on_text(data):
    text = [st.stem(word) for word in data]
    return data

#%%
data['text']= data['text'].apply(lambda x: stemming_on_text(x))

#%%
data['text'].head()


#%%
import nltk 
nltk.download('wordnet')
lm = nltk.WordNetLemmatizer()

#%%
def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return data
#%%
data['text'] = data['text'].apply(lambda x: lemmatizer_on_text(x))

#%%
data['text'].head()

#%%
X=data.text
y=data.label

#%%
max_len = 500
tok = Tokenizer(num_words=2000)
tok.fit_on_texts(X)
sequences = tok.texts_to_sequences(X)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
#%%
sequences_matrix.shape

#%%
data = pd.read_csv("commentsPython.csv", encoding = "ISO-8859-1", engine="python")

X_train, X_test, Y_train, Y_test = train_test_split(sequences_matrix, y, test_size=data, random_state=2)

#%%
def tensorflow_based_model(): 
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(2000,50,input_length=max_len)(inputs) 
    layer = LSTM(64)(layer) 
    layer = Dense(256,name='FC1')(layer) 
    layer = Activation('relu')(layer) 
    layer = Dropout(0.5)(layer) 
    layer = Dense(1,name='out_layer')(layer) 
    layer = Activation('sigmoid')(layer) 
    model = Model(inputs=inputs,outputs=layer) 
    return model

#%%
model = tensorflow_based_model() 
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])


#%%

history=model.fit(X_train,Y_train,batch_size=80,epochs=6, validation_split=0.1)
print('Training finished !!')

#%%%
accr1 = model.evaluate(X_test,Y_test)


#%%
print('Test set\n  Accuracy: {:0.2f}'.format(accr1[1]))

#%%
y_pred = model.predict(X_test) 
y_pred = (y_pred > 0.5)

#%%
X_test.shape

#%%
print('\n')
print("confusion matrix")
print('\n')
CR=confusion_matrix(Y_test, y_pred)
print(CR)
print('\n')
#%%
fig, ax = plot_confusion_matrix(conf_mat=CR,figsize=(10, 10),
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
plt.show()



#%%
######################### Model create
 #how to save model
model_json = model.to_json()
with open("model_aqeel.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_aqeel.h5")
print("Saved model to disk")


#%%

#how to read model
# load json and create model
from keras.models import model_from_json
json_file = open('model_aqeel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_aqeel.h5")
print("Loaded model from disk")

#%%
comment = np.array(["he is a good boy"]) ##
ser = pd.Series(comment)##
ser=ser.str.lower() ##

tokenizer = RegexpTokenizer(r'\w+')
ser = ser.apply(tokenizer.tokenize)
ser

ser = ser.apply(lambda x: stemming_on_text(x))
ser

ser = ser.apply(lambda x: lemmatizer_on_text(x))
ser

max_len = 500
tok = Tokenizer(num_words=2000)
tok.fit_on_texts(ser)
ser1 = tok.texts_to_sequences(ser)
commentSer = sequence.pad_sequences(ser1,maxlen=max_len)
y_pred = loaded_model.predict(commentSer) 

#%%

print(y_pred)
if y_pred>0.66:
    print("good comment")
elif y_pred>0.5 and y_pred<0.65:
    print("Neutral Comments")
elif y_pred<0.4:
    print("Negative Comment")
#y_pred = (y_pred > 0.5) 
#y_pred
#%%

y_pred = model.predict(X_test) 
fpr, tpr, thresholds = roc_curve(Y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()

































