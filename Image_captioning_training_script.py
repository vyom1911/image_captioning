# ## Import libraries

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals


# In[2]:


import tensorflow as tf
import efficientnet.tfkeras as eff
from efficientnet.tfkeras import preprocess_input

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm.notebook import tqdm
import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle


# In[3]:


# to plot an image
from matplotlib import pyplot as plt
def plot_image(name):
    filename = "gs://dataset_collection/coco/train2014/" + name
    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image,channels=3)
    plt.imshow(image)


# ## Reading captions and image paths

# Import the Google Cloud client library and JSON library
from google.cloud import storage
import json

# Instantiate a Google Cloud Storage client and specify required bucket and file
storage_client = storage.Client()
bucket = storage_client.get_bucket('dataset_collection')
blob = bucket.blob('coco/annotations/captions_train2014.json')

# Download the contents of the blob as a string and then parse it using json.loads() method
annotations = json.loads(blob.download_as_string(client=None))

image_gcs_path = "gs://dataset_collection/coco/train2014/"

npy_files_path = "/home/jupyter/image_npy/"


# Store captions and image names in vectors
all_captions = []
all_img_name_vector = []
filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

for annot in annotations['annotations']:
    text=annot['caption']
    for f in filters:
        text = text.replace(f,'')
    caption = '<start> ' + text.lower() + ' <end>'
    image_id = annot['image_id']  
    img_filename = 'COCO_train2014_' + '%012d.jpg' % (image_id)
    
    if img_filename == "COCO_train2014_000000281091.jpg":
        print("Skipping:","COCO_train2014_000000281091.jpg")
        continue
    all_img_name_vector.append(img_filename)
    all_captions.append(caption)


# Shuffle captions and image_names together
# Set a random state
train_captions, img_name_vector = shuffle(all_captions,
                                          all_img_name_vector,
                                          random_state=1)

# # Select the first 10000 captions from the shuffled set
# num_examples = 30000
# train_captions = train_captions[:num_examples]
# img_name_vector = img_name_vector[:num_examples]


# ## Extracting image Features and saving as npy

# In[8]:


def load_image(image_path):
    img = tf.io.read_file(image_gcs_path+image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (331, 331))
    img = preprocess_input(img)
    return img, image_path


# In[9]:


image_model = eff.EfficientNetB4(weights='noisy-student',include_top=False,input_shape=(331,331,3))
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


# In[10]:


# Get unique images
encode_train = sorted(set(img_name_vector))

# Feel free to change batch_size according to your system configuration
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(
  load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(8)


# In[11]:


from tensorflow.python.lib.io import file_io


# In[12]:


total_imgs = int(len(encode_train)/8)+1
for img, path in tqdm(image_dataset,to`tal=total_imgs):
    batch_features = image_features_extract_model(img)
    batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))
  
    
    for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        np.save(npy_files_path+path_of_feature, bf.numpy())


def to_vocabulary(captions):
    # build a list of all description strings
    all_captions = set()
    for k in captions:
        [all_captions.update(k.split())]
    return all_captions


# In[14]:


# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)


# In[15]:


top_k = len(to_vocabulary(train_captions))
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
train_seqs = tokenizer.texts_to_sequences(train_captions)
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'
#Create the tokenized vectors
train_seqs = tokenizer.texts_to_sequences(train_captions)


# In[16]:


tokenizer_json = tokenizer.to_json()
with open('./saved_tokenizer/tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))


# ##--- GPT2 ENCODING ---
# from gpt2.src import encoder
# enc = encoder.get_encoder('124M','gpt2_models')

# train_seqs = [enc.encode(c) for c in train_captions]

# In[17]:


# Pad each vector to the max_length of the captions
# If you do not provide a max_length value, pad_sequences calculates it automatically
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')


# In[18]:


# Calculates the max_length, which is used to store the attention weights
max_length = calc_max_length(train_seqs)


# In[19]:


# Create training and validation sets using an 80-20 split
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                    cap_vector,
                                                                    test_size=0.01,
                                                                    random_state=0)


# In[20]:


print("total train images:" ,len(cap_train),"\ntotal validation images:", len(cap_val))


# In[21]:


# Feel free to change these parameters according to your system's configuration
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = top_k + 1
num_steps = len(img_name_train) // BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = hidden_layer.shape[3]
attention_features_shape = hidden_layer.shape[1]*hidden_layer.shape[2]


# In[22]:


print("vocab_size:",vocab_size,"\nfeatures_shape:",features_shape,"\nattention_features_shape:",attention_features_shape)


# In[23]:


# Load the numpy files
def map_func(img_name, cap):
    img_tensor = np.load(npy_files_path+img_name.decode('utf-8')+'.npy')
    return img_tensor, cap


# In[24]:


dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# In[25]:


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# In[26]:


class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


# In[27]:


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size*2, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


# In[28]:


encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)


# In[29]:


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


# In[30]:


checkpoint_path = "./checkpoints_small/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)


# In[31]:


start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint)


# In[32]:


print("start epoch:",  start_epoch)


# In[33]:


# adding this in a separate cell because if you run the training cell
# many times, the loss_plot array will be reset
loss_plot = []


# In[34]:


@tf.function
def train_step(img_tensor, target):
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


# In[35]:


EPOCHS = 100

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0

    for (batch, (img_tensor, target)) in enumerate(tqdm(dataset,total=1+len(cap_train)//BATCH_SIZE)):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss

        if batch % 1000 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(
              epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / num_steps)

    #if epoch % 5 == 0:
    ckpt_manager.save()

    print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                         total_loss/num_steps))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


# In[ ]:


plt.plot(loss_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.savefig('loss_plot_effb4_noisy.png')
plt.show()


# In[ ]:


def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


# In[ ]:


def plot_attention(image, result, attention_plot):
    filename = "gs://dataset_collection/coco/train2014/" + image
    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image,channels=3)
    
    temp_image = np.array(image)
    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


# In[ ]:


# captions on the validation set
rid = np.random.randint(0, len(img_name_val))
image = img_name_val[rid]
real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
result, attention_plot = evaluate(image)

print ('Real Caption:', real_caption)
print ('Prediction Caption:', ' '.join(result))
plot_attention(image, result, attention_plot)


# In[ ]:


plot_image(image)


# In[ ]:


get_ipython().system('zip -r checkpoints_small.zip checkpoints_small ')


# ### Predicting captions for all images

# In[ ]:


# captions on the validation set
#rid = np.random.randint(0, len(img_name_val)) # -- Just picking 1 image
rid=None
predicted_captions = {}
real_caption = None
for rid,image in tqdm(enumerate(img_name_val)):
    real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
    result, attention_plot = evaluate(image)
    predicted_captions[image.split("/")[-1]] = result

print ('Real Caption:', real_caption)
print ('Prediction Caption:', ' '.join(result))
#plot_attention(image, result, attention_plot)


# In[ ]:


with open('predicted_caption_effnetb4.json','w') as j:
    json.dump(predicted_captions,j)


# ### Loading Predicted captions and creating reference and candidate for BLEU Score

# In[ ]:


with open("predicted_caption_effnetb4.json", 'r') as f:
    predicted_captions = json.load(f)


# In[ ]:


def modify_name(x):
    return x.split("/")[-1]


# In[ ]:


bleu_score_list = {}
img_set = set([modify_name(img_name) for img_name in img_name_val])
for annot in annotations["annotations"]:
    img_name = 'COCO_train2014_' + '%012d.jpg' % (annot["image_id"])
    if img_name in img_set:
        if img_name in bleu_score_list:
            bleu_score_list[img_name]["real"].append(annot["caption"].split())
        else:
            bleu_score_list[img_name] = {"predicted":predicted_captions[img_name]
                                            ,"real":[annot["caption"].split()]}


# In[ ]:


with open('predicted_caption_efficientnetB4_with_real_captions.json','w') as j:
    json.dump(bleu_score_list,j)


# In[ ]:


### Calculating BLEU Scores


# In[ ]:


from nltk.translate.bleu_score import sentence_bleu


# In[ ]:


bleu_scores_for_images = {}


# In[ ]:


with open("predicted_caption_efficientnetB4_with_real_captions.json", 'r') as f:
    bleu_scores_for_images = json.load(f)


# In[ ]:


for img, captions in bleu_score_list.items():
    reference = captions["real"]
    candidate = [ x for x in captions["predicted"] if x not in ["<start>","<end>","<unk>"] ]
    
    bleu_scores_for_images[img] = sentence_bleu(reference, candidate,weights=(1,0,0,0))


# In[ ]:


avg_bleu = 0
for k,v in bleu_scores_for_images.items():
    avg_bleu=avg_bleu+v
avg_bleu = avg_bleu/len(bleu_scores_for_images)


# In[ ]:


avg_bleu


# In[ ]:


print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))
