import tensorflow as tf
import numpy as np
import json
from encoder_model import CNN_Encoder
from decoder_model import RNN_Decoder
import efficientnet.tfkeras as eff
from efficientnet.tfkeras import preprocess_input

image_features_extract_model = None
encoder = None
decoder = None
def preprocess_image(image):
    img = tf.image.resize(image, (331, 331))
    img = preprocess_input(img)
    return img

def predict_caption(image,max_length,encoder,decoder,image_features_extract_model,tokenizer):

    attention_plot = np.zeros((max_length, 121))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(preprocess_image(image), 0)
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

def predict_caption_for_image(image):
    global encoder
    global decoder
    global image_features_extract_model
    BUFFER_SIZE = 1000
    embedding_dim = 256
    units = 512
    vocab_size = 25243
    max_length=47
    # Shape of the vector extracted from InceptionV3 is (64, 2048)
    # These two variables represent that vector shape
    features_shape = 1792
    attention_features_shape = 121
    # Feel free to change these parameters according to your system's configuration
    with open('./saved_tokenizer/tokenizer.json') as f:
        data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

    image_model = eff.EfficientNetB4(weights='noisy-student',include_top=False,input_shape=(331,331,3))
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    optimizer = tf.keras.optimizers.Adam()

    checkpoint_path = "./checkpoints_small/train"
    ckpt = tf.train.Checkpoint(encoder=encoder,
                            decoder=decoder,
                            optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
      # restoring the latest checkpoint in checkpoint_path
      ckpt.restore(ckpt_manager.latest_checkpoint)

    results,attention_plot = predict_caption(image,max_length,encoder,decoder,image_features_extract_model,tokenizer)
    return ' '.join(results)
