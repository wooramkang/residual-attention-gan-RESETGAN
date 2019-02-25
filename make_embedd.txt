import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

np.random.seed(10)
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

# Compute a representation for each message, showing various lengths supported.
word = "king"
word2 = "queen"
word3 = "man"
word4 = "girl"

sentence = "I am red"
sentence2 = "I am blue"
sentence3 = "I am green"

messages = [word, word2, word3, word4, sentence, sentence2, sentence3]

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

list_embedding = []

with tf.Session() as session:
   session.run([tf.global_variables_initializer(), tf.tables_initializer()])
   message_embeddings = session.run(embed(messages))

   for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
       print("Message: {}".format(messages[i]))
       print("Embedding size: {}".format(len(message_embedding)))
       list_embedding.append(message_embedding)
       message_embedding_snippet = ", ".join(
           (str(x) for x in message_embedding[:3]))
       print("Embedding: [{}, ...]\n".format(message_embedding_snippet))
       # Compute a representation for each message, showing various lengths supported.

messages = ["That band rocks!", "That song is really cool."]

temp = np.array(list_embedding[0]) - np.array(list_embedding[2]) + np.array(list_embedding[3])

print(temp)
print(np.array(list_embedding[1]))

with tf.Session() as session:
   session.run([tf.global_variables_initializer(), tf.tables_initializer()])
   message_embeddings = session.run(embed(messages))
   print(message_embeddings)
   embed_size = embed.get_output_info_dict()['default'].get_shape()[1].value
   print(embed_size)