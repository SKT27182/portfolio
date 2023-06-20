import tensorflow as tf
import numpy as np
import tensorflow_hub as hub

encoder_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1", input_shape=[], dtype=tf.string)



class MemNN:

    def __init__(self, max_story_len, max_query_len=512, vocab_size=1000, embedding_size=512):
        self.max_story_len = max_story_len
        self.max_query_len = max_query_len
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

    def embedding(self, inputs, query=False):

        x = inputs

        if query:
            for i in range(len(x)):
                x[i] = encoder_layer(x[i]) # return a 512 dim vector
        else:
            x = encoder_layer(x)

        return x
    
    def I(self, inputs):

        # inputs: story, query

        story, query = inputs

        embedded_story = self.embedding(story)

        embedded_query = self.embedding(query, query=True)

        return embedded_story, embedded_query
    
    def G(self, embedded_story):

        # store the story in memory and return the memory

        memory = tf.keras.layers.LSTM(self.embedding_size)(embedded_story)

        return memory
    
    def O(self, memory, embedded_querry, k):

        # inputs: memory, embedded_query, k

        # k is the number of hops

        output = embedded_querry

        for i in range(k):

            # match the output with the memory, return the index of the memory that matches the output the most

            similarity = tf.keras.layers.dot([output, memory], axes=-1)

            similarity = tf.keras.layers.Activation('softmax')(similarity)

            # add the memory to the output with max similarity
            max_mem = memory[tf.argmax(similarity)]

            output = tf.keras.layers.add([output, max_mem])

        return output
    
    def R(self, output):
        
        # inputs: output, memory

        answer = tf.keras.layers.LSTM(self.vocab_size)(output)
        answer = tf.keras.layers.Dense(self.vocab_size, activation='softmax')(answer)

        return output
    
    def model(self, k=3):
        
        # inputs: story, query

        story = tf.keras.Input(shape=[], dtype=tf.string, name='story')
        query = tf.keras.Input(shape=[], dtype=tf.string, name='query')

        embedded_story, embedded_query = self.I([story, query])

        # do the posional encoding here to the embedded_story

        memory = self.G(embedded_story)

        output = self.O(memory, embedded_query, k=k)

        answer = self.R(output)


        model = tf.keras.Model(inputs=[story, query], outputs=answer)

        self.model = model
    
    def compile(self, model, optimizer, loss, metrics):

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        
    def fit(self, model, story, query, answer, epochs, batch_size, validation_split, **kwargs):

        self.model.fit([story, query], answer, epochs=epochs, batch_size=batch_size, validation_split=validation_split, **kwargs)

    def predict(self, story, query):

        return self.model.predict([story, query])