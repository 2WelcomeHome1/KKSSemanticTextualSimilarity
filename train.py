import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from model import STSModel
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

class TrainModel():
    def __init__(self) -> None:
        self.tokenizer = Tokenizer()
        pass
    
    def load_data(self, trainset_path: str ,validset_path:str, testset_path: str):
        df_train = pd.read_csv(trainset_path, sep=',', encoding='cp1251')
        df_valid = pd.read_csv(validset_path, sep=',', encoding='cp1251')
        df_test = pd.read_csv(testset_path, sep=',', encoding='cp1251')
        return df_train, df_valid, df_test

    def create_vocab(self, texts):    # Создание словаря для преобразования слов в числа
        self.tokenizer.fit_on_texts(texts)
        return len(self.tokenizer.word_index) + 1

    def get_max_seq_length(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        max_sequence_length = max(len(seq) for seq in sequences)
        return max_sequence_length, sequences

    def pad_seq(self, sequences, max_sequence_length):
        return pad_sequences(sequences, maxlen=max_sequence_length)

    def preprocess(self, texts):
        vocab_size = self.create_vocab(texts)
        max_sequence_length, sequences = self.get_max_seq_length(texts)
        return self.pad_seq(sequences, max_sequence_length), vocab_size, max_sequence_length

    def preprocess_data(self, df_train, df_valid, df_test):
        df_train["combined_text"] = df_train["Исходные данные"] + " " + df_train["Референс"]
        df_valid["combined_text"] = df_valid["Исходные данные"] + " " + df_valid["Референс"]
        df_test["combined_text"] = df_test["Исходные данные"] + " " + df_test["Референс"]
        train, vocab_size, max_sequence_length = self.preprocess(np.array(df_train['combined_text']))
        valid, vocab_size, max_sequence_length = self.preprocess(np.array(df_valid['combined_text']))
        test, vocab_size, max_sequence_length = self.preprocess(np.array(df_test['combined_text']))
        return train, valid, test, vocab_size, max_sequence_length
    
    def get_labels(self, df_train, df_valid, df_test):
        train_labels = np.array(df_train['Классификатор'])
        valid_labels = np.array(df_valid['Классификатор'])
        test_labels = np.array(df_test['Классификатор'])
        return train_labels, valid_labels, test_labels
    
    def _visualise_result(self, model):
        plt.figure(0)
        plt.plot(model.history.history['accuracy'], label='training accuracy')
        plt.plot(model.history.history['val_accuracy'], label='validation accuracy')
        plt.title('Accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()


        plt.figure(1)
        plt.plot(model.history.history['loss'], label='training loss')
        plt.plot(model.history.history['val_loss'], label='val loss')
        plt.title('Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

    def create_model(self, vocab_size, max_sequence_length):
        return STSModel(vocab_size, max_sequence_length).model()

    def run_train(self, model, train, train_labels, valid, valid_labels, epochs:int = 10, batch_size:int = 20):
        Checkpoint = tf.keras.callbacks.ModelCheckpoint("saved_models/checkpoints/model-{epoch:02d}-{val_accuracy:.4f}.h5", save_best_only=True, monitor="val_accuracy", save_weights_only=True)
        Best_Checkpoint = tf.keras.callbacks.ModelCheckpoint("saved_models/best_model.h5", save_best_only=True, save_weights_only=True, monitor="val_accuracy")
        model.fit(train, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(valid, valid_labels), callbacks = [Checkpoint, Best_Checkpoint])
        return model

if __name__ == "__main__":
    ModelCoach  = TrainModel()
    print('Загружаем данные...')
    df_train, df_valid, df_test = ModelCoach.load_data('./Data/DataSet.csv', './Data/validset.csv', './Data/testset.csv')
    print('Выполняем предобработку данных...')
    train, valid, test, vocab_size, max_sequence_length =  ModelCoach.preprocess_data(df_train, df_valid, df_test)
    train_labels, valid_labels, test_labels= ModelCoach.get_labels(df_train, df_valid, df_test)
    print('Загружаем модель...')
    model = ModelCoach.create_model(vocab_size, max_sequence_length)
    model.summary()
    print('Начинаем обучение...')
    model = ModelCoach.run_train(model, train, train_labels, valid, valid_labels, epochs = 20, batch_size=25)
    print('Готово !')
    # model.save_weights('weights.h5')
    ModelCoach._visualise_result(model)