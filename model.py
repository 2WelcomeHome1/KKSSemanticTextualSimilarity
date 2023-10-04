from keras.models import Sequential
from keras.layers import Embedding, Dense, Conv1D, GlobalAveragePooling1D, Dropout, MaxPooling1D
from keras.optimizers import Adam

class STSModel (Sequential):
    def __init__(self, vocab_size:int = 391410, max_sequence_length:int = 12, output_dim:int = 6) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.output_dim = output_dim
        pass

    def model(self):
        self.add(Embedding(self.vocab_size, self.output_dim, input_length=self.max_sequence_length))
        self.add(Conv1D(16, kernel_size=3, activation='relu',padding="same"))
        self.add(Conv1D(16, kernel_size=3, activation='relu',padding="same"))
        self.add(MaxPooling1D(pool_size=2))

        self.add(Conv1D(32, kernel_size=3, activation='relu',padding="same"))
        self.add(Conv1D(32, kernel_size=3, activation='relu',padding="same"))
        self.add(MaxPooling1D(pool_size=2))
        self.add(Dropout(0.4))

        self.add(Conv1D(64, kernel_size=3, activation='relu',padding="same"))
        self.add(Conv1D(64, kernel_size=3, activation='relu',padding="same"))
        self.add(MaxPooling1D(pool_size=2))
        self.add(Dropout(0.6))

        self.add(Conv1D(128, kernel_size=3, activation='relu',padding="same"))
        self.add(Conv1D(128, kernel_size=3, activation='relu',padding="same"))
        self.add(Dropout(0.8))

        self.add(GlobalAveragePooling1D())
        self.add(Dense(1, activation='sigmoid'))
        self.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

        return self