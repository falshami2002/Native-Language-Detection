import pandas as pd
from tokenize_text import TokenizeText
from proximity_array import Proximity
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, Dropout, Input, Embedding, LSTM, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Concatenate, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import class_weight

POS_TAGS = [
    'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD',
    'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR',
    'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
    'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '$', "''", '(', ')', ',', '--', '.',
    ':', '``', '#'
]
pos_to_int = {tag: idx + 1 for idx, tag in enumerate(POS_TAGS)}

df = pd.read_csv('ef_POStagged_original_corrected.csv')
df = df.dropna(subset=["original", "nationality"])
print(df['nationality'].value_counts())

# Encode Labels
label_encoder = LabelEncoder()
df["encoded_label"] = label_encoder.fit_transform(df["nationality"])
num_classes = len(label_encoder.classes_)

# Split beforehand to avoid shuffling confusion in generator
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Tokenizer + Proximity objects (shared across generator)
tokenizer = TokenizeText()
proximity = Proximity()

def tf_data_wrapper(df_slice, batch_size):
    def gen():
        for start in range(0, len(df_slice), batch_size):
            end = min(start + batch_size, len(df_slice))
            batch_texts = df_slice.iloc[start:end]["original"]
            batch_labels = df_slice.iloc[start:end]["encoded_label"]

            batch_arrays = []
            batch_pos_seqs = []
            batch_ys = []

            for text, label in zip(batch_texts, batch_labels):
                # Co-occurrence matrix
                matrix = proximity.getNumpyArray(proximity.getProximityArray(tokenizer.tokenize(text)))
                matrix = matrix / np.max(matrix)
                batch_arrays.append(matrix[..., np.newaxis])  # add channel dim

                # POS sequence → tag list → int IDs
                pos_seq = tokenizer.tokenize(text)
                pos_seq_encoded = [pos_to_int.get(tag, 0) for tag in pos_seq]
                batch_pos_seqs.append(pos_seq_encoded)
                batch_ys.append(label)

            # Pad POS sequences
            padded_seqs = tf.keras.preprocessing.sequence.pad_sequences(batch_pos_seqs, padding='post')

            X_batch = np.array(batch_arrays, dtype=np.float32)
            Y_batch = np.array(batch_ys, dtype=np.int32)
            
            yield ({'pos_sequence': padded_seqs, 'pos_matrix': X_batch}, Y_batch)

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            {
                'pos_sequence': tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                'pos_matrix': tf.TensorSpec(shape=(16, 46, 46, 1), dtype=tf.float32),
            },
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )

# POS Sequence Input (variable-length)
pos_seq_input = Input(shape=(None,), name='pos_sequence')
embedding = Embedding(len(POS_TAGS) + 2, 32, mask_zero=True, name='embedding')(pos_seq_input)
lstm_out = Bidirectional(LSTM(64))(embedding)

# Co-occurence matrix input
pos_matrix_input = Input(shape=(46, 46, 1), name='pos_matrix')  # Add channel dim for Conv
x = Conv2D(32, (3, 3), activation='relu', padding='same')(pos_matrix_input)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = GlobalMaxPooling2D()(x)

# === Combine both inputs ===
num_classes = len(label_encoder.classes_)
combined = Concatenate()([lstm_out, x])
dense = Dense(64, activation='relu')(combined)
output = Dense(num_classes, activation='softmax')(dense)

# === Build and compile the model ===
model = Model(inputs=[pos_seq_input, pos_matrix_input], outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Parameters
batch_size = 16
steps_per_epoch = len(train_df) // batch_size
validation_steps = len(test_df) // batch_size

# Fit model using generators
train_data = tf_data_wrapper(train_df, batch_size)
val_data = tf_data_wrapper(test_df, batch_size)

# Balance nationalities
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(df['nationality']), y=df['nationality'])
class_weights_dict = dict(enumerate(class_weights))

model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    steps_per_epoch=len(train_df) // batch_size,
    validation_steps=len(test_df) // batch_size,
    class_weight=class_weights_dict
)