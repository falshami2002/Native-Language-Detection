import numpy as np
from collections import defaultdict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

window_size = 5

co_occurrence = defaultdict(lambda: defaultdict(int))

# Loop through the text and count co-occurrences within a window
for i, (word, tag) in enumerate(tagged_tokens):
    for j in range(i + 1, min(i + window_size, len(tagged_tokens))):
        word_j, tag_j = tagged_tokens[j]
        if tag != tag_j:  # Only count co-occurrences of different POS
            # Apply distance weight (closer words get higher weight)
            distance = j - i
            weight = 1 / (distance + 1)  # Weight by inverse of distance
            co_occurrence[tag][tag_j] += weight
            co_occurrence[tag_j][tag] += weight  # Make it symmetric

# Example co-occurrence matrix (simplified)
matrix = np.zeros((len(co_occurrence), len(co_occurrence)))

# Fill the matrix with the co-occurrence values
tags = list(co_occurrence.keys())
tag_to_index = {tag: idx for idx, tag in enumerate(tags)}

for tag1 in co_occurrence:
    for tag2 in co_occurrence[tag1]:
        i, j = tag_to_index[tag1], tag_to_index[tag2]
        matrix[i][j] = co_occurrence[tag1][tag2]

# Reshape matrix to be in the form (batch_size, height, width, channels)
matrix_reshaped = matrix.reshape(1, matrix.shape[0], matrix.shape[1], 1)  # Example for a single sentence

# Ensure the input is a NumPy array for processing by a CNN
matrix_reshaped = np.array(matrix_reshaped)

print(matrix_reshaped)

# Define CNN Model
model = Sequential()

# Example CNN architecture
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(matrix.shape[0], matrix.shape[1], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification (e.g., language detection)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model (assuming X_train and y_train are prepared)
# model.fit(X_train, y_train, epochs=10, batch_size=32)