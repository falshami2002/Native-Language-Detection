import numpy as np
from collections import defaultdict

ALL_TAGS = [
    'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD',
    'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR',
    'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
    'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '$', "''", '(', ')', ',', '--', '.',
    ':', '``', '#'
]

class Proximity:
    def __init__(self):
        self.window_size = 5
        self.all_tags = ALL_TAGS
        self.tag_to_index = {tag: i for i, tag in enumerate(self.all_tags)}
    def getProximityArray(self, pos):
        coOccurrence = defaultdict(lambda: defaultdict(int))
        for i in range(len(pos)):
            for j in range(i + 1, min(i + self.window_size, len(pos))):
                jPos = pos [j]
                iPos = pos [i]
                if iPos != jPos:  # Only count co-occurrences of different POS
                    # Apply distance weight (closer words get higher weight)
                    distance = j - i
                    weight = 1 / (distance + 1)  # Weight by inverse of distance
                    coOccurrence[iPos][jPos] += weight
                    coOccurrence[jPos][iPos] += weight  # Make it symmetric
        return coOccurrence
    
    def getNumpyArray(self, coOccurrence):
        size = len(self.all_tags)
        matrix = np.zeros((size, size), dtype=np.float32)

        for tag1 in coOccurrence:
            for tag2 in coOccurrence[tag1]:
                i, j = self.tag_to_index[tag1], self.tag_to_index[tag2]
                matrix[i][j] = coOccurrence[tag1][tag2]

        # Reshape matrix (height, width, channels)
        matrix_reshaped = matrix.reshape(matrix.shape[0], matrix.shape[1]) 

        matrix_reshaped = np.array(matrix_reshaped)
        return matrix_reshaped