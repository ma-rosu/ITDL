import numpy as np
import os
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.saving.save import load_model # Used for loading a pre-trained model for evaluation

# Path for  data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect (these define the output classes of the model)
actions = np.array(['sitting', 'sittingbed', 'notfalling', 'falling'])

# Number of sequences (videos) expected for each action during data loading
no_sequences = 60

# Each sequence (video) is composed of this many frames (keypoint sets)
sequence_length = 90


def get_train_test_split():
    """
    Loads keypoint data from disk, preprocesses it, and splits it into
    training and testing sets.

    Returns:
        tuple: X_train, X_test, y_train, y_test - numpy arrays for training and testing data.
    """
    # Create a mapping from action names to numerical labels
    label_map = {label: num for num, label in enumerate(actions)}

    sequences, labels = [], []
    # Iterate through each action to load its corresponding keypoint sequences
    for action in actions:
        # Iterate through each sequence (video) folder for the current action
        # Convert folder names to integers for sorting/iteration
        for sequence_num in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
            window = []
            # Load each frame's keypoints within the current sequence
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence_num), "{}.npy".format(frame_num)))
                window.append(res) # Add the frame's keypoints to the current sequence window
            sequences.append(window) # Add the completed sequence window to the list of all sequences
            labels.append(label_map[action]) # Append the numerical label for the current action

    # Convert lists to numpy arrays
    X = np.array(sequences)
    # Convert labels to one-hot encoded format (e.g., 0 -> [1,0,0,0], 1 -> [0,1,0,0])
    y = to_categorical(labels).astype(int)

    # Split the data into training and testing sets (95% training, 5% testing)
    return train_test_split(X, y, test_size=0.05)


# Prepare the data for training and testing
X_train, X_test, y_train, y_test = get_train_test_split()


# Setup TensorBoard callback for visualizing training progress
log_dir = os.path.join('Logs') # Directory where TensorBoard logs will be saved
tb_callback = TensorBoard(log_dir=log_dir)


def train():
    """
    Defines, compiles, and trains the LSTM neural network model.
    """
    model = Sequential()
    # First LSTM layer: processes sequences, outputs sequences for the next LSTM
    # Input shape: (sequence_length, number_of_keypoints_per_frame)
    # 1662 is derived from (33*4 for pose + 468*3 for face + 21*3 for left hand + 21*3 for right hand)
    model.add(LSTM(64, return_sequences=True, input_shape=(sequence_length, 1662)))
    model.add(Dropout(0.1)) # Dropout layer to prevent overfitting
    # Second LSTM layer: processes sequences, outputs sequences
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.1)) # Dropout layer
    # Third LSTM layer: processes sequences, but outputs only the last state (not sequences)
    model.add(LSTM(64, return_sequences=False))
    # Dense layers (fully connected) for classification
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    # Output layer: number of neurons equals the number of actions, softmax for probability distribution
    model.add(Dense(actions.shape[0], activation='softmax'))

    # Compile the model with Adam optimizer, categorical crossentropy loss (for multi-class classification)
    # and categorical accuracy as the metric
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # EarlyStopping callback: stops training if validation loss doesn't improve for 30 epochs
    # and restores the best weights found during training
    early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

    # Train the model
    # validation_split=0.2 means 20% of X_train, y_train will be used for validation during training
    model.fit(X_train, y_train, epochs=500, callbacks=[tb_callback, early_stop], validation_split=0.2)

    # Save the trained model in TensorFlow SavedModel format
    model.save('action_detection_model', save_format='tf')


def load_saved_model():
    """
    Loads a previously saved Keras model.
    """
    # Ensure the model file name matches what was saved in the train function
    return load_model('action_detection_model', compile=False)


# Load the trained model
model = load_saved_model()


def test():
    """
    Evaluates the loaded model on the test dataset and prints prediction results
    against actual labels.
    """
    res = model.predict(X_test) # Make predictions on the test set

    # Iterate through each prediction and compare it with the actual label
    for i in range(len(res)):
        predicted_action_index = np.argmax(res[i])
        actual_action_index = np.argmax(y_test[i])

        # Print True if prediction matches actual, False otherwise
        print(True) if actions[predicted_action_index] == actions[actual_action_index] else print(False)
        print('{')
        print('\t', 'Prediction:', actions[predicted_action_index]) # Predicted action name
        print('\t', 'Actual:', actions[actual_action_index]) # Actual action name
        print('}\n')


# Call the training function only when you want to train and save the model
train()

# Call the test function to evaluate the trained model only when you want to test the model
test()