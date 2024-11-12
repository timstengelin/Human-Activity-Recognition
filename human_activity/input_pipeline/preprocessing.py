import os
import gin
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import stats


@gin.configurable
def create_tfrecord_files(data_dir, window_size, window_shift):
    '''
    Creates TFRecord files of original files

    Args:
        data_dir (string): directory where the original data is stored
        window_size: length of the sliding window
        window_shift: amount of shift/slide of the sliding window
    '''


    def load_labels(data_dir):
        '''
        Loads and organizes labels

        Args:
            data_dir (string): directory where the original data is stored

        Returns:
            experiment_labels_list (list):
                A list of 61 sublists, where each sublist contains labels for a specific experiment
        '''

        # Load the labels file using pandas
        labels_dataframe = pd.read_csv("{}/HAPT_dataset/RawData/labels.txt".format(data_dir), sep=' ', header=None)

        # Create empty list
        experiment_labels_list = [[] for _ in range(61)]

        # Iterate over each label entry and append it to the corresponding experiment's sublist
        for _, label in labels_dataframe.iterrows():
            experiment_number_id = label[0] # Extract the experiment ID
            idx = experiment_number_id - 1  # Convert 1-based ID to 0-based index
            experiment_labels_list[idx].append(label.tolist())

        return experiment_labels_list


    def load_data_files(data_dir):
        '''
        Retrieves accelerometer and gyroscope files

        Args:
            data_dir (string): directory where the original data is stored

        Returns:
            acc_files (list): A list of accelerometer file names
            gyro_files (list): A list of gyroscope file names
        '''

        # Get a list of all files in the raw data directory that start with 'acc' (for accelerometer)
        acc_files = sorted([f for f in os.listdir('{}/HAPT_dataset/RawData/'.format(data_dir)) if f.startswith('acc')])

        # Get a list of all files in the raw data directory that start with 'gyro' (for gyroscope)
        gyro_files = sorted([f for f in os.listdir('{}/HAPT_dataset/RawData/'.format(data_dir)) if f.startswith('gyro')])

        return acc_files, gyro_files


    def normalize_data(acc_data, gyro_data):
        '''
        Normalizes accelerometer and gyroscope data using Z-score normalization

        Args:
            acc_data (pandas.DataFrame): A DataFrame containing accelerometer data with columns 'x', 'y', 'z'
            gyro_data (pandas.DataFrame): A DataFrame containing gyroscope data with columns 'x', 'y', 'z'

        Returns:
            (numpy.ndarray): 2D array with 6 columns representing the 6 features and each row representing a data point
        '''

        # Normalize accelerometer data for each axis ('x', 'y', 'z') using Z-score
        normalized_acc_x = stats.zscore(acc_data['x'])
        normalized_acc_y = stats.zscore(acc_data['y'])
        normalized_acc_z = stats.zscore(acc_data['z'])

        # Normalize gyroscope data for each axis ('x', 'y', 'z') using Z-score
        normalized_gyro_x = stats.zscore(gyro_data['x'])
        normalized_gyro_y = stats.zscore(gyro_data['y'])
        normalized_gyro_z = stats.zscore(gyro_data['z'])

        # Combine the normalized accelerometer and gyroscope data into a single 2D array
        return np.array([normalized_acc_x, normalized_acc_y, normalized_acc_z,
                             normalized_gyro_x, normalized_gyro_y, normalized_gyro_z]).T


    def assign_activitynumberids_to_array(experiment_length, experiment_labels):
        '''
        Assign activity number ids to an array of the experiment length

        Args:
            experiment_length (int): The length/number of data points in an experiment
            experiment_labels (list): The labels for a specific experiment

        Returns:
            labels_sequence (): An array of the experiment length with assigned activity number ids
        '''

        labels_sequence = np.zeros(experiment_length, dtype=int)

        # Loop over all experiment labels
        for experiment_label in experiment_labels:
            label_start_point, label_end_point, activity_number_id\
                = experiment_label[3], experiment_label[4], experiment_label[2]

            # Assign the activity number id to the positions in 'labels' from label start point to label end point
            labels_sequence[label_start_point:label_end_point + 1] = activity_number_id

        return labels_sequence


    def split_features_and_labels(acc_files, gyro_files, experiment_labels_list, data_dir):
        '''
        Splits the features and labels for accelerometer and gyroscope data based on experiment IDs into
        training, validation, and test sets

        Args:
            acc_files (list): A list of accelerometer file names
            gyro_files (list): A list of gyroscope file names
            experiment_labels_list (list):
                A List of 61 sublists, where each sublist contains labels for a specific experiment
            data_dir (string): directory where the original data is stored

        Returns:
            train_features (numpy.ndarray): An array of feature data for training
            train_labels (numpy.ndarray): An array of label data corresponding to features for training
            val_features (numpy.ndarray): An array of feature data for validation
            val_labels (numpy.ndarray): An array of label data corresponding to features for validation
            test_features (numpy.ndarray): An array of feature data for testing
            test_labels (numpy.ndarray): An array of label data corresponding to features for testing
        '''

        def load_and_process_data(experiment_id):
            '''
            Loads accelerometer and gyroscope data for a specific experiment
            '''

            # Load file names for specific experiment
            acc_file = acc_files[experiment_id - 1]
            gyro_file = gyro_files[experiment_id - 1]

            # Load data for a specific experiment
            acc_data = pd.read_csv(
                '{}/HAPT_dataset/RawData/{}'.format(data_dir, acc_file), sep=' ', names=['x', 'y', 'z'])
            gyro_data = pd.read_csv(
                '{}/HAPT_dataset/RawData/{}'.format(data_dir, gyro_file), sep=' ', names=['x', 'y', 'z'])

            # Normalize data
            data = normalize_data(acc_data, gyro_data)

            # Assign activity number ids to an array of the experiment length
            labels_sequence = assign_activitynumberids_to_array(len(acc_data), experiment_labels_list[experiment_id - 1])

            return data, labels_sequence

        def split_data(experiment_ids):
            '''
            Splits data and labels according to experiment ranges for predefined split of the dataset
            '''
            data_part, labels_sequence_part = zip(*[load_and_process_data(experiment_id) for experiment_id in experiment_ids])
            return np.concatenate(data_part), np.concatenate(labels_sequence_part)

        return split_data(train_experiment_ids), split_data(val_experiment_ids), split_data(test_experiment_ids)


    def create_tfrecord_dataset(features, labels, window_size, window_shift):
        '''
        Creates a TensorFlow dataset from features and labels using a sliding window approach

        Args:
            features (numpy.ndarray): An array of feature data
            labels (numpy.ndarray): An array of label data
            window_size (int): Length of the sliding window
            window_shift (int): Amount of shift/slide of the sliding window

        Returns:
            (tf.data.Dataset): A TensorFlow dataset with sliding windows of features and labels
        '''

        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))

        # Apply a sliding window to the dataset
        dataset_windows = dataset.window(size=window_size, shift=window_shift, drop_remainder=True)

        # Flatten the dataset of windows
        dataset_flattened_windows = dataset_windows.flat_map(
            lambda x, y: tf.data.Dataset.zip((x.batch(window_size), y.batch(window_size))))

        return dataset_flattened_windows


    def create_tfrecord_file(dataset, filepath):
        '''
        Writes a TensorFlow dataset to a TFRecord file

        Args:
            dataset (tf.data.Dataset): A TensorFlow dataset with sliding windows of features and labels
            filepath (): The path to save the TFRecord file

        References:
            Website: Tensorflow
            Title: TFRecord and tf.train.Example
            URL: https://www.tensorflow.org/tutorials/load_data/tfrecord
        '''

        # Open a TFRecordWriter
        with tf.io.TFRecordWriter(filepath) as writer:

            # Iterate through each (feature_window, label_window) pair in the dataset
            for window_feature, window_label in dataset:
                serialized_window_feature = tf.io.serialize_tensor(window_feature).numpy()
                serialized_window_label = tf.io.serialize_tensor(window_label).numpy()

                # Create a TFRecord Example containing the serialized feature and label tensors
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'window_features': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[serialized_window_feature])
                            ),
                            'window_label': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[serialized_window_label])
                            )
                        }
                    )
                )

            # Write serialized Example to the TFRecord file
            writer.write(example.SerializeToString())


    # Set experiment ranges for predefined split of dataset
    train_experiment_ids = range(1, 43+1) # Train dataset: (user-01 to user-21) leads to (experiment-01 to experiment-43)
    val_experiment_ids = range(56, 61+1) # Validation dataset: (user-28 to user-30) leads to (experiment-56 to experiment-61)
    test_experiment_ids = range(44, 55+1) # Test dataset: (user-22 to user-27) leads to (experiment-44 to experiment-55)
    total_experiment_ids = range(1, 61+1) # Total dataset: (user-01 to user-30) leads to (experiment-01 to experiment-61)

    # Load and organize labels
    experiment_labels_list = load_labels(data_dir)

    # Retrieve accelerometer and gyroscope files
    acc_files, gyro_files = load_data_files(data_dir)

    # Split the features and labels for accelerometer and gyroscope data based on experiment IDs into
    # training, validation, and test sets
    (train_features, train_labels), (val_features, val_labels), (test_features, test_labels)\
        = split_features_and_labels(acc_files, gyro_files, experiment_labels_list, data_dir)

    # Define directory of TFRecord files
    data_dir_tfrecords = './input_pipeline/tfrecord_files/window_size_{}_shift_{}'.format(window_size, window_shift)

    # Create a TensorFlow dataset from features and labels using a sliding window approach
    ds_train = create_tfrecord_dataset(train_features, train_labels, window_size, window_shift)
    ds_val = create_tfrecord_dataset(val_features, val_labels, window_size, window_shift)
    ds_test = create_tfrecord_dataset(test_features, test_labels, window_size, window_shift)

    # Write a TensorFlow dataset to a TFRecord file
    create_tfrecord_file(ds_train, '{}/train.tfrecord'.format(data_dir_tfrecords))
    create_tfrecord_file(ds_val, '{}/validation.tfrecord'.format(data_dir_tfrecords))
    create_tfrecord_file(ds_test, '{}/test.tfrecord'.format(data_dir_tfrecords))