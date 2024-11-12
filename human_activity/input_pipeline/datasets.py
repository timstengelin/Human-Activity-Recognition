import os
import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds

from human_activity.input_pipeline.preprocessing import create_tfrecord_files

@gin.configurable
def load(name, data_dir, window_size, window_shift, tfrecord_files_exist, batch_size, buffer_size):
    '''
    Loads and prepares data from files

    Args:
        name (string): abbreviation of the data set name
        data_dir (string): directory where the original raw data is stored
        window_size (int): length of the sliding window
        window_shift (int): amount of shift/slide of the sliding window
        batch_size (int): size of the batch
        buffer_size (int): size of the buffer

    Returns:
        ds_train (tf.data.Dataset): training dataset
        ds_val (tf.data.Dataset): validation dataset
        ds_test (tf.data.Dataset): test dataset
        ds_info (tf.data.Dataset): dataset info and structure

    '''

    if name != 'hapt':
        raise ValueError('Unsupported dataset with name \'{}\'. Currently, only \'hapt\' is supported.'.format(name))

    logging.info('Loading and preparing dataset \'{}\'...'.format(name))

    # Define directory of TFRecord files
    data_dir_tfrecords = './input_pipeline/tfrecord_files/window_size_{}_shift_{}'.format(window_size, window_shift)
    os.makedirs(data_dir_tfrecords, exist_ok=True)

    # Create new TFRecord files if they do not exist
    if not tfrecord_files_exist:
        create_tfrecord_files(data_dir, window_size, window_shift)

    # Dataset file paths
    ds_train_path = os.path.join(data_dir_tfrecords, 'train.tfrecord')
    ds_val_path = os.path.join(data_dir_tfrecords, 'validation.tfrecord')
    ds_test_path = os.path.join(data_dir_tfrecords, 'test.tfrecord')

    # Define parsing structure
    ds_info = {
        'window_features': tf.io.FixedLenFeature([], tf.string),
        'window_label': tf.io.FixedLenFeature([], tf.string)
    }

    def parse_example(example):
        '''
        Parses TFRecord example

        Args:
            example (tf.Tensor): A serialized example from a TFRecord file

        Returns:
            window_feature (tf.Tensor): The deserialized tensor of feature data
            window_label (tf.Tensor): The deserialized tensor of label data
        '''

        # Parse serialized example to extract the byte-encoded tensors
        parsed_features = tf.io.parse_single_example(example, ds_info)

        # Deserialize the byte-encoded tensors back to their original tensor formats
        window_feature = tf.io.parse_tensor(parsed_features['window_features'], out_type=tf.float64)
        window_label = tf.io.parse_tensor(parsed_features['window_label'], out_type=tf.int32)

        return window_feature, window_label

    # Load datasets
    ds_train = tf.data.TFRecordDataset(ds_train_path).map(parse_example)
    ds_val = tf.data.TFRecordDataset(ds_val_path).map(parse_example)
    ds_test = tf.data.TFRecordDataset(ds_test_path).map(parse_example)

    # Shuffle, batch, repeat training dataset
    ds_train = ds_train.shuffle(buffer_size).batch(batch_size).repeat()
    # Batch the validation and test datasets
    ds_val = ds_val.batch(batch_size)
    num_test_samples = sum(1 for _ in ds_test)
    ds_test = ds_test.batch(num_test_samples)

    logging.info('Dataset \'{}\' loaded and prepared.'.format(name))

    return ds_train, ds_val, ds_test, ds_info


