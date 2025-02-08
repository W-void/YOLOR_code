# -*- coding: utf-8 -*-

import tensorflow as tf

from utils import set_logger

logger = set_logger()

FLAGS = tf.flags.FLAGS


def generate_parse_fn(params, feature_protocal, label_protocal):
    batch_size = params['batch_size']
    logger.info('batch_size: %s', batch_size)

    def read_examples(serialized_example):

        parsed_features = tf.parse_example(serialized=serialized_example, features=feature_protocal)
        fea_dict = {c: parsed_features[c] for c in feature_protocal.keys()}

        parsed_labels = tf.parse_example(serialized=serialized_example, features=label_protocal)
        label_dict = {c: parsed_labels[c] for c in label_protocal.keys()}
        
        logger.info(fea_dict)
        logger.info(label_dict)
        return fea_dict, label_dict

    return read_examples


def input_fn(params, feature_protocal, label_protocal):
    with tf.name_scope("dataset"):
        _parse_tf_record = generate_parse_fn(params, feature_protocal, label_protocal)

        dataset = tf.data.AfoDataset()
        dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
        dataset = dataset.map(_parse_tf_record, num_parallel_calls=4)
        dataset = dataset.prefetch(buffer_size=4, prefetch_to_gpu=True)  # 设置太大，evaluator会core dump

        return dataset
