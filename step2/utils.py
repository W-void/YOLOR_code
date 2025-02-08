# encoding=utf-8
import collections
import logging
import os
import sys

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow_estimator.python.estimator.canned import metric_keys

# from model import SPA_FEATURE_NUM_TERABYTE

_EVENT_FILE_GLOB_PATTERN = 'events.out.tfevents.*'


def process_cmd(command):
    print(command)
    os.system(command)


def _summaries(eval_dir):
    if tf.gfile.Exists(eval_dir):
        for event_file in tf.gfile.Glob(
                os.path.join(eval_dir, _EVENT_FILE_GLOB_PATTERN)):
            print('event_file', event_file)
            for event in tf.train.summary_iterator(event_file):
                yield event


def read_eval_metrics(eval_dir):
    eval_metrics_dict = {}
    for event in _summaries(eval_dir):
        if not event.HasField('summary'):
            continue
        for value in event.summary.value:
            cur_key = None
            cur_value = None
            if value.HasField('simple_value'):
                cur_key = value.tag
                cur_value = value.simple_value
            elif value.HasField('tensor'):
                cur_key = value.tag
                cur_value = value.tensor.string_val[0]
            if not cur_key:
                continue
            if event.step not in eval_metrics_dict:
                eval_metrics_dict[event.step] = {cur_key: cur_value}
            else:
                eval_metrics_dict[event.step][cur_key] = cur_value
    return collections.OrderedDict(
        sorted(eval_metrics_dict.items(), key=lambda t: t[0]))


def UncertaintyWeightLoss(loss_list, name_scope):
    length = len(loss_list)
    sigma_list = []
    for i in range(length):
        sigma_list.append(slim.variable('Sigma_' + name_scope + str(i),
                                        dtype=tf.float32,
                                        shape=[],
                                        initializer=tf.initializers.random_uniform(minval=0.2, maxval=1))
                          )

    factor = tf.div(1.0, tf.multiply(2.0, sigma_list[0]))
    loss = tf.add(tf.multiply(factor, loss_list[0]), tf.log(sigma_list[0]))
    for i in range(1, length):
        factor = tf.div(1.0, tf.multiply(2.0, sigma_list[i]))
        loss = tf.add(loss, tf.add(tf.multiply(factor, loss_list[i]), tf.log(sigma_list[i])))
    return loss


def clipUncertaintyWeightLoss(loss_list, name_scope):
    length = len(loss_list)
    sigma_list = []
    for i in range(length):
        sigma_list.append(slim.variable('Sigma_' + name_scope + str(i),
                                        dtype=tf.float32,
                                        shape=[],
                                        initializer=tf.initializers.random_uniform(minval=0.2, maxval=1))
                          )
    factor = tf.div(1.0, tf.multiply(2.0, sigma_list[0]))
    clip_factor = tf.clip_by_value(factor, 1e-7, factor)
    clip_log_sigma = tf.clip_by_value(tf.log(sigma_list[0]), 1e-7, tf.log(sigma_list[0]))
    loss = tf.add(tf.multiply(clip_factor, loss_list[0]), clip_log_sigma)
    for i in range(1, length):
        factor = tf.div(1.0, tf.multiply(2.0, sigma_list[i]))
        clip_factor = tf.clip_by_value(factor, 1e-7, factor)
        clip_log_sigma = tf.clip_by_value(tf.log(sigma_list[i]), 1e-7, tf.log(sigma_list[i]))
        loss = tf.add(loss, tf.add(tf.multiply(clip_factor, loss_list[i]), clip_log_sigma))
    return loss


def expUncertaintyWeightLoss(loss_list, name_scope):
    length = len(loss_list)
    sigma_list = []
    for i in range(length):
        sigma_list.append(slim.variable('Sigma_' + name_scope + str(i),
                                        dtype=tf.float32,
                                        shape=[],
                                        initializer=tf.initializers.random_uniform(minval=0.2, maxval=1))
                          )
    inv_sig_sq = tf.math.exp(-1 * sigma_list[0])
    loss = loss_list[0] * inv_sig_sq + sigma_list[0]
    for i in range(1, length):
        inv_sig_sq = tf.math.exp(-1 * sigma_list[i])
        loss += loss_list[i] * inv_sig_sq + sigma_list[i]
    return loss


def set_logger():
    logger = logging.getLogger("tensorflow")
    if len(logger.handlers) == 1:
        logger.handlers = []
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - [%(filename)s:%(lineno)d] - %(name)s - %(levelname)s - %(message)s")
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        fh = logging.FileHandler('tensorflow.log')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)
        # print len(logger.handlers)

    return logger


def mish(x):
    return tf.multiply(x, tf.tanh(tf.math.softplus(x)))


def parametric_relu(_x, name):
    alphas = tf.get_variable('alpha' + name, [_x.get_shape()[-1]],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg


def dice(_x, axis=-1, epsilon=0.000000001, name='', is_training=True):
    logger = set_logger()

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('alpha' + name, _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
    pop_mean = tf.Variable(tf.zeros([1, _x.get_shape().as_list()[-1]]), trainable=False)
    pop_std = tf.Variable(tf.ones([1, _x.get_shape().as_list()[-1]]), trainable=False)
    logger.info("pop_mean %s", pop_mean)
    logger.info("pop_std %s", pop_std)

    reduction_axes = 0
    broadcast_shape = [1, _x.shape.as_list()[-1]]
    decay = 0.999
    logger.info("broadcast_shape %s", broadcast_shape)
    if is_training:
        mean = tf.reduce_mean(_x, axis=reduction_axes)
        brodcast_mean = tf.reshape(mean, broadcast_shape)
        std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
        std = tf.sqrt(std)
        brodcast_std = tf.reshape(std, broadcast_shape)
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + brodcast_mean * (1 - decay))
        train_std = tf.assign(pop_std,
                              pop_std * decay + brodcast_std * (1 - decay))
        with tf.control_dependencies([train_mean, train_std]):
            x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
    else:
        x_normed = (_x - pop_mean) / (pop_std + epsilon)
    # x_normed = tf.layers.batch_normalization(_x, center=False, scale=False)
    x_p = tf.sigmoid(x_normed)
    return alphas * (1.0 - x_p) * _x + x_p * _x


# class InitHashTableHook(tf.train.SessionRunHook):
#
#     def __init__(self):
#         tf.logging.info("Create InitHashTableHook.")
#         super(InitHashTableHook, self).__init__()
#
#     def begin(self):
#         self._task_index = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
#         self._total_task = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
#         self._insert_ops = []
#         if context.context().get_pipeline_mode() != context.PipelineMode.MainGraph:
#             tf.logging.info('Begin to create HashTable init ops.')
#             embed_tables = tf.get_collection(tf.GraphKeys.MERGEABLE_HASHTABLE_COLLECTION)
#             for i in range(len(SPA_FEATURE_NUM_TERABYTE)):
#                 total_id = tf.range(SPA_FEATURE_NUM_TERABYTE[i], dtype=tf.int64) + (i + 1) * (2 ** 48)  # id 添加偏移
#                 keys = tf.boolean_mask(total_id, tf.equal(tf.mod(total_id, self._total_task),
#                                                           self._task_index))  # 处理后的id mod 卡id，得到本张卡所需插入的id
#
#                 merge_tag = tem.get_merged_table_name(embed_tables[i].name)  # 这里的table.name是合表之前的table的name
#                 var_store = get_variable_store()
#                 merged_table = var_store._vars[merge_tag]  # 这个就是merged_table
#
#                 values = tf.random_uniform(shape=[tf.shape(keys)[0], 128],
#                                            minval=-1 / np.sqrt(SPA_FEATURE_NUM_TERABYTE[i]),
#                                            maxval=1 / np.sqrt(SPA_FEATURE_NUM_TERABYTE[i]),
#                                            seed=self._task_index * 100 + i,
#                                            dtype=tf.dtypes.float32) + tf.zeros([tf.shape(keys)[0], 128])
#                 insert_op = merged_table.insert(keys, values, pipeline_queue=False)
#                 self._insert_ops.append(insert_op)
#
#     def after_create_session(self, session, coord):
#         if context.context().get_pipeline_mode() != context.PipelineMode.MainGraph:
#             # When this is called, the graph is finalized and ops can no longer be added to the graph.
#             tf.logging.info("Begin to insert HashTable")
#             session.run(self._insert_ops)
#             tf.logging.info("Insert HashTable finished")


def auc_bigger(best_eval_result, current_eval_result):
    """Compares two evaluation results and returns true if the 2nd one is smaller.

    Both evaluation results should have the values for MetricKeys.AUC, which are
    used for comparison.

    Args:
      best_eval_result: best eval metrics.
      current_eval_result: current eval metrics.

    Returns:
      True if the AUC of current_eval_result is bigger; otherwise, False.

    Raises:
      ValueError: If input eval result is None or no AUC is available.
    """
    default_key = 'auc/ctr'
    if not best_eval_result or default_key not in best_eval_result:
        raise ValueError(
            'best_eval_result cannot be empty or no %s is found in it.'%(default_key))

    if not current_eval_result or default_key not in current_eval_result:
        raise ValueError(
            'current_eval_result cannot be empty or no %s is found in it.'%(default_key))

    return best_eval_result[default_key] < current_eval_result[default_key]


def gauc_bigger(best_eval_result, current_eval_result):
    """Compares two evaluation results and returns true if the 2nd one is smaller.

    Both evaluation results should have the values for "gauc", which are
    used for comparison.

    Args:
      best_eval_result: best eval metrics.
      current_eval_result: current eval metrics.

    Returns:
      True if the AUC of current_eval_result is bigger; otherwise, False.

    Raises:
      ValueError: If input eval result is None or no AUC is available.
    """
    default_key = 'gauc'
    if not best_eval_result or default_key not in best_eval_result:
        raise ValueError(
            'best_eval_result cannot be empty or no GAUC is found in it.')

    if not current_eval_result or default_key not in current_eval_result:
        raise ValueError(
            'current_eval_result cannot be empty or no GAUC is found in it.')

    return best_eval_result[default_key] < current_eval_result[default_key]


def cal_ssl_loss(emd_x, emd_y, temp):
    """计算（自监督）对比学习损失

    Args:
        emd_x:  数据增强样本X
        emd_y:  数据增强样本Y
        temp:   温度系数

    Returns:
    """
    normalize_emd_x = tf.nn.l2_normalize(emd_x, 1)
    normalize_emd_y = tf.nn.l2_normalize(emd_y, 1)

    normalize_emb_x_neg = normalize_emd_y

    pos_score = tf.reduce_sum(tf.multiply(normalize_emd_x, normalize_emd_y), axis=1)
    ttl_score = tf.matmul(normalize_emd_x, normalize_emb_x_neg, transpose_a=False, transpose_b=True)

    pos_score = tf.exp(pos_score / temp)
    ttl_score = tf.reduce_sum(tf.exp(ttl_score / temp), axis=1)

    ssl_loss = -tf.reduce_mean(tf.log(pos_score / ttl_score))

    return ssl_loss


def cal_ssl_loss_v2(emd_x, emd_y, temp, weight):
    """计算（自监督）对比学习损失，并对每个样本赋予不同loss权重

    Args:
        emd_x:  数据增强样本X
        emd_y:  数据增强样本Y
        temp:   温度系数
        weight: 样本权重

    Returns:
    """
    normalize_emd_x = tf.nn.l2_normalize(emd_x, 1)
    normalize_emd_y = tf.nn.l2_normalize(emd_y, 1)

    normalize_emb_x_neg = normalize_emd_y

    pos_score = tf.reduce_sum(tf.multiply(normalize_emd_x, normalize_emd_y), axis=1)
    ttl_score = tf.matmul(normalize_emd_x, normalize_emb_x_neg, transpose_a=False, transpose_b=True)

    pos_score = tf.exp(pos_score / temp)
    ttl_score = tf.reduce_sum(tf.exp(ttl_score / temp), axis=1)

    score = tf.log(pos_score / ttl_score)
    score = tf.multiply(score, weight)
    ssl_loss = -tf.reduce_mean(score)

    return ssl_loss


def cal_bpr_loss_v1(emd_pos, emd_neg, gamma=1e-10):
    """Compute BPRLoss, based on Bayesian Personalized Ranking

    Args:
        emd_pos: 正样本
        emd_neg: 负样本
        gamma: Small value to avoid division by zero

    Returns:
    """
    normalize_emd_pos = tf.nn.l2_normalize(emd_pos, 1)
    normalize_emd_neg = tf.nn.l2_normalize(emd_neg, 1)

    score = tf.log(tf.sigmoid(normalize_emd_pos - normalize_emd_neg) + gamma)
    bpr_loss = -tf.reduce_mean(score)

    return bpr_loss


"""
=======================
    Transformer
=======================
"""


def scaled_dot_product_attention(q, k, v, mask=None):
    """计算注意力权重。
    q, k, v 必须具有匹配的前置维度。
    k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
    虽然 mask 根据其类型（填充或前瞻）有不同的形状，
    但是 mask 必须能进行广播转换以便求和。

    Args:
        q: 请求的形状 == (..., seq_len_q, depth)
        k: 主键的形状 == (..., seq_len_k, depth)
        v: 数值的形状 == (..., seq_len_v, depth_v)
        mask: Float 张量，其形状能转换成 (..., seq_len_q, seq_len_k)。默认为None。

    Returns:
        注意力权重
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # 缩放 matmul_qk
    d_k = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)

    # 将 mask 加入到缩放的张量上。
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax 在最后一个轴（seq_len_k）上归一化，因此分数相加等于1。
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, name=name + '_wq')
        self.wk = tf.keras.layers.Dense(d_model, name=name + '_wk')
        self.wv = tf.keras.layers.Dense(d_model, name=name + '_wv')

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """分拆最后一个维度到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, name, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, name)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, name, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, name + str(i), rate)
                           for i in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # seq_len = tf.shape(x)[1]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class EncoderLayerV2(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayerV2, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, encoder_output, training, mask):
        attn_output, _ = self.mha(x, encoder_output, encoder_output, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class EncoderV2(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(EncoderV2, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.enc_layers = [EncoderLayerV2(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, mask):
        # seq_len = tf.shape(x)[1]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, enc_output, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, name, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads, name)
        self.mha2 = MultiHeadAttention(d_model, num_heads, name)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, name, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, name + str(i), rate)
                           for i in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        # x (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


def calc_gauc(res_dict):
    num = 0
    gauc_all = 0.
    label_sum = 0
    uuid_num, pv_num, used_uuid_num, used_pv_num = 0, 0, 0, 0
    for k, v in res_dict.iteritems():
        cur_len = len(v[0])   ###包含的元素个数
        label_sum = np.array(v[0]).sum()
        uuid_num += 1
        pv_num += cur_len
        if (label_sum > 0) and (label_sum < cur_len): ###label不全为1且label不全为0
            cur_gauc = metrics.roc_auc_score(v[0], v[1])
            num += cur_len
            gauc_all += cur_len * cur_gauc
            used_uuid_num += 1
            used_pv_num += cur_len
    return gauc_all / num, uuid_num, pv_num, used_uuid_num, used_pv_num

def calc_pred_result(preds):
    labels = np.reshape(np.array(preds['labels']).astype(np.float64), [-1])
    pvids = np.reshape(np.array(preds['pvid']).astype(np.float32).astype(np.int32), [-1])
    pctrs = np.reshape(np.array(preds["probabilities"]).astype(np.float64), [-1])
    print('#WSL:labels.shape is {}'.format(labels.shape))

    label_mean, label_var = labels.mean(), labels.var()
    pctr_mean, pctr_var = pctrs.mean(), pctrs.var()

    fpr, tpr, thresholds = metrics.roc_curve(labels, pctrs)
    auc = metrics.auc(fpr, tpr)

    loss = metrics.log_loss(labels, pctrs)
    mae = np.abs(labels - pctrs).mean()

    poi_gauc_dict = {}
    poi_pre = []
    poi_label = []
    for i in range(0, len(pvids)):
        if pvids[i] in poi_gauc_dict:
            poi_label = poi_gauc_dict[pvids[i]][0]
            poi_pre = poi_gauc_dict[pvids[i]][1]
        else:
            poi_pre = []
            poi_label = []
        poi_label.append(labels[i])
        poi_pre.append(pctrs[i])
        poi_gauc_dict[pvids[i]] = [poi_label,poi_pre]
    gauc, uuid_num, pv_num, used_uuid_num, used_pv_num = calc_gauc(poi_gauc_dict)

    result_dict = {}
    result_dict['auc'] = auc
    result_dict['gauc'] = gauc
    result_dict['loss'] = loss
    result_dict['mae'] = mae
    result_dict['label_mean'] = label_mean
    result_dict['label_var'] = label_var
    result_dict['pctr_mean'] = pctr_mean
    result_dict['pctr_var'] = pctr_var
    result_dict['data_num'] = len(labels)
    result_dict['uuid_num'] = uuid_num
    result_dict['pv_num'] = pv_num
    result_dict['used_uuid_num'] = used_uuid_num
    result_dict['used_pv_num'] = used_pv_num

    return result_dict
