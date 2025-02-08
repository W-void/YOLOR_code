# -*- coding: utf-8 -*-
import functools
import math

import numpy as np
from tensorflow.contrib.lookup.lookup_ops import get_mutable_dense_hashtable
from tensorflow.contrib.opt import HashAdamOptimizer, HashAdagradOptimizer, HashMomentumOptimizer
from tensorflow.python.ops import array_ops
import tensorflow_probability as tfp
import pandas as pd


from utils import *

logger = set_logger()


class DeepFM(object):
    def __init__(self, params):
        self.params = params

        self.embed_dim = params['embed_dim']
        self.emb_hashtable = None
        self.fm_hashtable = None
        self.random_seed = int(params['random_seed'])
        self.batch_size = int(params['batch_size'])
        self.freq_threshold = int(params['freq_threshold'])

        self.dict_bucket_file = params['dict_bucket_file']
        self.src_filename = params['src_filename']

        self.learning_rate = params['learning_rate']
        self.hash_learning_rate = params['hash_learning_rate']
        self.lr_decay = bool(params['lr_decay'])
        self.optimizer = params['optimizer']
        self.l2_reg = float(params['l2_reg'])
        self.activation = params['activation']

        # kv memory
        self.is_kv_memory = params['is_kv_memory']
        self.kv_mem_num = params['kv_mem_num']
        self.kv_embed_dim = params['kv_embed_dim']
        if not self.is_kv_memory:
            self.kv_embed_dim = 1

        # 各任务网络参数
        self.deep_layers_size_list = [int(element) for element in params['deep_layers'].split(",")]
        self.dropout = [float(element) for element in params['dropout'].split(",")]
        self.batch_norm = [eval(x) for x in params['batch_norm'].split(',')]

        # 输出
        self.label_dim = 1

        tf.set_random_seed(self.random_seed)
        np.random.seed(self.random_seed)

        # self.quantiles_mat = self.get_quantiles_mat_from_csv(self.dict_bucket_file)

    def Print(self, val, context, mode=tf.estimator.ModeKeys.TRAIN, content=None):
        if mode != tf.estimator.ModeKeys.PREDICT:
            if content is None:
                content = val
            val = tf.Print(val, ['#WSL:%s'%context, content], first_n=1000, summarize=40)
        return val

    def read_csv_files(self, directory):
        # 获取指定目录下所有的.csv文件
        csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        logger.info("#WSL:csv_files is %s"%csv_files)

        # 创建一个空的DataFrame用于存储数据
        all_data = pd.DataFrame()

        for csv_file in csv_files:
            # 读取.csv文件并将数据添加到all_data中
            df = pd.read_csv(os.path.join(directory, csv_file), engine='python', dtype=str)
            df = df.reset_index(drop=True)  # 重置索引
            all_data = pd.concat([all_data, df])

        return all_data
    def get_quantiles_map_from_csv(self, file_path):
        df = self.read_csv_files(file_path)
        quantiles_dict = {name: list(map(float, percentile.split(','))) for name, percentile in
                        zip(df[df.columns[0]], df[df.columns[1]])}
        return quantiles_dict

    def get_quantiles_mat_from_csv(self, file_path):
        df = self.read_csv_files(file_path)
        quantiles_list = [list(map(float, percentile.split(','))) for percentile in df[df.columns[1]]]
        quantiles_list = [percentile + [percentile[-1]] * (99 - len(percentile)) for percentile in quantiles_list]
        logger.info("#WSL:len of quantiles_list is %d"%(len(quantiles_list)))
        quantiles = tf.convert_to_tensor(quantiles_list, dtype=tf.float32)
        logger.info("#WSL:quantiles is %s"%(quantiles))
        return quantiles_list

    def get_quatile_emb(self, features, quantiles_list, name='0'):
        # features : [B, fea_num]
        # quantiles_list : [fea_num, 100]
        with tf.variable_scope('get_quatile_emb_%s'%name, reuse=tf.AUTO_REUSE):
            logger.info("#WSL:features is %s, len quantiles_list is %s"%(features, len(quantiles_list)))
            fea_num = features.shape[-1]
            percentile_dis = tf.reduce_sum(tf.cast(tf.expand_dims(features, -1) > quantiles_list, tf.int32), -1)
            # percentile_dis = tf.Print(percentile_dis, ['#WSL:percentile_dis', percentile_dis], first_n=1000, summarize=40)
            percentile_dis_one_hot = tf.one_hot(percentile_dis, depth=100)
            percentile_dis_emb_default = tf.get_variable('percentile_dis_emb_default',
                                                        shape=[100, fea_num * self.embed_dim],
                                                        initializer=tf.random_uniform_initializer(minval=-0.2, maxval=0.2))
            # percentile_dis_emb = tf.einsum('bnc,ncd->bnd', percentile_dis_one_hot, percentile_dis_emb_default)
            percentile_dis_emb = tf.transpose(tf.linalg.diag_part(tf.transpose(tf.reshape(tf.matmul(percentile_dis_one_hot, percentile_dis_emb_default), [-1, fea_num, fea_num, self.embed_dim]), [0, 3, 2, 1])), [0, 2, 1])
            result = tf.reshape(percentile_dis_emb, [-1, fea_num, self.embed_dim])
            logger.info("#WSL:percentile_dis_emb is %s", result)
            return result

    def create_features_for_each_bit(self, features, feature_name, split_feature_list):
        for i, fea in enumerate(split_feature_list):
            features[fea] = tf.expand_dims(features[feature_name][:, i], -1)
        return features

    def create_num_embedding_features(self, features, feature_name_list, embedding_dim):
        with tf.variable_scope('numeric_column_embedding'):
            boundaries_map = self.get_quantiles_map_from_csv(self.dict_bucket_file)
            emb_outputs = []
            for feature_name in feature_name_list:
                feature = tf.feature_column.numeric_column(feature_name)
                bucketized_feature = tf.feature_column.bucketized_column(feature, boundaries=boundaries_map[feature_name])
                embedding_feature = tf.feature_column.embedding_column(bucketized_feature, dimension=embedding_dim,
                                                                       initializer=tf.contrib.layers.xavier_initializer(),
                                                                       trainable=True)
                emb_output = tf.feature_column.input_layer(features, [embedding_feature])
                emb_outputs.append(emb_output)

            result = tf.concat(emb_outputs, axis=1)
            result = tf.reshape(result, [-1, self.numerical_fea_num, self.embed_dim])
            return result

    def norm_numerical_features(self, num_fea_col, is_training):
        """对连续性特征进行标准化，以及KV embedding（可选）"""
        is_kv_memory = self.is_kv_memory
        kv_mem_num = self.kv_mem_num
        kv_embed_dim = self.kv_embed_dim
        epsilon = 1e-8
        # mean_vec, var_vec = load_normalization_parameter(self.mean_var_filename, self.total_numerical_fea_num,
        #                                                  self.numerical_fea_idx)
        # mean, var = tf.nn.moments(num_fea_col, axes=[0])

        # mean_vec, var_vec = self.mean_var_stat
        # normalized_numerical_fea_col = (num_fea_col - mean_vec) / (tf.sqrt(var_vec) + epsilon)  # 120-1
        # normalized_numerical_fea_col = bn_layer(num_fea_col, training=is_training, name="BN_numerical_fea_col")
        normalized_numerical_fea_col = tf.contrib.layers.batch_norm(inputs=num_fea_col, decay=0.9,
                                                                    updates_collections=None,
                                                                    trainable=True,
                                                                    is_training=is_training)

        # 对标准化后的特征进行kv-memory
        if is_kv_memory:
            kv_index = tf.constant([float(i) / kv_mem_num for i in range(kv_mem_num + 1)], dtype=tf.float32)  # [K]
            tanh_normalized_numerical_fea_col = tf.nn.tanh(normalized_numerical_fea_col)  # tanh
            num_norm_fea = (tanh_normalized_numerical_fea_col + 1) / 2  # [B, N]
            num_distance = 1.0 / (tf.abs(tf.reshape(num_norm_fea, [-1, self.numerical_fea_num, 1]) -
                                         tf.reshape(kv_index, [-1, 1, kv_mem_num + 1])) + epsilon)
            num_weight = tf.nn.softmax(num_distance, -1)  # [B, N, K+1]
            num_weight = tf.expand_dims(num_weight, -1)  # [B, N, K+1, 1]
            logger.info(
                "#Z-kv_index:{},kv_index,num_norm_fea:{},num_distance:{},num_weight:{}".format(kv_index, num_norm_fea,
                                                                                               num_distance,
                                                                                               num_weight))
            num_kv_emb_matrix = tf.get_variable(shape=[self.numerical_fea_num, kv_mem_num + 1, kv_embed_dim],
                                                initializer=tf.truncated_normal_initializer(0.0, 1e-2),
                                                trainable=True,
                                                name='user_kv_emb_matrix')  # [N, K+1, D]
            num_kv_emb_matrix = tf.expand_dims(num_kv_emb_matrix, 0)  # [1, N, K+1, D]
            num_kv_embedding = tf.reduce_sum(num_weight * num_kv_emb_matrix, -2)  # [B, N, D]
            reshaped_num_kv_emb = tf.reshape(num_kv_embedding, [-1, self.numerical_fea_num * kv_embed_dim])
            logger.info(
                "#Z-num_kv_emb_matrix:{},num_kv_embedding:{}, reshaped_num_kv_emb:{}".format(num_kv_emb_matrix,
                                                                                             num_kv_embedding,
                                                                                             reshaped_num_kv_emb))
            normalized_numerical_fea_col = reshaped_num_kv_emb

        return normalized_numerical_fea_col

    def percentile_dis(self, num_fea_col, name, is_training):
        # normalized_numerical_fea_col = tf.contrib.layers.batch_norm(inputs=num_fea_col, decay=0.9,
        #                                                             updates_collections=None,
        #                                                             trainable=True,
        #                                                             is_training=is_training)
        percentile_data = num_fea_col
        param_shape = percentile_data.shape[-1]
        all_perc = []
        with tf.variable_scope('percentile_dis_%s'%name, reuse=tf.AUTO_REUSE):
            # for perc in range(1, 100, 1):

            batch_perc = tfp.stats.percentile(percentile_data, q=range(1, 100, 1), axis=[0])

            ## 声明BN中唯一需要学习的两个参数，y=gamma*x+beta
            # gamma = tf.get_variable('gamma_%d'%(perc), param_shape, initializer=tf.constant_initializer(1))
            # beta  = tf.get_variable('beta_%d'%(perc), param_shape, initializer=tf.constant_initializer(0))

            ## 计算当前整个batch的均值与方差
            # batch_mean, batch_var = tf.nn.moments(percentile_data, [0], name='moments')

            # 采用滑动平均更新均值与方差
            ema = tf.train.ExponentialMovingAverage(0.999, name='MA_%d'%(100))

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_perc])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_perc)

            # 训练时，更新均值与方差，测试时使用之前最后一次保存的均值与方差
            all_perc = tf.cond(tf.equal(is_training, True), mean_var_with_update, lambda:(ema.average(batch_perc)))
            all_perc = tf.transpose(all_perc, [1, 0])
            logger.info("#WSL:all_perc %s", all_perc)
            # all_perc = tf.Print(all_perc, ['#WSL:all_perc', all_perc], first_n=1000, summarize=40)
            percentile_dis = tf.reduce_sum(tf.cast(tf.expand_dims(percentile_data, -1) > all_perc, tf.int32), -1)
            # percentile_dis = tf.Print(percentile_dis, ['#WSL:percentile_dis', percentile_dis], first_n=1000, summarize=40)
            percentile_dis_one_hot = tf.one_hot(percentile_dis, depth=100)
            percentile_dis_emb_default = tf.get_variable('percentile_dis_emb_default',
                                                        shape=[100, param_shape * self.embed_dim],
                                                        initializer=tf.random_uniform_initializer(minval=-0.2, maxval=0.2))
            # percentile_dis_emb = tf.einsum('bnc,ncd->bnd', percentile_dis_one_hot, percentile_dis_emb_default)
            percentile_dis_emb = tf.transpose(tf.linalg.diag_part(tf.transpose(tf.reshape(tf.matmul(percentile_dis_one_hot, percentile_dis_emb_default), [-1, param_shape, param_shape, self.embed_dim]), [0, 3, 2, 1])), [0, 2, 1])
            result = tf.reshape(percentile_dis_emb, [-1, param_shape, self.embed_dim])
            logger.info("#WSL:percentile_dis_emb is %s", result)
            # result = percentile_dis / 100
            # result = tf.Print(result, ['#WSL:result', result], first_n=1000, summarize=40)
        return result

    def inference(self, features, name, is_training):
        with tf.variable_scope('input_layer'):
            cate_fea_col = tf.cast(features['cate_fea'], tf.int32, name='cate_fea_col')
            num_fea_col = tf.cast(features['num_fea'], tf.float32, name='num_fea_col')
            
            target_poi_id = tf.cast(features['poi_id'], tf.int32, name='poi_id_col')
            target_poi_name = tf.cast(features['poi_name'], tf.int32, name='poi_name_col')
            seq_fea_col = tf.cast(features['seq_fea'], tf.int32, name='seq_fea_col')
            seq_poi_id, seq_ts = tf.split(seq_fea_col, 2, axis=1)
            seq_poi_name = tf.cast(features['seq_name'], tf.int32, name='seq_name_col')
            dianjin_seq_col = tf.cast(features['dianjin_seq_fea'], tf.int32, name='dianjin_seq_fea_col')
            dj_seq_poi_id, dj_seq_ts = tf.split(dianjin_seq_col, 2, axis=1)
            dj_seq_poi_name = tf.cast(features['dianjin_seq_name'], tf.int32, name='dj_seq_name_col')

            with tf.variable_scope('embedding_hashtable'):
                # 所有特征共用hashtable
                # initializer = tf.truncated_normal_initializer(0.0, 1e-2)
                initializer = tf.zeros_initializer()
                emb_hashtable = get_mutable_dense_hashtable(tf.int64,
                                                            tf.float32,
                                                            shape=tf.TensorShape(self.embed_dim),
                                                            name="emb_hashtable",
                                                            initializer=initializer,
                                                            fusion_optimizer_var=True,
                                                            export_optimizer_var=True,
                                                            )
                self.emb_hashtable = emb_hashtable
                tf.add_to_collection("HASHTABLE", emb_hashtable)

            # 类别特征Embedding
            cate_embed = tf.nn.embedding_lookup_hashtable_v2(emb_hashtable,
                                                             cate_fea_col,
                                                             unique_internal=True,
                                                             threshold=self.freq_threshold,
                                                             name='cate_fea_embed_lookup')
            reshaped_cate_embed = tf.reshape(cate_embed, [-1, self.cate_fea_num * self.embed_dim])

            seq_fea_list = [target_poi_id, target_poi_name, seq_poi_id, seq_poi_name, dj_seq_poi_id, dj_seq_poi_name]

            tgt_poi_emb, tgt_name_emb, seq_poi_emb, seq_name_emb, dj_seq_poi_emb, dj_seq_name_emb = \
                self.table_lookup(emb_hashtable, seq_fea_list, self.freq_threshold, 'seq_fea_embed_lookup', False, self.embed_dim)
            tgt_poi_emb = tf.Print(tgt_poi_emb, ["#WSL:tgt_poi_emb", tgt_poi_emb], first_n=1000, summarize=80)
            
        with tf.variable_scope('Seq'):
            # tgt_name_emb, seq_name_emb, dj_seq_name_emb = [tf.reduce_mean(tf.reshape(name_emb, [name_emb.shape[0].value, -1, 5, name_emb.shape[-1].value]), axis=2) for name_emb in (tgt_name_emb, seq_name_emb, dj_seq_name_emb)]
            tgt_name_emb = tf.reshape(tgt_name_emb, [-1, 1, 5, self.embed_dim])
            seq_name_emb = tf.reshape(seq_name_emb, [-1, 30, 5, self.embed_dim])
            dj_seq_name_emb = tf.reshape(dj_seq_name_emb, [-1, 20, 5, self.embed_dim])
            tgt_name_emb, seq_name_emb, dj_seq_name_emb = [tf.reduce_mean(name_emb, axis=-2) for name_emb in (tgt_name_emb, seq_name_emb, dj_seq_name_emb)]

            mask = tf.count_nonzero(seq_poi_id, axis=1, keep_dims=True)
            din_output = self.easy_attention_layer(tgt_poi_emb, seq_poi_emb, mask, 'din_layer')
            din_output_name = self.easy_attention_layer(tgt_name_emb, seq_name_emb, mask, 'din_layer_name')

            dj_mask = tf.count_nonzero(dj_seq_poi_id, axis=1, keep_dims=True)
            dj_din_output = self.easy_attention_layer(tgt_poi_emb, dj_seq_poi_emb, dj_mask, 'dj_din_layer')
            dj_din_output_name = self.easy_attention_layer(tgt_name_emb, dj_seq_name_emb, dj_mask, 'dj_din_layer_name')

        with tf.variable_scope('FM'):
            with tf.variable_scope('first_order'):
                first_term = tf.layers.dense(inputs=reshaped_cate_embed,
                                             units=1,
                                             kernel_initializer=tf.truncated_normal_initializer(0.0, 1e-2),
                                             name='%s_first_term' % name)

            with tf.variable_scope('second_order'):
                # embedding矩阵列的和平方, shape(None, embedding_size)
                square_of_sum = tf.square(tf.reduce_sum(
                    cate_embed, axis=1, keep_dims=True))
                # embedding矩阵列的平方和, shape(None, embedding_size)
                sum_of_square = tf.reduce_sum(
                    cate_embed * cate_embed, axis=1, keep_dims=True)
                # 二阶项 = 0.5 * (embedding矩阵列的和平方 - embedding矩阵列的平方和), shape(None, 1)
                second_term = 0.5 * tf.reduce_sum(tf.subtract(square_of_sum, sum_of_square), axis=2,
                                                  keep_dims=False)

        with tf.variable_scope('Deep'):
            # Deep输入：类别+数值特征+序列特征
            # deep = tf.concat([first_order, seq_embedding], axis=1)
            # num_embed = self.norm_numerical_features(num_fea_col, is_training)
            # reshaped_num_embed = tf.reshape(num_embed, [-1, self.numerical_fea_num * self.kv_embed_dim])
            # num_embed = self.percentile_dis(num_fea_col, is_training)
            # reshaped_num_embed = tf.reshape(num_embed, [-1, self.numerical_fea_num * self.embed_dim])
            num_embed = self.get_quatile_emb(num_fea_col, self.quantiles_mat)
            reshaped_num_embed = tf.reshape(num_embed, [-1, self.numerical_fea_num * self.embed_dim])

            # num_fea_list = self.num_fea.split(",")
            # features = self.create_features_for_each_bit(features, 'num_fea', num_fea_list)
            # normalized_num_fea_col = self.create_num_embedding_features(features, num_fea_list, self.embed_dim)
            # reshaped_num_embed = tf.reshape(normalized_num_fea_col, [-1, self.numerical_fea_num * self.embed_dim])

            deep_list = [reshaped_cate_embed, reshaped_num_embed]
            deep_list += [tf.squeeze(tgt_poi_emb, axis=[1]), tf.squeeze(tgt_name_emb, axis=[1])]
            seq_list = [din_output, din_output_name]
            seq_list_weight = tf.get_variable('seq_list_weight', shape=[len(seq_list)], initializer=tf.random_normal_initializer())
            seq_list_weight = tf.layers.dropout(seq_list_weight, 0.5)
            seq_list = [seq_list[i] * seq_list_weight[i] for i in range(len(seq_list))]
            deep = tf.concat(deep_list, axis=1)
            # deep = self.se_block(deep, self.embed_dim, 'deep')

            deep_layers_list = self.deep_layers_size_list
            dropout_deep = self.dropout
            output_dim = self.label_dim
            activation = self.activation
            if activation == "leakyRelu":
                activation_func = functools.partial(tf.nn.leaky_relu, alpha=0.3)
            else:
                activation_func = tf.nn.relu

            for i in range(len(deep_layers_list)):
                deep = tf.layers.dense(inputs=deep, units=deep_layers_list[i],
                                       activation=activation_func,
                                       kernel_initializer=tf.truncated_normal_initializer(0.0, 1e-2),
                                       name='%s_fc_%d' % (name, i))
                if dropout_deep[i] > 0:
                    deep = tf.layers.dropout(deep, dropout_deep[i])

            deep = tf.layers.dense(inputs=deep, units=output_dim,
                                   kernel_initializer=tf.truncated_normal_initializer(0.0, 1e-2),
                                   name='%s_fc_output' % name)

        output = tf.sigmoid(deep)
        # output = tf.sigmoid(tf.add_n([first_term, second_term, deep]))
        # output = tf.layers.dense(inputs=tf.concat([first_term, second_term, deep], -1),
        #                          units=self.label_dim,
        #                          activation=tf.nn.sigmoid,
        #                          name='%s_output' % name)

        return output

    def loss_function(self, y, y_, weights=1.0):
        with tf.variable_scope('task_loss'):
            loss = tf.reduce_mean(tf.losses.log_loss(y, y_, weights=weights))
            # if self.l2_reg > 0:
            #     l2 = tf.contrib.layers.l2_regularizer(self.l2_reg)
            #     loss += l2(self.fm_weights['category_feature_weights'])
            tf.summary.scalar('task_loss', loss)
        return loss
    
    def table_lookup(self, table, list_ids, threshold, v_name, flatten, embedding_size):
        def _do_lookup(ids):
            _embed = tf.nn.embedding_lookup_hashtable_v2(emb_tables=table,
                                                        ids=ids,
                                                        unique_internal=True,
                                                        threshold=threshold,
                                                        name=v_name)
            return _embed

        if not isinstance(list_ids, list):
            list_ids = [list_ids]

        list_embed = []
        for ids in list_ids:
            # num = ids.shape[1].value
            # uniq_ids = tf.reshape(ids, [-1])
            _embed = _do_lookup(ids)
            # if flatten:
            #     _embed = tf.reshape(_embed, [-1, num * embedding_size])
            # else:
            #     _embed = tf.reshape(_embed, [-1, num, embedding_size])
            list_embed.append(_embed)
        return list_embed
    
    def easy_dense(self, inputs, output_dims, name, activation=tf.nn.relu):
        with tf.variable_scope('easy_dense_' + name, reuse=tf.AUTO_REUSE):
            output = tf.layers.dense(inputs=inputs, units=output_dims,
                                    kernel_initializer=tf.truncated_normal_initializer(0.0, 1e-2),
                                    activation=activation, name='dense_layer_' + name)
            return output
        
    def easy_attention_layer_ts(self, cur_poi_seq_fea_col, hist_poi_seq_fea_col, mask, att_type):   
        seq_len = hist_poi_seq_fea_col.shape[-1] # [B, sel_len]
        din_deep_layers, din_activation = [64, 32], 'relu'
        # if len(cur_poi_seq_fea_col.shape) == 2:
        #     cur_poi_seq_fea_col = tf.expand_dims(cur_poi_seq_fea_col, 1)

        with tf.variable_scope("attention_layer_%s" % att_type, reuse=tf.AUTO_REUSE):

            ts_dif = cur_poi_seq_fea_col - hist_poi_seq_fea_col
            ts_dif_day = ts_dif // (3600 * 24)
            ts_dif_weekin = ts_dif_day % 7
            ts_dif_weekout = tf.clip_by_value(ts_dif_day // 7, 0, 12)
            ts_dif_mouth = tf.clip_by_value(ts_dif_day // 30, 0, 3)
            ts_dif_weekin_emb_default = tf.get_variable('ts_dif_weekin_emb',
                                            shape=[7, self.embed_dim],
                                            initializer=tf.random_uniform_initializer(minval=-1e-4, maxval=1e-4))
            ts_dif_weekout_emb_default = tf.get_variable('ts_dif_weekout_emb',
                                            shape=[13, self.embed_dim],
                                            initializer=tf.random_uniform_initializer(minval=-1e-4, maxval=1e-4))
            ts_dif_mouth_emb_default = tf.get_variable('ts_dif_mouth_emb',
                                            shape=[4, self.embed_dim],
                                            initializer=tf.random_uniform_initializer(minval=-1e-4, maxval=1e-4))
            ts_dif_weekin_emb = tf.gather(ts_dif_weekin_emb_default, ts_dif_weekin)
            ts_dif_weekout_emb = tf.gather(ts_dif_weekout_emb_default, ts_dif_weekout)
            ts_dif_mouth_emb = tf.gather(ts_dif_mouth_emb_default, ts_dif_mouth)
            ts_emb = tf.concat([ts_dif_weekin_emb, ts_dif_weekout_emb, ts_dif_mouth_emb], axis=-1)
            logger.info("ts_dif_weekin is {}, ts_dif_weekin_emb_default is {}, ts_dif_weekin_emb is {}".format(ts_dif_weekin, ts_dif_weekin_emb_default, ts_dif_weekin_emb))

            return ts_emb
    
    def easy_attention_layer(self, cur_poi_seq_fea_col, hist_poi_seq_fea_col, mask, att_type, aux_info=None):  
        '''input shape
        cur_poi_seq_fea_col : [B, emb_dim] or [B, 1, emb_dim]
        hist_poi_seq_fea_col : [B, seq_len, emb_dim]
        ''' 
        seq_len = hist_poi_seq_fea_col.shape[-2]
        din_deep_layers, din_activation = [64, 32], 'relu'
        if len(cur_poi_seq_fea_col.shape) == 2:
            cur_poi_seq_fea_col = tf.expand_dims(cur_poi_seq_fea_col, 1)

        with tf.variable_scope("attention_layer_%s" % att_type, reuse=tf.AUTO_REUSE):
            position_emb = tf.get_variable('position_emb',
                                            shape=[1, seq_len, self.embed_dim],
                                            initializer=tf.random_uniform_initializer(minval=-1e-4, maxval=1e-4))
            position_emb_tile = tf.zeros_like(hist_poi_seq_fea_col) + position_emb
            cur_poi_emb_rep = tf.tile(cur_poi_seq_fea_col, [1, seq_len, 1])
            din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col, position_emb_tile], axis=-1)

            activation = tf.nn.relu if din_activation == "relu" else tf.nn.tanh
            input_layer = din_all
            if aux_info is not None:
                din_all = tf.concat([din_all, aux_info], axis=-1)

            for i in range(len(din_deep_layers)):
                deep_layer = tf.layers.dense(input_layer, int(din_deep_layers[i]), activation=activation,
                                             name=att_type + 'f_%d_att' % i)
                input_layer = deep_layer

            din_output_layer = tf.layers.dense(input_layer, 1, activation=None, name=att_type + 'fout_att')
            din_output_layer = tf.reshape(din_output_layer, [-1, 1, seq_len])  # None, 1, seq_len

            # Mask
            if mask is None:
                mask = tf.count_nonzero(tf.reduce_sum(hist_poi_seq_fea_col, -1), axis=1, keep_dims=True)
            if mask is not None:
                if len(mask.shape) == 2:
                    key_masks = tf.sequence_mask(mask, seq_len)  # [B, 1, seq_len]
                else:
                    key_masks = mask

                outputs = tf.cast(key_masks, tf.float32) * din_output_layer
            else:
                outputs = din_output_layer

            weighted_outputs = tf.matmul(outputs, hist_poi_seq_fea_col)  # N, 1, seq_len, (N, seq_len , 24)= (N, 1, 24)

            weighted_outputs = tf.reshape(weighted_outputs, [-1, hist_poi_seq_fea_col.shape[-1]])  # N, 8

            if len(mask.shape) == 2:
                logger.info('cnt mode')
                mask = tf.cast(mask, tf.float32)
                seq_len = tf.cast(seq_len, tf.float32)
                weighted_outputs = mask / seq_len * weighted_outputs + (1- mask / seq_len) * tf.squeeze(cur_poi_seq_fea_col, 1)

            return weighted_outputs
    
    def multi_head_self_attention_qkv(self, query, values, head_num=1, mask=None, name='0'):
        with tf.variable_scope('multi_head_attention_qkv_' + name, reuse=tf.AUTO_REUSE):
            if head_num == 1:
                return self.self_attention_qkv(query, values, mask, name)
            def split_heads(x):
                len_q = x.get_shape().as_list()[-2]
                len_emb = x.get_shape().as_list()[-1]
                x = tf.reshape(x, (-1, len_q, head_num, len_emb // head_num))
                x = tf.transpose(x, perm=[0, 2, 1, 3])
                return x
            query, values = split_heads(query), split_heads(values)
            mask = mask[:, None, :, None]
            output, _ = self.self_attention_qkv(query, values, mask, name)
            output = tf.transpose(output, perm=[0, 2, 1, 3])

            len_q = query.get_shape().as_list()[-2]
            len_emb = query.get_shape().as_list()[-1]
            return tf.reshape(output, (-1, len_q, head_num * len_emb))

    def self_attention_qkv(self, query, values, mask=None, name='0'):
        # query : [B, seq_q, emb]
        # key : [B, seq_k, emb]
        # mask : [B, 1], dtype=float32
        with tf.variable_scope('attention_qkv_' + name, reuse=tf.AUTO_REUSE):
            num_units = query.get_shape().as_list()[-1]
            Q = self.easy_dense(query, num_units, 'query')
            #Q = tf.expand_dims(Q, 1)
            K = self.easy_dense(values, num_units, 'key')
            V = self.easy_dense(values, num_units, 'value')
            scores = tf.matmul(Q, K, transpose_b=True) # [B, seq_q, seq_k]
            logger.info("#Z scores:{}".format(scores))

            att_scores = tf.nn.softmax(scores)  # [B, seq_q, seq_k]
            seq_k = values.get_shape().as_list()[-2]
            seq_q = query.get_shape().as_list()[-2]
            if mask is not None:
                # key_masks = tf.sequence_mask(mask, seq_k)  # [B, 1, seq_k]
                # key_masks = tf.tile(key_masks, [1, seq_q, 1])

                # paddings = tf.zeros_like(att_scores)
                # att_scores = tf.where(mask, att_scores, paddings) 
                if len(mask.shape) == 2:
                    mask = mask[:, :, None]
                att_scores = att_scores * mask
                # att_scores = tf.math.divide_no_nan(att_scores, tf.cast(tf.reduce_sum(mask, -2, keep_dims=True), tf.float32))
                output = tf.matmul(att_scores, V)   # B, seq_q, emb
            else:
                output = tf.matmul(att_scores, V) / seq_q
            output = self.easy_dense(output, num_units, 'qkv_ffn')
            output = output * mask
            return output, att_scores
    
    def group_attention_layer(self, cur_poi_seq_fea_col, hist_poi_seq_fea_col, mask, att_type, group_flag=None):   
        seq_len = hist_poi_seq_fea_col.shape[-2]
        din_deep_layers, din_activation = [64, 32], 'relu'
        if len(cur_poi_seq_fea_col.shape) == 2:
            cur_poi_seq_fea_col = tf.expand_dims(cur_poi_seq_fea_col, 1)

        with tf.variable_scope("group_attention_layer_%s" % att_type, reuse=tf.AUTO_REUSE):
            position_emb = tf.get_variable('position_emb',
                                            shape=[1, seq_len, self.embed_dim],
                                            initializer=tf.random_uniform_initializer(minval=-1e-4, maxval=1e-4))
            position_emb_tile = tf.zeros_like(hist_poi_seq_fea_col) + position_emb
            cur_poi_emb_rep = tf.tile(cur_poi_seq_fea_col, [1, seq_len, 1])
            din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col, position_emb_tile], axis=-1)

            activation = tf.nn.relu if din_activation == "relu" else tf.nn.tanh
            input_layer = din_all

            for i in range(len(din_deep_layers)):
                deep_layer = tf.layers.dense(input_layer, int(din_deep_layers[i]), activation=activation,
                                             name=att_type + 'f_%d_att' % i)
                input_layer = deep_layer

            din_output_layer = tf.layers.dense(input_layer, 1, activation=None, name=att_type + 'fout_att')
            din_output_layer = tf.reshape(din_output_layer, [-1, 1, seq_len])  # None, 1, seq_len

            # Mask
            if mask is not None:
                if len(mask.shape) == 2:
                    key_masks = tf.sequence_mask(mask, seq_len)  # [B, 1, seq_len] 这个已经是三维的了
                else:
                    key_masks = mask

                outputs = tf.cast(key_masks, tf.float32) * din_output_layer
            else:
                outputs = din_output_layer
            
            outputs_group = tf.math.divide_no_nan(tf.exp(outputs), tf.transpose(tf.matmul(tf.cast(group_flag, tf.float32), tf.transpose(tf.exp(outputs), [0, 2, 1])), [0, 2, 1]))

            weighted_outputs = tf.matmul(outputs_group, hist_poi_seq_fea_col)  # N, 1, seq_len, (N, seq_len , 24)= (N, 1, 24)

            weighted_outputs = tf.reshape(weighted_outputs, [-1, hist_poi_seq_fea_col.shape[-1]])  # N, 8

            if mask is not None and len(mask.shape) == 2:
                mask = tf.cast(mask, tf.float32)
                seq_len = tf.cast(seq_len, tf.float32)
                weighted_outputs = mask / seq_len * weighted_outputs + (1- mask / seq_len) * tf.squeeze(cur_poi_seq_fea_col, 1)
            else:
                weighted_outputs = weighted_outputs

            return weighted_outputs
            
    def din_repeat(self, tgt_poi_emb, seq_poi_emb, mask, att_type):
         with tf.variable_scope("din_repeat_unit_%s" % att_type, reuse=tf.AUTO_REUSE):
            qkv_output, qkv_score = self.self_attention_qkv(seq_poi_emb, seq_poi_emb, mask, att_type)
            repeat_flag = tf.linalg.band_part(tf.cast(qkv_score > tf.expand_dims(tf.linalg.diag_part(qkv_score), -1) * 0.9, tf.int32), 0, -1) # [B, L, L]
            group_din_output = self.group_attention_layer(tgt_poi_emb, seq_poi_emb, mask, att_type+'_group', repeat_flag)
            group_cnt = tf.reduce_sum(repeat_flag, -1)
            repeat_mask = tf.cast(tf.reduce_sum(repeat_flag, -2, keep_dims=True) > 1, tf.int32) # [B, 1, L]
            group_cnt_emb_default = tf.get_variable('group_cnt_emb_default',
                                                    shape=[30, self.embed_dim],
                                                    initializer=tf.random_uniform_initializer(minval=-1e-4, maxval=1e-4))
            group_cnt_emb = tf.gather(group_cnt_emb_default, group_cnt)
            cnt_din_output = self.easy_attention_layer(tgt_poi_emb, seq_poi_emb, repeat_mask, att_type+'_cnt', group_cnt_emb)
            return group_din_output, cnt_din_output

    def deep_inference(self, deep, name='0', aux_info=None):
        with tf.variable_scope("deep_inference_%s" % name, reuse=tf.AUTO_REUSE):
            deep_layers_list = self.deep_layers_size_list
            dropout_deep = self.dropout
            output_dim = self.label_dim
            activation = self.activation
            if activation == "leakyRelu":
                activation_func = functools.partial(tf.nn.leaky_relu, alpha=0.3)
            else:
                activation_func = tf.nn.relu

            # deep_layers_list.append(output_dim)
            # dropout_deep.append(-1)
            for i in range(len(deep_layers_list)):
                deep = tf.layers.dense(inputs=deep, units=deep_layers_list[i],
                                       activation=activation_func,
                                       kernel_initializer=tf.truncated_normal_initializer(0.0, 1e-2),
                                       name='%s_fc_%d' % (name, i))
                # if dropout_deep[i] > 0:
                #     deep = tf.layers.dropout(deep, dropout_deep[i])
            
            deep = tf.layers.dense(inputs=deep, units=output_dim,
                                   kernel_initializer=tf.truncated_normal_initializer(0.0, 1e-2),
                                   name='%s_fc_output' % name)
            return deep
    
    def multi_task_inference(self, deep_list, task_num=2, name='0', aux_info=None, early_stop=False):
        with tf.variable_scope("multi_task_inference_%s" % name, reuse=tf.AUTO_REUSE):
            deep_layers_list = self.deep_layers_size_list
            dropout_deep = self.dropout
            output_dim = self.label_dim
            activation = self.activation
            if activation == "leakyRelu":
                activation_func = functools.partial(tf.nn.leaky_relu, alpha=0.3)
            else:
                activation_func = tf.nn.relu

            # deep_layers_list.append(output_dim)
            # dropout_deep.append(-1)
            transfer_list = []
            deep = tf.concat(deep_list, -1)
            for i in range(task_num + 1):
                transfer_layer = tf.layers.dense(inputs=deep, units=deep_layers_list[0]//2,
                                        activation=activation_func,
                                        kernel_initializer=tf.truncated_normal_initializer(0.0, 1e-2),
                                        name='transfer_layer_%d' % (i))
                transfer_list.append(transfer_layer)

            output = []
            for t in range(task_num):
                deep = tf.concat([transfer_list[-1], transfer_list[t]], -1)
                for i in range(len(deep_layers_list)):
                    deep = tf.layers.dense(inputs=deep, units=deep_layers_list[i],
                                        activation=activation_func,
                                        kernel_initializer=tf.truncated_normal_initializer(0.0, 1e-2),
                                        name='task_%d_fc_%d' % (t, i))
                    if dropout_deep[i] > 0:
                        deep = tf.layers.dropout(deep, dropout_deep[i])
                if not early_stop:
                    deep = tf.layers.dense(inputs=deep, units=output_dim, activation=tf.nn.sigmoid,
                                        kernel_initializer=tf.truncated_normal_initializer(0.0, 1e-2),
                                        name='task_%d_fc_output' % t)
                    deep = tf.squeeze(deep, -1)
                output.append(deep)
            return output
    
    def se_block(self, input, emb_dim, name_scope, scene_feature2=None):
        se_ratio = 4
        act_batch_size = input.shape[0].value
        fea_num = input.shape[1].value / emb_dim
        se_input = tf.reshape(input, [-1, fea_num, emb_dim])
        neuron_num = max(1, int(math.ceil(1.0 * fea_num / se_ratio)))
        logger.info(
            "[name_scope=%s] se_block, neuron_num %s, input %s, se_input %s, act_batch_size %s, emb_dim %s",
            name_scope, neuron_num, input, se_input, act_batch_size, emb_dim)
        # b,n,d -> b,n

        '''AUTOFeas
        aim : add scene info to se_block
        self.ep_feature_emb2 : [batch, scene_len * emb_dim]
        scene_output : [batch, emb_dim]
        '''
        if scene_feature2 is not None:
            scene_output = tf.layers.dense(scene_feature2,
                                        emb_dim,
                                        name=name_scope + "_scene_fc")
            squeeze_output = tf.reduce_mean(se_input, axis=2)
            squeeze_output = tf.concat([scene_output, squeeze_output], axis=-1)
        else:
            squeeze_output = tf.reduce_mean(se_input, axis=2)

        with tf.variable_scope("se_block_" + name_scope):
            deep_layer = tf.layers.dense(squeeze_output,
                                            neuron_num,
                                            activation=tf.nn.relu,
                                            name="fc0")
            excitation_out = tf.layers.dense(deep_layer,
                                            fea_num,
                                            # activation=tf.nn.relu,
                                            activation=tf.nn.sigmoid,
                                            name="fc1")

            excitation_out = tf.expand_dims(excitation_out, axis=-1)
            reweight_out = tf.multiply(2 * excitation_out, se_input)
            output = tf.reshape(reweight_out, [-1, fea_num * emb_dim])
            return output

    def _count_param(self):  # 计算网络参数量
        total_parameters = 0
        for v in tf.trainable_variables():
            shape = v.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters
        
    def get_optimizer(self, all_loss, run_mode):
        # 学习率设置
        learning_rate = self.learning_rate
        hash_learning_rate = self.hash_learning_rate
        lr_decay = self.lr_decay
        optimizer = self.optimizer

        logger.info('#WSL all_loss is %s'%all_loss)
        logger.info('#WSL run_mode is %s'%run_mode)

        # if lr_decay:
        #     learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
        #                                                global_step=tf.train.get_global_step(),
        #                                                decay_steps=1000,
        #                                                decay_rate=0.98,
        #                                                staircase=False)

        # lr = get_lr(learning_rate)
        lr = learning_rate
        slr = hash_learning_rate
        tf.summary.scalar('learning_rate', lr)

        # optimizer设置
        if optimizer == 'momentum':
            sparse_optimizer = HashMomentumOptimizer(learning_rate=slr, momentum=0)
            dense_optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0)
        elif optimizer == 'adagrad':
            sparse_optimizer = HashAdagradOptimizer(learning_rate=slr)
            dense_optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
        elif optimizer == 'adam':
            sparse_optimizer = HashAdagradOptimizer(slr)
            dense_optimizer = tf.contrib.opt.LazyAdamOptimizer(lr)
        else:
            sparse_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            dense_optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        sparse_tables = [tf.get_collection("HASHTABLE")]
        var_list_not_in_hashtable = []
        with tf.control_dependencies(update_ops):
            for v in tf.trainable_variables():
                if 'emb' in v.name:
                    sparse_tables.append(v)
                else:
                    var_list_not_in_hashtable.append(v)
            logger.info('HASHTABLE %s', sparse_tables)
            logger.info('var_list_not_in_hashtable %s', var_list_not_in_hashtable)

            sparse_var_grads = sparse_optimizer.compute_gradients(all_loss, var_list=sparse_tables)
            sparse_op = sparse_optimizer.apply_gradients(sparse_var_grads)
            dnn_var_grads = dense_optimizer.compute_gradients(all_loss, var_list=var_list_not_in_hashtable)
            dnn_op = dense_optimizer.apply_gradients(dnn_var_grads)

            if run_mode == 'G_train':
                G_vars = [tf.trainable_variables(scope='Generator')]
                G_vars_grads = dense_optimizer.compute_gradients(all_loss, var_list=G_vars)
                if not G_vars_grads:
                    raise ValueError("Gradient list is empty.")
                for grad, var in G_vars_grads:
                    if grad is None:
                        logger.info("Gradient for variable {} is None.".format(var.name))
                G_op = dense_optimizer.apply_gradients(G_vars_grads)
                logger.info("#WSL#G_vars:%s", G_vars)

                global_step_op = tf.assign_add(tf.train.get_global_step(), 1)
                train_ops = tf.group([update_ops, G_op, global_step_op])
            else:
                global_step_op = tf.assign_add(tf.train.get_global_step(), 1)
                train_ops = tf.group([update_ops, sparse_op, dnn_op, global_step_op])
            return train_ops
        
    def model_fn(self, features, labels, mode):
        is_training = True if mode == tf.estimator.ModeKeys.TRAIN else False
        # cate_embed, num_embed = model.process_data(features, is_training)
        click_probs, order_probs, price_pred, cand_poi_list = self.inference(features, name='DeepFM', is_training=is_training)
        click_mask = tf.concat([tf.zeros_like(click_probs)[:, :-20, :], tf.ones_like(click_probs)[:, -20:, :]], 1)

        click_probs = tf.sigmoid(click_probs)
        order_probs = tf.sigmoid(order_probs)

        click_probs = tf.squeeze(click_probs, -1)
        order_probs = tf.squeeze(order_probs, -1)

        def cal_rank_score(click_probs, order_probs, price_pred):
            bid = 1
            k1 = 0.7
            k2 = 0.4
            rs1 = bid * tf.maximum(3500 * order_probs / bid, 1)
            rs2 = k1 * click_probs * bid + k2 * click_probs * order_probs * price_pred
            rs = rs1 + rs2
            return rs
        
        rs = cal_rank_score(click_probs, order_probs, price_pred)
        rs_sort_idx = tf.argsort(rs, -1, direction='DESCENDING')
        logger.info('#WSL:tf.argsort is %s'%rs_sort_idx)
        poi_sort = tf.gather(cand_poi_list, rs_sort_idx, batch_dims=1)
        poi_sort = tf.reshape(poi_sort, [-1, self.params['len_pv']])
        poi_sort = tf.identity(tf.concat([tf.cast(rs_sort_idx, tf.float32), click_probs, order_probs, rs], -1), name='rerank_result')

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'rerank_result': poi_sort,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        click_probs = tf.reshape(click_probs, (-1, 1))
        pctr_logits = tf.identity(click_probs, name='click_probs')
        order_probs = tf.reshape(order_probs, (-1, 1))
        pcvr_logits = tf.identity(order_probs, name='order_probs')

        label_list = tf.reshape(labels['label_list'], (-1, 1))
        click_list = tf.reshape(labels['click_list'], (-1, 1))
        click_mask = tf.reshape(click_mask, (-1, 1))
        if mode == tf.estimator.ModeKeys.TRAIN:
            if "loss" in self.params['taskName']:
                gamma = 2
                weights = 10 * tf.where(tf.equal(click_list, 1), tf.pow(tf.maximum(0.7-click_probs, 0), gamma-0.4), tf.pow(click_probs, gamma))
                with tf.variable_scope('loss'):
                    loss = self.loss_function(tf.reshape(click_list, shape=[-1, 1]),
                                            tf.reshape(click_probs, shape=[-1, 1]), weights * click_mask)
            else:
                with tf.variable_scope('loss'):
                    loss = self.loss_function(tf.reshape(click_list, shape=[-1, 1]),
                                            tf.reshape(click_probs, shape=[-1, 1]), click_mask)
                    loss += self.loss_function(tf.reshape(label_list, shape=[-1, 1]),
                                            tf.reshape(order_probs, shape=[-1, 1]))


            res = self._count_param()
            logger.info('--------网络总参数量---------')
            logger.info(res)
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=self.get_optimizer(loss), training_hooks=[])

        if mode == tf.estimator.ModeKeys.EVAL:
            losses = self.loss_function(click_list, click_probs, click_mask) + self.loss_function(label_list, order_probs)

            
            metrics = {}
            metrics['logloss'] = tf.metrics.mean(losses)
            metrics['pctr'] = tf.metrics.mean(click_probs)
            metrics['ctr'] = tf.metrics.mean(click_list)
            metrics['auc'] = tf.metrics.auc(labels=click_list, predictions=click_probs, weights=click_mask, num_thresholds=5000, name='auc')
            metrics['auc_cvr'] = tf.metrics.auc(labels=label_list, predictions=order_probs, weights=click_mask, num_thresholds=5000, name='auc_cvr')
            
            # threshold_weights = {
            #     # 'auc_1e-2': ~(tf.equal(clk_labels, 0) & tf.less(click_probs, 1e-2)),
            #     # 'auc_2e-2': ~(tf.equal(clk_labels, 0) & tf.less(click_probs, 2e-2)),
            #     # 'auc_3e-2': ~(tf.equal(clk_labels, 0) & tf.less(click_probs, 3e-2)),
            #     'auc_4e-2': ~(tf.equal(click_list, 0) & tf.less(click_probs, 4e-2)),
            # }
            # for name, weight in threshold_weights.items():
            #     metrics[name] = tf.metrics.auc(labels=click_list, predictions=click_probs, weights=weight, num_thresholds=5000, name=name)


            return tf.estimator.EstimatorSpec(mode, loss=losses, eval_metric_ops=metrics)

