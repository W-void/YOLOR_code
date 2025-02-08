# -*- coding: utf-8 -*-
#%%
import functools
import math

import numpy as np
from tensorflow.contrib.lookup.lookup_ops import get_mutable_dense_hashtable
from tensorflow.contrib.opt import HashAdamOptimizer, HashAdagradOptimizer, HashMomentumOptimizer
from tensorflow.python.ops import array_ops
import tensorflow_probability as tfp
import pandas as pd

from model_base import *
from utils import *
import random
from itertools import combinations

#%%

logger = set_logger()


class DeepFMExp(DeepFM):
    def __init__(self, params):
        super(DeepFMExp, self).__init__(params)
        self.hierarchical_level = 3
        self.pv_len = 2 ** self.hierarchical_level
        self.ori_list = list(range(self.pv_len))

    def list_hash(self, ori_list):
        return [1<<i for i in ori_list]
    
    def build_context_cache(self):
        cache_key1 = []
        for l in range(self.hierarchical_level):
            cache_key1 += [combinations(self.ori_list, self.pv_len // 2**l)]
        cache_key2 = [sum(c) for c in cache_key1]
        cache_key_map = dict(zip(cache_key2, range(len(cache_key2))))
        return cache_key_map
    
    def HCEM(self, X_s, list_index, context_cache, poi_mask, level=3):
        # X_s : [B, len_pv, hidden_layer]
        # list_index : [self.pv_len]
        # context_cache : {int, tensor}
        # poi_mask : [B, len_pv]
        with tf.variable_scope('HCEM', reuse=tf.AUTO_REUSE):
            outputs = []
            len_pv = X_s.shape[-2]
            for i in range(level):
                tmp_X = tf.split(X_s, num_or_size_splits=2**i, axis=1)
                tmp_mask = tf.split(poi_mask, num_or_size_splits=2**i, axis=1)
                split_index = np.array_split(list_index, 2**i)
                logger.info("#WSL: tmp_X is %s"%tmp_X)
                split_index = [s*m for s, m in zip(split_index, tmp_mask)]
                index_2_key = [tf.reduce_sum(s) for s in split_index]

                # context = [tf.reduce_sum(self.multi_head_self_attention_qkv(X, X, 1, M, name='HCEM_%d'%i)[0], 1) for X, M in zip(tmp_X, tmp_mask)] # [B, hidden_layer]
                context = []
                for j, k in enumerate(index_2_key):
                    if k in context_cache.keys():
                        value = context_cache[k]
                    else:
                        X, M = tmp_X[j], tmp_mask[j]
                        value = tf.reduce_sum(self.multi_head_self_attention_qkv(X, X, 1, M, name='HCEM_%d'%i)[0], 1)
                        context_cache[k] = value
                    context.append(value)
                                
                context_reshape = tf.concat([tf.tile(C[:, None], [1, math.ceil(8 / 2**i), 1]) for C in context], axis=1) # [B, len_pv, hidden_layer]
                logger.info("#WSL: context_reshape is %s"%context_reshape)
                outputs.append(context_reshape[:, :len_pv])
            return tf.concat(outputs, -1)

    def loss_bpr(self, y, y_):
        with tf.variable_scope('task_loss'):
            dif_y = tf.expand_dims(y, -1) - tf.expand_dims(y, 1) # [B, pv_len, pv_len]
            dif_y_ = tf.expand_dims(y_, -1) - tf.expand_dims(y_, 1) # [B, pv_len, pv_len]
            loss = tf.reduce_mean(-tf.log(tf.sign(dif_y) * dif_y_))
        return loss

    def inference(self, features, name, is_training, mode):
        with tf.variable_scope('input_layer'):
            hist_poi_list = tf.cast(features['hist_poi_list'], tf.int64, name='his_poi_list')
            hist_poi_name_list = tf.cast(features['hist_poi_name_list'], tf.int64, name='his_poi_name_list')
            dj_poi_list = tf.cast(features['dj_poi_list'], tf.int64, name='dj_poi_list')
            dj_poi_name_list = tf.cast(features['dj_poi_name_list'], tf.int64, name='dj_poi_name_list')
            user_cat_fea_list = tf.cast(features['user_cat_fea_list'], tf.int64, name='user_cat_fea_list')
            user_num_fea_list = tf.cast(features['user_num_fea_list'], tf.float32, name='user_num_fea_list')

            cand_poi_list = tf.cast(features['cur_poi_list'], tf.int64, name='cand_poi_list')
            cand_poi_name_list = tf.cast(features['cand_poi_name_list'], tf.int64, name='cand_poi_name_list')
            poi_cat_fea_list = tf.cast(features['poi_cat_fea_list'], tf.int64, name='poi_cat_fea_list')
            poi_num_fea_list = tf.cast(features['poi_num_fea_list'], tf.float32, name='poi_num_fea_list')

            cand_poi_list = tf.reshape(cand_poi_list, [-1, self.params['len_pv'], 1])
            cand_poi_name_list = tf.reshape(cand_poi_name_list, [-1, self.params['len_pv'], 5])
            poi_cat_fea_list = tf.reshape(poi_cat_fea_list, [-1, self.params['len_pv'], self.params['len_pc']])
            poi_num_fea_list = tf.reshape(poi_num_fea_list, [-1, self.params['len_pv'], self.params['len_pn']])

            hist_poi_name_list = tf.reshape(hist_poi_name_list, [-1, 30, 5])
            dj_poi_name_list = tf.reshape(dj_poi_name_list, [-1, 20, 5])

            price_pred = poi_num_fea_list[:, :, 45]

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
                                                            export_optimizer_var=True
                                                            )
                self.emb_hashtable = emb_hashtable
                tf.add_to_collection("HASHTABLE", emb_hashtable)

            # 类别特征Embedding
            # reshaped_cate_embed = tf.reshape(cate_embed, [-1, self.cate_fea_num * self.embed_dim])

            seq_fea_list = [hist_poi_list, hist_poi_name_list, dj_poi_list, dj_poi_name_list, user_cat_fea_list, \
                            cand_poi_list, cand_poi_name_list, poi_cat_fea_list]

            hist_poi_list_emb, hist_poi_name_list_emb, dj_poi_list_emb, dj_poi_name_list_emb, user_cat_fea_list, \
                cand_poi_list_emb, cand_poi_name_list_emb, poi_cat_fea_list_emb = \
                self.table_lookup(emb_hashtable, seq_fea_list, self.freq_threshold, 'seq_fea_embed_lookup', True, self.embed_dim)
            
            # user_num_fea_emb = self.get_quatile_emb(tf.reshape(user_num_fea_list, [-1, self.params['len_un']]), self.quantiles_mat[:self.params['len_un']], name='user_num_fea')
            user_num_fea_emb = self.percentile_dis(tf.reshape(user_num_fea_list, [-1, self.params['len_un']]), 'user_num_fea', is_training)
            user_num_fea_emb = tf.reshape(user_num_fea_emb, [-1, self.params['len_un'], self.embed_dim])
            
            # poi_num_fea_emb = self.get_quatile_emb(tf.reshape(poi_num_fea_list, [-1, self.params['len_pn']]), self.quantiles_mat[self.params['len_un']:], name='poi_num_fea')
            poi_num_fea_emb = self.percentile_dis(tf.reshape(poi_num_fea_list, [-1, self.params['len_pn']]), 'poi_num_fea', is_training)
            poi_num_fea_emb = tf.reshape(poi_num_fea_emb, [-1, self.params['len_pv'], self.params['len_pn'], self.embed_dim])

        if mode == tf.estimator.ModeKeys.PREDICT:
            poi_mask = tf.where(tf.equal(cand_poi_list, tf.expand_dims(cand_poi_list[:, 0], 1)), tf.zeros_like(cand_poi_list), tf.ones_like(cand_poi_list))
            poi_mask = tf.concat([tf.ones_like(poi_mask[:, 0]), poi_mask[:, 1:, 0]], -1)
            poi_mask = tf.cast(poi_mask, tf.float32)
            logger.info('#WSL:poi_mask is %s'%poi_mask)
        else: 
            poi_mask = tf.where(tf.equal(cand_poi_list, 0), tf.zeros_like(cand_poi_list), tf.ones_like(cand_poi_list))
            poi_mask = tf.cast(poi_mask, tf.float32)
            poi_mask = tf.squeeze(poi_mask, -1)
            logger.info('#WSL:poi_mask is %s'%poi_mask)
        
        with tf.variable_scope('Seq'):
            def tiled_attention_layer(cand_poi_list_emb, hist_poi_list_emb, layer_name):
                ''' input shape
                    cand_poi_list_emb : [B, len_pv, 1, emb_dim]
                    hist_poi_list_emb : [B, seq_len, emb_dim]
                '''
                logger.info('#WSL: cand_poi_list_emb is {}, hist_poi_list_emb is {}.'.format(cand_poi_list_emb, hist_poi_list_emb))
                hist_poi_emb_tile = tf.tile(hist_poi_list_emb[:, None, :, :], [1, cand_poi_list_emb.shape[1], 1, 1])
                cand_poi_emb_din = self.easy_attention_layer(tf.reshape(cand_poi_list_emb, [-1, self.embed_dim]), tf.reshape(hist_poi_emb_tile, [-1, hist_poi_emb_tile.shape[-2], self.embed_dim]), None, layer_name)
                cand_poi_emb_din = tf.reshape(cand_poi_emb_din, [-1, self.params['len_pv'], 1, self.embed_dim]) 
                return cand_poi_emb_din

            cand_poi_emb_din = tiled_attention_layer(cand_poi_list_emb, hist_poi_list_emb, 'cand_din')
            cand_poi_emb_din_name = tiled_attention_layer(tf.reduce_mean(cand_poi_name_list_emb, -2, keep_dims=True), tf.reduce_mean(hist_poi_name_list_emb, -2), 'cand_din_name')

            dj_cand_poi_emb_din = tiled_attention_layer(cand_poi_list_emb, dj_poi_list_emb, 'dj_cand_din')
            dj_cand_poi_emb_din_name = tiled_attention_layer(tf.reduce_mean(cand_poi_name_list_emb, -2, keep_dims=True), tf.reduce_mean(dj_poi_name_list_emb, -2), 'dj_cand_din_name')


        with tf.variable_scope('Deep'):
            user_fea_emb = tf.concat([user_cat_fea_list, user_num_fea_emb], -2)
            poi_fea_emb = tf.concat([cand_poi_list_emb, cand_poi_name_list_emb, cand_poi_emb_din, cand_poi_emb_din_name, dj_cand_poi_emb_din, dj_cand_poi_emb_din_name, poi_cat_fea_list_emb, poi_num_fea_emb], -2)
            
            user_fea_emb = tf.reshape(user_fea_emb, [-1, 1, user_fea_emb.shape[-2] * user_fea_emb.shape[-1]])
            logger.info('#WSL, user_fea_emb is %s'%user_fea_emb)
            user_fea_emb_tile = tf.tile(user_fea_emb, [1, self.params['len_pv'], 1])
            poi_fea_emb = tf.reshape(poi_fea_emb, [-1, self.params['len_pv'], poi_fea_emb.shape[-2] * poi_fea_emb.shape[-1]])
            logger.info('#WSL, poi_fea_emb is %s'%poi_fea_emb)
            
        # X^s
        X_s, _ = self.multi_task_inference([user_fea_emb_tile, poi_fea_emb], early_stop=True) # [B, len_pv, hidden_layer]
        X_s = X_s[:, 9:17] # [B, 8, hidden_layer]
        poi_mask = poi_mask[:, 9:17]

        # X^C 
        context_cache = {}
        X_C = self.HCEM(X_s, self.list_hash(self.ori_list), context_cache, poi_mask, 3) # [B, 8, hidden_layer]

        # E^p
        position_emb_default = tf.get_variable('position_emb_default',
                                        shape=[1, 8, self.embed_dim],
                                        initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
        E_p = tf.zeros_like(X_s[:, :, :self.embed_dim]) + position_emb_default

        E_ctr = self.easy_dense(tf.concat([X_s, X_C, E_p], -1), 1, 'E_ctr', tf.nn.sigmoid) # [B, 8, 1]

        return E_ctr, poi_mask, X_s, X_C, E_p, context_cache

    
    def model_fn(self, features, labels, mode):
        is_training = True if mode == tf.estimator.ModeKeys.TRAIN else False
        # cate_embed, num_embed = model.process_data(features, is_training)
        click_probs, poi_mask, X_s, X_C, E_p, context_cache  = self.inference(features, 'DeepFM', is_training, mode)
        logger.info('#WSL run_mode is %s'%self.params['run_mode'])

        cur_poi_bid_list = features['cur_poi_bid_list'] # [B, len_pv]
                
        poi_mask = self.Print(poi_mask, "poi_mask", mode, poi_mask)

        if mode == tf.estimator.ModeKeys.PREDICT:            
            # 以精排返回的顺序为贪心结果，遍历其他排列
            max_index = tf.expand_dims(self.ori_list, 0) + tf.cast(tf.zeros_like(tf.squeeze(click_probs)), tf.int32)# [B, 8]
            max_reward = tf.reduce_max(click_probs, 1) # [B, 1]

            for _ in range(100000):
                shuffle_index = random.sample(self.ori_list, self.pv_len)
                tmp_index = tf.expand_dims(shuffle_index, 0) + tf.zeros_like(max_index)

                tmp_X_C = self.HCEM(tf.gather(X_s, tmp_index, batch_dims=1), self.list_hash(shuffle_index), context_cache, poi_mask, 3) # [B, 8, hidden_layer]
                tmp_X_s = tf.gather(X_s, tmp_index, batch_dims=1)
                tmp_E_p = tf.gather(E_p, tmp_index, batch_dims=1)

                tmp_E_ctr = self.easy_dense(tf.concat([tmp_X_s, tmp_X_C, tmp_E_p], -1), 1, 'E_ctr', tf.nn.sigmoid) # [B, 8, 1]
                tmp_reward = tf.reduce_max(tmp_E_ctr, 1) # [B, 1]

                select_index = tf.where(max_reward < tmp_reward, tf.ones_like(max_reward), tf.zeros_like(max_reward)) # [B, 1]
                select_index = tf.squeeze(tf.cast(select_index, tf.int32), 1)
                max_reward = tf.gather(tf.concat([max_reward, tmp_reward], 1), select_index[:, None], batch_dims=1)
                max_index = tf.gather(tf.stack([max_index, tmp_index], 1), select_index, batch_dims=1)

            # rerank_result = tf.identity(tf.concat([tf.cast(rs_sort_idx, tf.float32), click_probs, rs], -1), name='rerank_result')
            rerank_result = tf.identity(max_index, name='rerank_result')
            predictions = {
                'rerank_result': rerank_result,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        click_list = tf.reshape(labels['click_list'], (-1, self.params['len_pv']))
        click_list = click_list[:, 9:17]

        loss_bpr = self.loss_bpr(click_list, click_probs)
        tf.summary.scalar('loss/bpr', loss_bpr)

        click_probs = tf.reshape(click_probs, (-1, 1))        
        click_list = tf.reshape(click_list, (-1, 1))
        
        loss_ctr = self.loss_function(click_list, click_probs)
        tf.summary.scalar('loss/ctr', loss_ctr)
        alpha = 0.05
        loss = loss_ctr + alpha * loss_bpr

        if mode == tf.estimator.ModeKeys.TRAIN:
            res = self._count_param()
            logger.info('--------网络总参数量---------')
            logger.info(res)
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=self.get_optimizer(loss, self.params['run_mode']), training_hooks=[])

        if mode == tf.estimator.ModeKeys.EVAL:
            metrics = {}
            metrics['avg/ctr_p'] = tf.metrics.mean(click_probs)
            metrics['avg/ctr'] = tf.metrics.mean(click_list)

            metrics['auc/ctr'] = tf.metrics.auc(labels=click_list, predictions=click_probs, num_thresholds=5000, name='auc')
            
            metrics['loss/ctr'] = tf.metrics.mean(loss_ctr)
                  
            # clk_labels = tf.reshape(click_list, (-1, self.params['len_pv']))[:, 7:-2]
            # click_probs = tf.reshape(click_probs, (-1, self.params['len_pv']))[:, 7:-2]
            # poi_mask = tf.reshape(poi_mask, (-1, self.params['len_pv']))[:, 7:-2]
            # sort_idx = tf.argsort(click_probs + (1 - poi_mask) * 2)
            # pos_M, neg_N = tf.reduce_sum(clk_labels, -1), tf.reduce_sum((1 - clk_labels) * poi_mask, -1)
            # gauc = tf.math.divide_no_nan(tf.reduce_sum(tf.cast(1 + sort_idx, tf.float32) * clk_labels, -1) - pos_M * (pos_M + 1) / 2, pos_M * neg_N)
            # list_mask = tf.cast(pos_M * neg_N > 0, tf.float32)
            # metrics['gauc'] = tf.metrics.mean(gauc, weights=list_mask)


            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

