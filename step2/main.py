# -*- coding: utf-8 -*-
from __future__ import print_function

from data import *
from model_exp import *
from utils import *
import time
from sklearn import metrics
import collections
import re

_EVENT_FILE_GLOB_PATTERN = 'events.out.tfevents.*'

flags = tf.flags

flags.DEFINE_string("worker_hosts", "", "")
flags.DEFINE_string("evaluator_hosts", "", "")
flags.DEFINE_string("run_mode", "train_and_evaluate", "run mode in [train, train_and_evaluate, predict]")
flags.DEFINE_string("job_name", "worker", "One of 'ps', 'worker', 'chief', 'evaluator'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")
flags.DEFINE_string("event_file_pattern", "eval/*.tfevents.*", "eval data")
flags.DEFINE_string("taskName", "", "")

# 数据处理
flags.DEFINE_integer("len_pv", 20, "len of one pv")
flags.DEFINE_integer("len_uc", 20, "len of user cate feature")
flags.DEFINE_integer("len_un", 20, "len of user num feature")
flags.DEFINE_integer("len_pc", 20, "len of poi cate feature")
flags.DEFINE_integer("len_pn", 20, "len of poi num feature")
# flags.DEFINE_string("total_cate_fea", "", "total cate feats")
# flags.DEFINE_string("total_num_fea", "", "total num feats")
# flags.DEFINE_string("cate_fea", "", "test cate feats")
# flags.DEFINE_string("num_fea", "", "test num feats")
# flags.DEFINE_string("bias_cate_fea", "", "bias cate fea")
# flags.DEFINE_string("bias_dense_fea", "", "bias dense fea")

flags.DEFINE_integer("freq_threshold", 1000, "poiid freq_threshold")
flags.DEFINE_string("dict_bucket_file", "dict_bucket_file", "dict_bucket_file")
flags.DEFINE_string("src_filename", "src_filename", "src stat data")

# kv memory
flags.DEFINE_boolean("is_kv_memory", False, "is or not kv memory")
flags.DEFINE_integer("kv_mem_num", 20, "kv memory number")
flags.DEFINE_integer("kv_embed_dim", 8, "the dim of embedding numeric features")

# 模型参数
flags.DEFINE_integer("embed_dim", 8, "embed_dim")
flags.DEFINE_integer("random_seed", 2017, "random seed")

flags.DEFINE_integer("batch_size", 256, "batch size")
flags.DEFINE_float("learning_rate", 0.02, "learning rate")
flags.DEFINE_float("hash_learning_rate", 0.02, "hash_learning rate")
flags.DEFINE_boolean("lr_decay", False, "lr_decay")
flags.DEFINE_string("optimizer", 'adam', "optimizer")

flags.DEFINE_string("deep_layers", "128,64", "deep layer size")
flags.DEFINE_string('activation', 'tf.nn.relu', 'nn activation')
flags.DEFINE_string('batch_norm', 'False,False', 'batch_norm')
flags.DEFINE_string("dropout", "1.0,1.0,1.0", "dropout")
flags.DEFINE_float('l2_reg', 0.0001, 'l2 regularization')

# 任务参数
flags.DEFINE_integer("epoch", 100, "epoch")
# flags.DEFINE_string("task", 'train', "task: train, evaluate or test")
flags.DEFINE_string("save_param_filename", "model_file", "log directory")
# flags.DEFINE_integer("train_steps", 3000000, "total steps for train")
# flags.DEFINE_integer("valid_steps", 3000000, "total steps for valid")
flags.DEFINE_string("model_dir", "./model_dir", "dir for model")
flags.DEFINE_string("pretrain_model_dir", "", "dir for warmstart model")
flags.DEFINE_string("dict_dir", "", "dir for dict")
flags.DEFINE_string("pred_res_dir", "", "pred_res_dir")

FLAGS = tf.flags.FLAGS
tf.logging.set_verbosity(tf.logging.DEBUG)
tf.logging.info("FLAGS: %s" % FLAGS)

class Trainer(object):
    def __init__(self):
        self.job_name = FLAGS.job_name
        self.task_index = FLAGS.task_index
        self.worker_hosts = FLAGS.worker_hosts.split(",")
        self.evaluator_hosts = FLAGS.evaluator_hosts.split(",")
        tf.logging.info('Worker hosts are: %s' % self.worker_hosts)
        tf.logging.info('eval hosts are: %s' % self.evaluator_hosts)

        tf.logging.info('job_name : %s' % self.job_name)
        tf.logging.info('task_index : %d' % self.task_index)

        save_checkpoints_steps = 800
        if 'daily' in FLAGS.taskName:
            save_checkpoints_steps = 300
        session_config = tf.ConfigProto(allow_soft_placement=True)
        self.run_config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir,
                                                 save_summary_steps=save_checkpoints_steps,
                                                 save_checkpoints_steps=save_checkpoints_steps,
                                                 keep_checkpoint_max=0,
                                                 session_config=session_config,
                                                 )
        
        self.feature_protocal = {
            'userid': tf.FixedLenFeature([1], tf.int64),
            # 'pageid': tf.FixedLenFeature([1], tf.int64),
            'hist_poi_list': tf.FixedLenFeature([30], tf.int64),
            'hist_poi_name_list': tf.FixedLenFeature([150], tf.int64),
            'dj_poi_list': tf.FixedLenFeature([20], tf.int64),
            'dj_poi_name_list': tf.FixedLenFeature([100], tf.int64),
            'user_cat_fea_list': tf.FixedLenFeature([FLAGS.len_uc], tf.int64),
            'user_num_fea_list': tf.FixedLenFeature([FLAGS.len_un], tf.float32),

            'cur_poi_list': tf.FixedLenFeature([FLAGS.len_pv], tf.int64),
            'cand_poi_name_list': tf.FixedLenFeature([FLAGS.len_pv * 5], tf.int64),
            'poi_cat_fea_list': tf.FixedLenFeature([FLAGS.len_pc * FLAGS.len_pv], tf.int64),
            'poi_num_fea_list': tf.FixedLenFeature([FLAGS.len_pn * FLAGS.len_pv], tf.float32),

            'cur_poi_bid_list': tf.FixedLenFeature([FLAGS.len_pv], tf.float32),
            'cur_poi_charge_rate_list': tf.FixedLenFeature([FLAGS.len_pv], tf.float32),
        }
        self.label_protocal = {
            'label_list': tf.FixedLenFeature([FLAGS.len_pv], tf.float32),
            'click_list': tf.FixedLenFeature([FLAGS.len_pv], tf.float32),
        }

        # self.total_cate_fea_num = self.get_fea_num(FLAGS.total_cate_fea)
        # self.total_numerical_fea_num = self.get_fea_num(FLAGS.total_num_fea)
        # self.cate_fea_num = self.get_fea_num(FLAGS.cate_fea)
        # self.numerical_fea_num = self.get_fea_num(FLAGS.num_fea)
        # self.cate_fea_idx = self.get_fea_idx(FLAGS.total_cate_fea, FLAGS.cate_fea, 'cate_idx')
        # self.numerical_fea_idx = self.get_fea_idx(FLAGS.total_num_fea, FLAGS.num_fea, 'num_idx')

        self.params = {
            'run_mode' : FLAGS.run_mode,
            'len_pv': FLAGS.len_pv,  # 29
            'len_uc': FLAGS.len_uc,  # 21
            'len_un': FLAGS.len_un,  # 17
            'len_pc': FLAGS.len_pc,  # 90
            'len_pn': FLAGS.len_pn,  # 61
            'taskName': FLAGS.taskName,

            # 'total_cate_fea_num': self.total_cate_fea_num,
            # 'total_numerical_fea_num': self.total_numerical_fea_num,
            # 'cate_fea_num': self.cate_fea_num,
            # 'numerical_fea_num': self.numerical_fea_num,
            # 'cate_idx': self.cate_fea_idx,
            # 'num_idx': self.numerical_fea_idx,
            # 'num_fea': FLAGS.num_fea,

            'freq_threshold': FLAGS.freq_threshold,

            'dict_bucket_file': FLAGS.dict_bucket_file,
            'src_filename': FLAGS.src_filename,

            'is_kv_memory': FLAGS.is_kv_memory,
            'kv_mem_num': FLAGS.kv_mem_num,
            'kv_embed_dim': FLAGS.kv_embed_dim,

            'embed_dim': FLAGS.embed_dim,
            'random_seed': FLAGS.random_seed,
            'batch_size': FLAGS.batch_size,
            'learning_rate': FLAGS.learning_rate,
            'hash_learning_rate': FLAGS.hash_learning_rate,
            'lr_decay': FLAGS.lr_decay,
            'optimizer': FLAGS.optimizer,
            'activation': FLAGS.activation,
            'l2_reg': FLAGS.l2_reg,
            'deep_layers': FLAGS.deep_layers,
            'batch_norm': FLAGS.batch_norm,
            'dropout': FLAGS.dropout
        }
        self.model = DeepFMExp(self.params)
        self.model_fn = self.model.model_fn
        self.estimator = self.customer_estimator()

        tf.logging.info("params: %s" % self.params)

    def get_fea_num(self, fea_str):
        fea_ary = self.feat_split(fea_str)
        return fea_ary.size

    def get_fea_idx(self, total_fea_str, test_fea_str, fea_type):
        total_fea_ary = self.feat_split(total_fea_str)
        test_fea_ary = self.feat_split(test_fea_str)

        for i in test_fea_ary:
            if i not in total_fea_ary:
                raise ValueError('{} not in total feature list.'.format(i))

        idx = np.where(np.isin(total_fea_ary, test_fea_ary))[0]
        # return tf.constant(idx, tf.int32, name='{}_feature_index'.format(fea_type))
        return idx

    def feat_split(self, feat_str):
        fea_list = feat_str.split(',')
        return np.array(fea_list)

    def build_serving_input(self):
        """用于构建保存模型的输入格式"""
        # cate_fea_num = self.params['cate_fea_num']
        # num_fea_num = self.params['numerical_fea_num']

        feature_spec = {k: tf.placeholder(dtype=v.dtype, shape=[None, v.shape[0]], name=k) for k, v in self.feature_protocal.items()}

        # feature_spec['cur_poi_bid_list'] = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.len_pv], name='cur_poi_bid_list')
        feature_spec['cur_poi_type_list'] = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.len_pv], name='cur_poi_type_list')
        # feature_spec['cur_poi_charge_rate_list'] = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.len_pv], name='cur_poi_charge_rate_list')

        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

        return serving_input_receiver_fn

    def save_eval_res(self):
        """从tensorboard文件读取最优ckpt并保存，生成最终发布的模型文件"""
        logger.info('model_dir %s', FLAGS.model_dir)

        logger.info('ckpt dir %s', os.path.join(FLAGS.model_dir, 'eval'))
        eval_metrics_dict = read_eval_metrics(os.path.join(FLAGS.model_dir, 'eval'))
        logger.info('eval_metrics_dict %s', eval_metrics_dict)
        checkpoint_path = self.get_best_eval_checkpoint(eval_metrics_dict, self.params['src_filename'], True,
                                                   FLAGS.model_dir)
    
    def get_best_eval_checkpoint(self, eval_metrics_dict, src_filename, isSave, save_dir):
        """从tensorboard文件读取测试结果，返回最优ckpt，并保存测试结果至save_dir"""
        # result_best_auc = sorted([(step, metrics['checkpoint_path'],
        #                         metrics['auc'], metrics['logloss'], metrics['pctr'], metrics['ctr'],
        #                         abs(metrics['pctr'] / metrics['ctr'] - 1), metrics.get('gauc', 0), metrics['auc2']
        #                         ) for step, metrics in eval_metrics_dict.items()], key=lambda x: x[2] * (1 - x[3]),
        #                         reverse=True)[0]

        # if isSave:
        #     file_dir = os.path.join(save_dir, 'val_res')
        #     with tf.gfile.GFile(file_dir, 'w') as f:
        #         f.write('best checkpoint_path: %s\n' % result_best_auc[1].decode('UTF-8'))
        #         f.write('best_step: %s \n' % result_best_auc[0])
        #         f.write("best_auc\t%0.5f\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f\n" % (
        #             float(result_best_auc[2]), float(result_best_auc[3]), float(result_best_auc[4]),
        #             float(result_best_auc[5]), float(result_best_auc[6]), float(result_best_auc[7]), float(result_best_auc[8])))

        # checkpoint_path = result_best_auc[1].decode('UTF-8')

        group_list = [k[6:] for k, v in list(eval_metrics_dict.values())[0].items() if k.startswith('group_')]
        group_set = set([re.sub(r'\d+$', '', s) for s in group_list])
        # metrics_group = {pattern+'_group': sum(metrics[group] * metrics['group_'+group] for group in group_list if pattern in group) for pattern in group_set}
        # metrics_group = {}
        # for partten in group_set:
        #     tmp = 0.0
        #     for group in group_list:
        #         if partten in group:
        #             tmp += metrics[group] * metrics['group_' + group]
        #     metrics_group[partten+'_group'] = tmp

        result_best_auc = sorted([ dict(list({'step': step}.items()) + list(metrics.items()) +  
                                        list({pattern+'_group': sum(metrics[group] * metrics['group_'+group] for group in group_list if pattern in group) for pattern in group_set}.items())) 
                                  for step, metrics in eval_metrics_dict.items()], key=lambda x: x['auc/ctr'],
                                reverse=True)[0]

        if isSave:
            file_dir = os.path.join(save_dir, 'val_res')
            with tf.gfile.GFile(file_dir, 'w') as f:
                checkpoint_path = result_best_auc.pop('checkpoint_path').decode('UTF-8')
                f.write('best_checkpoint_path: %s\n' % checkpoint_path)
                f.write('best_step: %s \n' % result_best_auc.pop('step'))
                f.write("best_auc\t%s"%('\t'.join(str(k)+':'+str(round(float(result_best_auc[k]), 4)) for k in sorted(result_best_auc))))

        return checkpoint_path


    
    def customer_estimator(self):
        if len(FLAGS.pretrain_model_dir) > 10:
            logger.info('###train_mode### is daily')
            # pretrain_model_dir = os.path.join(FLAGS.model_dir, 'best_ckpt')
            pretrain_model_dir = FLAGS.pretrain_model_dir
            logger.info('WarmStartSettings use pretrain model: %s', pretrain_model_dir)
            ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=pretrain_model_dir, vars_to_warm_start=['.*'])
            # if 'ctr_model_seq3_hluc_daily' in self.params['taskName']:
            #     ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=pretrain_model_dir, vars_to_warm_start='.*', vars_not_warm_start=['.*position_emb.*'], hashtables_to_warm_start='input_layer/embedding_hashtable/emb_hashtable', var_name_to_prev_var_name={'input_layer/embedding_hashtable/emb_hashtable': 'merged_hashtable_0'})
            #     ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=pretrain_model_dir, vars_to_warm_start=['.*'], vars_not_warm_start=['.*position_emb.*'])
            return tf.estimator.Estimator(model_fn=self.model_fn, config=self.run_config, warm_start_from=ws)
        else:
            return tf.estimator.Estimator(model_fn=self.model_fn, config=self.run_config)


def main(_):
    # DO NOT MODIFY
    # if 'hluc' in FLAGS.taskName:
    #     tf.enable_gpu_booster(mode=2, enable_var_fusion=False, enable_hashtable_fusion=False)
    # else:
    tf.enable_gpu_booster(mode=2, enable_var_fusion=False, enable_xla=False)
    os.environ['TF_SIMPLE_WATCHER_LEVEL'] = '2'

    # if tf.gfile.Exists(FLAGS.model_dir):
    #     tf.gfile.DeleteRecursively(FLAGS.model_dir)
    # tf.gfile.MakeDirs(FLAGS.model_dir)

    # Init trainer
    trainer = Trainer()
    input_fn_with_params = functools.partial(input_fn, trainer.params, trainer.feature_protocal, trainer.label_protocal)

    hook = tf.train.ProfilerHook(
        save_steps=20,
        output_dir=os.path.join(FLAGS.model_dir, "tracing"),
        show_dataflow=True,
        show_memory=True)
    hooks = [hook]

    # Train Spec
    # hooks = []
    # hooks.append(InitHashTableHook()) 引擎自动做了，使用方不关心
    train_spec = tf.estimator.TrainSpec(input_fn=input_fn_with_params,
                                        embed_hooks=None)

    # Eval Spec
    exporter = tf.estimator.BestExporter(name='best_auc_exporter',
                                         serving_input_receiver_fn=trainer.build_serving_input(),
                                         compare_fn=auc_bigger,
                                         exports_to_keep=1)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_with_params, steps=None,
                                      start_delay_secs=0,
                                      throttle_secs=10,
                                      exporters=exporter)

    if FLAGS.run_mode == 'train':
        trainer.estimator.train(input_fn=input_fn)
        if trainer.task_index == 0:
            trainer.save_eval_res()
        tf.logging.info("Train Finished!")
    elif FLAGS.run_mode == 'train_and_evaluate' or FLAGS.run_mode == 'G_train':
        # if FLAGS.job_name == 'evaluator':
        #     export_dir = trainer.estimator.export_saved_model(FLAGS.model_dir + '/exported_model', trainer.build_serving_input())
        #     logger.info("export_dir is {}".format(export_dir))
        tf.estimator.train_and_evaluate(trainer.estimator, train_spec=train_spec, eval_spec=eval_spec)
        # if trainer.task_index == 0:
        if FLAGS.job_name == 'evaluator':
            trainer.save_eval_res()
        tf.logging.info("Train and evaluate Finished!")
    elif FLAGS.run_mode == 'predict':
        predictions = trainer.estimator.predict(input_fn=input_fn)
        index = 0
        try:
            for p in predictions:
                if index % 1000000 == 0:
                    tf.logging.info('current predict count = %d' % index)
                index += 1
        except Exception as e:
            tf.logging.error("Prediction error: %s", str(e))
        tf.logging.info("Prediction Finished!")
    else:
        tf.logging.error("Unknow run_mode!")


if __name__ == '__main__':
    tf.app.run()
