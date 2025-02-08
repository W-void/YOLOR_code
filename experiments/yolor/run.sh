#!/bin/bash
set -x
set +u
set -o pipefail
echo "`date`"

sudo -iu sankuai
export HOME='/home/sankuai'
whoami

# export today=20240211
# # if [ -n "${platTrainData}" ]; then
#     export today=$(date "+%Y%m%d")
# # fi
# today=`date -d"${today}  -2day" +%Y%m%d`
# today=20240506
curPwd=`dirname $0`
curPwd=${curPwd}/..
task=${1:-'train_and_evaluate'}
expId=v3_metis
expId=$(pwd | rev | cut -d "/" -f 1 | rev)
sample_type=1
total_sample_type=100

source ${curPwd}/../step1/step1.conf ${sample_type}
logPwd=${curPwd}/../logs/step2
step2Script=${curPwd}/../step2/step2_train_eval_save_hope.sh

trainStartDate=${start_date}
trainEndDate=${train_end_date}
evalStartDate=${eval_start_date}
evalEndDate=${end_date}

trainModelSaveDate=${today}

project=${tfrecord_path}
model_project=${tfrecord_path}

dataDir=viewfs://hadoop-meituan${project}
dictDir=viewfs://hadoop-meituan${project}/dict
modelDir=viewfs://hadoop-meituan${model_project}/${expId}/model
modelDir=viewfs://hadoop-meituan/user/hadoop-waimai-ad-cd/wangshuli03/data/tfrecord/dspad_self_coupon_rerank/${expId}


QUEUE_NAME=root.zw03_training02.hadoop-odghoae.job
QUEUE_NAME=root.zw05_training_cluster.hadoop-odghoae.job
# QUEUE_NAME=root.zw05_training_cluster.hadoop-waimai.ad

taskName=${misName}_rerank_model_${expId}
if [[ $task == 'G_train' ]]; then
    taskName=${taskName}_G_train
fi

params='{
    "len_pv":'${len_pv}',
    "len_uc":'${len_uc}',
    "len_un":'${len_un}',
    "len_pc":'${len_pc}',
    "len_pn":'${len_pn}',

    "dataDir":"'${dataDir}'",
    "modelDir":"'${modelDir}'",
    "sample_type":'${sample_type}',
    "total_sample_type":'${total_sample_type}',
    "trainModelSaveDate":"'${trainModelSaveDate}'",

    "freq_threshold":3,

    "embed_dim":8,

    "xml":"afo_settings_gpu1.xml",
    "task":"'${task}'",
    "taskName":"'${taskName}'",
    "misName":"'${misName}'",
    "queue":"'${QUEUE_NAME}'",

    "batch_size":1024,
    "eval_batch_size":1024,
    "epoch":4,
    "optimizer":"adagrad",
    "hash_learning_rate":0.03,
    "learning_rate":0.03,
    "deep_layers":"256,128",
    "dropout":"0,0",
    "l2_reg":0.001,
    "lr_decay":"False",
    "batch_norm":"False,False,False",
    "activation":"relu",

    "is_kv_memory":"False",
    "kv_mem_num":6,
    "kv_embed_dim": 6

}'
echo $params

sh ${step2Script} ${expId} ${trainStartDate} ${trainEndDate} ${evalStartDate} ${evalEndDate} ${params}

if [ $? -ne 0 ];then
    echo "train model task  failed"
    exit -1
fi
