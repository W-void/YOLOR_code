#!/bin/bash
set -x
set -u
set -o pipefail

#--task params------
expId=$1
trainStartDate=$2
trainEndDate=$3
evalStartDate=$4
evalEndDate=$5
shift 5
params=$*

echo $params

#------get pwd---------
curPwd=$(dirname $0)
step1Pwd=${curPwd}/../step1
export step2Pwd=${curPwd}
expPwd=${curPwd}/../experiments/${expId}
saveModelPwd=${curPwd}/../models

#----环境部分-----
export HADOOP_HOME=/opt/meituan/hadoop
export AFO_TF_HOME=/opt/meituan/tensorflow-release
export HADOOP_USER=hadoop-waimai-ad-cd
source /opt/meituan/hadoop-gpu/bin/hadoop_user_login_centos7.sh ${HADOOP_USER}
source ${AFO_TF_HOME}/local_env.sh

tensorflow_submit=${AFO_TF_HOME}/bin/mpi-submit.sh

#----数据/脚本参数----
#-----特征-----
source ${step1Pwd}/step1.conf
# cate_fea=${cate_fea}
# num_fea=${num_fea}
# total_cate_fea=${cate_fea}
# total_num_fea=${num_fea}

freq_threshold=$(echo $params | jq -r '.freq_threshold')

#------解析参数------
len_pv=$(echo $params | jq -r '.len_pv')
len_uc=$(echo $params | jq -r '.len_uc')
len_un=$(echo $params | jq -r '.len_un')
len_pc=$(echo $params | jq -r '.len_pc')
len_pn=$(echo $params | jq -r '.len_pn')

#----数据/脚本参数----
model_hdfs=$(echo $params | jq -r '.modelDir')
mean_var_filename=$(echo $params | jq -r '.numerical_mean_var_file')
src_filename=$(echo $params | jq -r '.src_filename')
dataDir=$(echo $params | jq -r '.dataDir')
dictDir=$(echo $params | jq -r '.dictDir')
trainModelSaveDate=$(echo $params | jq -r '.trainModelSaveDate')

main_script=$(echo $params | jq -r '.main_script')
model_dataset_script=$(echo $params | jq -r '.model_dataset_script')

#---xml参数---
task=$(echo $params | jq -r '.task')
taskName=$(echo $params | jq -r '.taskName')
misName=$(echo $params | jq -r '.misName')
queue=$(echo $params | jq -r '.queue')
config_xml=${step2Pwd}/$(echo $params | jq -r '.xml')
#---模型参数---
embed_dim=$(echo $params | jq -r '.embed_dim')

train_steps=$(echo $params | jq -r '.train_steps')
valid_steps=$(echo $params | jq -r '.valid_steps')
batch_size=$(echo $params | jq -r '.batch_size')
eval_batch_size=$(echo $params | jq -r '.eval_batch_size')
epoch=$(echo $params | jq -r '.epoch')
optimizer=$(echo $params | jq -r '.optimizer')
learning_rate=$(echo $params | jq -r '.learning_rate')
hash_learning_rate=$(echo $params | jq -r '.hash_learning_rate')
deep_layers=$(echo $params | jq -r '.deep_layers')
dropout=$(echo $params | jq -r '.dropout')
l2_reg=$(echo $params | jq -r '.l2_reg')
batch_norm=$(echo $params | jq -r '.batch_norm')
lr_decay=$(echo $params | jq -r '.lr_decay')
activation=$(echo $params | jq -r '.activation')
#----kv-memory----
is_kv_memory=$(echo $params | jq -r '.is_kv_memory')
kv_mem_num=$(echo $params | jq -r '.kv_mem_num')
kv_embed_dim=$(echo $params | jq -r '.kv_embed_dim')


#----dict----
local_dict_path=${expPwd}/dict
mkdir -p ${local_dict_path}
#rm ${local_dict_path}/${mean_var_filename}
dict_files="${local_dict_path}/${dict_bucket_file}"
rm -rf ${dict_files}
/opt/meituan/hadoop/bin/hadoop fs -get ${dict_hdfs} ${local_dict_path}

#----训练集----
cur_date=$trainStartDate
train_data=""
# dataDir=$(echo "$dataDir" | sed 's/v6\/100/v6_log\/100/')
while [ $cur_date -le $trainEndDate ]; do
  echo "date", $cur_date
  train_data=${train_data}",${dataDir}/${cur_date}"
  cur_date=$(date -d"$cur_date +1 day" '+%Y%m%d')
done
train_data=${train_data#,}

#----验证集----
cur_date=$evalStartDate
valid_data=""
# validDataDir=$(echo "$dataDir" | sed 's/v6\/100/v6_log\/100/')
validDataDir=$dataDir
while [ $cur_date -le $evalEndDate ]; do
  echo "date", $cur_date
  valid_data=${valid_data}",${validDataDir}/${cur_date}"
  cur_date=$(date -d"$cur_date +1 day" '+%Y%m%d')
done
valid_data=${valid_data#,}

# if [[ ${expId} =~ "daily" ]]; then
  trainDataDir=$(echo "$dataDir" | sed 's/v6\/100/v6_log\/100/')
  # train_data=${trainDataDir}/$(date -d"$evalEndDate -1 day" '+%Y%m%d')
  # valid_data=${validDataDir}/${evalEndDate}
# fi

start_time=$(date +"%s" '+%Y-%m-%d %H:%M:%S')
echo "start_time $start_time"


echo $train_data
echo $valid_data

#----脚本/数据文件----
# （本地路径，上传到服务器）
script_files="${step2Pwd}/main.py,${step2Pwd}/model_base.py,${step2Pwd}/model_exp.py,${step2Pwd}/utils.py,${step2Pwd}/data.py"
runScript="python -u main.py"
#data_files="${local_dict_path}/${mean_var_filename},${local_dict_path}/${src_filename}"

#----hope提交afo作业----
hope_submit=/opt/meituan/anaconda3/bin/hope
submit_mode=mpi
# user_group="hadoop-hmart-waimaiad"
user_group=$HADOOP_USER
# mkdir an-empty-dir/
# cd an-empty-dir/
#----hope免密登陆----

# 模型保存路径
cur_model_hdfs=${model_hdfs}/${trainModelSaveDate}
online_model_ckpt="${cur_model_hdfs%/*}/online"
/opt/meituan/hadoop/bin/hadoop fs -ls ${online_model_ckpt}
is_exist=`/opt/meituan/hadoop/bin/hadoop fs -ls ${online_model_ckpt} | wc -l`
if [[ ${is_exist} -gt 1 ]]; then
  # train_data=${trainDataDir}/$(date -d"$evalEndDate -1 day" '+%Y%m%d')
  echo "nothing to do"
else
  online_model_ckpt=""
fi

#-----Generator热加载最优ckpt----
if [[ ${task} == 'G_train' ]]; then
  cur_model_hdfs="${cur_model_hdfs%/*}_G/${trainModelSaveDate}"
  # /opt/meituan/hadoop/bin/hadoop fs -mkdir $cur_model_hdfs
  # /opt/meituan/hadoop/bin/hadoop fs -cp ${online_model_ckpt}/* ${cur_model_hdfs}/
else
  online_model_ckpt=""
  echo "pass"
fi

/opt/meituan/hadoop/bin/hadoop fs -ls $cur_model_hdfs
if [[ $? -eq 0 ]]; then
  /opt/meituan/hadoop/bin/hadoop fs -rm -r $cur_model_hdfs
fi
/opt/meituan/hadoop/bin/hadoop fs -mkdir -p $cur_model_hdfs

#-----天级更新加载最优ckpt----
local_path=${expPwd}
mkdir -p ${local_path}
rm ${local_path}/val_res_yestoday
cur_day=${trainModelSaveDate}
lastday=$(date -d "${cur_day} -1 day" +%Y%m%d)


if [[ ${expId} =~ "base" ]]; then
  train_data=${trainDataDir}/$(date -d"$evalEndDate -1 day" '+%Y%m%d')
fi


################################################################################
###############################增加代码「开始」####################################
# ----队列资源判定（在资源不够且有fake任务时，kill掉fake任务）----
/opt/meituan/hadoop/bin/hadoop fs -get viewfs://hadoop-meituan/user/hadoop-waimai-ad-cd/utils/check_gpu_wsl.sh
sh check_gpu_wsl.sh gcores40g
rm check_gpu_wsl.sh
###############################增加代码「结束」####################################
################################################################################


${hope_submit} run --xml ${config_xml} \
  --files ${script_files},${dict_files} \
  --usergroup ${user_group} \
  -Ddistribute.mode=${submit_mode} \
  -Dafo.engine.wait_for_job_finished=true \
  -Dboard.log_dir=${cur_model_hdfs} \
  -Dafo.app.name=${taskName} \
  -Dafo.app.queue=${queue} \
  -Dafo.xm.notice.receivers.account=${misName} \
  -Dworker.script="${runScript}" \
  -Devaluator.script="${runScript}" \
  -Dargs.taskName=${taskName} \
  -Dargs.dict_bucket_file=${dict_bucket_file} \
  -Dargs.model_dir=${cur_model_hdfs} \
  -Dargs.pretrain_model_dir=${online_model_ckpt} \
  -Dargs.model_ckpt_dir=${cur_model_hdfs} \
  -Dargs.run_mode=${task} \
  -Dargs.dict_dir=${dictDir} \
  -Dargs.train_data=${train_data} \
  -Dargs.valid_data=${valid_data} \
  -Dargs.mean_var_filename=${mean_var_filename} \
  -Dargs.src_filename=${src_filename} \
  -Dargs.train_steps=$train_steps \
  -Dargs.valid_steps=$valid_steps \
  -Dargs.batch_size=${batch_size} \
  -Dargs.evaluator.batch_size=${eval_batch_size} \
  -Dargs.embed_dim=${embed_dim} \
  -Dargs.epoch=${epoch} \
  -Dafo.data.max.epoch=${epoch} \
  -Dafo.data.evaluator.max.epoch=1 \
  -Dargs.optimizer=${optimizer} \
  -Dargs.l2_reg=${l2_reg} \
  -Dargs.lr_decay=${lr_decay} \
  -Dargs.activation=${activation} \
  -Dargs.learning_rate=${learning_rate} \
  -Dargs.hash_learning_rate=${hash_learning_rate} \
  -Dargs.deep_layers=${deep_layers} \
  -Dargs.batch_norm=${batch_norm} \
  -Dargs.dropout=${dropout} \
  -Dargs.freq_threshold=${freq_threshold} \
  -Dargs.is_kv_memory=${is_kv_memory} \
  -Dargs.kv_mem_num=${kv_mem_num} \
  -Dargs.kv_embed_dim=${kv_embed_dim} \
  -Dafo.data.agent.memory=20480 \
  -Dafo.data.dispatch.policy=com.meituan.hadoop.afo.tensorflow.data.policy.ShufflePolicy \
  -Dafo.app.marking-finish-roles=evaluator \
  -Dargs.len_pv=${len_pv} \
  -Dargs.len_uc=${len_uc} \
  -Dargs.len_un=${len_un} \
  -Dargs.len_pc=${len_pc} \
  -Dargs.len_pn=${len_pn}

if [[ $? -ne 0 ]]; then
  sh ${curPwd}/alarm.sh ${misName} "【FAIL】【${taskName}】$task $start_date~$end_date fail!"
  echo "$task $start_date~$end_date fail"
  exit -1
fi

local_path=${expPwd}
mkdir -p ${local_path}
rm ${local_path}/val_res
/opt/meituan/hadoop/bin/hadoop fs -get $cur_model_hdfs/val_res ${local_path}/
cat ${local_path}/val_res
best_step=$(cat ${local_path}/val_res | grep best_step | awk '{print $2}')
best_auc_result=$(cat ${local_path}/val_res | grep best_auc | awk '{$1=""; print}')
sh ${curPwd}/alarm.sh ${misName} "【SUCCESS ${trainModelSaveDate}】【${taskName} best_step:${best_step}, ${best_auc_result}, hash_learning_rate: ${hash_learning_rate}"

# 给定的字符串
input_str=$(cat ${local_path}/val_res | grep best_auc | awk '{$1=""; print}')

# 声明关联数组
declare -A result_map

# 使用循环处理字符串中的每个键值对
for kv in $input_str; do
  # 使用冒号':'分割键和值
  IFS=':' read -r key value <<< "$kv"
  # 将键值对存储到关联数组中
  result_map["$key"]="$value"
done

# 打印关联数组的内容（用于验证）
for key in "${!result_map[@]}"; do
  echo "$key: ${result_map[$key]}"
done

# best_step=${result_map["step"]}
auc=${result_map["auc/cvr"]}

# <<'train'
if [[ best_step -lt 500 ]]; then
  sh ${curPwd}/alarm.sh ${misName} "【FAIL】【${taskName}】 best auc step is $best_step!"
  echo "$task $start_date~$end_date fail"
  exit -1
fi

# AUC阈值设定
if [[ auc < 0.78 ]]; then
  sh ${curPwd}/alarm.sh ${misName} "【FAIL】【${taskName}】 best auc is too small!"
  echo "$task $start_date~$end_date fail"
  exit -1
fi

# 更新online_model_ckpt
online_model_ckpt="${cur_model_hdfs%/*}/online"
/opt/meituan/hadoop/bin/hadoop fs -rm -r ${online_model_ckpt}_backup
/opt/meituan/hadoop/bin/hadoop fs -mv ${online_model_ckpt} ${online_model_ckpt}_backup
/opt/meituan/hadoop/bin/hadoop fs -mkdir ${online_model_ckpt}
/opt/meituan/hadoop/bin/hadoop fs -cp $cur_model_hdfs/model.ckpt-${best_step}* $online_model_ckpt/
/opt/meituan/hadoop/bin/hadoop fs -cp $cur_model_hdfs/checkpoint_info.json $online_model_ckpt/
echo "model_checkpoint_path: \"model.ckpt-${best_step}\"" > best_ckpt_${best_step}
/opt/meituan/hadoop/bin/hadoop fs -put best_ckpt_${best_step} $online_model_ckpt/checkpoint
/opt/meituan/hadoop/bin/hadoop fs -cp $cur_model_hdfs/export/best_auc_exporter/*/* $online_model_ckpt/

# fetch model pb to local dir
#rm -rf ${saveModelPwd}/${today}/
#mkdir -p ${saveModelPwd}/${today}/
#/opt/meituan/hadoop/bin/hadoop fs -get $cur_model_hdfs/export/best_auc_exporter/*/* ${saveModelPwd}/${today}/

# 将模型保存到固定路径，用于算法平台打包
/opt/meituan/hadoop/bin/hadoop fs -rm -r ${platModelPath}
/opt/meituan/hadoop/bin/hadoop fs -mkdir -p ${platModelPath}
/opt/meituan/hadoop/bin/hadoop fs -cp $cur_model_hdfs/export/best_auc_exporter/*/* ${platModelPath}


sh ${curPwd}/alarm.sh ${misName} "【SUCCESS $(date "+%Y%m%d")】【${taskName}】"
