#!/bin/bash
set -x
set +u
set -o pipefail

sudo -iu sankuai
export HOME='/home/sankuai'
whoami

DATA_FLAG=${1:-true}

## 离线训练平台内置变量 begin ##
export platTrainData=${PLAT_TRAIN_DATA} # 训练样本HDFS路径
export platModelPath=${PLAT_MODEL_PATH} # 生成模型HDFS路径
export platConfPath=${PLAT_CONF_PATH} #模型训练配置文件路径
echo "platTrainData dir $platTrainData"
echo "platModelData dir $platModelPath"
echo "platConfPath dir $platConfPath"
## 离线训练平台内置变量 end ##

export today=20240119
if [ -n "${platTrainData}" ]; then
    export today=$(date "+%Y%m%d")
fi
today=`date -d"${today}  -2day" +%Y%m%d`

curPwd=`dirname $0`
curPwd=${curPwd}/..
step1Pwd=${curPwd}/../step1
step2Pwd=${curPwd}/../step2
source ${step1Pwd}/step1.conf 100
logPwd1=${curPwd}/../logs/step1

saveModelPwd=${curPwd}/../models
model_name=dspad_cps_cpc_self_coupon_ctr_v3
model_dir_local=${saveModelPwd}/${today}

mkdir -p ${curPwd}/../models
mkdir -p ${curPwd}/../logs/step1
mkdir -p ${curPwd}/../logs/step2

function checkData {
    # check数据表是否就绪
    is_exist=`/opt/meituan/hadoop/bin/hadoop fs -ls /zw02nn45/warehouse/$1 | grep dt=${today} | wc -l`
    if [[ ${is_exist} -ne 1 ]]; then
        echo "$1 ${today} not ready",
        return 1
    fi
    return 0
}


# -----step 1-----
# # check数据表是否就绪
source /opt/meituan/hadoop-gpu/bin/hadoop_user_login_centos7.sh hadoop-waimai-ad-cd
# for((i=0;i<66;i++))
# do
#     echo `date +"%Y-%m-%d %H:%M:%S"`
#     checkData mart_waimai_ad_cd.db/hoae_cps_self_coupon_ctr_sample_v3
#     if [[ $? -ne 0 ]]; then
#         sleep 600;
#     else
#         break;
#     fi
# done
# checkData mart_waimai_ad_cd.db/hoae_cps_self_coupon_ctr_sample_v3
# if [[ $? -ne 0 ]]; then
#     echo "${model_name} sample data not ready"
#     exit -1;
# fi
checkData mart_waimai_ad_cd.db/${table}_step2
if [[ $? -ne 0 ]]; then
    # ----------准备样本表，增量生成一天----------
    # sh ${step1Pwd}/part1_feature_extract.sh ${today} 
    sh ${step1Pwd}/part1_feature_extract_step1.sh ${today}
    sh ${step1Pwd}/part1_feature_extract_step2.sh ${today}

    # ----------准备tfrecords，增量生成一天----------
    # sh ${step1Pwd}/part2_tfrecord.sh ${today}
    sh ${step1Pwd}/part2_tfrecord_log.sh ${today}

    # ----------准备dict(mean_var, src file)----------
#    sh ${step1Pwd}/part3_dict.sh
fi


# -----step 2-----
# ----------模型训练----------
echo `date +"%Y-%m-%d %H:%M:%S"`
sh ./run.sh
# sh ./run.sh G_train
if [ $? -ne 0 ];then
    echo "train model task  failed"
    exit -1
fi

# -----step 3-----
# ----------模型推送文件----------
#cp -r ${curPwd}/../${model_name} ${model_dir_local}
#mkdir -p ${model_dir_local}/${model_name}/${model_name}
#mv ${model_dir_local}/*.pb ${model_dir_local}/${model_name}/${model_name} && mv ${model_dir_local}/variables/ ${model_dir_local}/${model_name}/${model_name}
#mv ${curPwd}/val_res ${model_dir_local}

# done文件
# echo "done" > _done
# /opt/meituan/hadoop/bin/hadoop fs -rm ${platModelPath}/_done
# /opt/meituan/hadoop/bin/hadoop fs -put _done ${platModelPath}
