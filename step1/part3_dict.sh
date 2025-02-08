#!/bin/sh
# 生成连续型特征均值方差文件、以及验证集上的分渠道信息（点击、曝光量）

set -x
set -u
set -o pipefail
curPwd=`dirname $0`
source ${curPwd}/step1.conf 100
# logPwd1=${logPwd}/step1


isDict=true

## 离线时：可任意指定日期
#dict_start_date=20210824
#dict_train_end_date=202101008
#dict_end_date=20211009

outputPath=${dict_hdfs}
targetTable=mart_waimai_ad_cd.dspad_cpc_cvr_v7_log2

if [[ $isDict == true ]]
then
#    hadoop fs -ls ${dict_file_path}
#    if [[ $? -eq 0 ]]
#    then
#        /opt/meituan/hadoop/bin/hadoop fs -rmr ${dict_file_path}/${sample_type}
#    fi
    # /opt/meituan/hadoop/bin/hadoop fs -mkdir -p ${dict_file_path}/${sample_type}
#    /opt/meituan/hadoop/bin/hadoop fs -chmod -R 777 ${dict_file_path}

#    nohup hope exec --jobname hadoop-waimai-ad-cd.spark.Hive2Dict --args="startDate=\"${dict_start_date}\" trainEndDate=\"${dict_train_end_date}\" endDate=\"${dict_end_date}\" sourceTable=\"${targetTable}\" mean_var_col=\"${mean_var_col}\" inner_src_col=\"${inner_src_col}\" dict_file_path=\"${dict_file_path}\" dict_mean_var_file=\"${dict_mean_var_file}\" dict_src_file=\"${dict_src_file}\" label=\"${label}\" sampleType=\"${sample_type}\" condition=\"${condition}\" " >${logPwd1}/log.dict.${sample_type}.${date} 2>&1 &
    # 注意：上线日更时，此处不能用nohup及输出重定向，否则下游任务不会等待该任务完成而直接开始
    hope_submit=/opt/meituan/anaconda3/bin/hope
    type /opt/meituan/yuhaopeng/anaconda3/bin/hope
    if [ $? -ne 1 ]; then
        hope_submit=/opt/meituan/yuhaopeng/anaconda3/bin/hope
    fi
    ${hope_submit} exec --jobname hadoop-waimai-ad-cd.pyspark.model_feature_binning_repo --args="-startDate=\"${dict_start_date}\" -endDate=\"${dict_end_date}\" -sourceTable=\"${targetTable}\" -feaStr=\"${num_fea_log}\" -outputPath=\"${outputPath}\" -bucketNum=\"${bucketNum}\""
    if [ $? -ne 0 ];then
        # sh ${curPwd}/alarm.sh ${misName} "【FAIL】【${:q}】part3_dict fail!"
        echo "dict_bucket failed"
        exit -1
    fi
fi
