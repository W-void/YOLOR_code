#!/bin/sh
# 按天生成tfrecords文件

set -x
set -u
set -o pipefail
curPwd=`dirname $0`
source ${curPwd}/step1.conf 100
# logPwd1=${logPwd}/step1



isTfrecord=true
isParallel=true  # 是否按天并行执行，并行数<30天

# 上线时：仅增量生成一天
end_date=${1:-20230511}
tfrecord_start_date=$end_date
tfrecord_end_date=$end_date

# 离线时：可任意指定日期
#tfrecord_start_date=$start_date
#tfrecord_end_date=$end_date
targetTable=${targetTable}
tfrecord_path=${tfrecord_path}

date=${tfrecord_start_date}

hope_submit=/opt/meituan/anaconda3/bin/hope
type /opt/meituan/yuhaopeng/anaconda3/bin/hope
if [ $? -ne 1 ]; then
    hope_submit=/opt/meituan/yuhaopeng/anaconda3/bin/hope
fi
${hope_submit} exec --jobname hadoop-waimai-ad-cd.spark.OfflineTfrecordWsl2 --args="startDate=\"${date}\" endDate=\"${date}\" sourceTable=\"${targetTable}_step2\" label=\"${label}\" feature_list=\"${feature_list}\" condition=\"${condition}\" output_path=\"${tfrecord_path}\""


# nohup sh part2_tfrecord_log.sh 20240505 &
# nohup sh task_parallel.sh 20240701 20240817 part2_tfrecord_log.sh &