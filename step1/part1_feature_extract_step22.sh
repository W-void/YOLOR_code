#!/bin/sh
set -x
set -u
set -o pipefail
curPwd=`dirname $0`
source ${curPwd}/step1.conf 100



# 上线时：仅增量生成一天
feature_extract_startDate=$end_date
feature_extract_endDate=$end_date

# 离线时：可任意指定日期
#feature_extract_startDate=$start_date
#feature_extract_endDate=$end_date

#clk_feature_extract_startDate=`date -d"${feature_extract_startDate}  -1day" +%Y%m%d`
#clk_feature_extract_endDate=`date -d"${feature_extract_endDate}  -1day" +%Y%m%d`

len_uc=$(echo $user_and_context_cat_fea | awk -F"," '{print NF}')
len_un=$(echo $user_and_context_num_fea | awk -F"," '{print NF}')
len_pc=$(echo $poi_cat_fea | awk -F"," '{print NF}')
len_pn=$(echo $poi_num_fea | awk -F"," '{print NF}')

len_EverDayCoupon=7
len_SelfCoupon=22


today=${1:-20240511}
# today=$end_date
lastday=`date -d"${today}  -1day" +%Y%m%d`

arg="
partition==dt
===
rtFeature=="${rtFeature}"
===
sqlStr==
\"
select *
from
(
    select userid
        ,concat_ws('\'','\'', collect_list(pageid)) as pageid
        ,concat_ws('\'','\'', collect_list(page_rank)) as page_rank
        ,concat_ws('\'','\'', collect_list(hist_poi_list)) as hist_poi_list
        ,concat_ws('\'','\'', collect_list(hist_poi_name_list)) as hist_poi_name_list
        ,concat_ws('\'','\'', collect_list(dianjin_poi_list)) as dianjin_poi_list
        ,concat_ws('\'','\'', collect_list(dianjin_poi_name_segment_list)) as dianjin_poi_name_segment_list
        ,concat_ws('\'','\'', collect_list(user_cat_fea_list)) as user_cat_fea_list
        ,concat_ws('\'','\'', collect_list(user_num_fea_list)) as user_num_fea_list
        ,concat_ws('\'','\'', collect_list(cur_poi_list)) as cur_poi_list
        ,concat_ws('\'','\'', collect_list(label_list)) as label_list
        ,concat_ws('\'','\'', collect_list(click_list)) as click_list
        ,concat_ws('\'','\'', collect_list(poi_name_list)) as poi_name_list

        ,concat_ws('\'','\'', collect_list(poi_cat_fea_list)) as poi_cat_fea_list
        ,concat_ws('\'','\'', collect_list(poi_num_fea_list)) as poi_num_fea_list

        ,$today as dt
    from
    (
        select userid
            ,pvid
            ,pageid
            ,row_number() OVER (PARTITION BY userid ORDER BY pageid) as page_rank
            ,hist_poi_list
            ,hist_poi_name_list
            ,dianjin_poi_list
            ,dianjin_poi_name_segment_list
            ,user_cat_fea_list
            ,user_num_fea_list

            ,case when pageid = '\''cpsEverDayCoupon'\'' then PaddingOperator(cur_poi_list, $len_EverDayCoupon)
                else PaddingOperator(cur_poi_list, $len_SelfCoupon)
                end as cur_poi_list
            ,case when pageid = '\''cpsEverDayCoupon'\'' then PaddingOperator(label_list, $len_EverDayCoupon)
                else PaddingOperator(label_list, $len_SelfCoupon)
                end as label_list
            ,case when pageid = '\''cpsEverDayCoupon'\'' then PaddingOperator(click_list, $len_EverDayCoupon)
                else PaddingOperator(click_list, $len_SelfCoupon)
                end as click_list
            ,case when pageid = '\''cpsEverDayCoupon'\'' then PaddingOperator(poi_name_list, $len_EverDayCoupon*5)
                else PaddingOperator(poi_name_list, $len_SelfCoupon*5)
                end as poi_name_list
            
            ,case when pageid = '\''cpsEverDayCoupon'\'' then PaddingOperator(poi_cat_fea_list, $len_EverDayCoupon*$len_pc)
                else PaddingOperator(poi_cat_fea_list, $len_SelfCoupon*$len_pc)
                end as poi_cat_fea_list
            ,case when pageid = '\''cpsEverDayCoupon'\'' then PaddingOperator(poi_num_fea_list, $len_EverDayCoupon*$len_pn)
                else PaddingOperator(poi_num_fea_list, $len_SelfCoupon*$len_pn)
                end as poi_num_fea_list
        from ${targetTable}_step1
        where dt = $today
          and pv_rank = 1
    )
    group by userid
)
where page_rank='\''1,2'\''
\"
=== 
targetTable==${targetTable}_step22
=== 
dropTable==False
"
hope_submit=/opt/meituan/anaconda3/bin/hope
type /opt/meituan/yuhaopeng/anaconda3/bin/hope
if [ $? -ne 1 ]; then
    hope_submit=/opt/meituan/yuhaopeng/anaconda3/bin/hope
fi
${hope_submit}  exec --jobname hadoop-waimai-ad-cd.spark.OfflineDataProcessWsl --args="${arg}"
# sql_file_name="./sql_data/sql_${today}_feature_extract.txt"
# echo "${arg}" > ${sql_file_name}
# hdfs_sql_path=${tfrecord_path}/txt
# hdfs_sql_file="${hdfs_sql_path}/sql_${today}_feature_extract.txt"

# hadoop fs -test -e $hdfs_sql_file && ( hadoop fs -rm  $hdfs_sql_file )
# hadoop fs -mkdir ${hdfs_sql_path}
# hadoop fs -put ${sql_file_name} ${hdfs_sql_file}
# hope exec --jobname hadoop-hmart-waimaiad.spark.list_offline_slot_data_wsl02  --args="${arg}

if [ $? -ne 0 ];then
    echo "feature extract task  failed"
    exit -1
fi

# nohup sh part1_feature_extract_step2.sh 20240505 &
# nohup sh task_parallel.sh 20240401 20240607 part1_feature_extract_step2.sh &