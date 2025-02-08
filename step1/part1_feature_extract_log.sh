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

today=${1:-20230511}
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
        ,pvid
        ,pageid
        ,concat_ws('\'','\'', collect_list(poiid)) as cur_poi_list
        ,concat_ws('\'','\'', collect_list(label)) as label_list
        ,concat_ws('\'','\'', collect_list(label_recv)) as click_list
        ,concat_ws('\'','\'', collect_list(position_rank)) as rank_list
        ,concat_ws('\'','\'', collect_list(position)) as position_list
        ,row_number() OVER (PARTITION BY userid,pvid ORDER BY requesttime) as rank

        ,concat_ws('\'','\'', collect_list(poi_name_segment_id)) as poi_name_list
        ,concat_ws('\'','\'', collect_list(poi_cat_fea)) as poi_cat_fea_list
        ,concat_ws('\'','\'', collect_list(poi_num_fea)) as poi_num_fea_list
        ,max(poi_list) as hist_poi_list
        ,max(poi_name_segment_list) as poi_name_segment_list
        ,max(dianjin_poi_list) as dianjin_poi_list
        ,max(dianjin_poi_name_segment_list) as dianjin_poi_name_segment_list
        ,max(user_and_context_cat_fea) as user_cat_fea_list
        ,max(user_and_context_num_fea) as user_num_fea_list
        ,$today as dt
    from
    (
        select userid
            ,pvid
            ,pageid
            ,poiid
            ,label
            ,label_recv
            ,requesttime
            ,row_number() OVER (PARTITION BY userid,pvid,pageid ORDER BY cast(position as int)) as position_rank

            ,concat_ws('\'','\'', ${user_and_context_cat_fea}) as user_and_context_cat_fea
            ,concat_ws('\'','\'', ${user_and_context_num_fea}) as user_and_context_num_fea
            ,concat_ws('\'','\'', ${poi_cat_fea}) as poi_cat_fea
            ,concat_ws('\'','\'', ${poi_num_fea}) as poi_num_fea
            ,poi_name_segment_id
            ,poi_list
            ,poi_name_segment_list
            ,dianjin_poi_list
            ,dianjin_poi_name_segment_list
        from 
        (
            select userid
                    ,pvid
                    ,poiid
                    ,requesttime
                    ,pageid
                    ,label
                    ,label_recv
                    ,cast(position as int) as position
                    ,json_tuple(features, AAA) as (BBB)
                from mart_waimai_ad_cd.cps_market_dcvr_sample_log
                where dt = $today
                and pageid in ('\''cpsSelfCouponAll'\'', '\''cpsEverDayCoupon'\'')
        )
    )
    group by pvid, userid, pageid
)
where rank = 1
\"
=== 
targetTable==${targetTable}
=== 
dropTable==False
"
hope_submit=/opt/meituan/anaconda3/bin/hope
type /opt/meituan/yuhaopeng/anaconda3/bin/hope
if [ $? -ne 1 ]; then
    hope_submit=/opt/meituan/yuhaopeng/anaconda3/bin/hope
fi
${hope_submit}  exec --jobname hadoop-waimai-ad-cd.spark.OfflineDataProcessV2 --args="${arg}"
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

# nohup sh part1_feature_extract_log.sh 20240227 &
# nohup sh task_parallel.sh 20240122 20240124 part1_feature_extract_log.sh &