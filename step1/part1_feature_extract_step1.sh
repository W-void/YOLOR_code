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

    select *, row_number() OVER (PARTITION BY userid,pageid ORDER BY requesttime) as pv_rank
    from
    (
        select userid
            ,pvid
            ,ori_pageid as pageid
            ,concat_ws('\'','\'', collect_list(poiid)) as cur_poi_list
            ,concat_ws('\'','\'', collect_list(label)) as label_list
            ,concat_ws('\'','\'', collect_list(label_recv)) as click_list
            ,concat_ws('\'','\'', collect_list(is_imp)) as imp_list
            ,concat_ws('\'','\'', collect_list(position_rank)) as rank_list
            ,concat_ws('\'','\'', collect_list(position)) as position_list
            ,concat_ws('\'','\'', collect_list(poi_bid)) as cur_poi_bid_list
            ,concat_ws('\'','\'', collect_list(charge_rate)) as cur_poi_charge_rate_list
            ,min(requesttime) as requesttime

            ,concat_ws('\'','\'', collect_list(poi_name_segment_id)) as poi_name_list
            ,concat_ws('\'','\'', collect_list(poi_cat_fea)) as poi_cat_fea_list
            ,concat_ws('\'','\'', collect_list(poi_num_fea)) as poi_num_fea_list
            ,max(poi_list) as hist_poi_list
            ,max(poi_name_segment_list) as hist_poi_name_list
            ,max(dianjin_poi_list) as dianjin_poi_list
            ,max(dianjin_poi_name_segment_list) as dianjin_poi_name_segment_list
            ,max(user_and_context_cat_fea) as user_cat_fea_list
            ,max(user_and_context_num_fea) as user_num_fea_list
            ,$today as dt
        from
        (
            select userid
                ,pvid
                ,ori_pageid
                ,poiid
                ,label
                ,label_recv
                ,is_imp
                ,requesttime
                ,position
                ,poi_bid
                ,charge_rate
                ,row_number() OVER (PARTITION BY userid,pvid,ori_pageid ORDER BY cast(position as int)) as position_rank

                ,concat_ws('\'','\'', ${user_and_context_cat_fea}) as user_and_context_cat_fea
                ,concat_ws('\'','\'', ${user_and_context_num_fea}) as user_and_context_num_fea
                ,concat_ws('\'','\'', ${poi_cat_fea}) as poi_cat_fea
                ,concat_ws('\'','\'', ${poi_num_fea}) as poi_num_fea
                ,substring(poi_name_segment_id, 2, length(poi_name_segment_id)-2) as poi_name_segment_id
                ,substring(poi_list, 2, length(poi_list)-2) as poi_list
                ,substring(poi_name_segment_list, 2, length(poi_name_segment_list)-2) as poi_name_segment_list
                ,substring(dianjin_poi_list, 2, length(dianjin_poi_list)-2) as dianjin_poi_list
                ,substring(dianjin_poi_name_segment_list, 2, length(dianjin_poi_name_segment_list)-2) as dianjin_poi_name_segment_list
            from 
            (
                select userid
                        ,pvid
                        ,poiid
                        ,requesttime
                        ,pageid as ori_pageid
                        ,label
                        ,label_recv
                        ,is_imp
                        ,cast(position as int) as position
                        ,json_tuple(features, AAA) as (BBB)
                        ,get_json_object(fvs2, '\''$.poi_bid_v3'\'') as poi_bid
                        ,get_json_object(fvs2, '\''$.charge_rate'\'') as charge_rate
                    from mart_waimai_ad_cd.cps_market_dcvr_sample_log
                    where dt = $today
                    and pageid in ('\''cpsSelfCouponAll'\'', '\''cpsEverDayCoupon'\'')
            )
        )
        group by pvid, userid, ori_pageid
    )

\"
=== 
targetTable==${targetTable}_step1
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

# nohup sh part1_feature_extract_step1.sh 20240808 &
# nohup sh task_parallel.sh 20240701 20240817 part1_feature_extract_step1.sh &