set -x
set -u
set -o pipefail

sdate=$1
edate=$2
task=$3
date=$sdate
a=0
mkdir logs
while [ $date -le $edate ];do
    nohup sh $task $date > logs/$(echo $task | cut -d . -f1).${date}.log 2>&1 &
    a=$[a+1]
    echo ${a}
    date=`date -d"$date 1 day" '+%Y%m%d'`
    echo ${date}
    sleep 60
done

# nohup sh task_parallel.sh 20230510 20230625 part0_user_feature1.sh &
# nohup sh task_parallel.sh 20230511 20230625 part1_feature_extract.sh &
# nohup sh task_parallel.sh 20230511 20230625 part2_tfrecord.sh &
# nohup sh task_parallel.sh 20230410 20230625 part0_dianjin_user_feature.sh &
# nohup sh task_parallel.sh 20230510 20230625 part0_dianjin_seq_feature.sh &

# nohup sh task_parallel.sh 20230626 20230815 part0_dianjin_user_feature.sh &
# nohup sh task_parallel.sh 20230626 20230815 part0_user_feature1.sh &
# nohup sh task_parallel.sh 20230626 20230815 part0_dianjin_seq_feature.sh &
# nohup sh task_parallel.sh 20240106 20240108 part1_feature_extract.sh &
# nohup sh task_parallel.sh 20240106 20240108 part2_tfrecord.sh &

# nohup sh task_parallel.sh 20230813 20230815 part1.2_feature_extract_group.sh &