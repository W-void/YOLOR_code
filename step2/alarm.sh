#!/usr/bin/env bash

misID=$1
desc=$2
DATE=`/usr/bin/env TZ='GMT' LANG=en_US date "+%a, %d %b %Y %H:%M:%S %Z"`
#KEY=033096123A421420
#SECRET=793e82b0bb37b817372b0e453dad57d6
KEY=5j23212411x06160
SECRET=b4f571f6b5244bdc5c83551908f21b56
P=/api/pub/push
method=PUT
SIGN=`echo -ne "$method $P\n$DATE"|openssl sha1 -hmac "$SECRET" -binary|base64`

misIDs=(`echo ${misID} | tr ',' ' '`)

for var in "${misIDs[@]}"
do
   curl -i -H 'Content-Type: application/json; charset=utf-8' -H "Date:$DATE" -H "Authorization:MWS $KEY:$SIGN" -X PUT \
    -d '{"body":{"bold":true,"fontName":"宋体","fontSize":12,"text":"'"$desc"'"},"fromName":"cptming","fromUid":137495622611,"messageType":"text","receivers":["'"$var"'@sankuai.info"],"toAppId":1}' \
     http://dxw-in.sankuai.com$P

done

