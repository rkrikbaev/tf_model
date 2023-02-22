#!/bin/bash

curl --location --request POST 'http://127.0.0.1:8008/action' \
--header 'Content-Type: application/json' \
--data-raw '{
    "model_config": {
        "input_window": 96,
        "output_window": 48,
        "granularity": "H"
    },
    "model_uri": "/mlruns/nodes_agadyr_f_u220_q_load/61ab33826501446f9a050cd7aef8935d/mlmodel",
    "metadata": null,
    "period": null,
    "dataset": [
        [
            1673784000000,
            52.931968783286194
        ],
        [
            1673787600000,
            52.931993428020306
        ],
        [
            1673791200000,
            52.93200210602827
        ],
        [
            1673794800000,
            52.93190966560801
        ],
        [
            1673798400000,
            52.93195732200139
        ],
        [
            1673802000000,
            52.93211528153801
        ],
        [
            1673805600000,
            52.93182190949362
        ],
        [
            1673809200000,
            52.932202168277335
        ],
        [
            1673812800000,
            52.93177348134335
        ],
        [
            1673816400000,
            52.93196458098744
        ],
        [
            1673820000000,
            52.93195961983908
        ],
        [
            1673823600000,
            52.931999207996526
        ],
        [
            1673827200000,
            52.93201107092791
        ],
        [
            1673830800000,
            52.931872507599394
        ],
        [
            1673834400000,
            52.93191526540794
        ],
        [
            1673838000000,
            52.93198273538607
        ],
        [
            1673841600000,
            52.932048340126265
        ],
        [
            1673845200000,
            52.93202517333596
        ],
        [
            1673848800000,
            52.932038381496646
        ],
        [
            1673852400000,
            52.93192075668633
        ],
        [
            1673856000000,
            52.931964774677205
        ],
        [
            1673859600000,
            52.93200659918726
        ],
        [
            1673863200000,
            52.93189194949596
        ],
        [
            1673866800000,
            52.93191090517906
        ],
        [
            1673870400000,
            52.93193875640723
        ],
        [
            1673874000000,
            52.93206219242927
        ],
        [
            1673877600000,
            52.932010474126656
        ],
        [
            1673881200000,
            52.93195052786537
        ],
        [
            1673884800000,
            52.93200828307589
        ],
        [
            1673888400000,
            52.932026816897135
        ],
        [
            1673892000000,
            52.93203999755631
        ],
        [
            1673895600000,
            52.931982248826365
        ],
        [
            1673899200000,
            52.93203715481111
        ],
        [
            1673902800000,
            52.93179149235384
        ],
        [
            1673906400000,
            52.9319414346924
        ],
        [
            1673910000000,
            52.93196336605687
        ],
        [
            1673913600000,
            52.93189627431053
        ],
        [
            1673917200000,
            52.93181782769331
        ],
        [
            1673920800000,
            52.93205332120597
        ],
        [
            1673924400000,
            52.931831579937246
        ],
        [
            1673928000000,
            52.93196714382431
        ],
        [
            1673931600000,
            52.931872482743415
        ],
        [
            1673935200000,
            52.93200397883619
        ],
        [
            1673938800000,
            52.93203557372623
        ],
        [
            1673942400000,
            52.93196912793337
        ],
        [
            1673946000000,
            52.93210994931525
        ],
        [
            1673949600000,
            52.93206409215749
        ],
        [
            1673953200000,
            52.93195812693129
        ],
        [
            1673956800000,
            52.931820644813314
        ],
        [
            1673960400000,
            52.931904323417264
        ],
        [
            1673964000000,
            52.93186752580174
        ],
        [
            1673967600000,
            52.93209530390266
        ],
        [
            1673971200000,
            52.93205390124286
        ],
        [
            1673974800000,
            52.931784338715964
        ],
        [
            1673978400000,
            52.93193675289872
        ],
        [
            1673982000000,
            52.931956524982894
        ],
        [
            1673985600000,
            52.93192869346667
        ],
        [
            1673989200000,
            52.9318327296344
        ],
        [
            1673992800000,
            52.931988192365715
        ],
        [
            1673996400000,
            52.932097280904166
        ],
        [
            1674000000000,
            52.932140531369484
        ],
        [
            1674003600000,
            52.93205756848233
        ],
        [
            1674007200000,
            52.93191575021731
        ],
        [
            1674010800000,
            52.93210610355097
        ],
        [
            1674014400000,
            52.932009261712864
        ],
        [
            1674018000000,
            52.93205179502076
        ],
        [
            1674021600000,
            52.93194875046365
        ],
        [
            1674025200000,
            52.93203251012455
        ],
        [
            1674028800000,
            52.932019794710854
        ],
        [
            1674032400000,
            52.93194457892173
        ],
        [
            1674036000000,
            52.93197361885461
        ],
        [
            1674039600000,
            52.93194329250808
        ],
        [
            1674043200000,
            52.93182191762808
        ],
        [
            1674046800000,
            52.89473135963788
        ],
        [
            1674050400000,
            52.93200155476497
        ],
        [
            1674054000000,
            52.9988800563427
        ],
        [
            1674057600000,
            52.93204536692812
        ],
        [
            1674061200000,
            52.93209615958746
        ],
        [
            1674064800000,
            52.93190577828008
        ],
        [
            1674068400000,
            52.932060340704815
        ],
        [
            1674072000000,
            52.93211312025104
        ],
        [
            1674075600000,
            52.9319896297043
        ],
        [
            1674079200000,
            52.931928568565276
        ],
        [
            1674082800000,
            52.931976729693076
        ],
        [
            1674086400000,
            52.93192881545521
        ],
        [
            1674090000000,
            52.9320735246053
        ],
        [
            1674093600000,
            52.93205162193456
        ],
        [
            1674097200000,
            52.931962565974366
        ],
        [
            1674100800000,
            52.93182621710115
        ],
        [
            1674104400000,
            52.9784746391919
        ],
        [
            1674108000000,
            52.93196471097217
        ],
        [
            1674111600000,
            52.93207850064193
        ],
        [
            1674115200000,
            52.93200055975459
        ],
        [
            1674118800000,
            52.93196507409732
        ],
        [
            1674122400000,
            52.93190904501157
        ],
        [
            1674126000000,
            52.93190746964062
        ],
        [
            1674129600000,
            52.93193326118297
        ]
    ]
}'
