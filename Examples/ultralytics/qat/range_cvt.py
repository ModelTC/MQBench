import json


mqb = open("/mnt/share/yongyang/projects/mqb/L6/update/ultralytics/qat/yolo_trt_fixed_15_clip_ranges.json")
dip = open("/mnt/share/yongyang/projects/mqb/L6/update/ultralytics/qat/results/trt_clip_val.json")

range_mqb = json.load(mqb)
print(range_mqb)

range_dip = json.load(dip)
print(range_dip)

range_v2 = range_mqb.copy()

for k in range_dip['blob_range']:
    if k not in range_mqb['blob_range']:
        range_v2['blob_range'][k] = range_dip['blob_range'][k]


with open("range_15_v2.json", "w") as outfile: 
    json.dump(range_v2, outfile, indent=4)
