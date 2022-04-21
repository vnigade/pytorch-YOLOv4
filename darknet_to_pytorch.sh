export PYTHONPATH=`pwd`

: ${root_dir="$HOME/vnigade/video_lcp/"}
# : ${video_file="${root_dir}/datasets/pkummd/test_0200-M_0201-R_0210-L.avi"}
: ${video_file="${root_dir}/datasets/dds/trafficcam_3.mp4"}
: ${gpu_id=0}
: ${log_dir="./"}
: ${iter="1000"}
: ${video_file}
# python3 ./evaluate_on_coco.py --weights_file ${root_dir}/darknet/yolov4.weights --model_config ${root_dir}/cfg/yolov4_320.cfg --data-dir ${root_dir}/datasets/coco/val2017/images --ground_truth_annotations ${root_dir}/datasets/coco/val2017/annotations/instances_val2017.json --gpu 1

frame_sizes="128 160 192 224 256 288 320 352 384 416 448 480 512 544 576 608 640 1280_704" # All these model sizes work for batch size 12
# frame_sizes="128 160 192 224 256 288 320 352 384 416 448 480 512 544 576" # All these model sizes work for batch size 12
# frame_sizes="1280_704"
batch_size=1
for size in ${frame_sizes}; do 
echo "Processing yolov4_${size}.cfg..."
# python3 ./profile_accuracy.py --weights_file ${root_dir}/pytorch_yolov4/models/yolov4_${size}.pth --model_config ${root_dir}/cfg/yolov4_${size}.cfg --data-dir ${root_dir}/datasets/coco/val2017/images --ground_truth_annotations ${root_dir}/datasets/coco/val2017/annotations/instances_val2017.json --gpu 1 --batch_size ${batch_size} > acc_model_${size}.txt
# python3 ./profile_latency.py --weights_dir ${root_dir}/pytorch_yolov4/models/ --model_config_dir ${root_dir}/cfg/ --max_batch_size $batch_size --input_size $size --gt_annotations_path ${root_dir}/datasets/coco/val2017/annotations/instances_val2017.json --dataset_dir ${root_dir}/datasets/coco/val2017/images --total_iter ${iter} --gpu_id ${gpu_id} --log_dir ${log_dir}
python3 ./dump_detections.py --weights_dir ${root_dir}/pytorch_yolov4/models/ --model_config_dir ${root_dir}/cfg/ --input_size $size --video_file ${video_file} --log_dir detections_dump
done

# Convert darknet models to pytorch
# python3 ./convert_darknet2pytorch.py --weights_file ${root_dir}/darknet/yolov4.weights --model_config_dir ${root_dir}/cfg --dst_dir ${root_dir}/pytorch_yolov4/models/
