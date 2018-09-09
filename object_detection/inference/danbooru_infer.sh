# From tensorflow/models/research/oid
SPLIT=train
NUM_GPUS=2
NUM_SHARDS=1024

tmux new-session -d -s "inference"
function tmux_start { screen -S "GPU$1" "${*:2}; exec bash"; }

gpu_index=0
start_shard=0
end_shard=614
TF_RECORD_FILES=$(seq -s, -f "/media/jerryli27/Data_Disk_HDD_3/DanbooruData/DanbooruData/tfrecord_1m/${SPLIT}-%05.0f-of-$(printf '%05d' $NUM_SHARDS)" $start_shard $end_shard)
tmux_start 0 \
PYTHONPATH=$PYTHONPATH:$(readlink -f ..) CUDA_VISIBLE_DEVICES=$gpu_index \
python -m object_detection/inference/infer_detections \
--input_tfrecord_paths=$TF_RECORD_FILES \
--output_tfrecord_path=/media/jerryli27/Data_Disk_HDD_3/DanbooruData/DanbooruData/tfrecord_1m_object_detection/${SPLIT}_detections.tfrecord-$(printf "%05d" $gpu_index)-of-$(printf "%05d" $NUM_GPUS) \
--inference_graph=/mnt/E076DD8F76DD66B6/image2tag_checkpoints/anime_face_detection_faster_rcnn_inception_v3/frozen/frozen_inference_graph.pb \
--discard_image_pixels

gpu_index=1
start_shard=615  # To 351180, then oom due to too large image.
end_shard=1023
TF_RECORD_FILES=$(seq -s, -f "/media/jerryli27/Data_Disk_HDD_3/DanbooruData/DanbooruData/tfrecord_1m/${SPLIT}-%05.0f-of-$(printf '%05d' $NUM_SHARDS)" $start_shard $end_shard)
tmux_start 1 \
PYTHONPATH=$PYTHONPATH:$(readlink -f ..) CUDA_VISIBLE_DEVICES=$gpu_index \
python -m object_detection/inference/infer_detections \
--input_tfrecord_paths=$TF_RECORD_FILES \
--output_tfrecord_path=/media/jerryli27/Data_Disk_HDD_3/DanbooruData/DanbooruData/tfrecord_1m_object_detection/${SPLIT}_detections.tfrecord-$(printf "%05d" $gpu_index)-of-$(printf "%05d" $NUM_GPUS) \
--inference_graph=/mnt/E076DD8F76DD66B6/image2tag_checkpoints/anime_face_detection_faster_rcnn_inception_v3/frozen/frozen_inference_graph.pb \
--discard_image_pixels




PYTHONPATH=$PYTHONPATH:/home/jerryli27/workspace/models/research/slim


jerryli27@jerryli27-Ubuntu:~/workspace/models/research$ echo $SPLIT
train
jerryli27@jerryli27-Ubuntu:~/workspace/models/research$ SPLIT=train
jerryli27@jerryli27-Ubuntu:~/workspace/models/research$ echo $SPLIT
train



gpu_index=1
start_shard=830  # To 118500, then oom due to too large image. [25984, 14173, 3]
end_shard=1023
TF_RECORD_FILES=$(seq -s, -f "/media/jerryli27/Data_Disk_HDD_3/DanbooruData/DanbooruData/tfrecord_1m/${SPLIT}-%05.0f-of-$(printf '%05d' $NUM_SHARDS)" $start_shard $end_shard)
tmux_start 1 \
PYTHONPATH=$PYTHONPATH:$(readlink -f ..) CUDA_VISIBLE_DEVICES=$gpu_index \
python -m object_detection/inference/infer_detections \
--input_tfrecord_paths=$TF_RECORD_FILES \
--output_tfrecord_path=/media/jerryli27/Data_Disk_HDD_3/DanbooruData/DanbooruData/tfrecord_1m_object_detection/${SPLIT}_detections.tfrecord-$(printf "%05d" 2)-of-$(printf "%05d" $NUM_GPUS) \
--inference_graph=/mnt/E076DD8F76DD66B6/image2tag_checkpoints/anime_face_detection_faster_rcnn_inception_v3/frozen/frozen_inference_graph.pb \
--discard_image_pixels
# From tensorflow/models/research/oid
SPLIT=train
NUM_GPUS=2
NUM_SHARDS=1024

tmux new-session -d -s "inference"
function tmux_start { screen -S "GPU$1" "${*:2}; exec bash"; }

gpu_index=0
start_shard=0
end_shard=614  # All 665710 images
TF_RECORD_FILES=$(seq -s, -f "/media/jerryli27/Data_Disk_HDD_3/DanbooruData/DanbooruData/tfrecord_1m/${SPLIT}-%05.0f-of-$(printf '%05d' $NUM_SHARDS)" $start_shard $end_shard)
tmux_start 0 \
PYTHONPATH=$PYTHONPATH:$(readlink -f ..) CUDA_VISIBLE_DEVICES=$gpu_index \
python -m object_detection/inference/infer_detections \
--input_tfrecord_paths=$TF_RECORD_FILES \
--output_tfrecord_path=/media/jerryli27/Data_Disk_HDD_3/DanbooruData/DanbooruData/tfrecord_1m_object_detection/${SPLIT}_detections.tfrecord-$(printf "%05d" $gpu_index)-of-$(printf "%05d" $NUM_GPUS) \
--inference_graph=/mnt/E076DD8F76DD66B6/image2tag_checkpoints/anime_face_detection_faster_rcnn_inception_v3/frozen/frozen_inference_graph.pb \
--discard_image_pixels

gpu_index=1
start_shard=615  # To 351180, then oom due to too large image.
end_shard=1023
TF_RECORD_FILES=$(seq -s, -f "/media/jerryli27/Data_Disk_HDD_3/DanbooruData/DanbooruData/tfrecord_1m/${SPLIT}-%05.0f-of-$(printf '%05d' $NUM_SHARDS)" $start_shard $end_shard)
tmux_start 1 \
PYTHONPATH=$PYTHONPATH:$(readlink -f ..) CUDA_VISIBLE_DEVICES=$gpu_index \
python -m object_detection/inference/infer_detections \
--input_tfrecord_paths=$TF_RECORD_FILES \
--output_tfrecord_path=/media/jerryli27/Data_Disk_HDD_3/DanbooruData/DanbooruData/tfrecord_1m_object_detection/${SPLIT}_detections.tfrecord-$(printf "%05d" $gpu_index)-of-$(printf "%05d" $NUM_GPUS) \
--inference_graph=/mnt/E076DD8F76DD66B6/image2tag_checkpoints/anime_face_detection_faster_rcnn_inception_v3/frozen/frozen_inference_graph.pb \
--discard_image_pixels




PYTHONPATH=$PYTHONPATH:/home/jerryli27/workspace/models/research/slim


jerryli27@jerryli27-Ubuntu:~/workspace/models/research$ echo $SPLIT
train
jerryli27@jerryli27-Ubuntu:~/workspace/models/research$ SPLIT=train
jerryli27@jerryli27-Ubuntu:~/workspace/models/research$ echo $SPLIT
train



gpu_index=1
start_shard=830  # To 118500, then oom due to too large image. [25984, 14173, 3]
end_shard=1023
TF_RECORD_FILES=$(seq -s, -f "/media/jerryli27/Data_Disk_HDD_3/DanbooruData/DanbooruData/tfrecord_1m/${SPLIT}-%05.0f-of-$(printf '%05d' $NUM_SHARDS)" $start_shard $end_shard)
tmux_start 1 \
PYTHONPATH=$PYTHONPATH:$(readlink -f ..) CUDA_VISIBLE_DEVICES=$gpu_index \
python -m object_detection/inference/infer_detections \
--input_tfrecord_paths=$TF_RECORD_FILES \
--output_tfrecord_path=/media/jerryli27/Data_Disk_HDD_3/DanbooruData/DanbooruData/tfrecord_1m_object_detection/${SPLIT}_detections.tfrecord-$(printf "%05d" 2)-of-$(printf "%05d" $NUM_GPUS) \
--inference_graph=/mnt/E076DD8F76DD66B6/image2tag_checkpoints/anime_face_detection_faster_rcnn_inception_v3/frozen/frozen_inference_graph.pb \
--discard_image_pixels



gpu_index=0
start_shard=960  # A total of 69290 images.
end_shard=1023
TF_RECORD_FILES=$(seq -s, -f "/media/jerryli27/Data_Disk_HDD_3/DanbooruData/DanbooruData/tfrecord_1m/${SPLIT}-%05.0f-of-$(printf '%05d' $NUM_SHARDS)" $start_shard $end_shard)
tmux_start 1 \
PYTHONPATH=$PYTHONPATH:$(readlink -f ..) CUDA_VISIBLE_DEVICES=$gpu_index \
python -m object_detection/inference/infer_detections \
--input_tfrecord_paths=$TF_RECORD_FILES \
--output_tfrecord_path=/media/jerryli27/Data_Disk_HDD_3/DanbooruData/DanbooruData/tfrecord_1m_object_detection/${SPLIT}_detections.tfrecord-$(printf "%05d" 3)-of-$(printf "%05d" $NUM_GPUS) \
--inference_graph=/mnt/E076DD8F76DD66B6/image2tag_checkpoints/anime_face_detection_faster_rcnn_inception_v3/frozen/frozen_inference_graph.pb \
--discard_image_pixels
