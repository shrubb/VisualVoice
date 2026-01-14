input_path="/home/ubuntu/datasets/xyz.mp4"
output_path=test_videos/xyz

filename=$(basename -- "$input_path")
filename="${filename%.*}"

python ./utils/detectFaces.py \
	--video_input_path $input_path \
	--output_path $output_path \
	--number_of_speakers 2 \
	--scalar_face_detection 1.5 \
	--detect_every_N_frame 8

ffmpeg -y -i $input_path -vn -ar 16000 -ac 1 -ab 192k -f wav $output_path/audio.wav

python ./utils/crop_mouth_from_video.py \
	--video-direc $output_path/faces/ \
	--landmark-direc $output_path/landmark/ \
	--save-direc $output_path/mouthroi/ \
	--convert-gray \
	--filename-path $output_path/filename_input/$filename.csv

python testRealVideo.py \
	--mouthroi_root $output_path/mouthroi/ \
	--facetrack_root $output_path/faces/ \
	--audio_path $output_path/audio.wav \
	--weights_lipreadingnet pretrained_models/lipreading_best.pth \
	--weights_facial pretrained_models/facial_best.pth \
	--weights_unet pretrained_models/unet_best.pth \
	--weights_vocal pretrained_models/vocal_best.pth \
	--lipreading_config_path configs/lrw_snv1x_tcn2x.json \
	--num_frames 64 \
	--audio_length 2.55 \
	--hop_size 160 \
	--window_size 400 \
	--n_fft 512 \
	--unet_output_nc 2 \
	--normalization \
	--visual_feature_type both \
	--identity_feature_dim 128 \
	--audioVisual_feature_dim 1152 \
	--visual_pool maxpool \
	--audio_pool maxpool \
	--compression_type none \
	--reliable_face \
	--audio_normalization \
	--desired_rms 0.7 \
	--number_of_speakers 2 \
	--mask_clip_threshold 5 \
	--hop_length 2.55 \
	--lipreading_extract_feature \
	--number_of_identity_frames 1 \
	--output_dir_root $output_path/

