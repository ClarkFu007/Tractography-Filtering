# Tractography-Filtering

bash Anaconda3-2019.07-Linux-x86_64.sh

conda create -n nicr python=3.7 (nicr is your virtual environment's name)

conda activate nicr

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch (from https://pytorch.org/get-started/previous-versions/)

pip install nibabel
pip install dipy

(If there are still lacking some pacakages, simply copy the indication and Google it.)

Train the model:
python3 train_main.py --cuda_id 0 --epoch_num 1000 --model_case 'final_VAE'

After training, used to saved model to evaluate:
Step1: Save filtered reults after doing k-means clustering of the latent space.
python3 evaluate.py --model_path '/ifs/loni/faculty/shi/spectrum/Student_2020/yfu/nicr/autoencoder/saved_models/MS_normal_best_model.pth' --cuda_id 0 --cluster_num 100 
--percentage_value 0.9 --refer_subj_num 10 -refer_str_num 10 --sigma_value 0.9 --subject_name 'motor_sensory' --voxel_version 'normal' --sphere_version False

Step2: Get consistency measures.
bash evaluate.sh (About two days and only occupy less than 25% of CPU resources)
Notice: you need to change evaluate_main.py's filename, subject_name, voxel_version and so on accordingly.
