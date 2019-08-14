# training
# for momentum data
python train.py --data_type=momentum --train_batch_size=64 --xval_batch_size=64 --data_dir=/data/yuming/quantum-processed-data/momentum --output_dir=output/momentum/0 --gpus=0
# for position data
python train.py --data_type=position --train_batch_size=64 --xval_batch_size=64 --data_dir=/data/yuming/quantum-processed-data/position --output_dir=output/position/0 --gpus=0

# testing
# for normal data, test size is 736
# for transition data, test size is 937
# for momentum data
python test.py --data_type=momentum --test_size=0 --data_dir=/data/yuming/quantum-processed-data/momentum --output_dir=output/momentum/0 --gpus=0
# for position data
python test.py --data_type=position --test_size=0 --data_dir=/data/yuming/quantum-processed-data/position --output_dir=output/position/0 --gpus=0

# inspect som
python inspect_checkpoint.py --file_path=output/momentum/0 --output_dir=output-som/momentum/0 --tensor_name=model/som/w --analyze=1

# inspect coordinates of samples
python inspect_coord.py --data_type=momentum --data_dir=samples/C_0_0.tfr,samples/C_0_1.tfr --output_dir=output/momentum/21 --gpus=0

# for inspect outliers
python inspect_outlier.py  --data_type=momentum --test_size=754 --data_dir=/scratch/quantum-meta/momentum_transition --output_dir=output/momentum_transition/11 --gpus=0
python inspect_outlier.py  --data_type=position --test_size=754 --data_dir=/scratch/quantum-meta/position_transition --output_dir=output/position_transition/11 --gpus=0


# for inspect interactive data
python test_2.py --data_type=momentum --test_size=784 --data_dir=/scratch/quantum-data/interact/phi_0.01_momentum/quantum.tfr --output_dir=output/momentum/21 --gpus=0
