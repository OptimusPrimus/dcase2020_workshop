conda activate dcase2020_workshop

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m dcase2020_workshop.experiments.$1 with num_workers=4 machine_type=0 machine_id=0 $3
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m dcase2020_workshop.experiments.$1 with num_workers=4 machine_type=0 machine_id=2 $3
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m dcase2020_workshop.experiments.$1 with num_workers=4 machine_type=0 machine_id=4 $3
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m dcase2020_workshop.experiments.$1 with num_workers=4 machine_type=0 machine_id=6 $3

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m dcase2020_workshop.experiments.$1 with num_workers=4 machine_type=1 machine_id=0 $3
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m dcase2020_workshop.experiments.$1 with num_workers=4 machine_type=1 machine_id=2 $3
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m dcase2020_workshop.experiments.$1 with num_workers=4 machine_type=1 machine_id=4 $3
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m dcase2020_workshop.experiments.$1 with num_workers=4 machine_type=1 machine_id=6 $3

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m dcase2020_workshop.experiments.$1 with num_workers=4 machine_type=2 machine_id=0 $3
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m dcase2020_workshop.experiments.$1 with num_workers=4 machine_type=2 machine_id=2 $3
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m dcase2020_workshop.experiments.$1 with num_workers=4 machine_type=2 machine_id=4 $3
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m dcase2020_workshop.experiments.$1 with num_workers=4 machine_type=2 machine_id=6 $3

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m dcase2020_workshop.experiments.$1 with num_workers=4 machine_type=3 machine_id=1 $3
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m dcase2020_workshop.experiments.$1 with num_workers=4 machine_type=3 machine_id=2 $3
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m dcase2020_workshop.experiments.$1 with num_workers=4 machine_type=3 machine_id=3 $3
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m dcase2020_workshop.experiments.$1 with num_workers=4 machine_type=3 machine_id=4 $3

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m dcase2020_workshop.experiments.$1 with num_workers=4 machine_type=4 machine_id=1 $3
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m dcase2020_workshop.experiments.$1 with num_workers=4 machine_type=4 machine_id=2 $3
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m dcase2020_workshop.experiments.$1 with num_workers=4 machine_type=4 machine_id=3 $3

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m dcase2020_workshop.experiments.$1 with num_workers=4 machine_type=5 machine_id=0 $3
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m dcase2020_workshop.experiments.$1 with num_workers=4 machine_type=5 machine_id=2 $3
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m dcase2020_workshop.experiments.$1 with num_workers=4 machine_type=5 machine_id=4 $3
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m dcase2020_workshop.experiments.$1 with num_workers=4 machine_type=5 machine_id=6 $3