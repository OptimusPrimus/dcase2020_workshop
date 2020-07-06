conda activate dcase2020_workshop

valid_types=same_mic_all_types
experiment_name=samplesize
gpu=0
outlier_data_set_class=dcase2020_workshop.data_sets.ComplementMCMDataSet

for seed in 181096 130394 311491
do
  for num_samples in 64 128 256 512
  do
    ./scripts/per_id_run.sh classification_experiment $gpu "seed=$seed id=workshop_${experiment_name}_${outlier_data_set_class}_${valid_types} outlier_data_set_class=${outlier_data_set_class} outlier_settings.valid_types=$valid_types outlier_settings.num_samples=${num_samples} -m student2.cp.jku.at:27017:dcase2020_workshop_experiments_${experiment_name}" > /dev/null 2>&1 &
    gpu=`expr $gpu + 1`
  done

  gpu=0
  wait

done
