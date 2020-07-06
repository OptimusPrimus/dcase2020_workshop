gpu=0

for seed in 181096 130394 311491
do

  # baseline
  ./scripts/per_id_run.sh baseline_experiment $gpu "seed=$seed id=workshop_baseline -m student2.cp.jku.at:27017:dcase2020_workshop_experiments" &
  gpu=`expr $gpu + 1`

  # complements sets
  for valid_types in all same_mic_all_types same_mic_different_type same_mic_same_type different_mic
  do
    outlier_data_set_class=dcase2020_workshop.data_sets.ComplementMCMDataSet
    ./scripts/per_id_run.sh classification_experiment $gpu "seed=$seed id=workshop_${outlier_data_set_class}_${valid_types} outlier_data_set_class=${outlier_data_set_class} outlier_settings.valid_types=$valid_types -m student2.cp.jku.at:27017:dcase2020_workshop_experiments" > /dev/null 2>&1 &
    gpu=`expr $gpu + 1`
  done

  # audio set
  outlier_data_set_class=dcase2020_workshop.data_sets.AudioSet
  ./scripts/per_id_run.sh classification_experiment $gpu "seed=$seed id=workshop_${outlier_data_set_class}_balanced outlier_data_set_class=${outlier_data_set_class} -m student2.cp.jku.at:27017:dcase2020_workshop_experiments" > /dev/null 2>&1 &
  gpu=`expr $gpu + 1`

  # asc set
  outlier_data_set_class=dcase2020_workshop.data_sets.ASCSet
  ./scripts/per_id_run.sh classification_experiment $gpu "seed=$seed id=workshop_${outlier_data_set_class} outlier_data_set_class=${outlier_data_set_class} -m student2.cp.jku.at:27017:dcase2020_workshop_experiments" > /dev/null 2>&1 &

  gpu=0
  wait

done
