conda activate dcase2020_workshop

epochs=100
# valid_types in all same_mic_all_types same_mic_different_type same_mic_same_type different_mic
for valid_types in different_mic
do
  python -m dcase2020_workshop.experiments.classification_experiment with "id=initial_gs_$valid_types" machine_type=0 machine_id=0 training_settings.epochs=$epochs outlier_settings.valid_types=$valid_types -m student2.cp.jku.at:27017:initial_gs_
  python -m dcase2020_workshop.experiments.classification_experiment with "id=initial_gs_$valid_types" machine_type=1 machine_id=0 training_settings.epochs=$epochs outlier_settings.valid_types=$valid_types -m student2.cp.jku.at:27017:initial_gs_
  python -m dcase2020_workshop.experiments.classification_experiment with "id=initial_gs_$valid_types" machine_type=2 machine_id=0 training_settings.epochs=$epochs outlier_settings.valid_types=$valid_types -m student2.cp.jku.at:27017:initial_gs_
  python -m dcase2020_workshop.experiments.classification_experiment with "id=initial_gs_$valid_types" machine_type=3 machine_id=1 training_settings.epochs=$epochs outlier_settings.valid_types=$valid_types -m student2.cp.jku.at:27017:initial_gs_
  python -m dcase2020_workshop.experiments.classification_experiment with "id=initial_gs_$valid_types" machine_type=4 machine_id=1 training_settings.epochs=$epochs outlier_settings.valid_types=$valid_types -m student2.cp.jku.at:27017:initial_gs_
  python -m dcase2020_workshop.experiments.classification_experiment with "id=initial_gs_$valid_types" machine_type=5 machine_id=0 training_settings.epochs=$epochs outlier_settings.valid_types=$valid_types -m student2.cp.jku.at:27017:initial_gs_
done

