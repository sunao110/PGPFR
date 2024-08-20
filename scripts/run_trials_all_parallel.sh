#!/bin/bash

# Run experiment
src_dir=/ogr_cmu/src
scripts_dir=/ogr_cmu/scripts
cd ${scripts_dir}

# General config
split_type="agnostic"
CUDA_VISIBLE_DEVICES=0
gpu=0

datasets=("ego_gesture")
baselines=("Rdfcil")
n_trials=3
n_tasks=7

#Run trials
trial_id=0
./run_trial.sh $trial_id $src_dir $split_type $gpu "${datasets[*]}" "${baselines[*]}" &

trial_id=1
./run_trial.sh $trial_id $src_dir $split_type $gpu "${datasets[*]}" "${baselines[*]}" &

trial_id=2
./run_trial.sh $trial_id $src_dir $split_type $gpu "${datasets[*]}" "${baselines[*]}" &





