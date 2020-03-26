#!/bin/bash

for line in $(cat experiment-design-full.txt)
do
    IFS=';' # Delimiter
    read -ra ADDR <<< "$line" # line is read into an array as tokens separated by IFS
    echo "Line: ${ADDR[@]}"
    HYPER_N="${ADDR[0]}"
    SPLIT="${ADDR[1]}"
    BATCH_SIZE="${ADDR[2]}"
    LEARNING_RATE="${ADDR[3]}"
    EPOCHS="${ADDR[4]}"
    STEPS=1000

    sed -e "s|HYPER_N|${HYPER_N}|g" -e "s|SPLIT|${SPLIT}|g" -e "s|BATCH_SIZE|${BATCH_SIZE}|g" -e "s|LEARNING_RATE|${LEARNING_RATE}|g" -e "s|EPOCHS|${EPOCHS}|g" -e "s|STEPS|${STEPS}|g" template2.sbatch > /pylon5/hm5fp1p/results/campaign0/stamps_hyper_input/hyper${HYPER_N}/hyper${HYPER_N}.sbatch

    JID=$(sbatch --output=/pylon5/hm5fp1p/results/campaign0/stamps_hyper_results/hyper${HYPER_N}/hyper${HYPER_N}.out --error=/pylon5/hm5fp1p/results/campaign0/stamps_hyper_results/hyper${HYPER_N}/hyper${HYPER_N}.err /pylon5/hm5fp1p/results/campaign0/stamps_hyper_input/hyper${HYPER_N}/hyper${HYPER_N}.sbatch)
    
    echo $JID
    JOB_ID=${JID##* }
    touch job_ids.txt
    echo `date`" "${JOB_ID} >> job_ids.txt

    IFS=' ' # reset to default value after usage
done
