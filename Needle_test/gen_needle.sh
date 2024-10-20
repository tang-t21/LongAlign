# generate num_needles as a list of integers from 1 to 4096, with a step of 1
num_needles=(`seq 1 1 4096`)
mkdir -p needles
for i in ${num_needles[@]}; do
    python needle_gen.py --num_keys $i --output_dir needles
done