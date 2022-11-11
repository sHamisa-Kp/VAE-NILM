#!/bin/bash

for value in {0..50}
do
	echo $value
	sbatch flvae.sh
	sleep 10
done

echo All done
