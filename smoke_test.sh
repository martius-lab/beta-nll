#!/bin/bash

set -e

LOGDIR=./logs_smoke_test
N_EPOCHS=3

mkdir -p $LOGDIR

echo "##########################################"
echo "            Running smoke test            "
echo ""
echo "Logging to $LOGDIR"
echo "##########################################"

for loss_fn in likelihood mse moment_matching student_t vari_var_xvamp vari_var_xvamp_star vari_var_vbem vari_var_vbem_star
do
	echo "### Testing toy-sinusoidal with $loss_fn"
	python -m src.train --log_dir $LOGDIR --log_every=1 --n_epochs $N_EPOCHS \
		--name sine_11_$loss_fn \
		--dataset 11 \
		--training $loss_fn \
		--loss-weight 1 \
		--batch_size 100 \
		--lr 0.0005 \
		--hidden_activation tanh \
		--hidden_dims 64 64

	echo "### Testing 1D-Slide with $loss_fn"
	python -m src.train --log_dir $LOGDIR --log_every=1 --n_epochs $N_EPOCHS \
		--name 1dslide_$loss_fn \
		--dataset 1dslide \
		--data_variant random2k \
		--standardize-inputs  \
		--eval-test \
		--training $loss_fn \
		--loss-weight 1 \
		--batch_size 256 \
		--lr 0.0005 \
		--hidden_activation tanh \
		--hidden_dims 64 64  \
		--track-best-metrics eval_likelihood

	echo "### Testing FetchPickAndPlace with $loss_fn"
	python -m src.train --log_dir $LOGDIR --log_every=1 --n_epochs $N_EPOCHS \
		--name fpp_$loss_fn \
		--dataset fpp \
		--standardize-inputs  \
		--eval-test \
		--training $loss_fn \
		--loss-weight 0.5 \
		--batch_size 256 \
		--lr 0.0005 \
		--hidden_activation tanh \
		--hidden_dims 64 64  \
		--track-best-metrics eval_likelihood  \
		--train-split 0.7  \
		--test-split 0.15

	if [ $loss_fn != "mse" ]; then
		echo "### Testing MNIST with $loss_fn"
		python -m src.train --log_dir $LOGDIR --log_every=1 --n_epochs $N_EPOCHS \
				--name mnist_$loss_fn \
				--device cuda \
				--dataset mnist \
				--eval-test \
				--train-split 0.8  \
				--training $loss_fn \
				--loss-weight 0.5 \
				--batch_size 250 \
				--lr 0.0003 \
				--model-type VAE \
				--latent-dims 10 \
				--hidden_activation relu \
				--hidden_dims 512 256 128 \
				--early-stop-metric eval_likelihood \
				--early-stop-iters 2
	fi

	echo "### Testing UCI energy with $loss_fn"
	python -m src.train_uci --log_dir $LOGDIR --log_every=1 \
		--name uci_energy_$loss_fn \
		--data-variant energy \
		--n_splits 3 \
		--training $loss_fn \
		--loss-weight 0.5 \
		--batch_size 100 \
		--n_updates 100 \
		--hidden_dims 50
done

rm -r $LOGDIR