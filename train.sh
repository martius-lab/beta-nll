#!/bin/bash

LOGDIR=./logs

if [[ $# < 2 ]]; then
	echo "Usage: ./train.sh <dataset> <method> <beta>"
	echo "where <dataset> can be one of {Sine,SineDetlefsen,ObjectSlide,FPP,MNIST,FashionMNIST}"
	echo "      <method> can be one of {likelihood,moment_matching,mse,student_t,xvamp,xvamp_star,vbem,vbem_star}"
	echo "      <beta> in [0, 1]"
	exit 1
fi

if [[ $1 = "Sine" ]]; then
	dataset=11
elif [[ $1 = "SineDetlefsen" ]]; then
	dataset=3
elif [[ $1 = "ObjectSlide" ]]; then
	dataset=1dslide
elif [[ $1 = "FPP" ]]; then
	dataset=fpp
elif [[ $1 = "MNIST" ]]; then
	dataset=mnist
elif [[ $1 = "FashionMNIST" ]]; then
	dataset=fashion-mnist
else
   	echo "Unknown dataset $1."
   	exit 1
fi

if [[ $2 = "likelihood" ]]; then
	method=likelihood
elif [[ $2 = "moment_matching" ]]; then
	method=moment_matching
elif [[ $2 = "mse" ]]; then
	method=mse
elif [[ $2 = "student_t" ]]; then
	method=student_t
elif [[ $2 = "xvamp" ]]; then
	method=vari_var_xvamp
elif [[ $2 = "xvamp_star" ]]; then
	method=vari_var_xvamp_star
elif [[ $2 = "vbem" ]]; then
	method=vari_var_vbem
elif [[ $2 = "vbem_star" ]]; then
	method=vari_var_vbem_star
else
   	echo "Unknown method $2."
   	exit 1
fi

if [[ $# == 3 ]]; then
	loss_weight=$3
else
	loss_weight=0
fi


if [[ $1 = "Sine" ]]; then
	python -m src.train --log_dir $LOGDIR --log_every=10000 --n_epochs 1000000 \
			--name Sine_$method_$loss_weight \
			--dataset $dataset \
			--training $method \
			--loss-weight $loss_weight \
			--batch_size 100 \
			--lr 0.0005 \
			--hidden_activation tanh \
			--hidden_dims 128 128
elif [ $1 = "SineDetlefsen" ]; then
	python -m src.train --log_dir $LOGDIR --log_every=5000 --n_epochs 20000 \
			--name SineDetlefsen_$method_$loss_weight \
			--dataset $dataset \
			--training $method \
			--loss-weight $loss_weight \
			--batch_size 100 \
			--lr 0.0001 \
			--weight-init lecun \
			--hidden_activation tanh \
			--hidden_dims 128 128
elif [ $1 = "ObjectSlide" ]; then
	activation="relu"
	if [ $method = "likelihood" ] || [ $method = "mse" ] || [ $method = "moment_matching" ]; then
		hidden_dims="128 128 128"
		learning_rate="0.001"
	elif [ $method = "student_t" ]; then
		hidden_dims="386 386"
		learning_rate="0.001"
	elif [ $method = "vari_var_xvamp" ]; then
		hidden_dims="128 128 128 128"
		learning_rate="0.0001"
	elif [ $method = "vari_var_xvamp_star" ]; then
		hidden_dims="256 256 256"
		learning_rate="0.0001"
	elif [ $method = "vari_var_vbem" ]; then
		hidden_dims="256 256"
		learning_rate="0.0003"
		activation="tanh"
	elif [ $method = "vari_var_vbem_star" ]; then
		hidden_dims="386 386"
		learning_rate="0.001"
	fi

	python -m src.train --log_dir $LOGDIR --log_every=100 --n_epochs 5000 \
			--name ObjectSlide_$method_$loss_weight \
			--dataset $dataset \
			--data_variant random2k \
			--standardize-inputs \
			--training $method \
			--loss-weight $loss_weight \
			--batch_size 256 \
			--lr $learning_rate \
			--weight-init lecun \
			--hidden_activation $activation \
			--hidden_dims $hidden_dims \
			--track-best-metrics eval_likelihood eval_mse
elif [ $1 = "FPP" ]; then
	if [ $method = "likelihood" ] && [ $(echo "$loss_weight <= 0.5" | bc) = 1 ]; then
		learning_rate="0.0003"
	else
		learning_rate="0.001"
	fi

	if [ $method = "likelihood" ] || [ $method = "mse" ] || [ $method = "moment_matching" ]; then
		hidden_dims="128 128 128 128"
	elif [ $method = "student_t" ]; then
		hidden_dims="256 256 256"
		learning_rate="0.0003"
	else
		hidden_dims="386 386 386"
		learning_rate="0.0001"
	fi

	if [ $method = "vari_var_vbem" ]; then
		learning_rate="0.001"
	fi

	python -m src.train --log_dir $LOGDIR --log_every=10 --n_epochs 500 \
			--name FPP_$method_$loss_weight \
			--dataset $dataset \
			--train-split 0.7  \
			--test-split 0.15 \
			--standardize-inputs \
			--training $method \
			--loss-weight $loss_weight \
			--batch_size 100 \
			--lr $learning_rate \
			--weight-init lecun \
			--hidden_activation relu \
			--hidden_dims $hidden_dims \
			--track-best-metrics eval_likelihood eval_mse
elif [ $1 = "MNIST" ] || [ $1 = "FashionMNIST" ]; then
	python -m src.train --log_dir $LOGDIR --log_every=5 --n_epochs 1000 \
			--name $1_$method_$loss_weight \
			--device cuda \
			--dataset $dataset \
			--train-split 0.8  \
			--training $method \
			--loss-weight $loss_weight \
			--batch_size 250 \
			--lr 0.0003 \
			--model-type VAE \
			--latent-dims 10 \
			--hidden_activation relu \
			--hidden_dims 512 256 128 \
			--early-stop-metric eval_likelihood \
			--early-stop-iters 10
else
	echo "Unknown dataset $1."
   	exit 1
fi