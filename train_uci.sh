#!/bin/bash

LOGDIR=./logs

if [[ $# < 2 ]]; then
	echo "Usage: ./train.sh <dataset> <method> <beta>"
	echo "where <dataset> can be one of {carbon,concrete,housing,energy,kin8m,naval,power,protein,superconductivity,wine-red,wine-white,yacht}"
	echo "      <method> can be one of {likelihood,moment_matching,mse,student_t,xvamp,xvamp_star,vbem,vbem_star}"
	echo "      <beta> in [0, 1]"
	exit 1
fi

if [[ $1 = "protein" ]]; then
	n_splits=5
	n_updates=100000
	hidden_dims=100
elif [[ $1 = "carbon" ]] || [[ $1 = "power" ]] || [[ $1 = "kin8m" ]] || [[ $1 = "naval" ]] || [[ $1 = "superconductivity" ]]; then
	n_splits=20
	n_updates=100000
	hidden_dims=50
elif [[ $1 = "concrete" ]] || [[ $1 = "housing" ]] || [[ $1 = "energy" ]] || [[ $1 = "wine-red" ]] || [[ $1 = "wine-white" ]] || [[ $1 = "yacht" ]]; then
	n_splits=20
	n_updates=20000
	hidden_dims=50
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

python -m src.train_uci --log_dir $LOGDIR \
	--name uci_$1 \
	--data-variant $1 \
	--n_splits $n_splits \
	--training $method \
	--loss-weight $loss_weight \
	--batch_size 256 \
	--n_updates $n_updates \
	--hidden_dims $hidden_dims
