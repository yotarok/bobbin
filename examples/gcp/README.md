# GCP setup example

This document explains how to run the example script for LibriSpeech training
on TPU-VM in a distributed setting.

For completing the process described in this doc, you will need:

- a GCP project.
- two GCS buckets:
  1. For datasets (denoted as "DATASET\_BUCKET" in the followings),
  1. and for training results (denoted as "JOB\_LOG\_BUCKET).
- bobbin code repo cloned in the local machine.

## Setup client environment

The configuration is provided as an [Ansible](https://www.ansible.com/)
playbook. Therefore, you need to install Ansible for proceeding the process
described below. For Ubuntu Linux, Ansible can be installed by the following
command:

```
sudo apt install ansible
```

The additional required python packages can be installed to venv by running
the following script.

```
# Hereafter, all command-line snippets below assumes that the current working
# directory is the root directory of bobbin project.
./setup_devenv.sh
source devenv/bin/activate
```

You only need to set up the client once; however, whenever you close the shell
and return to work on this later, you have to call `source devenv/bin/activate`
for entering to the venv.

## Dataset preparation (LibriSpeech specific)

For LibriSpeech dataset, it is strongly recommended to use a pre-built dataset
directory. For building the dataset directory, you need to create a GCS bucket
(here denoted as "DATASET\_BUCKET"), and enter the following command.

```
tfds build --data_dir gs://DATASET_BUCKET/tensorflow_datasets \
    -c lazy_decode librispeech
```

## GCP configuration

First, you need to log in to your GCP account with the following command:

```
gcloud auth application-default login
```

Then, by running the following command, you have TPU-VM to be setup.

```
ansible-playbook ./examples/gcp/tpu_vm.playbook.yml --extra-vars \
    "zone=ZONE project=PROJECT node_name=bobbin-tpu accelerator_type=TPUTYPE"
```

"PROJECT", "ZONE", and "TPUTYPE" here are placeholders, and values must be
adapted according to your configurations.
For configuring "ZONE" and "TPUTYPE", you must check [TPU regions and
zones](https://cloud.google.com/tpu/docs/regions-zones) for availability.
One possible setup is "v2-32" for "TPUTYPE" and "us-central1-a" for "ZONE".

Here, we specified "node\_name=bobbin-tpu". This is also a configurable variable
that you can choose your favorite name.

## Run LibriSpeech experiments

Finally, we can run the following command for running training.

```
gcloud compute tpus tpu-vm ssh bobbin-tpu \
    --zone=ZONE --project=PROJECT --worker=all --command \
    'python3 bobbin/examples/librispeech/train.py \
        --tfds_data_dir gs://DATASET_BUCKET/tensorflow_datasets \
        --log_dir_path gs://JOB_LOG_BUCKET/librispeech/first_exp \
        --per_device_batch_size=16'
# Don't forget a single quote at the end of line.
```

For monitoring the training progress, you can also run TensorBoard process on
the local machine as follows:

```
tensorboard --logdir gs://JOB_LOG_BUCKET/librispeech/first_exp/tensorboard
```
