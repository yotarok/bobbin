# GCP setup example

This document explains how to run the example script for LibriSpeech training
on TPU-VM in a distributed setting.

For completing the process described in this doc, you will need:

- a GCP project.
- two GCS buckets:
  1. For datasets (denoted as "DATASET\_BUCKET" in the followings),
  1. and for training results (denoted as "JOB\_LOG\_BUCKET).
- bobbin code repo cloned in the local machine.
- installation of Google Cloud SDK.
- installation of some tools on the local machines: ansible, jq.

## Setup client environment

The configuration is provided as an [Ansible](https://www.ansible.com/)
playbook. Therefore, you need to install Ansible for proceeding the process
described below. Further within Ansible, the example uses "jq" for parsing
JSON obtained as outputs of "gcloud" command. For Ubuntu Linux, those tools can
be installed by the following command:

```
sudo apt install ansible jq
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

## Dataset preparation

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
    "zone=ZONE project=PROJECT node_name=bobbin-tpu accelerator_type=TPUTYPE ssh_user=SSHUSER"
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
    'source bobbin_env/bin/activate; \
     python bobbin/examples/librispeech/train.py \
        --tfds_data_dir gs://DATASET_BUCKET/tensorflow_datasets \
        --log_dir_path gs://JOB_LOG_BUCKET/librispeech/first_exp \
        --per_device_batch_size=16'
# Don't forget a single quote at the end of line.
```

The above example creates the following directories and writes the results:

- Training checkpoints: `gs://JOB_LOG_BUCKET/librispeech/first_exp/all_ckpts`
- Best checkpoint: `gs://JOB_LOG_BUCKET/librispeech/first_exp/best_ckpts`
- TensorBoard summaries: `gs://JOB_LOG_BUCKET/librispeech/first_exp/tensorboard`

For monitoring the training progress, you can also run TensorBoard server on
the local machine as follows:

```
tensorboard --logdir gs://JOB_LOG_BUCKET/librispeech/first_exp/tensorboard
```

## (Optional) Delete the TPU VM

Once you finished the experiment, you can delete the TPU VM you created.
This can be done by the follosing command:

```
gcloud compute tpus tpu-vm delete bobbin-tpu --zone ZONE --project
PROJECT
```
