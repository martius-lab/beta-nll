from src.data import one_d_slide
from src.data.fpp_dataset import FPPDataset
from src.data.toy_data import ToyDataset
from src.data.uci_datasets import UCICollection


def load_dataset(
    dataset_name,
    data_variant=None,
    train_split=None,
    test_split=None,
    data_only_random=False,
    use_diff_as_target=True,
    standardize_inputs=False,
    test_split_idx=None,
    data_noise_level=None,
    data_dir=None,
    load_test_set=False,
):
    try:
        dataset_id = int(dataset_name)
    except ValueError:
        dataset_id = -1

    dataset_name = dataset_name.lower()

    if 0 <= dataset_id < 20:
        dataset, eval_dataset = ToyDataset.load_data(dataset_id)
        eval_x, eval_y = eval_dataset.X, eval_dataset.Y
        inp_dim, outp_dim = 1, 1
    elif dataset_id == 21 or dataset_name == "1dslide":
        assert data_variant is not None
        assert data_variant in one_d_slide.DATASETS
        train_path, eval_path, test_path = one_d_slide.get_data_paths(
            data_variant, data_dir
        )
        dataset = one_d_slide.make_one_d_slide_dataset(
            train_path,
            only_random_actions=data_only_random,
            obs_noise_level=data_noise_level,
            only_goal_noise=True,
        )
        dataset.standardize_targets()

        eval_dataset = one_d_slide.make_one_d_slide_dataset(
            test_path if load_test_set else eval_path,
            only_random_actions=data_only_random,
            obs_noise_level=data_noise_level,
            only_goal_noise=True,
        )
        eval_x, eval_y = eval_dataset.inputs, eval_dataset.targets

        if standardize_inputs:
            x_mean, x_std = dataset.standardize_inputs()
            eval_dataset.standardize_inputs(x_mean, x_std)

        inp_dim = dataset[0][0].shape[0]
        outp_dim = dataset[0][1].shape[0]
    elif dataset_id == 22 or dataset_name == "uci":
        assert data_variant is not None, "data_variant parameter must be set!"
        assert train_split is not None, "train_split must be set!"
        uci_collection = UCICollection(
            data_variant,
            test_split_idx=test_split_idx,
            train_val_split=train_split,
            standardize_inputs=True,
            standardize_targets=True,
        )

        dataset = uci_collection.train_data
        if load_test_set:
            eval_dataset = uci_collection.test_data
        else:
            eval_dataset = uci_collection.eval_data

        eval_x, eval_y = (
            eval_dataset.tensors[0].numpy(),
            eval_dataset.tensors[1].numpy(),
        )
        inp_dim = uci_collection.input_dim
        outp_dim = uci_collection.target_dim
    elif dataset_id == 23 or dataset_name == "fpp":
        assert train_split is not None, "train_split must be set!"
        fpp_data = FPPDataset(
            use_diff_as_target,
            train_split,
            test_split=test_split,
            standardize_inputs=standardize_inputs,
            standardize_targets=True,
            obs_noise_level=data_noise_level,
            data_dir=data_dir,
            only_object_noise=True,
        )
        dataset = fpp_data.train_data
        if load_test_set:
            eval_dataset = fpp_data.test_data
            assert eval_dataset is not None, "No test set specified/available"
        else:
            eval_dataset = fpp_data.val_data
            assert eval_dataset is not None, "No val set specified/available"

        eval_x, eval_y = (
            eval_dataset.tensors[0].numpy(),
            eval_dataset.tensors[1].squeeze().numpy(),
        )
        inp_dim = fpp_data.input_dim
        outp_dim = fpp_data.target_dim
    elif dataset_name == "mnist":
        from src.data.reconstruction_datasets import ReconstructionDataset

        assert train_split is not None, "train_split must be set!"

        dataset = ReconstructionDataset(
            "mnist",
            split="train",
            train_split=train_split,
            flatten=True,
            data_dir=data_dir,
        )
        split = "test" if load_test_set or train_split == 1.0 else "val"
        eval_dataset = ReconstructionDataset(
            "mnist",
            split=split,
            train_split=train_split,
            flatten=True,
            data_dir=data_dir,
        )
        eval_x, eval_y = eval_dataset.to_tensors()
        eval_x = eval_x.numpy()
        eval_y = eval_y.numpy()
        inp_dim = dataset.input_dim
        outp_dim = dataset.target_dim
    elif dataset_name == "fashion-mnist":
        from src.data.reconstruction_datasets import ReconstructionDataset

        assert train_split is not None, "train_split must be set!"

        dataset = ReconstructionDataset(
            "fashion-mnist",
            split="train",
            train_split=train_split,
            flatten=True,
            data_dir=data_dir,
        )
        split = "test" if load_test_set or train_split == 1.0 else "val"
        eval_dataset = ReconstructionDataset(
            "fashion-mnist",
            split=split,
            train_split=train_split,
            flatten=True,
            data_dir=data_dir,
        )
        eval_x, eval_y = eval_dataset.to_tensors()
        eval_x = eval_x.numpy()
        eval_y = eval_y.numpy()
        inp_dim = dataset.input_dim
        outp_dim = dataset.target_dim
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    assert dataset[0][0].ndim == 1 and dataset[0][1].ndim == 1
    assert eval_x.ndim == 2 and eval_y.ndim == 2

    return dataset, (eval_dataset, eval_x, eval_y), inp_dim, outp_dim


def should_collect_likelihoods(dataset_name):
    try:
        dataset_id = int(dataset_name)
    except ValueError:
        dataset_id = -1

    dataset_name = dataset_name.lower()

    if 0 <= dataset_id < 20:
        return None
    elif dataset_id == 21 or dataset_name == "1dslide":
        return False
    else:
        return None
