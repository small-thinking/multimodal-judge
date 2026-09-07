from multimodal_judge.run_naming import make_run_name


def test_names_distinguish_job_model_dataset_and_split():
    train = make_run_name('train', 'Qwen/Qwen3-VL-2B-Instruct', 'data/training_data/v5')
    evaluation = make_run_name('eval', 'Qwen/Qwen3-VL-2B-Instruct', 'data/training_data/v5', 'test')
    assert train.startswith('train-Qwen3-VL-2B-Instruct-v5-')
    assert evaluation.startswith('eval-Qwen3-VL-2B-Instruct-v5-test-')
    assert '/' not in train + evaluation
