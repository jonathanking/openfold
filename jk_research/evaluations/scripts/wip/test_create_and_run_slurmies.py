from experiments import Experiment, TrainingExperiment, EvaluationExperiment


def test_import():
    assert Experiment
    assert TrainingExperiment
    assert EvaluationExperiment


def test_create_training_experiment():
    exp = TrainingExperiment(exp_name="baseline-scnmin-noomm-00",
                             wandb_id="3lzb4s3o",
                             location="g019",
                             notes="test recreating the baseline scnmin experiment")

    assert exp.gpus == 4
    assert exp.ntasks == 64
    assert exp.partition == "dept_gpu"
    assert exp.time == "14-00:00:00"
    assert exp.qos is None
    assert exp.cluster is None
    assert exp.nodelist == "g019"


if __name__ == "__main__":
    test_import()
    test_create_training_experiment()