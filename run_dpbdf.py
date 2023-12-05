import os
import logging
import sys
import click

import models.utils
import models.dataset
import models.metrics
import models.dpb


LOGGER = logging.getLogger(__name__)

@click.group(chain=True)
@click.argument("save_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--save_results", is_flag=True)
@click.option("--save_model", is_flag=True)
@click.option("--base_class", type=int, default=2)
@click.option("--incre_class", type=int, default=1)
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    base_class,
    incre_class, 
    save_path,
    gpu, 
    seed,
    save_results,
    save_model,
):
    methods = {key: item for (key, item) in methods}

    device = models.utils.set_torch_device(gpu)
    models.utils.fix_seeds(seed, device)

    train_dataloader, test_dataloader, data_info = methods["get_dataloaders"](base_class, incre_class)
    save_path = os.path.join(save_path, data_info["name"])
    os.makedirs(save_path, exist_ok=True)

    dpb, train = methods["get_dpb"](data_info["imagesize"], device)

    f1_score_list = []
    accuracy_list = []
    for i in range(train_dataloader.sampler.n_tasks):
        train_dataloader.sampler.set_task(i)
        test_dataloader.sampler.set_task(i)

        categories = train_dataloader.dataset.categories

        # training
        if train:
            train_task_classes = categories[:base_class] if i == 0 else categories[base_class+(i-1)*incre_class:base_class+i*incre_class]
            LOGGER.info(
                "Training on Task: %d, Class: %s" 
                % (i, ','.join(train_task_classes))
            )
            dpb.update(train_dataloader)
    
        # evaluation
        eval_task_classes = categories[:base_class] if i == 0 else categories[:base_class+i*incre_class]
        LOGGER.info(
            "Evaluating on Task: %d, Class: %s"
            % (i, ','.join(eval_task_classes))
        )
        preds, labels = dpb.evaluate(test_dataloader)
        f1_score = models.metrics.f1_score(preds, labels)
        accuracy = models.metrics.accuracy_score(preds, labels)
        LOGGER.info(
            "Results-----------------\nTask: %d, F1-score: %.2f, Accuracy: %.2f"
            % (i, f1_score, accuracy)
        )
        f1_score_list.append(f1_score)
        accuracy_list.append(accuracy)
    
    if save_model:
        model_save_path = os.path.join(save_path, "checkpoints")
        os.makedirs(model_save_path, exist_ok=True)
        dpb.save(model_save_path)

    if save_results:
        results = {"F1-score": f1_score_list, "Accuracy": accuracy_list}
        models.utils.save_results(save_path, results, method_name='DPB-DF', incre=True)



@main.command("dpb_df")
@click.option("--load_path", "-p", type=str) 
@click.option("--k1", type=int, default=5)
@click.option("--k2", type=int, default=40)
@click.option("--dsr_n", "-r", type=float, default=0.1)
@click.option("--dsr_d", type=float, default=0.5)
def dpb(
    load_path,
    k1,
    k2,
    dsr_n,
    dsr_d
):
    def get_dpb(input_shape, device):
        train = True
        dpb = models.dpb.DPB(
            k1, k2, dsr_n, dsr_d, input_shape=input_shape, device=device
        )
        if load_path:
            dpb.load(load_path, load_df=True)
            train = False
        return dpb, train
    return ("get_dpb", get_dpb)


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--batch_size", default=32, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--validation", is_flag=True)
@click.option("--train_val_split", default=1.0, type=float, show_default=True)
def dataset(
    name, 
    data_path, 
    batch_size, 
    resize, 
    imagesize,
    validation,
    train_val_split,
    num_workers, 
):
    def get_dataloaders(base_class, incre_class):
        train_dataloader, test_dataloader = models.utils.get_dataloaders(
            root=data_path, 
            d_name=name,
            validation=validation,
            train_val_split=train_val_split,
            base_class=base_class,
            incre_class=incre_class,
            batch_size=batch_size,
            resize=resize,
            imagesize=imagesize,
            num_workers=num_workers
        )
        data_info = {
            "name": name,
            "imagesize": imagesize
        }
        return train_dataloader, test_dataloader, data_info
    
    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()