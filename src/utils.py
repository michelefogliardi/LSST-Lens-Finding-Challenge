import os
import sys
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from prettytable import PrettyTable
import torch
from torchinfo import summary

import errno
import time
from collections import defaultdict, deque
import torch.distributed as dist

################################################################################

def count_parameters(model):
    """Return the number of all parameters of the model."""
    return sum(p.numel() for p in model.parameters())

################################################################################

def count_named_parameters(model):
    """Return the number of all the _named_ parameters of the model."""
    return sum(p.numel() for name, p in model.named_parameters())

################################################################################

def count_trainable_parameters(model):
    """Return the number of all the _trainable_ parameters of the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

################################################################################

def save_parameters(model, file_name=None):
    table = PrettyTable(["Modules", "Parameters", "Trainable"])
    total_params = 0
    total_trainable_params = 0
    # assert that all the parameters are NAMED:
    assert count_named_parameters(model)==count_parameters(model)
    for name, parameter in model.named_parameters():
        param = parameter.numel()
        total_params += param
        trainable = 'False'
        if parameter.requires_grad:
            trainable = 'True'
            total_trainable_params += param
        table.add_row([name, param, trainable])
    if file_name is None:
        print(table)
        print(f"TOTAL Params          : {total_params}")
        print(f"Total TRAINABLE Params: {total_trainable_params}")
    else:
        with open(file_name, 'w') as f:
            print(table, file=f)
            print(f"TOTAL Params          : {total_params}",           file=f)
            print(f"Total TRAINABLE Params: {total_trainable_params}", file=f)

################################################################################

def fix_all_seeds(seed):
    """Fix randomness. Usage: fix_all_seeds(42)"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark     = True

################################################################################

def show_gpu_memory(device):
    """Print on standard output information on the status of the memory of the
    GPU."""
    # Retrieve maximum GPU memory allocated by PyTorch
    max_memory_allocated = torch.cuda.max_memory_allocated()
    # Retrieve GPU memory statistics
    memory_stats = torch.cuda.memory_stats()
    # Calculate available GPU memory
    total_memory     = torch.cuda.get_device_properties(0).total_memory
    available_memory = total_memory - memory_stats["allocated_bytes.all.current"]
    # Print the result
    print(f"    [INFO] Allocated GPU memory: {torch.cuda.memory_allocated(device) / 1024 / 1024:.2f} MB")
    print(f"    [INFO] Peak GPU memory allocated by PyTorch: {max_memory_allocated / 1024**3:.2f} GB")
    print(f"    [INFO] Cached GPU memory   : {torch.cuda.memory_reserved(device)  / 1024 / 1024:.2f} MB")
    print(f"    [INFO] Available GPU memory: {available_memory / 1024**3:.2f} GB")

################################################################################

def create_model_dir(config, path='/dati4/mfogliardi/training/ggsl/', folder='models'):
    """Create the model directory (if it does not already exist), using the
    date and time of execution."""
    base_dir  = Path(path)
    model_dir = Path(base_dir/folder)
    model_dir.mkdir(parents=True, exist_ok=True)
    # Generate timestamp for the training session (Year-Month-Day_Hour-Minute-Second)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Create a directory to store the checkpoints if it does not already exist
    checkpoint_dir = Path(model_dir/f"{config.VERSION+'_'+timestamp}")
    # Create the checkpoint directory if it does not already exist
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Path of the model checkpoint file
    checkpoint_file = checkpoint_dir/f"{config.MODEL_NAME+'.pt'}"
    return checkpoint_file, timestamp

################################################################################

def save_df_results_csv(results, file_name):
    df_results = pd.DataFrame.from_dict(results)
    df_results.to_csv(file_name, index=False)
    return df_results

################################################################################

def save_fig_losses(df_results, file_name):
    """Plot losses."""
    plt.figure(figsize=(12,8))
    plt.plot(np.array(df_results['epoch']), df_results['train_loss_tot'],         ls='-',  c='k', lw=2.0,  alpha=1.0,  label='Train Loss (TOT)')
    plt.plot(np.array(df_results['epoch']), df_results['valid_loss_tot'],         ls='--', c='k', lw=2.0,  alpha=1.0,  label='Valid Loss (TOT)')
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(file_name)
    plt.grid()
    plt.legend(prop={'size': 8}, framealpha=0.9, ncol=6)
    plt.tight_layout()
    plt.savefig(file_name)

################################################################################

def save_config(config, file_name=None):
    """Print the config object."""
    #config_dict = config.__dict__
    attrs = vars(config)
    if file_name is None:
        for key,value in attrs.items():
            print(str(key) + ' = ' +str(value))
    else:
        with open(file_name, 'w') as f:
            for key,value in attrs.items():
                if str(key)[:2] == '__':
                    continue
                print(str(key) + ' = ' +str(value), file=f)

################################################################################

def save_model_summary(model, config, file_name=None):
    """Print a summary of the model using `torchinfo`."""
    model_summary = summary(model=model,
                            input_size=(config.BATCH_SIZE, 6, config.HEIGHT, config.WIDTH), # make sure this is "input_size", not "input_shape"
                            col_names=["input_size", "output_size", "num_params", "trainable"],
                            #col_width=25,
                            row_settings=["var_names"],
                            verbose=0)
    if file_name is None:
        print(model_summary)
    else:
        with open(file_name, 'w') as f:
            print(model_summary, file=f)

def save_model_summary3(model, config, file_name=None):
    """Print a summary of the model using `torchinfo`."""
    model_summary = summary(model=model,
                            input_size=(config.BATCH_SIZE, 3, config.HEIGHT, config.WIDTH), # make sure this is "input_size", not "input_shape"
                            col_names=["input_size", "output_size", "num_params", "trainable"],
                            #col_width=25,
                            row_settings=["var_names"],
                            verbose=0)
    if file_name is None:
        print(model_summary)
    else:
        with open(file_name, 'w') as f:
            print(model_summary, file=f)



################################################################################

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_model.pth")
    """
    ### Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    ### Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    ### Save the model `state_dict()`
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

################################################################################

def calculate_f1_score(precision, recall):
    """Compute the F1 score from Precision and Recall values."""
    if (precision + recall) > 0:
      f1_score = 2. * (precision * recall) / (precision + recall)
    else:
      f1_score = 0
    return f1_score

################################################################################
# SOURCE: https://github.com/pytorch/vision/blob/v0.15.2/references/detection/utils.py
################################################################################

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


def collate_fn(batch):
    return tuple(zip(*batch))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

################################################################################