
import psutil
import torch.multiprocessing as mp
from collections import defaultdict

def process_group(model, subset_data, complexity):
    return model(subset_data, complexity)

def redistribute_complexity_groups(grouped_data, resource_threshold=0.8):
    cpu_usage = psutil.cpu_percent(interval=1, percpu=False)
    if cpu_usage > resource_threshold * 100:
        largest_group = max(grouped_data, key=lambda k: len(grouped_data[k]))
        samples_to_move = len(grouped_data[largest_group]) // 2
        samples = grouped_data[largest_group][:samples_to_move]
        grouped_data[largest_group] = grouped_data[largest_group][samples_to_move:]
        target_group = min(grouped_data, key=lambda k: len(grouped_data[k]))
        grouped_data[target_group].extend(samples)
    return grouped_data

def process_batch_dynamic(model, data, complexities, device):
    grouped_data = defaultdict(list)
    for i, complexity in enumerate(complexities.values()):
        grouped_data[complexity.item()].append(i)
    grouped_data = redistribute_complexity_groups(grouped_data)
    output = torch.zeros(data.size(0), model.output_layer.out_features).to(device)
    model_fn = partial(process_group, model)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = []
        for complexity, indices in grouped_data.items():
            subset_data = data.index_select(0, torch.tensor(indices, dtype=torch.long, device=device))
            results.append(pool.apply_async(model_fn, args=(subset_data, complexity)))
        for complexity, indices in grouped_data.items():
            subset_output = results.pop(0).get()
            output.index_copy_(0, torch.tensor(indices, dtype=torch.long, device=device), subset_output)
    return output
    