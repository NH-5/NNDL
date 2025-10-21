import os
from datetime import datetime, timezone, timedelta

def write_logs(Loss, Accuracy, epochs, batch_size, lr):
    utc_plus_8 = timezone(timedelta(hours=8))
    now_utc_8 = datetime.now(utc_plus_8)
    formatted_time = now_utc_8.strftime('%Y-%m-%d %H:%M:%S UTC+8')

    filename = f'{formatted_time}.txt'
    folderpath = 'logs/'
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    logs_path = os.path.join(folderpath, filename)

    with open(logs_path, 'a') as file:
        file.write(f"Training Time is {formatted_time}.\n")
        file.write(f"HyperParameter : batch_size {batch_size} lr {lr}\n")
        for epoch in range(epochs):
            file.write(
                f"Epoch {epoch} : Loss is {Loss[epoch]}, Accuracy is {Accuracy[epoch]}.\n"
            )

    