from data_provider.data_loader import (
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom,
    Dataset_Solar,
    Dataset_PEMS,
    Dataset_Pred,
)
from torch.utils.data import DataLoader

data_dict = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "Solar": Dataset_Solar,
    "PEMS": Dataset_PEMS,
    "custom": Dataset_Custom,
}


def data_provider(args, flag):
    Data = Dataset_Pred if flag == "pred" else data_dict[args.data]

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=1 if args.embed == "timeF" else 0,
        freq=args.freq,
    )
    print(flag, len(data_set))

    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=True if flag == "train" else False,
        num_workers=args.num_workers,
    )

    return data_set, data_loader
