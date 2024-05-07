import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import numpy as np

if __name__ == "__main__":
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="Mamba")

    # basic config
    parser.add_argument(
        "--is_training", type=int, required=True, default=1, help="status"
    )
    parser.add_argument(
        "--model_id", type=str, required=True, default="test", help="model id"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="TimesMamba",
        help="model name, options: [TimesMamba]",
    )

    # data loader
    parser.add_argument(
        "--data", type=str, required=True, default="custom", help="dataset type"
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/electricity/",
        help="root path of the data file",
    )
    parser.add_argument(
        "--data_path", type=str, default="electricity.csv", help="data csv file"
    )
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
    )
    parser.add_argument(
        "--target", type=str, default="OT", help="target feature in S or MS task"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument(
        "--label_len", type=int, default=48, help="start token length"
    )  # no longer needed in inverted Transformers
    parser.add_argument(
        "--pred_len", type=int, default=96, help="prediction sequence length"
    )

    # model define
    parser.add_argument(
        "--no_norm", dest="use_norm", action="store_false", help="use norm and denorm"
    )
    parser.add_argument("--revin_affine", action="store_true", help="RevIN affine")
    parser.add_argument("--use_mark", action="store_true", help="use timestamp feature")
    parser.add_argument(
        "--channel_independence",
        action="store_true",
        help="whether to use channel_independence mechanism",
    )
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument(
        "--enc_m_in", type=int, default=4, help="encoder data marker input size"
    )
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument(
        "--c_out", type=int, default=7, help="output size"
    )  # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--ssm_expand", type=int, default=1, help="expand factor")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument(
        "--r_ff", type=int, default=4, help="ratio ffn hidden dimension / d_model"
    )
    parser.add_argument(
        "--moving_avg", type=int, default=25, help="window size of moving average"
    )
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument(
        "--nodistil",
        dest="distil",
        action="store_false",
        help="whether to use distilling in encoder, using this argument means not using distilling",
        default=True,
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="whether to output attention in ecoder",
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="whether to predict unseen future data",
    )

    # optimization
    parser.add_argument(
        "--num_workers", type=int, default=10, help="data loader num workers"
    )
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of train input data"
    )
    parser.add_argument("--warmup_epochs", type=int, default=0, help="warmup epochs")
    parser.add_argument(
        "--patience", type=int, default=-1, help="early stopping patience"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument(
        "--lradj", type=str, default="type1", help="adjust learning rate"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument(
        "--device", type=str, default="0", help="device ids of multile gpus"
    )

    # experiment
    parser.add_argument(
        "--exp_name",
        type=str,
        required=False,
        default="MTSF",
        help="experiemnt name, options:[MTSF, partial_train]",
    )
    parser.add_argument("--inverse", action="store_true", help="inverse output data")
    parser.add_argument(
        "--class_strategy",
        type=str,
        default="projection",
        help="projection/average/cls_token",
    )
    parser.add_argument(
        "--efficient_training",
        type=bool,
        default=False,
        help="whether to use efficient_training (exp_name should be partial train)",
    )
    parser.add_argument(
        "--partial_start_index",
        type=int,
        default=0,
        help="the start index of variates for partial training, "
        "you can select [partial_start_index, min(enc_in + partial_start_index, N)]",
    )

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print("Args in experiment:")
    print(args)

    # MTSF: multivariate time series forecasting
    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = f"{args.model_id}_{args.model}_{args.data}_{ii}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.r_ff}_fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}_{args.class_strategy}"

            exp = Exp(args)  # set experiments
            print(f">>>>>>>start training: {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.train(setting)

            print(f">>>>>>>testing: {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.test(setting)

            if args.do_predict:
                print(f">>>>>>>predicting: {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = f"{args.model_id}_{args.model}_{args.data}_{ii}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.r_ff}_fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}_{args.class_strategy}"

        exp = Exp(args)  # set experiments
        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
