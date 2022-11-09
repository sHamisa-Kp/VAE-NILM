from typing import List

import os
import datetime
import argparse
import random
import logging
import json
import copy
import torch
import wandb

import torch.nn.functional as F
import tensorflow as tf
import tensorflow.keras.backend as K

from scipy import linalg

from dtw import *
from VAE_functions import *
from NILM_functions import *

'''wandb config'''
os.environ['WANDB_API_KEY'] = "b76283bc6c04e2ce6611147c4d328f71af8c71ba"

tf.compat.v1.disable_eager_execution()

wandb.init(project="fed_vae")

# ?
ADD_VAL_SET = False

logging.getLogger('tensorflow').disabled = True

###############################################################################
# Config
###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default=0, type=int, help="Appliance to learn")
parser.add_argument("--config", default="", type=str, help="Path to the config file")
parser.add_argument('--fl', default=False, action='store_true')
parser.add_argument('--agg', default='att', type=str)
parser.add_argument('--global_epoch', default=50, type=int)
parser.add_argument('--local_epoch', default=1, type=int)
parser.add_argument('--step_s', default=1.2, type=float)
parser.add_argument('--metric', default=2, type=int)
parser.add_argument('--dp', default=0.001, type=float)
a = parser.parse_args()

# Select GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

# Random seed
np.random.seed(123)

# FL parameters
fl_mode = a.fl
agg = a.agg
global_epochs = a.global_epoch
local_epochs = a.local_epoch
step_s = a.step_s
metric = a.metric
dp = a.dp

thr_house_2 = dict(Fridge=50, WashingMachine=20, Dishwasher=100, Kettle=100, Microwave=200)

print("###############################################################################")
print("NILM DISAGGREGATION")
print("GPU: {}".format(a.gpu))
print("Config: {}".format(a.config))
print("FL mode: {}".format(a.fl))
if fl_mode:
    print("Aggregation mode: {}".format(a.agg))
print("###############################################################################")

with open(a.config) as data_file:
    nilm = json.load(data_file)

name = "NILM_Disag_{}".format(nilm["appliance"])
time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

for r in range(1, nilm["run"] + 1):
    ###############################################################################
    # Load dataset
    ###############################################################################
    if fl_mode:
        clients = load_data(nilm["model"], nilm["appliance"], nilm["dataset"], nilm["preprocessing"]["width"],
                            nilm["preprocessing"]["strides"], nilm["training"]["batch_size"], fl_mode, set_type="train")
        x_val, y_val, x_test, y_test = load_data(nilm["model"], nilm["appliance"], nilm["dataset"],
                                                 nilm["preprocessing"]["width"],
                                                 nilm["preprocessing"]["strides"], nilm["training"]["batch_size"],
                                                 fl_mode,
                                                 set_type="test")
    else:
        x_train, y_train = load_data(nilm["model"], nilm["appliance"], nilm["dataset"], nilm["preprocessing"]["width"],
                                     nilm["preprocessing"]["strides"], nilm["training"]["batch_size"], fl_mode,
                                     set_type="train")
        x_test, y_test = load_data(nilm["model"], nilm["appliance"], nilm["dataset"], nilm["preprocessing"]["width"],
                                   nilm["preprocessing"]["strides"], nilm["training"]["batch_size"], fl_mode,
                                   set_type="test")

    main_mean = nilm["preprocessing"]["main_mean"]
    main_std = nilm["preprocessing"]["main_std"]

    app_mean = nilm["preprocessing"]["app_mean"]
    app_std = nilm["preprocessing"]["app_std"]

    width = nilm["preprocessing"]["width"]
    stride = nilm["preprocessing"]["strides"]

    ###############################################################################
    # Training parameters
    ###############################################################################
    epochs = nilm["training"]["epoch"]
    batch_size = nilm["training"]["batch_size"]

    if fl_mode:
        c_n = clients.keys()
        temp = [clients[c][0].shape[0] for c in c_n]
        total_ins = sum(temp)
        STEPS_PER_EPOCH = total_ins // batch_size
    else:
        STEPS_PER_EPOCH = x_train.shape[0] // batch_size

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        float(nilm["training"]["lr"]),
        decay_steps=STEPS_PER_EPOCH * nilm["training"]["decay_steps"],
        decay_rate=1,
        staircase=False)


    ###############################################################################
    # Optimizer
    ###############################################################################
    def get_optimizer(opt):
        if opt == "adam":
            return tf.keras.optimizers.Adam(lr_schedule)
        else:
            return tf.keras.optimizers.RMSprop(lr_schedule)


    ###############################################################################
    # Create and initialize the model
    ###############################################################################
    if not fl_mode:
        if nilm["model"] == "VAE":
            model = create_model(nilm["model"], nilm["config"], nilm["preprocessing"]["width"],
                                 optimizer=get_optimizer(nilm["training"]["optimizer"]))
        elif nilm["model"] == "DAE":
            model = create_model(nilm["model"], nilm["config"], nilm["preprocessing"]["width"], optimizer="Adam")
        elif nilm["model"] == "S2P":
            model = create_model(nilm["model"], nilm["config"], nilm["preprocessing"]["width"],
                                 optimizer=tf.keras.optimizers.Adam(learning_rate=nilm["training"]["lr"], beta_1=0.9,
                                                                    beta_2=0.999))
        elif nilm["model"] == "S2S":
            model = create_model(nilm["model"], nilm["config"], nilm["preprocessing"]["width"],
                                 optimizer=tf.keras.optimizers.Adam(learning_rate=nilm["training"]["lr"], beta_1=0.9,
                                                                    beta_2=0.999))
    else:
        pass

    ###############################################################################
    # Callback checkpoint settings
    ###############################################################################
    list_callbacks = []

    # Create a callback that saves the model's weights
    if nilm["training"]["save_best"] == 1:
        checkpoint_path = "{}/{}/{}/logs/model/House_{}/{}/{}".format(name, nilm["dataset"]["name"], nilm["model"],
                                                                      nilm["dataset"]["test"]["house"][0], time,
                                                                      r) + "/checkpoint.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=0,
                                                         monitor="val_mean_absolute_error",
                                                         mode="min",
                                                         save_best_only=True)
    else:
        checkpoint_path = "{}/{}/{}/logs/model/House_{}/{}/{}".format(name, nilm["dataset"]["name"], nilm["model"],
                                                                      nilm["dataset"]["test"]["house"][0], time,
                                                                      r) + "/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=0,
                                                         period=1)

    # todo
    # list_callbacks.append(cp_callback)

    if nilm["training"]["patience"] > 0:
        patience = nilm["training"]["patience"]
        start_epoch = nilm["training"]["start_stopping"]

        print("Patience : {}, Start at : {}".format(patience, start_epoch))

        es_callback = CustomStopper(monitor='val_loss', patience=patience, start_epoch=start_epoch, mode="auto")

        # todo
        # list_callbacks.append(es_callback)

    ###############################################################################
    # Normalize Test Data and History Callback
    ###############################################################################
    if ADD_VAL_SET:
        if nilm["dataset"]["name"] == "ukdale":
            if nilm["model"] == "S2P":
                x_test_s2p, y_test_s2p = transform_s2p(x_test, y_test, nilm["preprocessing"]["width"],
                                                       nilm["training"]["S2P_strides"])
                history_cb = AdditionalValidationSets(
                    [((x_test_s2p - main_mean) / main_std, (y_test_s2p - app_mean) / app_std, 'House_2')], verbose=1)
            else:
                history_cb = AdditionalValidationSets(
                    [((x_test - main_mean) / main_std, (y_test - app_mean) / app_std, 'House_2')], verbose=1)

        # ?
        elif nilm["dataset"]["name"] == "house_2":
            history_cb = AdditionalValidationSets([(x_test, y_test, 'House_2')], verbose=1)
        elif nilm["dataset"]["name"] == "refit":
            history_cb = AdditionalValidationSets([(x_test, y_test, 'House_2')], verbose=1)

        # todo
        # list_callbacks.append(history_cb)

    ###############################################################################
    # Summary of all parameters
    ###############################################################################
    print("###############################################################################")
    print("Summary")
    print("###############################################################################")
    print("{}".format(nilm))
    print("Run number : {}/{}".format(r, nilm["run"]))
    print("###############################################################################")

    # if not os.path.exists("{}/{}/{}/logs/model/House_{}/{}".format(name, nilm["dataset"]["name"], nilm["model"],
    #                                                                nilm["dataset"]["test"]["house"][0], time)):
    #     os.makedirs("{}/{}/{}/logs/model/House_{}/{}".format(name, nilm["dataset"]["name"], nilm["model"],
    #                                                          nilm["dataset"]["test"]["house"][0], time))
    #
    # with open("{}/{}/{}/logs/model/House_{}/{}/config.txt".format(name, nilm["dataset"]["name"], nilm["model"],
    #                                                               nilm["dataset"]["test"]["house"][0], time),
    #           "w") as outfile:
    #     json.dump(nilm, outfile)

    ###############################################################################
    # Train Model
    ###############################################################################
    if not fl_mode:
        if nilm["dataset"]["name"] == "ukdale":
            ###############################################################################
            # Real Validation
            ###############################################################################
            if nilm["model"] == "S2P":
                x_train_s2p, y_train_s2p = transform_s2p(x_train, y_train, nilm["preprocessing"]["width"],
                                                         nilm["training"]["S2P_strides"])

                history = model.fit((x_train_s2p - main_mean) / main_std, (y_train_s2p - app_mean) / app_std,
                                    validation_split=nilm["training"]["validation_split"], shuffle=True,
                                    epochs=epochs, batch_size=batch_size, callbacks=list_callbacks, verbose=1,
                                    initial_epoch=0)

            elif nilm["model"] == "VAE":
                # x_train = (x_train-main_mean)/main_std
                # y_train = (y_train-app_mean)/app_std
                history = model.fit((x_train - main_mean) / main_std, (y_train - app_mean) / app_std,
                                    validation_split=nilm["training"]["validation_split"], shuffle=True,
                                    epochs=epochs, batch_size=batch_size, callbacks=list_callbacks, verbose=1,
                                    initial_epoch=0)

            elif nilm["model"] == "S2S":
                history = model.fit((x_train - main_mean) / main_std, (y_train - app_mean) / app_std,
                                    validation_split=nilm["training"]["validation_split"], shuffle=True,
                                    epochs=epochs, batch_size=batch_size, callbacks=list_callbacks, verbose=1,
                                    initial_epoch=0)

            elif nilm["model"] == "DAE":
                history = model.fit((x_train - main_mean) / main_std, (y_train - app_mean) / app_std,
                                    validation_split=nilm["training"]["validation_split"], shuffle=True,
                                    epochs=epochs, batch_size=batch_size, callbacks=list_callbacks, verbose=1,
                                    initial_epoch=0)

            ###############################################################################
            # Save history
            ###############################################################################
            # np.save(
            #     "{}/{}/{}/logs/model/House_{}/{}/{}/history.npy".format(name, nilm["dataset"]["name"], nilm["model"],
            #                                                             nilm["dataset"]["test"]["house"][0], time, r),
            #     history.history)
            # np.save("{}/{}/{}/logs/model/{}/{}/history_cb_{}.npy".format(name, nilm["dataset"]["name"],
            # nilm["model"], time, r, epochs), history_cb.history)

            print("Fit finished!")
        else:
            print("Error in dataset name!")
    else:
        decay_steps = STEPS_PER_EPOCH * nilm["training"]["decay_steps"]

        client_names = random.sample(clients.keys(), len(clients.keys()))
        # initialize global model
        global_model = create_model(nilm["model"], nilm["config"], nilm["preprocessing"]["width"],
                                    optimizer=tf.keras.optimizers.RMSprop(0.001))
        for epoch in range(global_epochs):
            # x_total = []
            # y_total_pred = []
            # y_total_true = []
            global_w = global_model.get_weights()
            global_w_dict = {f'layer_{i}': torch.from_numpy(elem) for i, elem in enumerate(global_w)}
            local_weights_dict: List[dict] = []
            random.shuffle(client_names)

            for client in client_names:
                step = epoch
                local_opt = tf.keras.optimizers.Adam(float(nilm["training"]["lr"]) / (1 + 1 * step / decay_steps))
                local_model = create_model(nilm["model"], nilm["config"], nilm["preprocessing"]["width"],
                                           optimizer=local_opt)

                local_model.set_weights(global_w)
                local_model.fit((clients[client][0] - main_mean) / main_std, (clients[client][1] - app_mean) / app_std,
                                validation_split=nilm["training"]["validation_split"], shuffle=True,
                                epochs=local_epochs, batch_size=batch_size, callbacks=list_callbacks, verbose=1,
                                initial_epoch=0)
                local_weights_dict.append(
                    {f'layer_{i}': torch.from_numpy(elem) for i, elem in enumerate(local_model.get_weights())})
                # K.clear_session()

            if agg == "att":
                global_w_tmp = aggregate_att(local_weights_dict, global_w_dict, step_s, metric, dp)
            else:
                global_w_tmp = average_weights(local_weights_dict, dp)

            global_w = [i.numpy() for i in global_w_tmp.values()]
            global_model.set_weights(global_w)

            print(f"Validation started...")
            y_pred = global_model.predict([(x_val-main_mean)/main_std], verbose=1)
            y_all_pred = reconstruct(y_pred[:]*app_std+app_mean, width, stride, "median")
            x_all = reconstruct(x_val[:], width, stride, "median")
            y_all_true = reconstruct(y_val[:], width, stride, "median")
            y_all_pred[y_all_pred < 15] = 0

            x_all = x_all.reshape([1, -1])
            y_all_pred = y_all_pred.reshape([1, -1])
            y_all_true = y_all_true.reshape([1, -1])

            # for i in range(x_all.shape[-1]):
            #     x_total.append(x_all[0, i])
            # for i in range(y_all_pred.shape[-1]):
            #     y_total_pred.append(y_all_pred[0, i])
            # for i in range(y_all_true.shape[-1]):
            #     y_total_true.append(y_all_true[0, i])
            #
            # del x_all
            # del y_all_pred
            # del y_all_true

            MAE_tot, MAE_app, MAE = MAE_metric(y_all_pred, y_all_true, disaggregation=True, only_power_on=False)
            acc_P_tot, acc_P_app, acc_P = acc_Power(y_all_pred, y_all_true, disaggregation=True)
            PR_app = PR_metric(y_all_pred, y_all_true, thr=thr_house_2[nilm["appliance"]])
            RE_app = RE_metric(y_all_pred, y_all_true, thr=thr_house_2[nilm["appliance"]])
            F1_app = F1_metric(y_all_pred, y_all_true, thr=thr_house_2[nilm["appliance"]])
            SAE_app = SAE_metric(y_all_pred, y_all_true)
            RETE = relative_error_total_energy(y_all_pred, y_all_true)

            print(f"MAE total: {MAE_tot} | MAE app: {MAE_app}")
            print(f"Accuracy total: {acc_P_tot} | Accuracy app: {acc_P_app}")
            print(f"Precision: {PR_app[0]}")
            print(f"Recall: {RE_app[0]}:")
            print(f"F1: {F1_app[0]}")
            print(f"SAE: {SAE_app[0]}")
            print(f"RETE: {RETE}")

            new_row = {' MAE': MAE_app, 'Accuracy': acc_P_app, 'Precision': PR_app[0], 'Recall': RE_app[0],
                       'F1': F1_app[0], 'SAE': SAE_app[0], 'RETE': RETE}
            wandb.log(new_row)


