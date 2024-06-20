import time
from options.train_options import TrainOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
from test import run_test
from validate import run_validation


def train_one_epoch():
    epoch_start_time = time.time()
    epoch_iter = 0
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.optimize_parameters()
        epoch_iter += opt.batch_size
        writer.plot_loss(model.loss, epoch,  epoch_iter, len(dataset))
    training_time = time.time() - epoch_start_time
    return training_time


def test():
    start_time = time.time()
    accuracy = run_test()
    return accuracy, time.time() - start_time


def validate():
    start_time = time.time()
    accuracy = run_validation()
    return accuracy, time.time() - start_time


def init_writer():
    logging_header = ("Epoch", "Training Loss", "Test Accuracy", "Training Time", "Test Time", "Total Time", "              Learn Rate", "Model saved")
    writer = Writer(opt, logging_header)
    return writer


def log_epoch_data():
    learn_rate = model.optimizer.param_groups[0]['lr']
    data = (epoch, model.loss.item(), test_accuracy, training_time, test_time, total_time, learn_rate, best_model_saved)
    writer.log_epoch_data(data)


if __name__ == '__main__':
    train_options = TrainOptions()
    opt = train_options.parse()
    dataset = DataLoader(opt)
    model = create_model(opt)

    writer = init_writer()
    writer.log_options(train_options)
    writer.log_model_description(model)
    writer.log_headline()


    best_accuracy = 0.0
    num_epochs = opt.niter + opt.niter_decay
    for epoch in range(opt.epoch_count, num_epochs + 1):
        start_time = time.time()

        training_time = train_one_epoch()
        model.save_network('latest')

        if epoch % opt.run_test_freq == 0:
            test_accuracy, test_time = test()

            best_model_saved = False
            if test_accuracy > best_accuracy:
                best_model_saved = True
                model.save_network('best')
                best_accuracy = test_accuracy

            total_time = time.time() - start_time
            log_epoch_data()
            writer.plot_acc(test_accuracy, epoch)

        model.update_learning_rate()

    writer.close()

