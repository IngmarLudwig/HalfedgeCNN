import logging
from random import randint

try:
    from tensorboardX import SummaryWriter
except ImportError as error:
    print('Tensorboard X not installed, visualizing wont be available')
    SummaryWriter = None


class Writer:
    def __init__(self, opt, logging_header =()):
        self.opt = opt
        self.nexamples = 0
        self.ncorrect = 0
        self.logfile = None

        if opt.is_train and not opt.no_vis and SummaryWriter is not None:
            settings_str = self.create_run_settings_str()
            self.display = SummaryWriter(comment=settings_str)

            logdir = self.display._get_file_writer().get_logdir()
            file = logdir + '/run.log'
            self.logfile = file
            logging.basicConfig(filename=file, encoding='utf-8',level=logging.INFO)
        else:
            self.display = None

        self.logging_header = logging_header


    def create_run_settings_str(self):
        id = randint(0, 10000000)
        name = "_"
        name += self.opt.name
        name += "_"
        name += "F" + str(self.opt.feat_selection)
        name += "N" + str(self.opt.nbh_size)
        name += "_"
        name += str(self.opt.pooling)
        name += "_"
        name += "id" + str(id)
        return name


    def log_options(self, opt):
        log_str = opt.formatted_str()
        print(log_str)
        if self.logfile is not None:
            logging.info(log_str)


    def log_headline(self):
        headline = "| "
        for header in self.logging_header:
            headline += header + " | "

        bar = ""
        for _ in range(len(headline)):
            bar += "-"

        print()
        print(headline)
        print(bar)

        if self.logfile is not None:
            logging.info("")
            logging.info(headline)
            logging.info(bar)


    def log_model_description(self, model):
        model_description = model.get_description()
        print(model_description)
        if self.logfile is not None:
            logging.info(model_description)


    # Deals with the case that there is no validation data leading to "No Data" the accuracy value.
    def handle_accuracy(self, accuracy):
        try:
            accuracy = float(accuracy)
            accuracy = '{:.3} %'.format(accuracy * 100.0)
        except ValueError:
            accuracy = 'No Val Data'
        return accuracy


    # "data_list” should be in same order as "logging_header".
    def log_epoch_data(self, data):
        msg = "|"
        epoch = 0
        for i in range(len(self.logging_header)):
            header = self.logging_header[i]
            date = str(data[i])
            if "Accuracy" in header:
                date = self.handle_accuracy(date)
            elif "Time" in header:
                date = '%.2f s' % data[i]
            elif "Loss" in header:
                date = '%.5f' % data[i]
            date +=" |"
            date = date.rjust(len(self.logging_header[i])+3)
            msg += date
        print(msg)
        if self.logfile is not None:
            logging.info(msg)


    def plot_acc(self, acc, epoch):
        if self.display:
            self.display.add_scalar('data/test_acc', acc, epoch)


    def plot_loss(self, loss, epoch, i, n):
        iters = i + (epoch - 1) * n
        if self.display:
            self.display.add_scalar('data/train_loss', loss, iters)


    def plot_model_wts(self, model, epoch):
        if self.opt.is_train and self.display:
            for name, param in model.net.named_parameters():
                self.display.add_histogram(name, param.clone().cpu().data.numpy(), epoch)


    def reset_counter(self):
        self.ncorrect = 0
        self.nexamples = 0

    def update_counter(self, ncorrect, nexamples):
        self.ncorrect += ncorrect
        self.nexamples += nexamples


    @property
    def acc(self):
        if self.nexamples == 0:
            return "No Data"
        return float(self.ncorrect) / self.nexamples

    def close(self):
        if self.display is not None:
            self.display.close()
