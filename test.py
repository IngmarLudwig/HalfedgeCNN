import os

from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer


def run_test_or_val(phase):
    opt = TestOptions().parse()
    opt.serial_batches = True
    opt.phase = phase

    # Set number of augmentations to 1 for test and val, because we do not want to augment the test and val data.
    opt.number_augmentations = 1

    # The logic for the segmentation is different than for the classification, therefore a special check is necessary.
    if opt.phase == "val" and opt.dataset_mode == "segmentation" and not os.path.isdir("datasets/human_seg/val"):
        return "No Data"

    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)

    writer.reset_counter()

    if (opt.dataset_mode == "classification" and opt.print_labels):
        # Print names of wrongly classified images and their assigned class.
        cnt = 0
        for i, data in enumerate(dataset):
            model.set_input(data)
            results = model.forward()
            pred_class = results.data.max(1)[1]
            labels = data['label']
            for i in range(len(results)):
                if pred_class[i] != labels[i]:
                    print(cnt,"Label: ", labels[i],"Predicted Class:",pred_class[i].item())
                cnt += 1

    for i, data in enumerate(dataset):
        model.set_input(data)
        ncorrect, nexamples = model.test()
        writer.update_counter(ncorrect, nexamples)
    return writer.acc


# Test with test data.
def run_test():
    accuracy = run_test_or_val('test')
    return accuracy


if __name__ == '__main__':
    print('Running Test')
    accuracy = run_test()
    print('Test accuracy: {:.5} %'.format(accuracy * 100))

