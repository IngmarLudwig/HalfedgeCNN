import numbers

from test import run_test_or_val


# Test with validation data
def run_validation():
    accuracy = run_test_or_val('val')
    return accuracy

if __name__ == '__main__':
    print('Running Validation')
    accuracy = run_validation()
    if isinstance(accuracy, numbers.Number):
        print('Validation accuracy: {:.5} %'.format(accuracy * 100))
    else:
        print('No validation data found.')

